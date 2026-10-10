#!/usr/bin/env python3
"""Non-confirmatory transport runner for Candidate 3 grid identification.

The design lives in spec/candidate3_grid_pilot.json. A real run is impossible while that file is
not frozen and hash-bound. The runner returns only fixed-size per-step bin accumulators; no array
whose leading dimension is the photon count leaves the device.

This program never computes equivalence, coverage or a Candidate 3 verdict.

    <python> -W error scripts/candidate3_grid_pilot.py --selftest
    <python> -W error scripts/candidate3_grid_pilot.py --freeze-status
    <python> -W error scripts/candidate3_grid_preflight.py --run --jaxpad-root PATH
    <python> -W error scripts/candidate3_grid_pilot.py --run --jaxpad-root PATH --out PATH
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PILOT_SPEC = os.path.join(ROOT, "spec", "candidate3_grid_pilot.json")
BASELINE_SPEC = os.path.join(ROOT, "spec", "baseline_spec.json")
NUISANCE_SPEC = os.path.join(ROOT, "spec", "nuisance_spec.json")
PLAN_DOC = os.path.join(ROOT, "docs", "planning", "CANDIDATE3_GRID_PILOT_PLAN.md")
ANALYSIS = os.path.join(ROOT, "scripts", "candidate3_grid_analysis.py")
AUDIT = os.path.join(ROOT, "scripts", "candidate3_grid_pilot_audit.py")
PREFLIGHT = os.path.join(ROOT, "scripts", "candidate3_grid_preflight.py")
PREFLIGHT_SBATCH = os.path.join(ROOT, "scripts", "candidate3_grid_preflight_fir.sbatch")
PILOT_SBATCH = os.path.join(ROOT, "scripts", "candidate3_grid_pilot_fir.sbatch")
_SPEC_OVERRIDE_DATA = None

OUTPUT_KEYS = ("sum_plus", "sum_minus", "sum_Y", "sum_Y2", "max_abs_Y")
DETERMINISTIC_FLAG = "xla_gpu_deterministic_ops=true"


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def canonical_digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def array_digest(*arrays):
    h = hashlib.sha256()
    for value in arrays:
        a = np.ascontiguousarray(np.asarray(value))
        h.update(str(a.dtype).encode("ascii"))
        h.update(str(a.shape).encode("ascii"))
        h.update(a.tobytes())
    return h.hexdigest()


def _normative_local_paths():
    return {
        "docs/planning/CANDIDATE3_GRID_PILOT_PLAN.md": PLAN_DOC,
        "scripts/candidate3_grid_analysis.py": ANALYSIS,
        "scripts/candidate3_grid_pilot_audit.py": AUDIT,
        "scripts/candidate3_grid_preflight.py": PREFLIGHT,
        "scripts/candidate3_grid_preflight_fir.sbatch": PREFLIGHT_SBATCH,
        "scripts/candidate3_grid_pilot_fir.sbatch": PILOT_SBATCH,
        "scripts/candidate3_executor.py": os.path.join(ROOT, "scripts",
                                                        "candidate3_executor.py"),
        "scripts/candidate3_grid_pilot.py": os.path.abspath(__file__),
        "scripts/nuisance_feasibility_gate.py": os.path.join(
            ROOT, "scripts", "nuisance_feasibility_gate.py"),
        "spec/baseline_spec.json": BASELINE_SPEC,
        "spec/nuisance_spec.json": NUISANCE_SPEC,
        "src/augmented.py": os.path.join(ROOT, "src", "augmented.py"),
    }


def freeze_material(spec, jaxpad_root):
    """Compute, but never write, the material that an independent audit would freeze."""
    local = {k: sha256(v) for k, v in _normative_local_paths().items()}
    jp = {
        "jaxpad:scripts/archive/inverse/multi_wl_v12_colab.py": os.path.join(
            jaxpad_root, "scripts", "archive", "inverse", "multi_wl_v12_colab.py"),
        "jaxpad:scripts/inverse/phase2_v12_runner.py": os.path.join(
            jaxpad_root, "scripts", "inverse", "phase2_v12_runner.py"),
        "jaxpad:src/pad/config_v4_633nm.py": os.path.join(
            jaxpad_root, "src", "pad", "config_v4_633nm.py"),
        "jaxpad:src/pad/mc_forward.py": os.path.join(jaxpad_root, "src", "pad", "mc_forward.py"),
        "jaxpad:src/pad/physics_engine_v4.py": os.path.join(
            jaxpad_root, "src", "pad", "physics_engine_v4.py"),
    }
    hashes = dict(local)
    for name, path in jp.items():
        hashes[name] = sha256(path) if os.path.exists(path) else f"unavailable:{path}"
    payload = json.loads(json.dumps(spec))
    payload["frozen"] = True
    payload["status"] = "FROZEN NON-CONFIRMATORY GRID-IDENTIFICATION PILOT"
    payload["freeze_hash"] = None
    payload["normative_input_hashes"] = hashes
    digest = hashlib.sha256(b"candidate3-grid-pilot-v1\0" + canonical_bytes(payload)).hexdigest()
    return {"normative_input_hashes": hashes, "freeze_hash": digest, "payload": payload}


def freeze_status(spec, jaxpad_root):
    mat = freeze_material(spec, jaxpad_root)
    problems = []
    if spec.get("frozen") is not True:
        problems.append("spec.frozen is not true")
    if spec.get("freeze_hash") != mat["freeze_hash"]:
        problems.append("freeze_hash is absent or does not match current normative inputs")
    if spec.get("normative_input_hashes") != mat["normative_input_hashes"]:
        problems.append("normative_input_hashes are absent or stale")
    if any(str(v).startswith("unavailable:") for v in mat["normative_input_hashes"].values()):
        problems.append("one or more normative jax-pad files are unavailable")
    return {"frozen": not problems, "problems": problems,
            "computed_freeze_hash": mat["freeze_hash"],
            "computed_input_hashes": mat["normative_input_hashes"]}


def _make_accumulator(kernel, n_steps, max_steps):
    """Create one six-step jitted accumulator around the production photon kernel."""
    if not callable(kernel):
        raise SystemExit("REFUSING: kernel must be callable")
    if isinstance(n_steps, (bool, np.bool_)) or not isinstance(n_steps, (int, np.integer)):
        raise SystemExit("REFUSING: n_steps must be an integer, never bool or float")
    if isinstance(max_steps, (bool, np.bool_)) or not isinstance(max_steps, (int, np.integer)):
        raise SystemExit("REFUSING: max_steps must be an integer, never bool or float")
    L, K = int(n_steps), int(max_steps)
    if L <= 0 or K <= 0:
        raise SystemExit("REFUSING: n_steps and max_steps must be positive")

    import jax
    import jax.numpy as jnp

    def body(keys, bounds_p, bounds_m, muss_p, muss_m, muas, gs, ns, edges, h, areas):
        trace = jax.vmap(kernel, in_axes=(0, None, None, None, None, None, None, None))
        M = areas.shape[0]
        sp, sm, sy, sy2, my = [], [], [], [], []

        def fold_pair(b, v, ok):
            same = (b[:, None] == b[None, :]) & ok[:, None] & ok[None, :]
            combined = (same * v[None, :]).sum(axis=1)
            first = jnp.argmax(same, axis=1) == jnp.arange(2)
            keep = ok & first
            return combined * keep, keep

        for l in range(L):
            dp, wp = trace(keys, muas, muss_p[l], gs, ns, bounds_p[l], edges, K)
            dm, wm = trace(keys, muas, muss_m[l], gs, ns, bounds_m[l], edges, K)

            op, om = dp >= 0, dm >= 0
            bp = jnp.where(op, dp, 0).astype(jnp.int32)
            bm = jnp.where(om, dm, 0).astype(jnp.int32)
            zp = jnp.zeros((M,), dtype=wp.dtype).at[bp].add(jnp.where(op, wp, 0.0))
            zm = jnp.zeros((M,), dtype=wm.dtype).at[bm].add(jnp.where(om, wm, 0.0))

            bins = jnp.stack([bp, bm], axis=1)
            ok = jnp.stack([op, om], axis=1)
            vals = jnp.where(
                ok, jnp.stack([wp, -wm], axis=1) / (2.0 * h[l] * areas[bins]), 0.0)
            combined, keep = jax.vmap(fold_pair)(bins, vals, ok)
            kept = jnp.where(keep, combined, 0.0)
            z = jnp.zeros((M,), dtype=wp.dtype)
            sp.append(zp)
            sm.append(zm)
            sy.append(z.at[bins].add(kept))
            sy2.append(z.at[bins].add(jnp.where(keep, combined * combined, 0.0)))
            my.append(z.at[bins].max(jnp.where(keep, jnp.abs(combined), 0.0)))
        return {"sum_plus": jnp.stack(sp), "sum_minus": jnp.stack(sm),
                "sum_Y": jnp.stack(sy), "sum_Y2": jnp.stack(sy2),
                "max_abs_Y": jnp.stack(my)}

    return jax.jit(body)


def _validate_and_run(accumulator, *, keys, n_photons, bounds_plus, bounds_minus,
                      muss_plus, muss_minus, muas, gs, ns, edges, h_grid, areas):
    import jax.numpy as jnp

    h, A, E = np.asarray(h_grid, float), np.asarray(areas, float), np.asarray(edges, float)
    L = h.size
    kshape, kdtype = getattr(keys, "shape", None), getattr(keys, "dtype", None)
    if kshape is None or kdtype is None or tuple(kshape) != (int(n_photons), 2):
        raise SystemExit(f"REFUSING: keys must have shape ({int(n_photons)}, 2); got {kshape}")
    if np.dtype(kdtype) != np.uint32:
        raise SystemExit(f"REFUSING: keys must have dtype uint32; got {kdtype}")
    if L < 4 or not np.all(np.isfinite(h)) or not np.all(h > 0) or len(np.unique(h)) != L:
        raise SystemExit(
            "REFUSING: h_grid must contain at least four distinct finite positive steps")
    arrays = {"bounds_plus": bounds_plus, "bounds_minus": bounds_minus,
              "muss_plus": muss_plus, "muss_minus": muss_minus}
    n_layers = np.asarray(muas).size
    for name, value in arrays.items():
        a = np.asarray(value, float)
        if a.shape != (L, n_layers) or not np.all(np.isfinite(a)):
            raise SystemExit(f"REFUSING: {name} must be finite with shape {(L, n_layers)}")
    for name in ("bounds_plus", "bounds_minus"):
        a = np.asarray(arrays[name], float)
        if np.any(a <= 0) or np.any(np.diff(a, axis=1) <= 0):
            raise SystemExit(f"REFUSING: {name} contains a non-positive or non-increasing stack")
    for name in ("muss_plus", "muss_minus"):
        if np.any(np.asarray(arrays[name], float) < 0):
            raise SystemExit(f"REFUSING: {name} contains negative scattering")
    if (A.ndim != 1 or np.any(~np.isfinite(A)) or np.any(A <= 0)
            or E.shape != (A.size + 1,) or np.any(np.diff(E) <= 0)):
        raise SystemExit("REFUSING: detector areas/edges are invalid")
    if (np.asarray(muas).shape != np.asarray(gs).shape
            or np.asarray(muas).shape != np.asarray(ns).shape):
        raise SystemExit("REFUSING: muas, gs and ns shapes differ")
    if (np.any(~np.isfinite(muas)) or np.any(np.asarray(muas) < 0)
            or np.any(~np.isfinite(gs)) or np.any(np.abs(gs) > 1)
            or np.any(~np.isfinite(ns)) or np.any(np.asarray(ns) < 1)):
        raise SystemExit("REFUSING: optical coefficients are outside their numeric domains")

    out = accumulator(keys, jnp.asarray(bounds_plus), jnp.asarray(bounds_minus),
                      jnp.asarray(muss_plus), jnp.asarray(muss_minus), jnp.asarray(muas),
                      jnp.asarray(gs), jnp.asarray(ns), jnp.asarray(edges), jnp.asarray(h),
                      jnp.asarray(A))
    host = {k: np.asarray(v) for k, v in out.items()}
    if set(host) != set(OUTPUT_KEYS):
        raise SystemExit(f"REFUSING: accumulator keys are {sorted(host)}, expected {OUTPUT_KEYS}")
    for name, value in host.items():
        if value.shape != (L, A.size) or not np.all(np.isfinite(value)):
            raise SystemExit(f"REFUSING: {name} has invalid shape or non-finite values")
    B = 1.0 / (2.0 * h[:, None] * A[None, :])
    scale = np.maximum(B, 1.0)
    if np.any(host["max_abs_Y"] > B + 64.0 * np.finfo(float).eps * scale):
        raise SystemExit("REFUSING: observed max_abs_Y exceeds the deterministic absolute bound")
    if np.any(host["sum_Y2"] < -64.0 * np.finfo(float).eps):
        raise SystemExit("REFUSING: sum_Y2 is negative")
    return host


def build_pilot_executor(jaxpad_root, design, nspec, bspec):
    """Build the production-symbol six-step accumulator. No transport runs during construction."""
    import jax
    import jax.numpy as jnp

    sys.path.insert(0, os.path.abspath(jaxpad_root))
    sys.path.insert(0, HERE)
    jax.config.update("jax_enable_x64", True)
    import nuisance_feasibility_gate as gate
    import candidate3_executor as c3
    from scripts.inverse.phase2_v12_runner import (Phase2RunConfig, scattering_for_transport,
                                                   _build_mu)
    import scripts.archive.inverse.multi_wl_v12_colab as v12
    from src.pad import mc_forward as mcf
    from src.pad import physics_engine_v4 as pe

    if mcf.simulate_one_photon is not pe.simulate_one_photon_rigorous:
        raise SystemExit("REFUSING: the pilot kernel is not the production mc_forward symbol")
    rd = design["run_design"]
    if list(map(float, rd["wavelengths_nm"])) != list(map(float, bspec["wavelengths_nm"])):
        raise SystemExit("REFUSING: pilot wavelengths differ from the production baseline")
    names = gate.eta_names(nspec)
    if rd["columns"] != names:
        raise SystemExit("REFUSING: pilot column order differs from nuisance_spec")
    prof = bspec["profiles"]["production"]
    radial, max_steps = prof["radial"], int(prof["max_steps"])
    n_photons = int(rd["n_photons_per_wavelength"])
    labels = bspec["parameters"]["psi_labels"]
    psi0 = np.array([bspec["parameters"]["nominal_psi"][k] for k in labels], float)
    kappa = bspec["blood_coordinate"]["legacy_to_physical_divisor"]
    cfg = Phase2RunConfig(n_photons_final=n_photons, seed_keys=0, seed_noise=1,
                          sto2_ubnp_true=psi0[0], sto2_dbnp_true=psi0[2],
                          b_ubnp_true=psi0[1] * kappa, b_dbnp_true=psi0[3] * kappa,
                          max_steps=max_steps)
    n_wl, n_eta = int(v12.N_WL), len(names)
    edges = jnp.linspace(radial["r_min_cm"], radial["r_max_cm"], radial["n_bins"] + 1,
                         dtype=jnp.float64)
    areas = jnp.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    mus_nom = [jnp.asarray(scattering_for_transport(v12.MUSS_WL[k], v12.GS_WL[k],
                                                    cfg.scattering_convention),
                           dtype=jnp.float64) for k in range(n_wl)]
    nom_b = jnp.asarray(gate.NOMINAL_BOUNDARIES_CM, dtype=jnp.float64)
    wl_j = jnp.asarray(bspec["wavelengths_nm"], dtype=jnp.float64)
    c = jnp.stack([kappa * psi0[1] * psi0[0], kappa * psi0[1] * (1.0 - psi0[0]),
                   kappa * psi0[3] * psi0[2], kappa * psi0[3] * (1.0 - psi0[2]),
                   psi0[4], psi0[5]])
    mu_list, _, _ = _build_mu(jnp.log(c + v12.EPS))

    def boundaries_jnp(eta):
        t = jnp.diff(jnp.concatenate([jnp.zeros(1, dtype=nom_b.dtype), nom_b]))
        s = jnp.exp(jnp.stack([eta[0], eta[1], eta[2]]))
        t = t.at[0].multiply(s[0]).at[1].multiply(s[0])
        t = t.at[2].multiply(s[1]).at[4].multiply(s[2])
        return jnp.cumsum(t)

    def mus_factor_jnp(eta):
        return jnp.exp(eta[3]) * (wl_j / gate.LAM0_NM) ** (-eta[4])

    h_grid = tuple(float(x) for x in rd["h_candidates"])
    accumulator = _make_accumulator(pe.simulate_one_photon_rigorous, len(h_grid), max_steps)

    def accumulate(eta_index, wavelength_index, keys):
        if not c3._is_index(eta_index) or not 0 <= int(eta_index) < n_eta:
            raise SystemExit(f"REFUSING: invalid eta_index {eta_index!r}")
        if not c3._is_index(wavelength_index) or not 0 <= int(wavelength_index) < n_wl:
            raise SystemExit(f"REFUSING: invalid wavelength_index {wavelength_index!r}")
        for h in h_grid:
            for sign in (+1.0, -1.0):
                eta = np.zeros(n_eta)
                eta[int(eta_index)] = sign * h
                chk = gate.check_point(eta, names, bspec["wavelengths_nm"])
                if not chk["feasible"]:
                    raise SystemExit(f"REFUSING: infeasible pilot perturbation: {chk['reasons']}")
        bp, bm, sp, sm = c3.column_inputs(
            jnp, eta_index=eta_index, h_grid=h_grid, n_eta=n_eta,
            boundaries_fn=boundaries_jnp,
            muss_fn=lambda e, k: mus_nom[k] * mus_factor_jnp(e)[k],
            wavelength_index=wavelength_index)
        t0 = time.time()
        out = _validate_and_run(
            accumulator, keys=keys, n_photons=n_photons,
            bounds_plus=bp, bounds_minus=bm, muss_plus=sp, muss_minus=sm,
            muas=mu_list[wavelength_index], gs=v12.GS_WL[wavelength_index],
            ns=v12.NS_WL[wavelength_index], edges=edges, h_grid=h_grid, areas=areas)
        return out, time.time() - t0, array_digest(bp, bm, sp, sm)

    return {"accumulate": accumulate,
            "split": lambda seed: jax.random.split(jax.random.PRNGKey(seed), n_photons),
            "backend": jax.default_backend(), "jax_version": jax.__version__,
            "areas": np.asarray(areas), "h_grid": h_grid,
            "n_wavelengths": n_wl, "n_eta": n_eta, "max_steps": max_steps}


def run_replicate(executor, design, replicate_index):
    rd = design["run_design"]
    r = int(replicate_index)
    if r < 0 or r >= len(rd["replicate_seeds"]):
        raise SystemExit(f"REFUSING: replicate index {r} is outside the frozen design")
    seed = rd["replicate_seeds"][r]
    keys = executor["split"](int(seed))
    key_fp = array_digest(np.asarray(keys))
    columns = {}
    for j, name in enumerate(rd["columns"]):
        by_key = {k: [] for k in OUTPUT_KEYS}
        timings, parameter_digests = [], []
        for k in range(executor["n_wavelengths"]):
            out, seconds, pd = executor["accumulate"](j, k, keys)
            for field in OUTPUT_KEYS:
                by_key[field].append(out[field])
            timings.append(seconds)
            parameter_digests.append(pd)
        columns[name] = {
            field: np.concatenate(by_key[field], axis=1).tolist()
            for field in OUTPUT_KEYS
        }
        columns[name].update({"wall_seconds_by_wavelength": timings,
                              "parameter_digests": parameter_digests,
                              "checks": {"shape_finite_range_pairing": True}})
    return {"replicate": r, "seed": int(seed), "key_fingerprint": key_fp,
            "columns": columns, "checks": {"all_passed": True}}


def pilot_artifact(design, replicates):
    rd = design["run_design"]
    return {"schema_version": "candidate3-grid-pilot-artifact-v1",
            "status": "NON-CONFIRMATORY GRID PILOT; NO CANDIDATE 3 VERDICT",
            "run_design_digest": canonical_digest(rd),
            "transport_run": True, "candidate3_verdict": None,
            "coverage_claimed": False, "replicates": list(replicates)}


def run_pilot(executor, design):
    reps = [run_replicate(executor, design, r)
            for r in range(design["run_design"]["n_replicates"])]
    return pilot_artifact(design, reps)


def shard_path(replicate_index):
    return os.path.join(ROOT, "results",
                        f"candidate3_grid_pilot_shard_{int(replicate_index)}.json")


def assemble_shards(design, preflight_artifact):
    rd = design["run_design"]
    if (preflight_artifact.get("schema_version") != "candidate3-grid-pilot-preflight-v1"
            or preflight_artifact.get("passes") is not True
            or preflight_artifact.get("freeze_hash") != design["freeze_hash"]
            or preflight_artifact.get("normative_input_hashes")
            != design["normative_input_hashes"]
            or preflight_artifact.get("pilot_transport_run") is not False
            or preflight_artifact.get("candidate3_verdict") is not None
            or preflight_artifact.get("coverage_claimed") is not False):
        raise SystemExit("REFUSING assembly: preflight artifact is absent, stale or invalid")
    expected_pf = canonical_digest(preflight_artifact)
    reps, environments, array_jobs = [], [], []
    for r, seed in enumerate(rd["replicate_seeds"]):
        path = shard_path(r)
        if not os.path.isfile(path):
            raise SystemExit(f"REFUSING assembly: missing shard {path}")
        shard = load_json(path)
        problems = []
        if shard.get("schema_version") != "candidate3-grid-pilot-shard-v1":
            problems.append("schema")
        if shard.get("freeze_hash") != design["freeze_hash"]:
            problems.append("freeze hash")
        if shard.get("normative_input_hashes") != design["normative_input_hashes"]:
            problems.append("normative input hashes")
        if shard.get("run_design_digest") != canonical_digest(rd):
            problems.append("run-design digest")
        if shard.get("preflight_artifact_digest") != expected_pf:
            problems.append("preflight digest")
        if shard.get("transport_run") is not True or shard.get("candidate3_verdict") is not None:
            problems.append("transport/verdict boundary")
        rep = shard.get("replicate") or {}
        if rep.get("replicate") != r or rep.get("seed") != seed:
            problems.append("replicate index/seed")
        if not rep.get("checks", {}).get("all_passed"):
            problems.append("structural checks")
        slurm = shard.get("slurm") or {}
        if slurm.get("array_task_id") != r or not slurm.get("array_job_id"):
            problems.append("Slurm array identity")
        if (shard.get("environment") or {}).get("backend") != "gpu":
            problems.append("GPU environment")
        if problems:
            raise SystemExit(f"REFUSING assembly: shard {r} failed " + ", ".join(problems))
        reps.append(rep)
        environments.append(shard.get("environment"))
        array_jobs.append(str(slurm["array_job_id"]))
    if len(set(array_jobs)) != 1:
        raise SystemExit("REFUSING assembly: shards do not belong to one Slurm array job")
    artifact = pilot_artifact(design, reps)
    artifact.update({"freeze_hash": design["freeze_hash"],
                     "normative_input_hashes": design["normative_input_hashes"],
                     "preflight_artifact_digest": expected_pf,
                     "preflight_projected_shard_wall_seconds":
                         preflight_artifact["projected_shard_wall_seconds"],
                     "execution": {"kind": "six-replicate Slurm array",
                                   "array_job_id": array_jobs[0],
                                   "shards": [os.path.basename(shard_path(r))
                                              for r in range(len(reps))]},
                     "environments": environments})
    return artifact


def selftest():
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    checks = []

    def ck(name, ok):
        checks.append((name, bool(ok)))

    def fake_kernel(key, muas, muss, gs, ns, bounds, edges, max_steps):
        del gs, ns, max_steps
        m = edges.shape[0] - 1
        det = (key[0] + jnp.asarray(jnp.round(1000.0 * bounds[0]), jnp.uint32)) % m
        weight = jnp.exp(-0.001 * (jnp.sum(muas) + jnp.sum(muss)))
        return det.astype(jnp.int32), weight

    L, N, M = 6, 40, 12
    h = np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.07])
    edges = np.linspace(0.0, 1.2, M + 1)
    areas = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    base = np.array([0.01, 0.02, 0.05])
    bp = np.stack([base + np.array([0.001 * x, 0.002 * x, 0.003 * x]) for x in h])
    bm = np.stack([base - np.array([0.001 * x, 0.002 * x, 0.003 * x]) for x in h])
    sp = np.stack([np.array([1.0, 1.1, 1.2]) * np.exp(x * 0.01) for x in h])
    sm = np.stack([np.array([1.0, 1.1, 1.2]) * np.exp(-x * 0.01) for x in h])
    muas, gs, ns = np.array([0.1, 0.2, 0.3]), np.array([0.8, 0.8, 0.8]), np.array([1.4] * 3)
    keys = jax.random.split(jax.random.PRNGKey(123), N)
    acc = _make_accumulator(fake_kernel, L, 20)
    out = _validate_and_run(acc, keys=keys, n_photons=N, bounds_plus=bp,
                            bounds_minus=bm, muss_plus=sp, muss_minus=sm, muas=muas,
                            gs=gs, ns=ns, edges=edges, h_grid=h, areas=areas)
    ck("the accumulator returns exactly five fixed-size arrays", set(out) == set(OUTPUT_KEYS)
       and all(v.shape == (L, M) for v in out.values()))
    ck("second moments and maxima are non-negative",
       np.all(out["sum_Y2"] >= 0) and np.all(out["max_abs_Y"] >= 0))
    ck("plus and minus sums never exceed the launch count",
       np.all(out["sum_plus"] <= N) and np.all(out["sum_minus"] <= N))
    B = 1.0 / (2.0 * h[:, None] * areas[None, :])
    ck("observed per-photon maxima obey the deterministic bound",
       np.all(out["max_abs_Y"] <= B + 1e-12 * np.maximum(B, 1.0)))

    # The central-difference arms must be combined for each photon/bin BEFORE squaring. A dense
    # oracle with both arms deliberately landing in the same bin catches the tempting but wrong
    # arm-wise sum of squares.
    def same_bin_kernel(key, muas_, muss_, gs_, ns_, bounds, edges_, max_steps_):
        del muas_, muss_, gs_, ns_, max_steps_
        det = key[0] % (edges_.shape[0] - 1)
        return det.astype(jnp.int32), jnp.asarray(0.5) + 0.2 * bounds[0]

    paired = _validate_and_run(
        _make_accumulator(same_bin_kernel, L, 20), keys=keys, n_photons=N,
        bounds_plus=bp, bounds_minus=bm, muss_plus=sp, muss_minus=sm, muas=muas,
        gs=gs, ns=ns, edges=edges, h_grid=h, areas=areas)
    ref_sum, ref_sum2 = np.zeros((L, M)), np.zeros((L, M))
    host_keys = np.asarray(keys)
    for l in range(L):
        for key in host_keys:
            det = int(key[0] % M)
            yp = 0.5 + 0.2 * bp[l, 0]
            ym = 0.5 + 0.2 * bm[l, 0]
            y = (yp - ym) / (2.0 * h[l] * areas[det])
            ref_sum[l, det] += y
            ref_sum2[l, det] += y * y
    ck("same-bin arms are folded per photon before the second moment",
       np.allclose(paired["sum_Y"], ref_sum, rtol=2e-13, atol=2e-13)
       and np.allclose(paired["sum_Y2"], ref_sum2, rtol=2e-13, atol=2e-13))

    def refuses(**changes):
        kw = dict(keys=keys, n_photons=N, bounds_plus=bp, bounds_minus=bm,
                  muss_plus=sp, muss_minus=sm, muas=muas, gs=gs, ns=ns,
                  edges=edges, h_grid=h, areas=areas)
        kw.update(changes)
        try:
            _validate_and_run(acc, **kw)
            return False
        except SystemExit:
            return True

    ck("wrong launch count is refused before transport", refuses(n_photons=N + 1))
    ck("missing step is refused", refuses(h_grid=h[:-1], bounds_plus=bp[:-1],
                                           bounds_minus=bm[:-1], muss_plus=sp[:-1],
                                           muss_minus=sm[:-1]))
    bad = bp.copy(); bad[0, 1] = bad[0, 0]
    ck("non-increasing layer boundaries are refused", refuses(bounds_plus=bad))
    ck("negative scattering is refused", refuses(muss_minus=-sm))
    design = load_json(PILOT_SPEC)
    draft_design = json.loads(json.dumps(design))
    draft_design.update({"frozen": False, "freeze_hash": None,
                         "normative_input_hashes": {}})
    fs = freeze_status(draft_design, os.path.normpath(os.path.join(ROOT, "..", "jax-pad")))
    ck("a real run is closed while the design is not frozen", not fs["frozen"]
       and "spec.frozen is not true" in fs["problems"])
    ck("the command line has no photons, replicates, seeds or h override",
       not ({"--photons", "--replicates", "--seed", "--h-grid"}
            & {o for a in _build_parser()._actions for o in a.option_strings}))
    expected_manifest = {
        "docs/planning/CANDIDATE3_GRID_PILOT_PLAN.md",
        "jaxpad:scripts/archive/inverse/multi_wl_v12_colab.py",
        "jaxpad:scripts/inverse/phase2_v12_runner.py",
        "jaxpad:src/pad/config_v4_633nm.py",
        "jaxpad:src/pad/mc_forward.py",
        "jaxpad:src/pad/physics_engine_v4.py",
        "scripts/candidate3_executor.py",
        "scripts/candidate3_grid_analysis.py",
        "scripts/candidate3_grid_pilot.py",
        "scripts/candidate3_grid_pilot_audit.py",
        "scripts/candidate3_grid_preflight.py",
        "scripts/candidate3_grid_preflight_fir.sbatch",
        "scripts/candidate3_grid_pilot_fir.sbatch",
        "scripts/nuisance_feasibility_gate.py",
        "spec/baseline_spec.json", "spec/nuisance_spec.json", "src/augmented.py"}
    ck("the freeze manifest binds every local and jax-pad dependency",
       set(fs["computed_input_hashes"]) == expected_manifest
       and all(not str(x).startswith("unavailable:")
               for x in fs["computed_input_hashes"].values()))

    # Exercise main's real ordering with a counting factory. Even though the self-test imported JAX
    # for the fake kernel above, this proves the --run control flow invokes no production builder
    # while the spec is unfrozen.
    global build_pilot_executor, _SPEC_OVERRIDE_DATA
    saved_builder, saved_argv = build_pilot_executor, sys.argv
    saved_spec_override = _SPEC_OVERRIDE_DATA
    calls = {"builder": 0}

    def counted_builder(*args, **kwargs):
        del args, kwargs
        calls["builder"] += 1
        raise AssertionError("the production builder must not be reached")

    try:
        build_pilot_executor = counted_builder
        _SPEC_OVERRIDE_DATA = draft_design
        sys.argv = ["candidate3_grid_pilot.py", "--run"]
        try:
            main()
            main_refused = False
        except SystemExit as exc:
            main_refused = str(exc).startswith("REFUSING to run transport")
    finally:
        build_pilot_executor, sys.argv = saved_builder, saved_argv
        _SPEC_OVERRIDE_DATA = saved_spec_override
    ck("main --run refuses before constructing the production executor",
       main_refused and calls["builder"] == 0)

    # Under a valid frozen design, a target-GPU preflight is still a separate hard gate. Exercise
    # both branches without importing JAX or calling the production builder.
    import tempfile
    material = freeze_material(design, os.path.normpath(os.path.join(ROOT, "..", "jax-pad")))
    frozen_design = material["payload"]
    frozen_design["freeze_hash"] = material["freeze_hash"]
    saved_flags = os.environ.get("XLA_FLAGS")
    saved_module = sys.modules.get("candidate3_grid_preflight")
    missing_module = saved_module is None
    gate_calls = {"preflight": 0, "builder": 0, "allow": False}

    class FakePreflight:
        @staticmethod
        def require_preflight_passed(*args, **kwargs):
            del args, kwargs
            gate_calls["preflight"] += 1
            if not gate_calls["allow"]:
                raise SystemExit("REFUSING to run pilot: target-GPU preflight is absent")
            return {"projected_pilot_wall_seconds": 1.0}

    def must_not_build(*args, **kwargs):
        del args, kwargs
        gate_calls["builder"] += 1
        raise AssertionError("builder reached after preflight")

    try:
        sys.modules["candidate3_grid_preflight"] = FakePreflight
        os.environ["XLA_FLAGS"] = f"--{DETERMINISTIC_FLAG}"
        build_pilot_executor = must_not_build
        _SPEC_OVERRIDE_DATA = frozen_design
        with tempfile.TemporaryDirectory() as tmp:
            sys.argv = ["candidate3_grid_pilot.py", "--run", "--out",
                        os.path.join(tmp, "pilot.json")]
            try:
                main()
                preflight_refused = False
            except SystemExit as exc:
                preflight_refused = "target-GPU preflight is absent" in str(exc)
            ck("a frozen run still refuses a missing target-GPU preflight before build",
               preflight_refused and gate_calls == {
                   "preflight": 1, "builder": 0, "allow": False})
            gate_calls["allow"] = True
            try:
                main()
                ordering_proved = False
            except AssertionError as exc:
                ordering_proved = str(exc) == "builder reached after preflight"
            ck("a passing preflight is checked exactly once before executor construction",
               ordering_proved and gate_calls["preflight"] == 2
               and gate_calls["builder"] == 1)
    finally:
        build_pilot_executor, sys.argv = saved_builder, saved_argv
        _SPEC_OVERRIDE_DATA = saved_spec_override
        if saved_flags is None:
            os.environ.pop("XLA_FLAGS", None)
        else:
            os.environ["XLA_FLAGS"] = saved_flags
        if missing_module:
            sys.modules.pop("candidate3_grid_preflight", None)
        else:
            sys.modules["candidate3_grid_preflight"] = saved_module
    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"  {len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


def _build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--freeze-status", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--run-shard", action="store_true")
    ap.add_argument("--assemble-shards", action="store_true")
    ap.add_argument("--jaxpad-root", default=os.environ.get(
        "JAXPAD_ROOT", os.path.normpath(os.path.join(ROOT, "..", "jax-pad"))))
    ap.add_argument("--out", default=os.path.join(ROOT, "results",
                                                   "candidate3_grid_pilot.json"))
    return ap


def main():
    a = _build_parser().parse_args()
    modes = sum(bool(x) for x in (a.selftest, a.freeze_status, a.run,
                                  a.run_shard, a.assemble_shards))
    if modes != 1:
        raise SystemExit("choose exactly one execution mode")
    if a.selftest:
        return selftest()
    spec = (json.loads(json.dumps(_SPEC_OVERRIDE_DATA))
            if _SPEC_OVERRIDE_DATA is not None else load_json(PILOT_SPEC))
    status = freeze_status(spec, a.jaxpad_root)
    if a.freeze_status:
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0 if status["frozen"] else 2
    # Critical ordering: refuse BEFORE build_pilot_executor imports JAX or production transport.
    if not status["frozen"]:
        raise SystemExit("REFUSING to run transport: the non-confirmatory grid pilot is not "
                         "frozen and hash-bound. " + "; ".join(status["problems"]))
    if a.run_shard:
        raw_index = os.environ.get("SLURM_ARRAY_TASK_ID")
        try:
            replicate_index = int(raw_index)
        except (TypeError, ValueError):
            raise SystemExit("REFUSING shard run: SLURM_ARRAY_TASK_ID is absent or non-integer")
        if replicate_index not in spec["execution_plan"]["array_indices"]:
            raise SystemExit("REFUSING shard run: array index is outside the frozen execution plan")
        output_path = shard_path(replicate_index)
    else:
        output_path = a.out
    if os.path.exists(output_path):
        raise SystemExit(f"REFUSING to overwrite an existing pilot artifact: {output_path}")
    if a.assemble_shards:
        pf = load_json(os.path.join(ROOT, "results", "candidate3_grid_pilot_preflight.json"))
        artifact = assemble_shards(spec, pf)
        with open(output_path, "x") as fh:
            json.dump(artifact, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"  wrote {os.path.relpath(output_path, ROOT)} from six complete shards")
        print("  NON-CONFIRMATORY; no Candidate 3 verdict or coverage claim")
        return 0
    if DETERMINISTIC_FLAG not in os.environ.get("XLA_FLAGS", ""):
        raise SystemExit("REFUSING to build the executor: XLA_FLAGS must contain "
                         f"{DETERMINISTIC_FLAG!r} before JAX is imported")
    import candidate3_grid_preflight as preflight
    pf = preflight.require_preflight_passed(spec, a.jaxpad_root, freeze_status)
    nspec, bspec = load_json(NUISANCE_SPEC), load_json(BASELINE_SPEC)
    ex = build_pilot_executor(a.jaxpad_root, spec, nspec, bspec)
    if ex["backend"] != "gpu":
        raise SystemExit(f"REFUSING to run the 200000-photon pilot on backend "
                         f"{ex['backend']!r}; the frozen resource design requires GPU")
    environment = {"backend": ex["backend"], "jax_version": ex["jax_version"],
                   "max_steps": ex["max_steps"]}
    if a.run_shard:
        rep = run_replicate(ex, spec, replicate_index)
        artifact = {"schema_version": "candidate3-grid-pilot-shard-v1",
                    "status": "NON-CONFIRMATORY GRID PILOT SHARD",
                    "freeze_hash": spec["freeze_hash"],
                    "normative_input_hashes": spec["normative_input_hashes"],
                    "run_design_digest": canonical_digest(spec["run_design"]),
                    "preflight_artifact_digest": canonical_digest(pf),
                    "preflight_projected_shard_wall_seconds":
                        pf["projected_shard_wall_seconds"],
                    "transport_run": True, "candidate3_verdict": None,
                    "coverage_claimed": False, "replicate": rep,
                    "environment": environment,
                    "slurm": {"array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                              "array_task_id": replicate_index}}
    else:
        artifact = run_pilot(ex, spec)
        artifact.update({"freeze_hash": spec["freeze_hash"],
                         "normative_input_hashes": spec["normative_input_hashes"],
                         "preflight_artifact_digest": canonical_digest(pf),
                         "preflight_projected_shard_wall_seconds":
                             pf["projected_shard_wall_seconds"],
                         "environment": environment})
    with open(output_path, "x") as fh:
        json.dump(artifact, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"  wrote {os.path.relpath(output_path, ROOT)}")
    print("  NON-CONFIRMATORY; no Candidate 3 verdict or coverage claim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
