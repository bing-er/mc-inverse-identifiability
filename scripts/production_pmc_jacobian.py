#!/usr/bin/env python3
"""production_pmc_jacobian.py -- the full-grid replay-pMC production Jacobian, versus AD.

The production spectrum and the headline bound are conditional on the pathwise-AD Jacobian.
MERGE_CHECKS item 1 closed with unresolved adjudication: AD and replay pMC are detectably
different and the independent-stream reference cannot say which is right at an acceptable budget.
This script asks the cheaper, answerable question instead: do the UNIFIED SPECTRUM FUNCTIONALS --
weakest direction, condition number, per-parameter CRLB -- change when the production Jacobian is
swapped for the replay-pMC one, computed on the SAME 32 frozen streams?

What this is: a same-stream estimator-robustness check of the spectrum functionals. The replayed
trajectories are the production trajectories (the replay gate established photon-level identity on
the production backend), so differences in the output are attributable to the derivative
construction, not to Monte Carlo noise.

What this is NOT: adjudication. Agreement between the AD and pMC spectra does not establish that
either Jacobian is correct -- the two constructions can share a bias, which is why the
ensemble-reference route existed and why its closure stands. The word "validated" must not appear
downstream of this result.

Phase A (--run) replays the 32 production streams with the per-layer pathlength accumulator and
writes one npz per seed to data/derived/production_pmc/. Resumable at seed granularity.
Fail-closed trajectory guard: the frozen GPU photon-level equivalence gate, its negative control,
its source hashes, and the production stream metadata must all match. The replayed aggregate R is
then compared with the archived production R at the gate's frozen reflectance tolerance. Aggregate
R is not required to be bitwise equal because the production artifact evaluates the primal beside
jacfwd whereas the replay evaluates R beside its moment reduction; XLA may reorder those two
mathematically identical reductions. The photon-level gate, not pooled agreement, establishes the
same-trajectory premise. --smoke (2000 photons, 200 steps) checks against a freshly computed
production forward on the current backend instead, into data/derived/production_pmc/_smoke/.

Phase B (--analyze) never runs transport. It pools the per-seed pMC Jacobians with the same mean
as the production pool, re-verifies the AD pool against results/production_crb.json, computes the
raw, diagonal-Poisson and fixed-launch multinomial branches for both Jacobians, and writes
results/production_pmc_jacobian.json.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import unify  # noqa: E402
from production_crb import LABELS, full_rank_crlb  # noqa: E402
from production_worker import (  # noqa: E402
    NOMINAL_BOUNDARIES, SEED_BASE, SEED_STRIDE, N_SEEDS, all_seeds)
from unify_spectrum import column_scale, load_spec  # noqa: E402

BSPEC = os.path.join(ROOT, "spec", "baseline_spec.json")
AD_STREAMS = os.path.join(ROOT, "data", "derived", "production_cuda129")
POOLED = os.path.join(ROOT, "data", "derived", "production_pooled.npz")
CRB = os.path.join(ROOT, "results", "production_crb.json")
OUT_JSON = os.path.join(ROOT, "results", "production_pmc_jacobian.json")
GPU_GATE = os.path.join(ROOT, "results", "replay_equivalence_gpu.json")
GPU_GATE_PROVENANCE = os.path.join(ROOT, "results", "replay_equivalence_gpu_provenance.json")
# Cross-platform LAPACK/SVD evaluation of the ill-conditioned CRLB can move in the last few
# decimal places. This is a regression tolerance, not an inferential or decision threshold.
AD_FROZEN_REGRESSION_RTOL = 1e-12


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path):
    with open(path) as fh:
        return json.load(fh)


def verified_gpu_gate(jaxpad_root, runtime_jax_version=None, runtime_backend=None):
    """Verify the frozen photon-level GPU gate and return its aggregate tolerance."""
    if not os.path.isfile(GPU_GATE) or not os.path.isfile(GPU_GATE_PROVENANCE):
        raise SystemExit("the frozen GPU replay-equivalence gate and provenance are required")
    gate, prov = _json(GPU_GATE), _json(GPU_GATE_PROVENANCE)
    recorded_gate_hash = prov.get("output_hashes", {}).get(
        "results/replay_equivalence_gpu.json")
    if recorded_gate_hash != sha256(GPU_GATE):
        raise SystemExit("GPU replay-equivalence gate hash does not match its provenance")
    summary = gate.get("summary", {})
    if not (summary.get("all_passed") and summary.get("negative_control_rejected")):
        raise SystemExit("GPU replay-equivalence gate did not pass fail-closed requirements")
    env = gate.get("environment", {})
    if env.get("jax_backend") != "gpu":
        raise SystemExit("replay-equivalence artifact is not a GPU gate")
    if runtime_backend is not None and env.get("jax_backend") != runtime_backend:
        raise SystemExit(f"GPU gate backend {env.get('jax_backend')} != runtime {runtime_backend}")
    if runtime_jax_version is not None and env.get("jax_version") != runtime_jax_version:
        raise SystemExit(
            f"GPU gate JAX {env.get('jax_version')} != runtime JAX {runtime_jax_version}")

    sources = {}
    sources.update({os.path.join(jaxpad_root, rel): want for rel, want in
                    prov.get("production_source_hashes", {}).items()})
    sources.update({os.path.join(ROOT, rel): want for rel, want in
                    prov.get("replay_source_hash", {}).items()})
    sources.update({os.path.join(ROOT, rel): want for rel, want in
                    prov.get("gate_script_hash", {}).items()})
    baseline_hash = prov.get("spec_hashes", {}).get("spec/baseline_spec.json")
    if baseline_hash:
        sources[BSPEC] = baseline_hash
    for path, want in sources.items():
        if not os.path.isfile(path) or sha256(path) != want:
            raise SystemExit(f"GPU replay-equivalence provenance mismatch: {path}")

    tol = float(gate.get("criteria", {}).get("reflectance_max_rel", -1.0))
    if not (0.0 < tol <= 1e-12):
        raise SystemExit(f"invalid frozen aggregate reflectance tolerance {tol!r}")
    return {"max_rel_tolerance": tol, "gate_sha256": sha256(GPU_GATE),
            "provenance_sha256": sha256(GPU_GATE_PROVENANCE),
            "n_photon_comparisons": int(summary.get("total_photon_comparisons", 0)),
            "det_id_mismatches": int(summary.get("det_id_mismatches_total", -1)),
            "escaped_flag_mismatches": int(summary.get("escaped_flag_mismatches_total", -1)),
            "weights_bitwise_identical": bool(
                summary.get("weight_bitwise_identical_in_all_cases"))}


def aggregate_reflectance_guard(replayed, archived, max_rel_tolerance):
    replayed, archived = np.asarray(replayed), np.asarray(archived)
    if replayed.shape != archived.shape:
        raise SystemExit(
            f"aggregate reflectance shape mismatch: {replayed.shape} != {archived.shape}")
    if not (np.all(np.isfinite(replayed)) and np.all(np.isfinite(archived))):
        raise SystemExit("aggregate reflectance contains a non-finite value")
    delta = np.abs(replayed - archived)
    scale = np.maximum(np.abs(archived), np.finfo(np.float64).tiny)
    max_abs = float(delta.max(initial=0.0))
    max_rel = float((delta / scale).max(initial=0.0))
    return {"passes": bool(max_rel <= max_rel_tolerance),
            "max_abs_diff": max_abs, "max_rel_diff": max_rel,
            "max_rel_tolerance": float(max_rel_tolerance),
            "bitwise_identical": bool(np.array_equal(replayed, archived))}


def pmc_dir(smoke):
    d = os.path.join(ROOT, "data", "derived", "production_pmc")
    return os.path.join(d, "_smoke") if smoke else d


def seed_path(smoke, seed):
    return os.path.join(pmc_dir(smoke), f"seed_{seed}{'_smoke' if smoke else ''}.npz")


# ---------------------------------------------------------------------------------------- phase A
def run(a):
    with open(BSPEC) as fh:
        spec = json.load(fh)
    prof = spec["profiles"]["production"]
    kappa = spec["blood_coordinate"]["legacy_to_physical_divisor"]
    nom, labels = spec["parameters"]["nominal_psi"], spec["parameters"]["psi_labels"]
    radial = prof["radial"]
    n_photons = 2000 if a.smoke else int(prof["n_transport_photons_per_wavelength_per_stream"])
    max_steps = 200 if a.smoke else int(prof["max_steps"])
    outdir = pmc_dir(a.smoke)
    os.makedirs(outdir, exist_ok=True)

    seeds = all_seeds() if a.seeds is None else [int(x) for x in a.seeds.split(",")]
    todo = [s for s in seeds if a.force or not os.path.exists(seed_path(a.smoke, s))]
    skipped = [s for s in seeds if s not in todo]
    if skipped:
        print(f"  already present, skipping: {skipped}")
    if not todo:
        print("  nothing to do")
        return 0
    print(f"  replay-pMC sweep over {len(todo)} seeds: {todo}")

    sys.path.insert(0, os.path.abspath(a.jaxpad_root))
    import importlib.util
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from scripts.inverse.phase2_v12_runner import (  # noqa: E402
        Phase2RunConfig, scattering_for_transport, _build_mu)
    import scripts.archive.inverse.multi_wl_v12_colab as v12  # noqa: E402

    gate = ({"max_rel_tolerance": 1e-12, "gate_sha256": "smoke",
             "provenance_sha256": "smoke", "n_photon_comparisons": 0,
             "det_id_mismatches": 0, "escaped_flag_mismatches": 0,
             "weights_bitwise_identical": True}
            if a.smoke else verified_gpu_gate(
                os.path.abspath(a.jaxpad_root), jax.__version__, jax.default_backend()))

    _rk_spec = importlib.util.spec_from_file_location(
        "mcii_replay_kernel", os.path.join(ROOT, "src", "replay_kernel.py"))
    _rk = importlib.util.module_from_spec(_rk_spec)
    _rk_spec.loader.exec_module(_rk)

    n_wl = int(v12.N_WL)
    if n_wl != len(spec["wavelengths_nm"]):
        raise SystemExit(f"v12 has {n_wl} wavelengths, spec declares {len(spec['wavelengths_nm'])}")
    edges = jnp.linspace(radial["r_min_cm"], radial["r_max_cm"], radial["n_bins"] + 1,
                         dtype=jnp.float64)
    e = np.asarray(edges)
    areas = np.pi * (e[1:] ** 2 - e[:-1] ** 2)
    psi0 = np.array([nom[k] for k in labels], dtype=float)
    cfg = Phase2RunConfig(n_photons_final=n_photons, seed_keys=SEED_BASE,
                          seed_noise=SEED_BASE + 10000,
                          sto2_ubnp_true=psi0[0], sto2_dbnp_true=psi0[2],
                          b_ubnp_true=psi0[1] * kappa, b_dbnp_true=psi0[3] * kappa,
                          max_steps=max_steps)
    bounds = jnp.array(NOMINAL_BOUNDARIES, dtype=jnp.float64)
    mus_list = [scattering_for_transport(v12.MUSS_WL[k], v12.GS_WL[k], cfg.scattering_convention)
                for k in range(n_wl)]

    def psi_to_theta(psi):
        s_u, f_u, s_d, f_d, dermr, subq = psi
        c = jnp.stack([kappa * f_u * s_u, kappa * f_u * (1.0 - s_u),
                       kappa * f_d * s_d, kappa * f_d * (1.0 - s_d), dermr, subq])
        return jnp.log(c + v12.EPS)

    def opt(k):
        return dict(muss=mus_list[k], gs=v12.GS_WL[k], ns=v12.NS_WL[k],
                    layer_boundaries=bounds, detector_r_edges=edges, max_steps=max_steps)

    psi_j = jnp.asarray(psi0)
    # chain rule factor, analytic through _build_mu: one (n_layers, 6) matrix per wavelength
    dmua_dpsi = [np.asarray(jax.jacfwd(
        lambda p, kk=k: _build_mu(psi_to_theta(p))[0][kk])(psi_j)) for k in range(n_wl)]

    def replay_all(psi, keys):
        mu_list, _, _ = _build_mu(psi_to_theta(psi))
        Rs, Jb = [], []
        for k in range(n_wl):
            R_k, S_k = _rk.replay_moments(muas=mu_list[k], keys=keys, **opt(k))
            Rs.append(R_k)
            Jb.append(-(S_k @ jnp.asarray(dmua_dpsi[k])))
        return jnp.concatenate(Rs), jnp.concatenate(Jb, axis=0)

    replay_jit = jax.jit(replay_all)

    fwd_jit = None
    if a.smoke:
        def refl_psi(psi, keys):
            mu_list, _, _ = _build_mu(psi_to_theta(psi))
            return jnp.concatenate([
                v12.forward_solver(muas=mu_list[k], keys=keys, **opt(k)) for k in range(n_wl)])
        fwd_jit = jax.jit(refl_psi)

    k0 = jax.random.split(jax.random.PRNGKey(todo[0]), n_photons)
    jax.block_until_ready(k0)
    t0 = time.perf_counter()
    compiled = replay_jit.lower(psi_j, k0).compile()
    print(f"  compiled once in {time.perf_counter() - t0:.1f}s "
          f"({n_photons} photons/wavelength, max_steps={max_steps}, "
          f"backend={jax.default_backend()})")

    for s in todo:
        keys = jax.random.split(jax.random.PRNGKey(s), n_photons)
        jax.block_until_ready(keys)
        t0 = time.perf_counter()
        R_j, J_j = compiled(psi_j, keys)
        R = np.asarray(jax.block_until_ready(R_j))
        J_pmc = np.asarray(jax.block_until_ready(J_j))
        t_exec = time.perf_counter() - t0

        # Fail closed on the aggregate consequence of the already-frozen photon-level gate.
        if a.smoke:
            smoke_arch = os.path.join(
                ROOT, "data", "derived", "production", "_smoke", f"seed_{s}_smoke.npz")
            if os.path.isfile(smoke_arch):
                with np.load(smoke_arch, allow_pickle=False) as d:
                    R_ref = np.asarray(d["R"])
            else:
                R_ref = np.asarray(jax.block_until_ready(fwd_jit(psi_j, keys)))
            guard = aggregate_reflectance_guard(R, R_ref, gate["max_rel_tolerance"])
        else:
            arch = os.path.join(AD_STREAMS, f"seed_{s}.npz")
            with np.load(arch, allow_pickle=False) as d:
                R_arch = np.asarray(d["R"])
                metadata = {k: d[k].item() for k in (
                    "seed", "n_transport_photons", "max_steps", "n_wavelengths", "profile",
                    "smoke", "jax_version", "jax_backend", "ptxas_version", "cuda_module")}
                archived_psi = np.asarray(d["psi_nominal"])
            expected_metadata = {
                "seed": s, "n_transport_photons": n_photons, "max_steps": max_steps,
                "n_wavelengths": n_wl, "profile": "production", "smoke": False,
                "jax_version": jax.__version__, "jax_backend": jax.default_backend(),
                "ptxas_version": os.environ.get("MCII_PTXAS_VERSION", ""),
                "cuda_module": os.environ.get("MCII_CUDA_MODULE", "")}
            if metadata != expected_metadata or not np.array_equal(archived_psi, psi0):
                raise SystemExit(
                    f"seed {s}: frozen production metadata does not match this replay runtime")
            guard = aggregate_reflectance_guard(R, R_arch, gate["max_rel_tolerance"])
        if not guard["passes"]:
            raise SystemExit(
                f"seed {s}: aggregate reflectance consequence of the photon-level gate failed "
                f"(max abs {guard['max_abs_diff']:.3e}, max rel {guard['max_rel_diff']:.3e}, "
                f"frozen tolerance {guard['max_rel_tolerance']:.3e}). Nothing was written.")

        out = seed_path(a.smoke, s)
        tmp = out + f".tmp.{os.getpid()}.npz"
        with open(tmp, "wb") as fh:
            np.savez_compressed(
                fh, seed=s, R=R, J_pmc=J_pmc, areas=areas, psi_nominal=psi0,
                estimator="replay_pmc_pathlength",
                n_transport_photons=n_photons, max_steps=max_steps, n_wavelengths=n_wl,
                blood_coordinate="physical_bvf", profile="production", smoke=a.smoke,
                jax_version=__import__("jax").__version__,
                jax_backend=jax.default_backend(),
                ptxas_version=os.environ.get("MCII_PTXAS_VERSION", ""),
                cuda_module=os.environ.get("MCII_CUDA_MODULE", ""),
                trajectory_gate="photon_level_gpu_gate_plus_aggregate_consequence",
                trajectory_gate_sha256=gate["gate_sha256"],
                trajectory_gate_provenance_sha256=gate["provenance_sha256"],
                trajectory_gate_max_rel_tolerance=guard["max_rel_tolerance"],
                aggregate_R_max_abs_diff=guard["max_abs_diff"],
                aggregate_R_max_rel_diff=guard["max_rel_diff"],
                aggregate_R_bitwise_identical=guard["bitwise_identical"])
        os.replace(tmp, out)
        print(f"  seed {s}: {t_exec:.1f}s, trajectory guard passed "
              f"(max |dR| {guard['max_abs_diff']:.3e}, max rel {guard['max_rel_diff']:.3e}), "
              f"wrote {os.path.basename(out)}")
    return 0


# ---------------------------------------------------------------------------------------- phase B
def weakest(V):
    v = np.asarray(V)[:, -1]
    return v / np.linalg.norm(v)


def analyze(a):
    spec = load_spec()
    prof = spec["profiles"]["production"]
    n_measurement = int(prof["n_measurement_photons_per_wavelength"])

    def seed_of(path):
        return int(os.path.basename(path).split("_")[1].split(".")[0])

    if a.smoke:
        ad_dir = os.path.join(ROOT, "data", "derived", "production", "_smoke")
        ad_files = sorted(glob.glob(os.path.join(ad_dir, "seed_*_smoke.npz")), key=seed_of)
        if not ad_files:
            raise SystemExit("smoke analyze needs AD smoke streams; run "
                             "'python3 scripts/production_worker.py --all --smoke' first")
        aR, aJ, expected_seeds = [], [], []
        for path in ad_files:
            with np.load(path, allow_pickle=False) as d:
                expected_seeds.append(int(d["seed"]))
                aR.append(np.asarray(d["R"], dtype=float))
                aJ.append(np.asarray(d["J_psi"], dtype=float))
                psi = np.asarray(d["psi_nominal"], dtype=float)
                areas = np.asarray(d["areas"], dtype=float)
                n_wavelengths = int(d["n_wavelengths"])
        n_expected = len(ad_files)
        R_ad, J_ad = np.stack(aR).mean(axis=0), np.stack(aJ).mean(axis=0)
        ad_stream_R = dict(zip(expected_seeds, aR))
    else:
        n_expected = int(prof["n_transport_streams"])
        pooled = np.load(POOLED, allow_pickle=False)
        expected_seeds = [int(x) for x in pooled["seeds"]]
        psi = np.asarray(pooled["psi_nominal"], dtype=float)
        areas = np.asarray(pooled["areas"], dtype=float)
        n_wavelengths = int(pooled["n_wavelengths"])
        R_ad = np.asarray(pooled["R"], dtype=float)
        J_ad = np.asarray(pooled["J_psi"], dtype=float)
        ad_stream_R = {}
        for s in expected_seeds:
            with np.load(os.path.join(AD_STREAMS, f"seed_{s}.npz"), allow_pickle=False) as d:
                ad_stream_R[s] = np.asarray(d["R"], dtype=float)

    files = sorted(glob.glob(os.path.join(pmc_dir(a.smoke), "seed_*.npz")), key=seed_of)
    if len(files) != n_expected:
        raise SystemExit(f"expected {n_expected} pMC stream files, found {len(files)}")

    gate = verified_gpu_gate(os.path.abspath(a.jaxpad_root)) if not a.smoke else {
        "max_rel_tolerance": 1e-12, "gate_sha256": "smoke",
        "provenance_sha256": "smoke"}
    Rs, Js, seeds, hashes, stored_guards = [], [], [], [], []
    for path in files:
        with np.load(path, allow_pickle=False) as d:
            if bool(d["smoke"]) != bool(a.smoke):
                raise SystemExit(f"{os.path.basename(path)}: smoke flag mismatch")
            seeds.append(int(d["seed"]))
            Rs.append(np.asarray(d["R"], dtype=float))
            Js.append(np.asarray(d["J_pmc"], dtype=float))
            stored_guards.append({
                "max_abs_diff": float(d["aggregate_R_max_abs_diff"]),
                "max_rel_diff": float(d["aggregate_R_max_rel_diff"]),
                "max_rel_tolerance": float(d["trajectory_gate_max_rel_tolerance"]),
                "gate_sha256": str(d["trajectory_gate_sha256"]),
                "provenance_sha256": str(d["trajectory_gate_provenance_sha256"])})
        hashes.append({"seed": seeds[-1], "sha256": sha256(path),
                       "bytes": os.path.getsize(path)})
    if seeds != expected_seeds:
        raise SystemExit(f"pMC seeds {seeds} do not match AD seeds {expected_seeds}")

    # Same-stream premise: photon identity is established by the frozen GPU gate. Aggregate R is
    # re-verified here only as its numerical consequence, using that gate's recorded tolerance.
    stream_guards = []
    for i, s in enumerate(seeds):
        guard = aggregate_reflectance_guard(
            Rs[i], ad_stream_R[s], gate["max_rel_tolerance"])
        stored = stored_guards[i]
        if (not guard["passes"] or stored["max_abs_diff"] != guard["max_abs_diff"]
                or stored["max_rel_diff"] != guard["max_rel_diff"]
                or stored["max_rel_tolerance"] != gate["max_rel_tolerance"]
                or stored["gate_sha256"] != gate["gate_sha256"]
                or stored["provenance_sha256"] != gate["provenance_sha256"]):
            raise SystemExit(f"seed {s}: stored trajectory guard does not re-verify")
        stream_guards.append({"seed": s, **guard})

    R_pmc = np.stack(Rs).mean(axis=0)
    J_pmc = np.stack(Js).mean(axis=0)
    pooled_guard = aggregate_reflectance_guard(
        R_pmc, R_ad, gate["max_rel_tolerance"])
    if not pooled_guard["passes"]:
        raise SystemExit("pooled replay R fails the frozen aggregate reflectance tolerance")

    D = column_scale(psi)
    with open(CRB) as fh:
        frozen = json.load(fh)
    out = {"status": "SAME-STREAM SPECTRUM ROBUSTNESS CHECK; NOT ADJUDICATION",
           "interpretation": (
               "Both Jacobians are plug-in estimates from the same 32 frozen production streams. "
               "Agreement of the spectrum functionals shows the production identifiability "
               "conclusion is insensitive to the choice between the pathwise-AD and replay-pMC "
               "derivative constructions; it does not establish that either construction is "
               "correct, and the unresolved independent-estimator adjudication stands."),
           "transport_rerun": False,
           "inputs": {"pooled_artifact": POOLED.replace(ROOT + os.sep, ""),
                      "pooled_sha256": sha256(POOLED),
                      "production_crb": CRB.replace(ROOT + os.sep, ""),
                      "production_crb_sha256": sha256(CRB),
                      "baseline_spec_sha256": sha256(BSPEC),
                      "pmc_stream_files": hashes},
           "same_stream_evidence": {
               "photon_level_gpu_gate": gate,
               "stream_aggregate_consequences": stream_guards,
               "pooled_aggregate_consequence": pooled_guard,
               "note": "The frozen photon-level GPU gate establishes trajectory identity. "
                       "Aggregate reflectance is a consequence and can differ at roundoff level "
                       "when XLA reduces the production primal beside jacfwd rather than moments."},
           "branches": {}}

    if a.smoke:
        print("  smoke: likelihood branches skipped (smoke R has empty bins; whitening is "
              "undefined there). Raw branch and discrepancy only.")
    lik_branches = () if a.smoke else ("multinomial_fixed_launch", "diagonal_poisson")
    for name in lik_branches:
        if name == "multinomial_fixed_launch":
            W_ad = unify.whiten_multinomial(J_ad, R_ad, areas, n_measurement, D, n_wavelengths)
            W_pm = unify.whiten_multinomial(J_pmc, R_ad, areas, n_measurement, D, n_wavelengths)
        else:
            W_ad = unify.whiten_poisson(J_ad, R_ad, areas, n_measurement, D)
            W_pm = unify.whiten_poisson(J_pmc, R_ad, areas, n_measurement, D)
        crlb_ad, (s_ad, V_ad, _) = full_rank_crlb(W_ad, D, psi)
        crlb_pm, (s_pm, V_pm, _) = full_rank_crlb(W_pm, D, psi)
        if name == "multinomial_fixed_launch":
            frozen_ad = frozen["branches"][name]["crlb"]["sqrt_crlb_psi"]
            recomputed_ad = crlb_ad["sqrt_crlb_psi"]
            rel = {k: float(abs(recomputed_ad[k] - frozen_ad[k]) / abs(frozen_ad[k]))
                   for k in LABELS}
            regression = {
                "passes": bool(all(np.isclose(recomputed_ad[k], frozen_ad[k], rtol=
                                                   AD_FROZEN_REGRESSION_RTOL, atol=0.0)
                                   for k in LABELS)),
                "relative_tolerance": AD_FROZEN_REGRESSION_RTOL,
                "recomputed_sqrt_crlb_psi": recomputed_ad,
                "frozen_sqrt_crlb_psi": frozen_ad,
                "relative_difference": rel,
                "max_relative_difference": max(rel.values()),
                "note": "Regression guard for platform-level linear-algebra roundoff; this is "
                        "not an inferential tolerance or a changed scientific decision rule."}
            out["frozen_ad_regression"] = regression
            if not a.smoke and not regression["passes"]:
                raise SystemExit(
                    "recomputed AD CRLB does not match production_crb.json within the frozen "
                    f"regression tolerance {AD_FROZEN_REGRESSION_RTOL:.1e}; max relative "
                    f"difference {regression['max_relative_difference']:.3e}")
        v_ad, v_pm = weakest(V_ad), weakest(V_pm)
        out["branches"][name] = {
            "ad": {"sqrt_crlb_psi": crlb_ad["sqrt_crlb_psi"],
                   "condition_number_Jw": crlb_ad["condition_number_Jw"],
                   "numerical_rank": crlb_ad["numerical_rank"],
                   "singular_values": crlb_ad["singular_values"],
                   "weakest_direction_scaled": v_ad.tolist()},
            "pmc": {"sqrt_crlb_psi": crlb_pm["sqrt_crlb_psi"],
                    "condition_number_Jw": crlb_pm["condition_number_Jw"],
                    "numerical_rank": crlb_pm["numerical_rank"],
                    "singular_values": crlb_pm["singular_values"],
                    "weakest_direction_scaled": v_pm.tolist()},
            "comparison": {
                "abs_cosine_weakest_directions": float(abs(v_ad @ v_pm)),
                "abs_StO2_D_coefficient_ad": float(abs(v_ad[LABELS.index("StO2_D")])),
                "abs_StO2_D_coefficient_pmc": float(abs(v_pm[LABELS.index("StO2_D")])),
                "sqrt_crlb_ratio_pmc_over_ad": {
                    k: float(crlb_pm["sqrt_crlb_psi"][k] / crlb_ad["sqrt_crlb_psi"][k])
                    for k in LABELS},
            },
        }

    # raw geometric branch: no likelihood, spectrum shape and weakest direction only
    raw = {}
    for tag, J in (("ad", J_ad), ("pmc", J_pmc)):
        _, s, Vt = np.linalg.svd(unify.whiten_raw(J, D), full_matrices=False)
        raw[tag] = {"normalized_singular_values": (s / s[0]).tolist(),
                    "weakest_direction_scaled": weakest(Vt.T).tolist()}
    va = np.asarray(raw["ad"]["weakest_direction_scaled"])
    vp = np.asarray(raw["pmc"]["weakest_direction_scaled"])
    raw["comparison"] = {"abs_cosine_weakest_directions": float(abs(va @ vp)),
                         "abs_StO2_D_coefficient_ad": float(abs(va[LABELS.index("StO2_D")])),
                         "abs_StO2_D_coefficient_pmc": float(abs(vp[LABELS.index("StO2_D")]))}
    out["branches"]["raw_geometric"] = raw

    # elementwise Jacobian discrepancy, for the record (this is where AD and pMC DO differ)
    scale = np.maximum(np.abs(J_ad), np.abs(J_pmc)).max(axis=0)
    out["pooled_jacobian_discrepancy"] = {
        "max_abs_elementwise_diff_per_column": {
            LABELS[j]: float(np.max(np.abs(J_ad[:, j] - J_pmc[:, j]))) for j in range(6)},
        "max_relative_diff_per_column_vs_column_max": {
            LABELS[j]: float(np.max(np.abs(J_ad[:, j] - J_pmc[:, j])) / scale[j])
            for j in range(6)},
        "note": ("relative to each column's own max magnitude; the paired pilot already "
                 "established the two estimators are detectably different per photon"),
    }

    dst = OUT_JSON if not a.smoke else OUT_JSON.replace(".json", "_smoke.json")
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"  wrote {dst}")
    for name in list(out["branches"]):
        c = out["branches"][name]["comparison"]
        print(f"  {name}: |cos(weakest_ad, weakest_pmc)| = "
              f"{c['abs_cosine_weakest_directions']:.6f}, "
              f"|StO2_D| coeff ad {c['abs_StO2_D_coefficient_ad']:.5f} "
              f"vs pmc {c['abs_StO2_D_coefficient_pmc']:.5f}")
    if "multinomial_fixed_launch" in out["branches"]:
        h = out["branches"]["multinomial_fixed_launch"]["comparison"]
        print(f"  headline sqrt CRLB ratio pmc/ad (StO2_D): "
              f"{h['sqrt_crlb_ratio_pmc_over_ad']['StO2_D']:.6f}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", action="store_true")
    g.add_argument("--analyze", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", help="comma-separated subset (default: all 32)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--jaxpad-root", default=os.environ.get(
        "JAXPAD_ROOT", os.path.normpath(os.path.join(ROOT, "..", "jax-pad"))))
    a = ap.parse_args()
    return run(a) if a.run else analyze(a)


if __name__ == "__main__":
    raise SystemExit(main())
