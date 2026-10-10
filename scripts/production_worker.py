#!/usr/bin/env python3
"""production_worker.py -- long-lived worker that compiles once and sweeps many seeds.

The per-seed collector pays the XLA compile cost once per process, which is wasted 31 times over a
32-seed pool. The cause is subtle: the PRNG key array was captured in a closure, so jit treated it
as a compile-time constant and a new seed forced a new trace. Here `keys` is a traced ARGUMENT of
the compiled function. Its shape is identical for every seed, so one compilation serves the whole
sweep and only the values change.

Each seed is written to its own file the moment it finishes, so the worker is interruptible and
resumable at seed granularity, and a partial pool is always a valid partial pool. For parallelism,
run two or three workers over disjoint seed subsets rather than one process per seed.

Timing synchronizes JAX on both sides of the measured region: nothing is outstanding when the clock
starts, and the result is materialized before it stops. Otherwise the asynchronous dispatch would
attribute one seed's compute to the next seed's timer.

  python3 scripts/production_worker.py --all
  python3 scripts/production_worker.py --worker-index 0 --n-workers 3
  python3 scripts/production_worker.py --seeds 44000,45000,46000
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from functools import partial

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NOMINAL_BOUNDARIES = [0.0020, 0.0100, 0.0250, 0.0330, 0.1830, 0.1910, 0.8000]
SEED_BASE, SEED_STRIDE, N_SEEDS = 43000, 1000, 32


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 ** 2) if platform.system() == "Darwin" else r / 1024


def all_seeds():
    return [SEED_BASE + i * SEED_STRIDE for i in range(N_SEEDS)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true")
    g.add_argument("--seeds", help="comma-separated")
    g.add_argument("--worker-index", type=int, help="with --n-workers, take a strided subset")
    g.add_argument("--index", type=int,
                   help="array task index in [0, 32): run exactly the one corresponding seed")
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--jaxpad-root", default=os.environ.get(
        "JAXPAD_ROOT", os.path.normpath(os.path.join(ROOT, "..", "jax-pad"))))
    ap.add_argument("--outdir", default=os.path.join(ROOT, "data", "derived", "production"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--diagnostic", action="store_true",
                    help="run the energy-accounting termination diagnostic. It costs ~190 s per "
                         "seed and is evidence about the step cap, not about the seed, so it is "
                         "off by default: run it on the verification seed and a spot check, not "
                         "all 32 times.")
    ap.add_argument("--diag-photons", type=int, default=20000)
    a = ap.parse_args()

    if a.index is not None:
        if not (0 <= a.index < N_SEEDS):
            ap.error(f"--index must lie in [0, {N_SEEDS})")
        seeds = [SEED_BASE + a.index * SEED_STRIDE]
    elif a.seeds:
        seeds = [int(x) for x in a.seeds.split(",")]
    elif a.worker_index is not None:
        if not (0 <= a.worker_index < a.n_workers):
            ap.error("--worker-index must lie in [0, --n-workers)")
        seeds = all_seeds()[a.worker_index::a.n_workers]
    else:
        seeds = all_seeds()

    spec = json.load(open(os.path.join(ROOT, "spec", "baseline_spec.json")))
    prof = spec["profiles"]["production"]
    kappa = spec["blood_coordinate"]["legacy_to_physical_divisor"]
    nom, labels = spec["parameters"]["nominal_psi"], spec["parameters"]["psi_labels"]
    radial = prof["radial"]
    n_photons = 2000 if a.smoke else prof["n_transport_photons_per_wavelength_per_stream"]
    max_steps = 200 if a.smoke else prof["max_steps"]

    outdir = os.path.join(a.outdir, "_smoke") if a.smoke else a.outdir
    os.makedirs(outdir, exist_ok=True)

    def path_for(s):
        return os.path.join(outdir, f"seed_{s}{'_smoke' if a.smoke else ''}.npz")

    todo = [s for s in seeds if a.force or not os.path.exists(path_for(s))]
    skipped = [s for s in seeds if s not in todo]
    if skipped:
        print(f"  already present, skipping: {skipped}")
    if not todo:
        print("  nothing to do")
        return 0
    print(f"  worker will sweep {len(todo)} seeds: {todo}")

    sys.path.insert(0, os.path.abspath(a.jaxpad_root))
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from scripts.inverse.phase2_v12_runner import (          # noqa: E402
        Phase2RunConfig, scattering_for_transport, _build_mu)
    import scripts.archive.inverse.multi_wl_v12_colab as v12  # noqa: E402
    from src.pad.physics_engine_v4 import (                   # noqa: E402
        simulate_one_photon_rigorous_energy as sim_energy)

    n_wl = int(v12.N_WL)
    edges = jnp.linspace(radial["r_min_cm"], radial["r_max_cm"], radial["n_bins"] + 1,
                         dtype=jnp.float64)
    e = np.asarray(edges)
    areas = np.pi * (e[1:] ** 2 - e[:-1] ** 2)
    areas_tiled = np.tile(areas, n_wl)
    per = radial["n_bins"]
    n_obs = per * n_wl
    if n_obs != prof["n_observables"]:
        raise ValueError(f"grid gives {n_obs} observables, spec declares {prof['n_observables']}")

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

    # keys is an ARGUMENT, not a closure constant: that is what makes one compilation serve every
    # seed. Its shape is the same for all of them, so no retrace occurs.
    def refl_psi(psi, keys):
        mu_list, _, _ = _build_mu(psi_to_theta(psi))
        return jnp.concatenate([
            v12.forward_solver(muas=mu_list[k], keys=keys, muss=mus_list[k], gs=v12.GS_WL[k],
                               ns=v12.NS_WL[k], layer_boundaries=bounds,
                               detector_r_edges=edges, max_steps=max_steps)
            for k in range(n_wl)])

    both = jax.jit(lambda p, keys: (refl_psi(p, keys),
                                    jax.jacfwd(refl_psi, argnums=0)(p, keys)))

    def diag_fn(psi, keys):
        mu_list, _, _ = _build_mu(psi_to_theta(psi))
        out = []
        for k in range(n_wl):
            out.append(jax.vmap(sim_energy, in_axes=(0, None, None, None, None, None, None, None))(
                keys, mu_list[k], mus_list[k], v12.GS_WL[k], v12.NS_WL[k], bounds, edges, max_steps))
        return out

    diag_jit = jax.jit(diag_fn)

    psi_j = jnp.asarray(psi0)
    n_diag = min(a.diag_photons, n_photons)

    # ---- compile once, on the first seed's key shape -------------------------------------------
    k0 = jax.random.split(jax.random.PRNGKey(todo[0]), n_photons)
    jax.block_until_ready(k0)
    t0 = time.perf_counter()
    compiled = both.lower(psi_j, k0).compile()
    t_compile_main = time.perf_counter() - t0
    compiled_diag, t_compile_diag = None, None
    if a.diagnostic:
        t0 = time.perf_counter()
        compiled_diag = diag_jit.lower(psi_j, k0[:n_diag]).compile()
        t_compile_diag = time.perf_counter() - t0
    print(f"  compiled once: main {t_compile_main:.1f}s"
          + (f", diagnostic {t_compile_diag:.1f}s" if a.diagnostic else ", diagnostic skipped")
          + f" ({n_photons} photons/wavelength, max_steps={max_steps})")

    results = []
    for s in todo:
        keys = jax.random.split(jax.random.PRNGKey(s), n_photons)
        jax.block_until_ready(keys)                    # nothing outstanding when the clock starts

        t0 = time.perf_counter()
        R_j, J_j = compiled(psi_j, keys)
        R = np.asarray(jax.block_until_ready(R_j))     # materialized before the clock stops
        J_psi = np.asarray(jax.block_until_ready(J_j))
        t_exec = time.perf_counter() - t0

        term, t_diag = None, None
        if a.diagnostic:
            t0 = time.perf_counter()
            raw = jax.block_until_ready(compiled_diag(psi_j, keys[:n_diag]))
            t_diag = time.perf_counter() - t0
            term = []
            for k in range(n_wl):
                det_id, w_bin, w_refl, w_tran, w_abs, r_sp = (np.asarray(x) for x in raw[k])
                entered = 1.0 - r_sp
                in_flight = entered - (w_refl + w_tran + w_abs)
                term.append({
                    "wavelength_nm": spec["wavelengths_nm"][k],
                    "mean_specular_reflected": float(r_sp.mean()),
                    "mean_diffuse_reflected": float(w_refl.mean()),
                    "mean_transmitted": float(w_tran.mean()),
                    "mean_absorbed": float(w_abs.mean()),
                    "mean_in_flight_at_termination": float(in_flight.mean()),
                    "in_flight_fraction_of_entered": float(in_flight.mean() / entered.mean()),
                    "fraction_detected": float((det_id >= 0).mean()),
                })

        Rm = R.reshape(n_wl, per)
        p_det = (areas_tiled * R).reshape(n_wl, per).sum(axis=1)
        refl_diag = {
            "min_R_per_wavelength": Rm.min(axis=1).tolist(),
            "max_R_per_wavelength": Rm.max(axis=1).tolist(),
            "n_zero_bins_per_wavelength": (Rm <= 0).sum(axis=1).tolist(),
            "n_zero_bins_total": int((R <= 0).sum()),
            "detection_probability_per_wavelength": p_det.tolist(),
        }

        out = path_for(s)
        # Atomic: write to a temporary name in the same directory, then rename. An interrupted write
        # must never leave a truncated file that the resume check would accept as complete.
        # numpy appends ".npz" to a path that lacks it, which would break the rename. Passing an
        # open handle avoids that rewrite entirely.
        tmp = out + f".tmp.{os.getpid()}.npz"
        with open(tmp, "wb") as _fh:
            np.savez_compressed(
                _fh, R=R, J_psi=J_psi, areas=areas_tiled,
                bin_centers=0.5 * (e[1:] + e[:-1]),
                psi_nominal=psi0, seed=np.array(s), n_transport_photons=np.array(n_photons),
                max_steps=np.array(max_steps), n_wavelengths=np.array(n_wl),
                kappa=np.array(kappa), blood_coordinate=np.array("physical_bvf"),
                profile=np.array("production"), smoke=np.array(bool(a.smoke)),
                ptxas_version=np.array(os.environ.get("MCII_PTXAS_VERSION", "unrecorded")),
                cuda_module=np.array(os.environ.get("MCII_CUDA_MODULE", "unrecorded")),
                jax_version=np.array(jax.__version__),
                jax_backend=np.array(jax.default_backend()),
                min_count_applied=np.array(False),
                epsilon_floor_applied=np.array(False),
                bins_dropped=np.array(0))
        os.replace(tmp, out)

        peak = peak_rss_mb()
        rep = {"profile": "production", "seed": s, "smoke": bool(a.smoke),
               "n_transport_photons_per_wavelength": n_photons, "max_steps": max_steps,
               "n_observables": n_obs, "blood_coordinate": "physical_bvf",
               "psi_labels": labels, "psi_nominal": psi0.tolist(), "kappa": kappa,
               "timing_s": {"jax_compile_main_shared": t_compile_main,
                            "jax_compile_diagnostic_shared": t_compile_diag,
                            "execution": t_exec, "energy_diagnostic": t_diag},
               "compile_shared_across_seeds": True,
               "peak_rss_mb": peak,
               "reflectance_diagnostics": refl_diag,
               "termination_diagnostics": {"n_photons_subsample": n_diag, "per_wavelength": term},
               "artifact": os.path.relpath(out, ROOT),
               "jax_version": jax.__version__, "jax_backend": jax.default_backend(),
               "ptxas_version": os.environ.get("MCII_PTXAS_VERSION", "unrecorded"),
               "cuda_module": os.environ.get("MCII_CUDA_MODULE", "unrecorded"),
               "min_count_applied": False, "epsilon_floor_applied": False, "bins_dropped": 0,
               "platform": platform.platform()}
        rp = os.path.join(ROOT, "results", "production",
                          f"seed_{s}{'_smoke' if a.smoke else ''}.json")
        os.makedirs(os.path.dirname(rp), exist_ok=True)
        with open(rp, "w") as fh:
            json.dump(rep, fh, indent=2)

        msg = (f"  seed {s}: exec {t_exec:7.1f}s  peak {peak:6.0f} MB  "
               f"zeros {refl_diag['n_zero_bins_total']:3d}/{n_obs}")
        if term is not None:
            msg += (f"  diag {t_diag:5.1f}s  in-flight "
                    + " ".join(f"{t['in_flight_fraction_of_entered']:.3f}" for t in term))
        print(msg)
        results.append({"seed": s, "execution_s": t_exec, "peak_rss_mb": peak})

    if results:
        ex = [r["execution_s"] for r in results]
        print(f"\n  {len(results)} seeds: execution median {np.median(ex):.1f}s, "
              f"total {sum(ex):.1f}s, compile paid once ({t_compile_main:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
