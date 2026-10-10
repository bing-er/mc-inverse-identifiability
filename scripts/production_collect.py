#!/usr/bin/env python3
"""production_collect.py -- per-seed, resumable collector for the production transport pool.

One seed per invocation, one file per seed, so the pool can be built incrementally and an array job
can fan out without any shared state. An existing output is left alone unless --force is given.

The Jacobian is taken directly with respect to the production coordinates

    psi = [StO2_U, f_U, StO2_D, f_D, mua_DermR, mua_SubQ]

with f in PHYSICAL blood volume fraction. The upstream engine is parameterized by legacy blood
concentrations, so the wrapper maps psi -> legacy log-concentrations with the documented factor
kappa = 5.355813953488371 and differentiates through that map. One autodiff pass, no hand-written
chain rule, and the blood-coordinate convention is handled in one place.

Recorded per seed: JAX compile time and execution time separately, peak resident memory,
per-wavelength min(R) and empty-bin counts, and an energy-accounting termination diagnostic.

The three spectral analyses downstream share this single pooled R/J. Transport is never repeated for
a different likelihood.

    python3 scripts/production_collect.py --seed 43000 --smoke     # validate the code path cheaply
    python3 scripts/production_collect.py --seed 43000             # full production parameters
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
SEED_BASE, SEED_STRIDE, N_SEEDS = 43000, 1000, 32     # matches P2's pool convention


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 ** 2) if platform.system() == "Darwin" else r / 1024


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--jaxpad-root", default=os.environ.get(
        "JAXPAD_ROOT", os.path.normpath(os.path.join(ROOT, "..", "jax-pad"))))
    ap.add_argument("--outdir", default=os.path.join(ROOT, "data", "derived", "production"))
    ap.add_argument("--force", action="store_true", help="recompute even if the seed file exists")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny photon count and step cap, to validate the path before a long run")
    ap.add_argument("--diag-photons", type=int, default=20000,
                    help="photon subsample for the energy-accounting diagnostic")
    a = ap.parse_args()

    spec = json.load(open(os.path.join(ROOT, "spec", "baseline_spec.json")))
    prof = spec["profiles"]["production"]
    kappa = spec["blood_coordinate"]["legacy_to_physical_divisor"]
    nom = spec["parameters"]["nominal_psi"]
    labels = spec["parameters"]["psi_labels"]
    radial = prof["radial"]

    n_photons = 2000 if a.smoke else prof["n_transport_photons_per_wavelength_per_stream"]
    max_steps = 200 if a.smoke else prof["max_steps"]

    os.makedirs(a.outdir, exist_ok=True)
    # A smoke run must never be mistakable for a pool member: it gets its own name and its own
    # directory, so the resumable check on the real pool is unaffected either way.
    if a.smoke:
        a.outdir = os.path.join(a.outdir, "_smoke")
        os.makedirs(a.outdir, exist_ok=True)
        out = os.path.join(a.outdir, f"seed_{a.seed}_smoke.npz")
    else:
        out = os.path.join(a.outdir, f"seed_{a.seed}.npz")
    if os.path.exists(out) and not a.force:
        print(f"  seed {a.seed}: {os.path.relpath(out, ROOT)} exists, skipping (use --force)")
        return 0

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
    if n_wl != len(spec["wavelengths_nm"]):
        raise ValueError(f"engine N_WL {n_wl} != spec {len(spec['wavelengths_nm'])}")

    # production grid: 30 bins to 1.5 cm, all retained
    edges = jnp.linspace(radial["r_min_cm"], radial["r_max_cm"], radial["n_bins"] + 1,
                         dtype=jnp.float64)
    e = np.asarray(edges)
    areas = np.pi * (e[1:] ** 2 - e[:-1] ** 2)
    n_obs = radial["n_bins"] * n_wl
    if n_obs != prof["n_observables"]:
        raise ValueError(f"grid gives {n_obs} observables, spec declares {prof['n_observables']}")

    psi0 = np.array([nom[k] for k in labels], dtype=float)

    cfg = Phase2RunConfig(n_photons_final=n_photons, seed_keys=a.seed, seed_noise=a.seed + 10000,
                          sto2_ubnp_true=psi0[0], sto2_dbnp_true=psi0[2],
                          b_ubnp_true=psi0[1] * kappa, b_dbnp_true=psi0[3] * kappa,
                          max_steps=max_steps)
    keys = jax.random.split(jax.random.PRNGKey(a.seed), n_photons)
    bounds = jnp.array(NOMINAL_BOUNDARIES, dtype=jnp.float64)

    solvers, mus_list = [], []
    for k in range(n_wl):
        mus = scattering_for_transport(v12.MUSS_WL[k], v12.GS_WL[k], cfg.scattering_convention)
        mus_list.append(mus)
        solvers.append(partial(v12.forward_solver, keys=keys, muss=mus, gs=v12.GS_WL[k],
                               ns=v12.NS_WL[k], layer_boundaries=bounds,
                               detector_r_edges=edges, max_steps=max_steps))

    def psi_to_theta(psi):
        """physical-BVF psi -> the engine's legacy log-concentrations."""
        s_u, f_u, s_d, f_d, dermr, subq = psi
        c = jnp.stack([kappa * f_u * s_u, kappa * f_u * (1.0 - s_u),
                       kappa * f_d * s_d, kappa * f_d * (1.0 - s_d), dermr, subq])
        return jnp.log(c + v12.EPS)

    def refl_psi(psi):
        mu_list, _, _ = _build_mu(psi_to_theta(psi))
        return jnp.concatenate([solvers[k](muas=mu_list[k]) for k in range(n_wl)])

    both = jax.jit(lambda p: (refl_psi(p), jax.jacfwd(refl_psi)(p)))

    print(f"  seed {a.seed}{' [SMOKE]' if a.smoke else ''}: {n_photons} photons/wavelength, "
          f"max_steps={max_steps}, {n_obs} observables, psi in physical BVF")

    t0 = time.perf_counter()
    compiled = both.lower(jnp.asarray(psi0)).compile()
    t_compile = time.perf_counter() - t0

    t0 = time.perf_counter()
    R_j, J_j = compiled(jnp.asarray(psi0))
    R = np.asarray(jax.block_until_ready(R_j))
    J_psi = np.asarray(jax.block_until_ready(J_j))
    t_exec = time.perf_counter() - t0

    # --- per-wavelength reflectance diagnostics --------------------------------------------------
    per = radial["n_bins"]
    Rm = R.reshape(n_wl, per)
    areas_tiled = np.tile(areas, n_wl)
    p_det = (areas_tiled * R).reshape(n_wl, per).sum(axis=1)
    refl_diag = {
        "min_R_per_wavelength": Rm.min(axis=1).tolist(),
        "max_R_per_wavelength": Rm.max(axis=1).tolist(),
        "n_zero_bins_per_wavelength": (Rm <= 0).sum(axis=1).tolist(),
        "n_zero_bins_total": int((R <= 0).sum()),
        "detection_probability_per_wavelength": p_det.tolist(),
    }

    # --- energy-accounting termination diagnostic, on a subsample --------------------------------
    n_diag = min(a.diag_photons, n_photons)
    kd = keys[:n_diag]
    mu_list, _, _ = _build_mu(psi_to_theta(jnp.asarray(psi0)))
    term = []
    t0 = time.perf_counter()
    for k in range(n_wl):
        det_id, w_bin, w_refl, w_tran, w_abs, r_sp = jax.vmap(
            sim_energy, in_axes=(0, None, None, None, None, None, None, None))(
            kd, mu_list[k], mus_list[k], v12.GS_WL[k], v12.NS_WL[k], bounds, edges, max_steps)
        w_refl, w_tran, w_abs = (np.asarray(x) for x in (w_refl, w_tran, w_abs))
        r_sp = np.asarray(r_sp)
        entered = 1.0 - r_sp                       # weight that actually enters the tissue
        accounted = w_refl + w_tran + w_abs
        in_flight = entered - accounted            # truncated by the step cap or the weight cutoff
        term.append({
            "wavelength_nm": spec["wavelengths_nm"][k],
            "mean_specular_reflected": float(r_sp.mean()),
            "mean_diffuse_reflected": float(w_refl.mean()),
            "mean_transmitted": float(w_tran.mean()),
            "mean_absorbed": float(w_abs.mean()),
            "mean_in_flight_at_termination": float(in_flight.mean()),
            "in_flight_fraction_of_entered": float(in_flight.mean() / entered.mean()),
            "max_abs_accounting_residual": float(np.max(np.abs(entered - accounted - in_flight))),
            "fraction_detected": float((np.asarray(det_id) >= 0).mean()),
        })
    t_diag = time.perf_counter() - t0

    peak_mb = peak_rss_mb()

    np.savez_compressed(
        out, R=R, J_psi=J_psi, areas=areas_tiled, bin_centers=0.5 * (e[1:] + e[:-1]),
        psi_nominal=psi0, seed=np.array(a.seed),
        n_transport_photons=np.array(n_photons), max_steps=np.array(max_steps),
        n_wavelengths=np.array(n_wl), kappa=np.array(kappa),
        blood_coordinate=np.array("physical_bvf"), profile=np.array("production"),
        smoke=np.array(bool(a.smoke)),
    )

    report = {
        "profile": "production", "seed": a.seed, "smoke": bool(a.smoke),
        "n_transport_photons_per_wavelength": n_photons, "max_steps": max_steps,
        "n_observables": n_obs, "blood_coordinate": "physical_bvf",
        "psi_labels": labels, "psi_nominal": psi0.tolist(), "kappa": kappa,
        "timing_s": {"jax_compile": t_compile, "execution": t_exec,
                     "energy_diagnostic": t_diag},
        "peak_rss_mb": peak_mb,
        "reflectance_diagnostics": refl_diag,
        "termination_diagnostics": {"n_photons_subsample": n_diag, "per_wavelength": term},
        "artifact": os.path.relpath(out, ROOT),
        "jax_version": jax.__version__, "platform": platform.platform(),
    }
    rep_path = os.path.join(ROOT, "results", "production",
                            f"seed_{a.seed}{'_smoke' if a.smoke else ''}.json")
    os.makedirs(os.path.dirname(rep_path), exist_ok=True)
    with open(rep_path, "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"  compile {t_compile:8.1f}s   execute {t_exec:8.1f}s   "
          f"diag {t_diag:6.1f}s   peak RSS {peak_mb:7.0f} MB")
    print(f"  zero bins: {refl_diag['n_zero_bins_per_wavelength']}  (total "
          f"{refl_diag['n_zero_bins_total']} of {n_obs})")
    print(f"  min R per wavelength: " + " ".join(f"{x:.3e}" for x in refl_diag["min_R_per_wavelength"]))
    print(f"  detection probability: " + " ".join(f"{x:.4f}" for x in p_det))
    print(f"  in-flight at termination: " + " ".join(
        f"{t['in_flight_fraction_of_entered']:.4f}" for t in term))
    print(f"  wrote {os.path.relpath(out, ROOT)} and {os.path.relpath(rep_path, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
