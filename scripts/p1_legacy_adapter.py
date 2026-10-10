#!/usr/bin/env python3
"""p1_legacy_adapter.py -- Step B of the p1_legacy profile.

Rerun P1's forward model at the NOMINAL operating point only (blood-volume scale = 1.0) and emit a
standardized artifact carrying both the Jacobian and the reflectance:

    data/derived/p1_legacy_nominal.npz   ->  J_phi (n_obs, 6), R (n_obs,), areas (n_obs,), metadata

P1's own script caches J but never stores R, because it forms no noise model. R is required here
because whitening needs the observation covariance. The six-point blood-volume scan is not rerun.

The upstream jax-pad repository is imported read-only and is never written to. Its constants are
asserted against spec/baseline_spec.json's p1_legacy profile, so a drift on either side is an error
rather than a silent mismatch.

The adapter is validated by comparing its Jacobian against P1's frozen J_scale_1.0.

    JAXPAD_ROOT=../jax-pad python3 scripts/p1_legacy_adapter.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from functools import partial

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

# P1's nominal operating point, from jax-pad/scripts/inverse/p1_jacobian_spectrum.py
NOMINAL_BOUNDARIES = [0.0020, 0.0100, 0.0250, 0.0330, 0.1830, 0.1910, 0.8000]
STO2_U_TRUE, STO2_D_TRUE = 0.95, 0.65
B_U_NOM, B_D_NOM = 0.30, 0.10          # legacy blood coordinate
NOMINAL_SCALE = 1.0
SEED_KEYS, SEED_NOISE = 43000, 53000


def load_profile(spec_path):
    with open(spec_path) as fh:
        spec = json.load(fh)
    return spec, spec["profiles"]["p1_legacy"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jaxpad-root", default=os.environ.get(
        "JAXPAD_ROOT", os.path.normpath(os.path.join(ROOT, "..", "jax-pad"))))
    ap.add_argument("--n-photons", type=int, default=None,
                    help="defaults to the spec's p1_legacy n_transport_photons")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="defaults to the spec's p1_legacy max_steps")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "derived", "p1_legacy_nominal.npz"))
    ap.add_argument("--report", default=os.path.join(ROOT, "results", "p1_legacy_stepB.json"))
    a = ap.parse_args()

    spec, prof = load_profile(os.path.join(ROOT, "spec", "baseline_spec.json"))
    n_photons = a.n_photons if a.n_photons is not None \
        else prof["n_transport_photons_per_wavelength_per_stream"]
    max_steps = a.max_steps if a.max_steps is not None else prof["max_steps"]

    if not os.path.isdir(a.jaxpad_root):
        raise FileNotFoundError(f"jax-pad not found at {a.jaxpad_root}")
    sys.path.insert(0, os.path.abspath(a.jaxpad_root))

    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from scripts.inverse.phase2_v12_runner import (       # noqa: E402
        Phase2RunConfig, scattering_for_transport, _build_mu)
    import scripts.archive.inverse.multi_wl_v12_colab as v12   # noqa: E402

    # --- gate: upstream constants must match the frozen p1_legacy profile -----------------------
    radial = prof["radial"]
    mismatches = []
    if float(v12.DETECTOR_R_MAX) != radial["r_max_cm"]:
        mismatches.append(f"DETECTOR_R_MAX {v12.DETECTOR_R_MAX} != {radial['r_max_cm']}")
    if float(v12.MAX_R_CUTOFF) != radial["bin_centers_cut_cm"]:
        mismatches.append(f"MAX_R_CUTOFF {v12.MAX_R_CUTOFF} != {radial['bin_centers_cut_cm']}")
    if int(v12.N_WL) != len(spec["wavelengths_nm"]):
        mismatches.append(f"N_WL {v12.N_WL} != {len(spec['wavelengths_nm'])}")
    if mismatches:
        raise ValueError("upstream jax-pad constants disagree with spec p1_legacy: "
                         + "; ".join(mismatches))

    edges = jnp.linspace(0.0, v12.DETECTOR_R_MAX, radial["n_bins"] + 1, dtype=jnp.float64)
    centers = np.asarray(v12.compute_bin_centers(edges))
    keep = centers <= v12.MAX_R_CUTOFF
    e = np.asarray(edges)
    areas_kept = (np.pi * (e[1:] ** 2 - e[:-1] ** 2))[keep]
    n_obs = int(keep.sum()) * int(v12.N_WL)
    if n_obs != prof["n_observables"]:
        raise ValueError(f"computed {n_obs} observables, spec declares {prof['n_observables']}")

    # --- build P1's solvers at the nominal point ------------------------------------------------
    bu, bd = B_U_NOM * NOMINAL_SCALE, B_D_NOM * NOMINAL_SCALE
    cfg = Phase2RunConfig(n_photons_final=n_photons, seed_keys=SEED_KEYS, seed_noise=SEED_NOISE,
                          sto2_ubnp_true=STO2_U_TRUE, sto2_dbnp_true=STO2_D_TRUE,
                          b_ubnp_true=bu, b_dbnp_true=bd, max_steps=max_steps)
    keys = jax.random.split(jax.random.PRNGKey(cfg.seed_keys), cfg.n_photons_final)
    b = jnp.array(NOMINAL_BOUNDARIES, dtype=jnp.float64)
    solvers = []
    for k in range(v12.N_WL):
        mus = scattering_for_transport(v12.MUSS_WL[k], v12.GS_WL[k], cfg.scattering_convention)
        solvers.append(partial(v12.forward_solver, keys=keys, muss=mus, gs=v12.GS_WL[k],
                               ns=v12.NS_WL[k], layer_boundaries=b, detector_r_edges=edges,
                               max_steps=cfg.max_steps))

    def theta_log(su, bu_, sd, bd_, dermr=0.49, subq=0.45):
        vals = [bu_ * su, bu_ * (1 - su), bd_ * sd, bd_ * (1 - sd), dermr, subq]
        return jnp.log(jnp.array(vals, dtype=jnp.float64) + v12.EPS)

    def refl(th):
        mu_list, _, _ = _build_mu(th)
        return jnp.concatenate([solvers[k](muas=mu_list[k])[keep] for k in range(v12.N_WL)])

    theta = theta_log(STO2_U_TRUE, bu, STO2_D_TRUE, bd)
    print(f"  nominal scale {NOMINAL_SCALE}: n_transport_photons={n_photons}, "
          f"max_steps={max_steps}, {n_obs} observables")
    R = np.asarray(refl(theta))
    J_phi = np.asarray(jax.jacfwd(refl)(theta))

    # --- physical sanity: A*R is a detection probability, strictly inside (0, 1) per wavelength --
    per = int(keep.sum())
    areas_tiled = np.tile(areas_kept, int(v12.N_WL))
    p = areas_tiled * R
    p_tot = p.reshape(int(v12.N_WL), per).sum(axis=1)
    if np.any(R < 0):
        raise ValueError("negative reflectance density returned by the forward solver")
    if not np.all((p_tot > 0) & (p_tot < 1)):
        raise ValueError(f"per-wavelength detection probability outside (0,1): {p_tot}")

    # --- validation: does this adapter reproduce P1's frozen Jacobian? --------------------------
    frozen_path = os.path.join(a.jaxpad_root, "outputs", "p1_jacobian_spectrum_jacobians.npz")
    val = {"frozen_available": os.path.exists(frozen_path)}
    if val["frozen_available"]:
        Jf = np.load(frozen_path)[f"J_scale_{NOMINAL_SCALE}"]
        val["frozen_shape"] = list(Jf.shape)
        if Jf.shape == J_phi.shape:
            scale_J = max(float(np.max(np.abs(Jf))), np.finfo(float).tiny)
            val["max_abs_diff"] = float(np.max(np.abs(J_phi - Jf)))
            val["max_abs_diff_over_max_abs_J"] = float(val["max_abs_diff"] / scale_J)
            val["bitwise_identical"] = bool(np.array_equal(J_phi, Jf))
        else:
            val["shape_mismatch"] = True

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(
        a.out,
        J_phi=J_phi, R=R, areas=areas_tiled, bin_centers_kept=centers[keep],
        phi_nominal=np.asarray(theta), psi_nominal=np.array(
            [STO2_U_TRUE, bu, STO2_D_TRUE, bd, 0.49, 0.45]),
        profile=np.array("p1_legacy"), blood_coordinate=np.array("legacy"),
        n_transport_photons=np.array(n_photons), max_steps=np.array(max_steps),
        n_wavelengths=np.array(int(v12.N_WL)), scale=np.array(NOMINAL_SCALE),
        seed_keys=np.array(SEED_KEYS),
    )

    report = {"step": "B", "profile": "p1_legacy", "transport_run": True,
              "scale": NOMINAL_SCALE, "n_transport_photons": n_photons, "max_steps": max_steps,
              "n_observables": n_obs, "per_wavelength_detection_probability": p_tot.tolist(),
              "R_min": float(R.min()), "R_max": float(R.max()),
              "jacobian_validation_against_frozen": val,
              "artifact": os.path.relpath(a.out, ROOT)}
    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    with open(a.report, "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"  per-wavelength detection probability: "
          f"{', '.join(f'{x:.4f}' for x in p_tot)}")
    if val.get("bitwise_identical"):
        print("  Jacobian vs frozen J_scale_1.0: BITWISE IDENTICAL")
    elif "max_abs_diff_over_max_abs_J" in val:
        print(f"  Jacobian vs frozen J_scale_1.0: max|dJ|/max|J| = "
              f"{val['max_abs_diff_over_max_abs_J']:.3e}")
    print(f"  wrote {os.path.relpath(a.out, ROOT)} and {os.path.relpath(a.report, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
