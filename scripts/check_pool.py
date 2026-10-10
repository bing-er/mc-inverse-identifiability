#!/usr/bin/env python3
"""check_pool.py -- gate the production pool before any spectral analysis runs.

The three analyses (raw-J SVD, diagonal-Poisson whitening, fixed-launch multinomial whitening) all
consume ONE pooled R and J. Whitening is undefined on a non-positive bin, so the pool must be proven
strictly positive in all 120 bins first. This script is that gate, and it also writes the pooled
artifact the analyses share, so transport is never rerun to change a likelihood.

    python3 scripts/check_pool.py
"""
from __future__ import annotations

import argparse, glob, json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=os.path.join(ROOT, "data", "derived", "production_cuda129"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "derived", "production_pooled.npz"))
    ap.add_argument("--report", default=os.path.join(ROOT, "results", "pool_check.json"))
    ap.add_argument("--require-ptxas", default="12.9.86")
    ap.add_argument("--allow-partial", action="store_true",
                    help="pool whatever is present instead of requiring the full 32")
    a = ap.parse_args()

    with open(os.path.join(ROOT, "spec", "baseline_spec.json"), encoding="utf-8") as stream:
        spec = json.load(stream)
    prof = spec["profiles"]["production"]
    n_expected = prof["n_transport_streams"]
    n_obs_expected = prof["n_observables"]

    files = sorted(glob.glob(os.path.join(a.pool, "seed_*.npz")))
    if not files:
        raise FileNotFoundError(f"no seed files in {a.pool}")
    if len(files) != n_expected and not a.allow_partial:
        raise SystemExit(f"  pool has {len(files)} of {n_expected} seeds; "
                         f"rerun the missing ones or pass --allow-partial")

    Rs, Js, seeds, prov = [], [], [], []
    ref = None
    required_ptxas = a.require_ptxas
    for f in files:
        with np.load(f) as d:
            if bool(d["smoke"]):
                raise SystemExit(f"  {os.path.basename(f)} is a smoke result and must not be pooled")
            meta = (int(d["n_transport_photons"]), int(d["max_steps"]), int(d["n_wavelengths"]),
                    str(d["blood_coordinate"]), str(d["profile"]))
            if ref is None:
                ref, areas, centers, psi0 = meta, d["areas"], d["bin_centers"], d["psi_nominal"]
                shape_ref = (d["R"].shape, d["J_psi"].shape)
            elif meta != ref:
                raise SystemExit(f"  {os.path.basename(f)} disagrees with the pool: {meta} vs {ref}")
            if (d["R"].shape, d["J_psi"].shape) != shape_ref:
                raise SystemExit(f"  {os.path.basename(f)} shape mismatch: "
                                 f"{(d['R'].shape, d['J_psi'].shape)} vs {shape_ref}")
            g = lambda k, dflt="unrecorded": str(d[k]) if k in d.files else dflt
            prov.append({"seed": int(d["seed"]), "ptxas": g("ptxas_version"),
                         "cuda": g("cuda_module"), "jax": g("jax_version"),
                         "backend": g("jax_backend"),
                         "min_count": g("min_count_applied", "unrecorded"),
                         "eps_floor": g("epsilon_floor_applied", "unrecorded"),
                         "bins_dropped": g("bins_dropped", "unrecorded")})
            Rs.append(d["R"]); Js.append(d["J_psi"]); seeds.append(int(d["seed"]))

    # --- seed completeness and uniqueness -------------------------------------------------------
    expected_seeds = [43000 + 1000 * i for i in range(n_expected)]
    dup = sorted({x for x in seeds if seeds.count(x) > 1})
    missing = [x for x in expected_seeds if x not in seeds]
    extra = [x for x in seeds if x not in expected_seeds]

    # --- compiler provenance --------------------------------------------------------------------
    bad_ptxas = [p for p in prov if p["ptxas"] != required_ptxas]
    non_gpu = [p for p in prov if p["backend"] != "gpu"]

    # --- no filtering was applied ---------------------------------------------------------------
    filtered = [p for p in prov
                if p["min_count"] != "False" or p["eps_floor"] != "False" or p["bins_dropped"] != "0"]

    R = np.mean(Rs, axis=0)
    J = np.mean(Js, axis=0)
    if R.size != n_obs_expected:
        raise SystemExit(f"  pooled R has {R.size} bins, spec declares {n_obs_expected}")

    n_wl = ref[2]; per = R.size // n_wl
    Rm = R.reshape(n_wl, per)
    n_nonpos = int((R <= 0).sum())
    per_wl_nonpos = (Rm <= 0).sum(axis=1).tolist()
    finite = bool(np.all(np.isfinite(R)) and np.all(np.isfinite(J)))
    p_det = (areas * R).reshape(n_wl, per).sum(axis=1)
    p_ok = bool(np.all((p_det > 0) & (p_det < 1)))

    gates = {
        "exactly_n_seeds": len(files) == n_expected,
        "no_missing_seeds": not missing,
        "no_duplicate_seeds": not dup,
        "no_unexpected_seeds": not extra,
        f"all_ptxas_{required_ptxas}": not bad_ptxas,
        "all_gpu_backend": not non_gpu,
        "config_and_shape_identical": True,          # enforced by raising above
        "pooled_R_strictly_positive_all_bins": n_nonpos == 0,
        "pooled_R_and_J_finite": finite,
        "detection_probability_in_open_unit_interval": p_ok,
        "no_min_count_or_epsilon_floor_or_bin_dropping": not filtered,
    }
    ok = all(gates.values())

    rep = {"gates": gates, "all_gates_passed": ok,
           "missing_seeds": missing, "duplicate_seeds": dup, "unexpected_seeds": extra,
           "provenance": prov,
           "n_seeds_pooled": len(files), "seeds": seeds,
           "n_seeds_expected": n_expected, "partial": len(files) != n_expected,
           "n_observables": int(R.size),
           "n_nonpositive_bins": n_nonpos,
           "nonpositive_per_wavelength": per_wl_nonpos,
           "min_R_per_wavelength": Rm.min(axis=1).tolist(),
           "detection_probability_per_wavelength": p_det.tolist(),
           "all_finite": finite,
           "strictly_positive_120_of_120": n_nonpos == 0,
           "whitening_eligible": ok,
           "n_transport_photons_per_wavelength_per_stream": ref[0],
           "pooled_total_photons_per_wavelength": ref[0] * len(files),
           "max_steps": ref[1], "blood_coordinate": ref[3]}

    if ok:
        with open(a.out + ".tmp.npz", "wb") as fh:
            np.savez_compressed(fh, R=R, J_psi=J, areas=areas, bin_centers=centers,
                                psi_nominal=psi0, seeds=np.array(seeds),
                                n_seeds=np.array(len(files)),
                                n_transport_photons=np.array(ref[0]),
                                max_steps=np.array(ref[1]), n_wavelengths=np.array(n_wl),
                                blood_coordinate=np.array(ref[3]), profile=np.array("production"))
        os.replace(a.out + ".tmp.npz", a.out)
        rep["pooled_artifact"] = os.path.relpath(a.out, ROOT)

    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    with open(a.report, "w") as fh:
        json.dump(rep, fh, indent=2)

    print(f"  pooled {len(files)} of {n_expected} seeds "
          f"({ref[0] * len(files)} photons/wavelength total, max_steps={ref[1]})")
    print(f"  non-positive bins: {n_nonpos} of {R.size}  per wavelength {per_wl_nonpos}")
    print(f"  min R per wavelength: " + " ".join(f"{x:.3e}" for x in Rm.min(axis=1)))
    print(f"  detection probability: " + " ".join(f"{x:.4f}" for x in p_det))
    print("  --- gates ---")
    for k, v in gates.items():
        print(f"    {'PASS' if v else 'FAIL'}  {k}")
    if missing: print(f"    missing seeds: {missing}")
    if dup:     print(f"    duplicate seeds: {dup}")
    if bad_ptxas: print(f"    wrong ptxas: {[(p['seed'], p['ptxas']) for p in bad_ptxas]}")
    print(f"  -> {'WHITENING ELIGIBLE' if ok else 'NOT ELIGIBLE'}")
    if ok:
        print(f"  wrote {os.path.relpath(a.out, ROOT)} for all three analyses to share")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
