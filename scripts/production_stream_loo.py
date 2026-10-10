#!/usr/bin/env python3
"""Leave-one-stream-out sensitivity analysis for the frozen production pool.

This script never runs photon transport.  It reconstructs the committed pooled artifact from the
32 archived stream files, requires every pooled array to match exactly, and then recomputes the
likelihood-weighted local-information summaries after omitting each stream in turn.  The resulting
range and jackknife standard error quantify finite-pool Monte Carlo sensitivity; they are not
sampling intervals for data and do not change the frozen measurement model.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import unify  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "scripts"))
from production_crb import LABELS, full_rank_crlb  # noqa: E402
from unify_spectrum import column_scale, load_spec  # noqa: E402


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jackknife_summary(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    mean = float(a.mean())
    se = float(np.sqrt((a.size - 1) / a.size * np.sum((a - mean) ** 2)))
    return {
        "n_leave_one_out_values": int(a.size),
        "minimum": float(a.min()),
        "maximum": float(a.max()),
        "mean": mean,
        "median": float(np.median(a)),
        "range": float(np.ptp(a)),
        "jackknife_standard_error": se,
    }


def branch_summary(name: str, J: np.ndarray, R: np.ndarray, areas: np.ndarray,
                   n_measurement: int, D: np.ndarray, psi: np.ndarray,
                   n_wavelengths: int) -> dict:
    if name == "multinomial_fixed_launch":
        W = unify.whiten_multinomial(J, R, areas, n_measurement, D, n_wavelengths)
    elif name == "diagonal_poisson":
        W = unify.whiten_poisson(J, R, areas, n_measurement, D)
    else:
        raise ValueError(f"unsupported likelihood branch {name!r}")
    crlb, (singular_values, V, _) = full_rank_crlb(W, D, psi)
    return {
        "deep_StO2_sqrt_crlb": float(crlb["sqrt_crlb_psi"]["StO2_D"]),
        "condition_number_Jw": float(singular_values[0] / singular_values[-1]),
        "near_null_abs_StO2_D_coefficient_in_scaled_coordinate": float(abs(V[2, -1])),
        "numerical_rank": int(crlb["numerical_rank"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--streams", default=os.path.join(
        ROOT, "data", "derived", "production_cuda129"))
    ap.add_argument("--pooled", default=os.path.join(
        ROOT, "data", "derived", "production_pooled.npz"))
    ap.add_argument("--crlb", default=os.path.join(ROOT, "results", "production_crb.json"))
    ap.add_argument("--out", default=os.path.join(
        ROOT, "results", "production_stream_loo.json"))
    a = ap.parse_args()

    spec = load_spec()
    prod = spec["profiles"]["production"]
    n_expected = int(prod["n_transport_streams"])
    n_measurement = int(prod["n_measurement_photons_per_wavelength"])

    pooled = np.load(a.pooled, allow_pickle=False)
    expected_seeds = [int(x) for x in pooled["seeds"]]
    files = sorted(glob.glob(os.path.join(a.streams, "seed_*.npz")),
                   key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
    if len(files) != n_expected:
        raise SystemExit(f"expected {n_expected} stream files, found {len(files)}")

    Rs, Js, seeds, hashes = [], [], [], []
    reference_meta = None
    reference_areas = None
    reference_psi = None
    for path in files:
        with np.load(path, allow_pickle=False) as d:
            seed = int(d["seed"])
            meta = {
                "n_transport_photons": int(d["n_transport_photons"]),
                "max_steps": int(d["max_steps"]),
                "n_wavelengths": int(d["n_wavelengths"]),
                "blood_coordinate": str(d["blood_coordinate"]),
                "profile": str(d["profile"]),
                "ptxas_version": str(d["ptxas_version"]),
                "cuda_module": str(d["cuda_module"]),
                "jax_version": str(d["jax_version"]),
                "jax_backend": str(d["jax_backend"]),
                "smoke": bool(d["smoke"]),
                "min_count_applied": bool(d["min_count_applied"]),
                "epsilon_floor_applied": bool(d["epsilon_floor_applied"]),
                "bins_dropped": int(d["bins_dropped"]),
                "R_shape": list(d["R"].shape),
                "J_psi_shape": list(d["J_psi"].shape),
            }
            if reference_meta is None:
                reference_meta = meta
                reference_areas = np.asarray(d["areas"]).copy()
                reference_psi = np.asarray(d["psi_nominal"]).copy()
            elif meta != reference_meta:
                raise SystemExit(f"stream metadata mismatch in {os.path.basename(path)}")
            Rs.append(np.asarray(d["R"], dtype=float))
            Js.append(np.asarray(d["J_psi"], dtype=float))
            seeds.append(seed)
        hashes.append({"seed": seed, "sha256": sha256(path), "bytes": os.path.getsize(path)})

    if seeds != expected_seeds:
        raise SystemExit(f"stream seeds {seeds} do not match pooled seeds {expected_seeds}")
    if reference_meta is None:
        raise SystemExit("no stream metadata")
    expected_meta = {
        "n_transport_photons": int(prod["n_transport_photons_per_wavelength_per_stream"]),
        "max_steps": int(prod["max_steps"]),
        "n_wavelengths": len(spec["wavelengths_nm"]),
        "blood_coordinate": "physical_bvf",
        "profile": "production",
        "ptxas_version": "12.9.86",
        "cuda_module": "cuda/12.9",
        "jax_version": "0.9.1",
        "jax_backend": "gpu",
        "smoke": False,
        "min_count_applied": False,
        "epsilon_floor_applied": False,
        "bins_dropped": 0,
        "R_shape": [int(prod["n_observables"])],
        "J_psi_shape": [int(prod["n_observables"]), len(LABELS)],
    }
    if reference_meta != expected_meta:
        raise SystemExit(f"stream metadata does not match frozen production profile: {reference_meta}")

    Rs_a, Js_a = np.stack(Rs), np.stack(Js)
    R_full, J_full = Rs_a.mean(axis=0), Js_a.mean(axis=0)
    exact = {
        "R": bool(np.array_equal(R_full, pooled["R"])),
        "J_psi": bool(np.array_equal(J_full, pooled["J_psi"])),
        "areas": bool(np.array_equal(pooled["areas"], reference_areas)),
        "psi_nominal": bool(np.array_equal(pooled["psi_nominal"], reference_psi)),
    }
    if not all(exact.values()):
        raise SystemExit(f"stream reconstruction does not exactly reproduce pooled artifact: {exact}")

    psi = np.asarray(pooled["psi_nominal"], dtype=float)
    areas = np.asarray(pooled["areas"], dtype=float)
    n_wavelengths = int(pooled["n_wavelengths"])
    D = column_scale(psi)
    branches = ("multinomial_fixed_launch", "diagonal_poisson")
    full = {name: branch_summary(
        name, J_full, R_full, areas, n_measurement, D, psi, n_wavelengths)
        for name in branches}

    with open(a.crlb) as fh:
        frozen_crlb = json.load(fh)
    frozen_headline = float(frozen_crlb["branches"]["multinomial_fixed_launch"]
                            ["crlb"]["sqrt_crlb_psi"]["StO2_D"])
    if full["multinomial_fixed_launch"]["deep_StO2_sqrt_crlb"] != frozen_headline:
        raise SystemExit("reconstructed full-pool headline does not exactly match production_crb.json")

    loo = []
    sum_R, sum_J = Rs_a.sum(axis=0), Js_a.sum(axis=0)
    for i, seed in enumerate(seeds):
        R_i = (sum_R - Rs_a[i]) / (n_expected - 1)
        J_i = (sum_J - Js_a[i]) / (n_expected - 1)
        if not np.all(R_i > 0):
            raise SystemExit(f"leave-one-out pool excluding seed {seed} has non-positive R")
        entry = {"omitted_seed": seed, "branches": {}}
        for name in branches:
            entry["branches"][name] = branch_summary(
                name, J_i, R_i, areas, n_measurement, D, psi, n_wavelengths)
        loo.append(entry)

    summaries = {}
    for name in branches:
        summaries[name] = {}
        for metric in ("deep_StO2_sqrt_crlb", "condition_number_Jw",
                       "near_null_abs_StO2_D_coefficient_in_scaled_coordinate"):
            values = [row["branches"][name][metric] for row in loo]
            s = jackknife_summary(values)
            theta = full[name][metric]
            s["full_pool_value"] = theta
            s["maximum_absolute_shift_from_full_pool"] = float(
                max(abs(v - theta) for v in values))
            s["maximum_relative_shift_from_full_pool"] = float(
                s["maximum_absolute_shift_from_full_pool"] / abs(theta))
            summaries[name][metric] = s

    payload = {
        "status": "READ-ONLY FINITE-POOL SENSITIVITY; NOT A SAMPLING INTERVAL",
        "transport_rerun": False,
        "measurement_model_changed": False,
        "interpretation": (
            "Leave-one-stream-out sensitivity of the frozen 32-stream Monte Carlo pool. The "
            "jackknife standard errors quantify finite-pool Monte Carlo variability of the "
            "reported functionals; they are not confidence intervals for measured data."),
        "inputs": {
            "stream_directory": os.path.relpath(a.streams, ROOT),
            "pooled_artifact": os.path.relpath(a.pooled, ROOT),
            "pooled_sha256": sha256(a.pooled),
            "production_crb": os.path.relpath(a.crlb, ROOT),
            "production_crb_sha256": sha256(a.crlb),
            "stream_files": hashes,
        },
        "frozen_metadata": reference_meta,
        "exact_reconstruction_of_committed_pool": exact,
        "n_streams": n_expected,
        "full_pool": full,
        "leave_one_stream_out_summary": summaries,
        "leave_one_stream_out": loo,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, a.out)

    h = summaries["multinomial_fixed_launch"]["deep_StO2_sqrt_crlb"]
    v = summaries["multinomial_fixed_launch"][
        "near_null_abs_StO2_D_coefficient_in_scaled_coordinate"]
    print(f"  exact pooled reconstruction: {all(exact.values())} ({n_expected} streams)")
    print(f"  headline deep-StO2 sqrt(CRLB): full {h['full_pool_value']:.9f}; "
          f"LOO {h['minimum']:.9f}--{h['maximum']:.9f}; "
          f"max relative shift {100*h['maximum_relative_shift_from_full_pool']:.3f}%")
    print(f"  near-null |StO2_D| coefficient: full {v['full_pool_value']:.9f}; "
          f"LOO {v['minimum']:.9f}--{v['maximum']:.9f}")
    print(f"  wrote {os.path.relpath(a.out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
