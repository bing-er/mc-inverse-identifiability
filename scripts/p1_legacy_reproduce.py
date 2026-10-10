#!/usr/bin/env python3
"""p1_legacy_reproduce.py -- Step A of the p1_legacy profile.

Reproduce P1's original raw-Jacobian SVD from the FROZEN Jacobian, running no photon transport.

This is the migration check: before anything is recomputed or reparameterized, the new repository
must be able to reproduce P1's published numbers exactly from P1's own cached artifact. Only the
nominal scale = 1.0 point is used; the six-point blood-volume scan is not rerun.

The upstream jax-pad repository is read only. Nothing here writes to it.

Reference quantities, all recomputed and compared against P1's frozen JSON:
  singular values, s_min/s_max, |<e_StO2, v_min>|, the weakest-1 and weakest-2 subspace projections,
  the per-direction responses ||J e||, and the two response ratios.

Additionally cross-checks that this repository's own phi-space StO2 direction construction agrees
with the direction vectors P1 stored alongside its Jacobians.

    python3 scripts/p1_legacy_reproduce.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import unify  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = os.path.normpath(os.path.join(ROOT, "..", "jax-pad", "outputs"))

# P1's nominal operating point, from jax-pad/scripts/inverse/p1_jacobian_spectrum.py
STO2_U_TRUE, STO2_D_TRUE = 0.95, 0.65
NOMINAL_SCALE = 1.0


def p1_direction(kind, s, offset):
    """Byte-for-byte reimplementation of P1's `direction()` helper, for the comparison."""
    e = np.zeros(6)
    if kind == "sto2":
        e[offset], e[offset + 1] = 1.0 / s, -1.0 / (1.0 - s)
    else:
        e[offset], e[offset + 1] = 1.0, 1.0
    return e / np.linalg.norm(e)


def recompute(J, e_su, e_sd):
    """Exactly P1's computation, in P1's order."""
    s = np.linalg.svd(J, compute_uv=False)
    _, _, Vt = np.linalg.svd(J)
    vmin = Vt[-1]
    e_bu = p1_direction("bv", None, 0)
    e_bd = p1_direction("bv", None, 2)
    resp = {k: float(np.linalg.norm(J @ v))
            for k, v in (("sto2U", e_su), ("bvU", e_bu), ("sto2D", e_sd), ("bvD", e_bd))}
    return {
        "response": resp,
        "resp_ratio_U": resp["sto2U"] / resp["bvU"],
        "resp_ratio_D": resp["sto2D"] / resp["bvD"],
        "singular_values": s.tolist(),
        "smallest_over_largest": float(s[-1] / s[0]),
        "align_sto2U_vmin": float(abs(e_su @ vmin)),
        "align_sto2D_vmin": float(abs(e_sd @ vmin)),
        "sto2U_in_weakest1": float(np.linalg.norm(Vt[-1:] @ e_su)),
        "sto2U_in_weakest2": float(np.linalg.norm(Vt[-2:] @ e_su)),
        "sto2D_in_weakest1": float(np.linalg.norm(Vt[-1:] @ e_sd)),
        "sto2D_in_weakest2": float(np.linalg.norm(Vt[-2:] @ e_sd)),
    }


def rel(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = np.maximum(np.abs(b), np.finfo(float).tiny)
    return float(np.max(np.abs(a - b) / denom))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", default=UPSTREAM,
                    help="read-only directory holding P1's frozen outputs")
    ap.add_argument("--tol", type=float, default=1e-12,
                    help="maximum relative residual accepted against P1's frozen JSON")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "p1_legacy_stepA.json"))
    a = ap.parse_args()

    npz_path = os.path.join(a.upstream, "p1_jacobian_spectrum_jacobians.npz")
    json_path = os.path.join(a.upstream, "p1_jacobian_spectrum.json")
    for p in (npz_path, json_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"frozen P1 artifact not found: {p}")

    d = np.load(npz_path)
    key = f"J_scale_{NOMINAL_SCALE}"
    if key not in d:
        raise KeyError(f"{npz_path} has no {key!r}; available: {list(d.files)}")
    J = d[key]
    e_su_frozen, e_sd_frozen = d["e_sto2U"], d["e_sto2D"]

    with open(json_path) as fh:
        ref_all = json.load(fh)
    ref = next((r for r in ref_all["rows"] if abs(r["scale"] - NOMINAL_SCALE) < 1e-12), None)
    if ref is None:
        raise KeyError(f"{json_path} has no row for scale={NOMINAL_SCALE}")

    # (i) this repository's own direction construction must agree with P1's stored vectors
    e_su_ours = unify.sto2_direction_phi(STO2_U_TRUE, 0)
    e_sd_ours = unify.sto2_direction_phi(STO2_D_TRUE, 2)
    dir_res_u = float(np.max(np.abs(e_su_ours - e_su_frozen)))
    dir_res_d = float(np.max(np.abs(e_sd_ours - e_sd_frozen)))

    # (ii) recompute P1's quantities from the frozen Jacobian
    got = recompute(J, e_su_frozen, e_sd_frozen)

    residuals = {
        "singular_values": rel(got["singular_values"], ref["singular_values"]),
        "smallest_over_largest": rel(got["smallest_over_largest"], ref["smallest_over_largest"]),
        "align_sto2U_vmin": rel(got["align_sto2U_vmin"], ref["align_sto2U_vmin"]),
        "align_sto2D_vmin": rel(got["align_sto2D_vmin"], ref["align_sto2D_vmin"]),
        "sto2U_in_weakest1": rel(got["sto2U_in_weakest1"], ref["sto2U_in_weakest1"]),
        "sto2U_in_weakest2": rel(got["sto2U_in_weakest2"], ref["sto2U_in_weakest2"]),
        "sto2D_in_weakest1": rel(got["sto2D_in_weakest1"], ref["sto2D_in_weakest1"]),
        "sto2D_in_weakest2": rel(got["sto2D_in_weakest2"], ref["sto2D_in_weakest2"]),
        "resp_ratio_U": rel(got["resp_ratio_U"], ref["resp_ratio_U"]),
        "resp_ratio_D": rel(got["resp_ratio_D"], ref["resp_ratio_D"]),
        "response": max(rel(got["response"][k], ref["response"][k]) for k in got["response"]),
        "direction_StO2_upper_vs_P1_stored": dir_res_u,
        "direction_StO2_deep_vs_P1_stored": dir_res_d,
    }
    worst = max(residuals.values())
    ok = worst < a.tol

    payload = {
        "step": "A",
        "profile": "p1_legacy",
        "transport_run": False,
        "source_npz": os.path.relpath(npz_path, ROOT),
        "source_json": os.path.relpath(json_path, ROOT),
        "upstream_n_photons": ref_all["n_photons"],
        "upstream_max_steps": ref_all["max_steps"],
        "scale": NOMINAL_SCALE,
        "J_shape": list(J.shape),
        "recomputed": got,
        "reference": {k: ref[k] for k in got},
        "relative_residuals": residuals,
        "worst_relative_residual": worst,
        "tolerance": a.tol,
        "pass": bool(ok),
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"  J {J.shape} from {os.path.basename(npz_path)} "
          f"(n_photons={ref_all['n_photons']}, max_steps={ref_all['max_steps']})")
    print(f"  {'quantity':<34} {'residual':>12}")
    for k, v in sorted(residuals.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<34} {v:12.3e}")
    print(f"\n  worst {worst:.3e}  tol {a.tol:.0e}  ->  {'PASS' if ok else 'FAIL'}")
    print(f"  wrote {os.path.relpath(a.out, ROOT)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
