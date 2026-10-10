#!/usr/bin/env python3
"""p2_legacy_reproduce.py -- reproduce P2's original diagonal-Poisson Fisher analysis.

Runs no photon transport. P2 cached both the reflectance Rcw and the Jacobian Gcw on its 120-bin
grid for 32 seeds, so the whole Fisher analysis is recomputable from the frozen pool.

P2 ships TWO pipelines over the same pool, and they do not agree on three points. Both are
reproduced here rather than silently picking one:

                        build_fim_stack.py          fim_analyze.py
  photon budget         N_TOT = 4e5 (total)         PROD_N = 1e5 (per wavelength)
  min-count filter      none                        drop bins with N*A*R < 5
  parameter order       PERM [0,2,1,3,4,5]          production order

Gcw is stored in the production order [StO2_U, B_U, StO2_D, B_D, DermR, SubQ]; fim_stack.npz stores
the permuted order [StO2_U, StO2_D, B_U, B_D, DermR, SubQ].

The upstream jax-pad-fisher repository is read only.

    python3 scripts/p2_legacy_reproduce.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = os.path.normpath(os.path.join(ROOT, "..", "jax-pad-fisher", "outputs", "fim_crb"))

PERM = [0, 2, 1, 3, 4, 5]                      # production order -> fim_stack order
PRODUCTION_ORDER = ["StO2_U", "B_U", "StO2_D", "B_D", "DermR", "SubQ"]
STACK_ORDER = ["StO2_U", "StO2_D", "B_U", "B_D", "DermR", "SubQ"]
MIN_COUNT = 5.0
RANK_TOL = 1e-10                               # relative to lambda_max, as in P2


def pool_seeds(pool_dir):
    files = sorted(glob.glob(os.path.join(pool_dir, "seed_*.npz")))
    if not files:
        raise FileNotFoundError(f"no seed_*.npz in {pool_dir}")
    d0 = np.load(files[0])
    R = np.zeros_like(d0["Rcw"], dtype=float)
    G = np.zeros_like(d0["Gcw"], dtype=float)
    for f in files:
        d = np.load(f)
        R += d["Rcw"]
        G += d["Gcw"]
    return R / len(files), G / len(files), len(files), int(d0["N_R"]), float(d0["R_MAX"])


def grid(n_r, r_max):
    e = np.linspace(0.0, r_max, n_r + 1)
    return 0.5 * (e[:-1] + e[1:]), np.pi * (e[1:] ** 2 - e[:-1] ** 2)


def fisher(R, G, centers, areas, n_wl, n_r, N, r_cut, min_count):
    """P2's Fisher accumulation. min_count None disables the filter."""
    F = np.zeros((n_wl, 6, 6))
    used = np.zeros((n_wl, n_r), dtype=bool)
    for k in range(n_wl):
        for j in range(n_r):
            b = k * n_r + j
            Rb = R[b]
            if centers[j] > r_cut or Rb <= 0:
                continue
            if min_count is not None and N * areas[j] * Rb < min_count:
                continue
            g = G[b]
            F[k] += (N * areas[j] / Rb) * np.outer(g, g)
            used[k, j] = True
    return F, used


def crb(F, rank_tol=RANK_TOL):
    """P2's estimability-aware bound: pseudo-inverse over eigenvalues above rank_tol * lambda_max."""
    lam, V = np.linalg.eigh(F)
    keep = lam > rank_tol * lam.max()
    inv = (V[:, keep] / lam[keep]) @ V[:, keep].T
    return np.sqrt(np.diag(inv)), int(keep.sum()), lam


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", default=UPSTREAM)
    ap.add_argument("--pool", default="raw_ms2000")
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "p2_legacy_reproduce.json"))
    a = ap.parse_args()

    pool_dir = os.path.join(a.upstream, a.pool)
    R, G_prod, n_seeds, n_r, r_max = pool_seeds(pool_dir)
    centers, areas = grid(n_r, r_max)
    n_wl = 4
    n_obs = R.size

    checks, residuals = {}, {}

    # --- (1) reproduce fim_stack.npz exactly -----------------------------------------------------
    stack_path = os.path.join(a.upstream, "fim_stack.npz")
    st = np.load(stack_path, allow_pickle=True)
    N_tot = float(st["N_tot"])
    G_stack = G_prod[:, PERM]
    Fk_mine, used_stack = fisher(R, G_stack, centers, areas, n_wl, n_r,
                                 N=N_tot, r_cut=1.5, min_count=None)
    den = np.maximum(np.abs(st["Fk"]), np.finfo(float).tiny)
    residuals["fim_stack_Fk"] = float(np.max(np.abs(Fk_mine - st["Fk"]) / den))
    checks["fim_stack_bitwise_identical"] = bool(np.array_equal(Fk_mine, st["Fk"]))
    checks["stack_theta_order_matches"] = [str(x) for x in st["theta_names"]] == STACK_ORDER
    checks["n_seeds_matches"] = int(st["n_seeds"]) == n_seeds

    # --- (2) the fim_analyze.py route: per-wavelength budget, with the min-count filter -----------
    N_per_wl = N_tot / n_wl
    F_mc, used_mc = fisher(R, G_prod, centers, areas, n_wl, n_r,
                           N=N_per_wl, r_cut=r_max, min_count=MIN_COUNT)
    F_nomc, used_nomc = fisher(R, G_prod, centers, areas, n_wl, n_r,
                               N=N_per_wl, r_cut=r_max, min_count=None)
    sd_mc, rank_mc, lam_mc = crb(F_mc.sum(axis=0))
    sd_nomc, rank_nomc, lam_nomc = crb(F_nomc.sum(axis=0))

    iu, idd = PRODUCTION_ORDER.index("StO2_U"), PRODUCTION_ORDER.index("StO2_D")

    # --- (3) compare against P2's frozen pooled bounds at the production budget -------------------
    ref_path = os.path.join(a.upstream, "fim_boundary_pooled.json")
    ref_row = None
    if os.path.exists(ref_path):
        with open(ref_path) as fh:
            rows = json.load(fh)["rows"]
        ref_row = next((r for r in rows if abs(r["x"] - 1.0) < 1e-12), None)
    if ref_row is not None:
        for tag, sd in (("min_count", sd_mc), ("no_min_count", sd_nomc)):
            residuals[f"sdU_{tag}_vs_frozen"] = abs(sd[iu] - ref_row["sdU"]) / ref_row["sdU"]
            residuals[f"sdD_{tag}_vs_frozen"] = abs(sd[idd] - ref_row["sdD"]) / ref_row["sdD"]

    # --- (4) the least-informed direction, in P2's normalized coordinates -------------------------
    psi_nom = {"StO2_U": 0.95, "B_U": 0.30, "StO2_D": 0.65, "B_D": 0.10,
               "DermR": 0.49, "SubQ": 0.45}
    scale = np.array([1.0, psi_nom["B_U"], 1.0, psi_nom["B_D"],
                      psi_nom["DermR"], psi_nom["SubQ"]])
    Fn = F_nomc.sum(axis=0) * np.outer(scale, scale)
    lam_n, V_n = np.linalg.eigh(Fn)
    vmin = V_n[:, 0]

    payload = {
        "profile": "p2_legacy",
        "transport_run": False,
        "pool": os.path.relpath(pool_dir, ROOT),
        "n_seeds": n_seeds,
        "n_observables": n_obs,
        "n_radial_bins": n_r,
        "r_max_cm": r_max,
        "N_tot": N_tot,
        "N_per_wavelength": N_per_wl,
        "parameter_order_production": PRODUCTION_ORDER,
        "parameter_order_fim_stack": STACK_ORDER,
        "checks": checks,
        "relative_residuals": residuals,
        "empty_bins": {"n_R_le_zero": int((R <= 0).sum()), "n_obs": n_obs},
        "min_count_filter": {
            "threshold": MIN_COUNT,
            "bins_used_with_filter": int(used_mc.sum()),
            "bins_used_without_filter": int(used_nomc.sum()),
            "bins_dropped_by_filter": int(used_nomc.sum() - used_mc.sum()),
        },
        "crb": {
            "with_min_count": {"sd": sd_mc.tolist(), "numerical_rank": rank_mc,
                               "eigenvalues": lam_mc.tolist()},
            "without_min_count": {"sd": sd_nomc.tolist(), "numerical_rank": rank_nomc,
                                  "eigenvalues": lam_nomc.tolist()},
        },
        "frozen_reference_x1": ref_row,
        "least_informed_direction_normalized": {
            "eigenvalue": float(lam_n[0]),
            "components": {PRODUCTION_ORDER[i]: float(vmin[i]) for i in range(6)},
        },
    }
    worst = max(residuals.values()) if residuals else 0.0
    payload["worst_relative_residual"] = worst
    payload["pass"] = bool(checks["fim_stack_bitwise_identical"]
                           and checks["stack_theta_order_matches"]
                           and checks["n_seeds_matches"])

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"  pooled {n_seeds} seeds, {n_obs} observables, {n_r} bins to {r_max} cm")
    print(f"  fim_stack.npz Fk: residual {residuals['fim_stack_Fk']:.3e}, "
          f"bitwise identical {checks['fim_stack_bitwise_identical']}")
    print(f"  empty bins (R <= 0): {payload['empty_bins']['n_R_le_zero']} of {n_obs}")
    f = payload["min_count_filter"]
    print(f"  min-count {MIN_COUNT}: uses {f['bins_used_with_filter']} of "
          f"{f['bins_used_without_filter']} bins (drops {f['bins_dropped_by_filter']})")
    if ref_row:
        print(f"  CRB vs frozen x=1.0  (sdU {ref_row['sdU']:.6f}, sdD {ref_row['sdD']:.6f})")
        for tag, sd in (("with min-count   ", sd_mc), ("without min-count", sd_nomc)):
            print(f"    {tag}  sdU {sd[iu]:.6f}  sdD {sd[idd]:.6f}   "
                  f"rel {abs(sd[idd]-ref_row['sdD'])/ref_row['sdD']:.3e} (deep)")
    print(f"  least-informed direction: " + " ".join(
        f"{k}={v:+.2f}" for k, v in payload["least_informed_direction_normalized"]["components"].items()))
    print(f"  wrote {os.path.relpath(a.out, ROOT)}")
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
