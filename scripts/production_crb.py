#!/usr/bin/env python3
"""production_crb.py -- Cramer-Rao lower bounds from the frozen pool, computed through the SVD.

What the numbers are, stated once so the field names can stay short. Every quantity here is a
LOWER BOUND on the standard deviation of a locally unbiased estimator at the nominal parameter
value, under the assumed photon budget and the assumed likelihood. It is not a confidence interval,
it is not an achieved error, and no claim is made that any estimator attains it. The span reported
alongside each bound is nominal +/- 1.96 sqrt(CRLB): a Gaussian-equivalent width, useful for seeing
how the bound compares with the parameter's own scale, and nothing more. Hence the field names
`sqrt_crlb` and `gaussian_equivalent_1p96_crlb_span`.

The covariance is built from the SVD of J_w and never from a plain matrix inverse:

    J_w = U diag(s) V^T,   C_z = V diag(1/s^2) V^T,   C_psi = D C_z D

z is the column-scaled coordinate, psi = D z, so a target g(psi) with gradient c = dg/dpsi becomes
(D c)^T z. Every target direction is mapped through D before it touches the singular subspace.

Numerical rank is computed from the singular values against a recorded tolerance and asserted, not
assumed. If the matrix were ever rank deficient this script would stop rather than report a
full-rank bound.

The estimability-aware analysis is reported SEPARATELY, as a curve over a threshold grid that is
fixed in this file before the threshold-sensitivity bounds are evaluated. For each threshold and
target it answers one question first: does the target lie inside the retained subspace? If it has
any component in the discarded subspace its bound is INFINITE, and the diagonal of a thresholded
pseudo-inverse is NOT reported as a finite bound on that parameter. That diagonal is a bound on the
variance of the target's projection onto the retained subspace, which is a different estimand.

    python3 scripts/production_crb.py --spectrum results/unify_spectrum.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import unify  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unify_spectrum import column_scale, jacobian_coordinates, load_spec  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = list(unify.PSI_LABELS)
Z = 1.959963984540054

# Fixed in this file before the threshold-sensitivity bounds below were evaluated. The whole curve
# is reported and no value is selected from it.
THRESHOLD_GRID = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-6, 1e-12)

# A target is deemed to leave the retained subspace once its discarded component exceeds this. It
# is a numerical tolerance on an exact geometric question, not a tunable knob.
LEAKAGE_TOL = 1e-10

# Numerical rank tolerance, the LAPACK convention: max(m, n) * eps * s_max.
RANK_TOL_RULE = "max(m, n) * finfo(float64).eps * s_max"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def numerical_rank(s, shape):
    """Rank from the singular values against an explicitly recorded tolerance."""
    tol = max(shape) * np.finfo(float).eps * float(s[0])
    return int((s > tol).sum()), float(tol)


# ---------------------------------------------------------------------------------------------
# Targets. Each is (value, gradient) at the nominal psi. Linear targets have constant gradients;
# S_combined is nonlinear and carries its analytic gradient, verified against finite differences.
# ---------------------------------------------------------------------------------------------

def s_combined(psi):
    """Blood-volume-weighted saturation, (f_U S_U + f_D S_D) / (f_U + f_D).

    A BVF-weighted descriptive contrast in the nominal two-plexus parameterization. It does not
    include layer-thickness, photon-pathlength, or measurement-sensitivity weights and is not
    asserted to equal an instrument-effective saturation.
    """
    S_U, f_U, S_D, f_D = psi[0], psi[1], psi[2], psi[3]
    return (f_U * S_U + f_D * S_D) / (f_U + f_D)


def s_combined_grad(psi):
    S_U, f_U, S_D, f_D = psi[0], psi[1], psi[2], psi[3]
    t = f_U + f_D
    g = np.zeros(6)
    g[0] = f_U / t                       # d/dS_U
    g[1] = f_D * (S_U - S_D) / t ** 2    # d/df_U
    g[2] = f_D / t                       # d/dS_D
    g[3] = f_U * (S_D - S_U) / t ** 2    # d/df_D
    return g


def verify_s_combined_grad(psi, rel_step=1e-6):
    num = np.zeros(6)
    for i in range(6):
        h = rel_step * max(abs(psi[i]), 1e-3)
        a, b = psi.copy(), psi.copy()
        a[i] += h
        b[i] -= h
        num[i] = (s_combined(a) - s_combined(b)) / (2 * h)
    ana = s_combined_grad(psi)
    denom = max(float(np.max(np.abs(ana))), 1e-30)
    return float(np.max(np.abs(ana - num)) / denom)


def targets(psi):
    out = {}
    for i, lab in enumerate(LABELS):
        c = np.zeros(6)
        c[i] = 1.0
        out[lab] = {"value": float(psi[i]), "gradient": c, "kind": "parameter"}
    c = np.zeros(6); c[2], c[0] = 1.0, -1.0
    out["StO2_D_minus_StO2_U"] = {"value": float(c @ psi), "gradient": c, "kind": "linear contrast"}
    c = np.zeros(6); c[3], c[1] = 1.0, -1.0
    out["f_D_minus_f_U"] = {"value": float(c @ psi), "gradient": c, "kind": "linear contrast"}
    out["S_combined"] = {
        "value": float(s_combined(psi)), "gradient": s_combined_grad(psi),
        "kind": "nonlinear contrast, analytic gradient propagated",
        "definition": "(f_U * StO2_U + f_D * StO2_D) / (f_U + f_D)",
    }
    return out


# ---------------------------------------------------------------------------------------------


def full_rank_crlb(W, D, psi):
    U_, s, Vt = np.linalg.svd(W, full_matrices=False)
    V = Vt.T
    rank, rank_tol = numerical_rank(s, W.shape)
    if rank != 6:
        raise SystemExit(f"numerical rank {rank} < 6 at tolerance {rank_tol:.3e}; "
                         "the full-rank CRLB does not apply and must not be reported")

    C_z = V @ np.diag(1.0 / s ** 2) @ V.T
    Dm = np.diag(D)
    C_psi = 0.5 * ((Dm @ C_z @ Dm) + (Dm @ C_z @ Dm).T)
    sd = np.sqrt(np.diag(C_psi))
    corr = C_psi / np.outer(sd, sd)

    # independent check: a Cholesky solve reaches the same covariance by a different route. The
    # reported values come from the SVD; this only shows the two agree.
    L = np.linalg.cholesky(W.T @ W)
    C_z_chol = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(6)))
    chol_res = float(np.max(np.abs(C_z_chol - C_z)) / np.max(np.abs(C_z)))

    tg = {}
    for name, t in targets(psi).items():
        c = t["gradient"]
        se = float(np.sqrt(c @ C_psi @ c))
        val = t["value"]
        e = {"value": val, "kind": t["kind"], "sqrt_crlb": se,
             "gaussian_equivalent_1p96_crlb_span": [val - Z * se, val + Z * se],
             "relative_sqrt_crlb": (se / abs(val)) if val != 0.0 else None}
        if "definition" in t:
            e["definition"] = t["definition"]
            e["gradient_wrt_psi"] = c.tolist()
            e["analytic_gradient_vs_finite_difference"] = verify_s_combined_grad(psi)
        tg[name] = e

    pairs = []
    for i in range(6):
        for j in range(i + 1, 6):
            pairs.append({"a": LABELS[i], "b": LABELS[j], "correlation": float(corr[i, j])})
    pairs.sort(key=lambda p: -abs(p["correlation"]))

    return {
        "quantity": "lower bound on the standard deviation of a locally unbiased estimator at the "
                    "nominal parameter value, under the assumed photon budget and likelihood",
        "not_a_confidence_interval": True,
        "not_an_achieved_estimator_error": True,
        "method": "C_z = V diag(1 / s^2) V^T from the SVD of J_w, then C_psi = D C_z D",
        "plain_matrix_inverse_used": False,
        "singular_values_truncated": False,
        "numerical_rank": rank,
        "numerical_rank_tolerance": rank_tol,
        "numerical_rank_tolerance_rule": RANK_TOL_RULE,
        "numerical_rank_asserted_equal_to_n_parameters": True,
        "independent_cholesky_check": {"normalized_max_diff": chol_res,
                                       "passes": bool(chol_res < 1e-8)},
        "singular_values": s.tolist(),
        "condition_number_Jw": float(s[0] / s[-1]),
        "sqrt_crlb_psi": {LABELS[i]: float(sd[i]) for i in range(6)},
        "gaussian_equivalent_1p96_crlb_span_psi": {
            LABELS[i]: [float(psi[i] - Z * sd[i]), float(psi[i] + Z * sd[i])] for i in range(6)},
        "relative_sqrt_crlb_psi": {LABELS[i]: float(sd[i] / psi[i]) for i in range(6)},
        "correlation_matrix": corr.tolist(),
        "correlation_labels": LABELS,
        "correlation_meaning": "local coupling of the bound at the nominal point: the directions "
                               "in which the Fisher information fails to separate two parameters. "
                               "It is a statement about this geometry and photon budget, not "
                               "evidence that two absorption spectra are similar or that the two "
                               "parameters share a single degree of freedom",
        "strongest_correlated_pairs": pairs,
        "targets": tg,
    }, (s, V, C_psi)


def estimability_curve(D, psi, s, V):
    rows = []
    rel = s / s[0]
    tgs = targets(psi)
    for thr in THRESHOLD_GRID:
        keep = rel >= thr
        Vk, sk = V[:, keep], s[keep]
        P_disc = np.eye(6) - Vk @ Vk.T
        per = {}
        for name, t in tgs.items():
            dz = D * t["gradient"]           # the target in z coordinates: (D c)^T z
            n = np.linalg.norm(dz)
            leak = float(np.linalg.norm(P_disc @ dz) / n) if n > 0 else 0.0
            inside = leak <= LEAKAGE_TOL
            w = Vk.T @ dz
            e = {"discarded_component": leak, "inside_retained_subspace": inside}
            if inside:
                e["sqrt_crlb"] = float(np.sqrt(np.sum((w / sk) ** 2)))
                e["bound"] = "finite"
            else:
                e["sqrt_crlb"] = None
                e["bound"] = "infinite"
                e["status"] = "not estimable under this threshold"
                e["projected_subspace_sqrt_crlb_not_a_parameter_bound"] = float(
                    np.sqrt(np.sum((w / sk) ** 2)))
            per[name] = e
        rows.append({"relative_threshold": thr, "retained_rank": int(keep.sum()),
                     "discarded_singular_values": s[~keep].tolist(), "targets": per})
    return {
        "threshold_grid": list(THRESHOLD_GRID),
        "threshold_grid_fixed_before_threshold_sensitivity_bounds_were_evaluated": True,
        "no_threshold_selected": True,
        "leakage_tolerance": LEAKAGE_TOL,
        "note": "reported as a curve; the headline bound is the full-rank one, because the "
                "numerical rank of J_w equals the number of parameters",
        "warning": "where a target has a discarded component its bound is infinite; the diagonal "
                   "of a thresholded pseudo-inverse is a bound on the variance of the target's "
                   "projection onto the retained subspace, a different estimand",
        "curve": rows,
    }


def cross_check_spectrum(path, branches, D, N, coords):
    """Read the spectrum artifact and assert it agrees with this recomputation, branch by branch."""
    with open(path) as fh:
        sp = json.load(fh)
    checks = {"artifact": os.path.relpath(path, ROOT), "role": "computational input, read and "
              "asserted against this recomputation", "per_branch": {}}
    ok = True

    def rel(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = max(float(np.max(np.abs(b))), np.finfo(float).tiny)
        return float(np.max(np.abs(a - b)) / d)

    c = {"coordinates": [sp["jacobian_coordinates"], coords],
         "reparameterize_applied": [sp["reparameterize_applied"], False],
         "column_scale_D": rel(sp["column_scale_D"], D),
         "n_measurement_photons_per_wavelength": [
             sp["n_measurement_photons_per_wavelength"], N]}
    g_ok = (c["coordinates"][0] == c["coordinates"][1]
            and c["reparameterize_applied"][0] is False and c["reparameterize_applied"][1] is False
            and c["column_scale_D"] < 1e-14
            and c["n_measurement_photons_per_wavelength"][0]
            == c["n_measurement_photons_per_wavelength"][1])
    c["passes"] = bool(g_ok)
    checks["global"] = c
    ok = ok and g_ok

    for name, s_here in branches.items():
        s_there = sp["branches"][name]["singular_values"]
        r = rel(s_there, s_here)
        p = bool(r < 1e-12)
        checks["per_branch"][name] = {"singular_values_normalized_max_diff": r, "passes": p}
        ok = ok and p
    checks["all_passed"] = bool(ok)
    if not ok:
        raise SystemExit(f"spectrum artifact disagrees with this recomputation: {checks}")
    return checks


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pooled", default=os.path.join(ROOT, "data", "derived",
                                                     "production_pooled.npz"))
    ap.add_argument("--spectrum", default=os.path.join(ROOT, "results", "unify_spectrum.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "production_crb.json"))
    a = ap.parse_args()

    spec = load_spec()
    prod = spec["profiles"]["production"]
    primary, secondary = prod["likelihood"], prod["likelihood_secondary"]
    N = prod["n_measurement_photons_per_wavelength"]

    d = np.load(a.pooled)
    coords, jname = jacobian_coordinates(d)
    if coords != "psi":
        raise SystemExit(f"expected a psi-coordinate pooled artifact, got {coords!r}")
    J = np.asarray(d[jname], float)
    psi = np.asarray(d["psi_nominal"], float)
    R, areas = np.asarray(d["R"], float), np.asarray(d["areas"], float)
    n_wl = int(d["n_wavelengths"])
    D = column_scale(psi)

    Jw = {
        "raw": unify.whiten_raw(J, D),
        "diagonal_poisson": unify.whiten_poisson(J, R, areas, N, D),
        "multinomial_fixed_launch": unify.whiten_multinomial(J, R, areas, N, D, n_wl),
    }

    branches, sv_here = {}, {}
    for name, W in Jw.items():
        _, s, _ = np.linalg.svd(W, full_matrices=False)
        sv_here[name] = s.tolist()
        rank, rank_tol = numerical_rank(s, W.shape)
        nn = unify.near_null_direction(W, D, J_psi=J)
        geom = {"singular_values": s.tolist(),
                "numerical_rank": rank, "numerical_rank_tolerance": rank_tol,
                "numerical_rank_tolerance_rule": RANK_TOL_RULE,
                "condition_number_Jw": float(s[0] / s[-1]),
                "condition_number_I": float((s[0] / s[-1]) ** 2),
                "near_null": nn,
                "near_null_direction_z_from_svd": {
                    LABELS[i]: nn["near_null_direction_z_from_svd"][i] for i in range(6)},
                "near_null_tangent_psi_per_unit_z": {
                    LABELS[i]: nn["near_null_tangent_psi_per_unit_z"][i] for i in range(6)},
                "near_null_tangent_psi_unit_norm_display_only": {
                    LABELS[i]: nn["near_null_tangent_psi_unit_norm_display_only"][i]
                    for i in range(6)},
                "near_null_coordinate_warning":
                    "the right singular vectors of J_w live in the scaled coordinate z, where "
                    "psi = D z. Any step taken in psi space must use the pushforward D v_z. The "
                    "earlier field name near_null_direction_psi_from_svd was wrong: it held v_z."}
        if name == "raw":
            branches[name] = {"geometry": geom, "crlb": None,
                              "crlb_omitted_because": "the raw branch has no likelihood, so there "
                                                      "is no Fisher information and no CRLB; only "
                                                      "the geometric spectrum is meaningful"}
            continue
        crlb, (s2, V, _) = full_rank_crlb(W, D, psi)
        branches[name] = {"geometry": geom, "crlb": crlb,
                          "estimability_aware_separate": estimability_curve(D, psi, s2, V)}

    xchk = cross_check_spectrum(a.spectrum, sv_here, D, N, coords)
    xchk["sha256"] = sha256(a.spectrum)

    payload = {
        "reported_quantity": "Cramer-Rao LOWER BOUNDS on the standard deviation of a locally "
                             "unbiased estimator at the nominal parameter value, under the assumed "
                             "photon budget and likelihood. Not confidence intervals, not achieved "
                             "estimator errors, and no claim that any estimator attains them.",
        "span_definition": "gaussian_equivalent_1p96_crlb_span is nominal +/- 1.96 sqrt(CRLB), a "
                           "Gaussian-equivalent width for comparing the bound with the parameter's "
                           "own scale. It is not a physical confidence interval and carries no "
                           "coverage guarantee, so it may extend outside a parameter's physical "
                           "range without that being a contradiction.",
        "inputs": {
            "pooled_artifact": os.path.relpath(a.pooled, ROOT),
            "pooled_sha256": sha256(a.pooled),
            "spectrum_cross_check": xchk,
            "transport_rerun": False,
        },
        "likelihood_selection": {
            "headline": primary,
            "sensitivity": secondary,
            "basis": "spec/baseline_spec.json profiles.production.likelihood, fixed in the spec "
                     "before the CRLB was computed",
        },
        "n_measurement_photons_per_wavelength": N,
        "psi_labels": LABELS, "psi_nominal": psi.tolist(), "column_scale_D": D.tolist(),
        "coordinate_note": "psi = D z, so a target with gradient c is (D c)^T z in the whitened "
                           "coordinate",
        "branches": branches,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(payload, fh, indent=2)

    tsv = os.path.join(os.path.dirname(a.out), "production_crb_table.tsv")
    with open(tsv, "w") as fh:
        fh.write("likelihood\trole\ttarget\tkind\tvalue\tsqrt_crlb\t"
                 "gauss_equiv_1p96_span_lo\tgauss_equiv_1p96_span_hi\trelative_sqrt_crlb\n")
        for name in (primary, secondary):
            role = "headline" if name == primary else "sensitivity"
            for t, e in branches[name]["crlb"]["targets"].items():
                r = "" if e["relative_sqrt_crlb"] is None else f"{e['relative_sqrt_crlb']:.6g}"
                lo, hi = e["gaussian_equivalent_1p96_crlb_span"]
                fh.write(f"{name}\t{role}\t{t}\t{e['kind']}\t{e['value']:.8g}\t"
                         f"{e['sqrt_crlb']:.8g}\t{lo:.8g}\t{hi:.8g}\t{r}\n")

    print(f"  headline {primary}   sensitivity {secondary}")
    print(f"  spectrum cross-check: {'PASS' if xchk['all_passed'] else 'FAIL'} "
          f"(coords, D, N, and singular values asserted per branch)")
    for name in (primary, secondary):
        c = branches[name]["crlb"]
        print(f"  {name:<26} rank {c['numerical_rank']} (tol {c['numerical_rank_tolerance']:.3e})"
              f"  sqrt-CRLB StO2_U {c['sqrt_crlb_psi']['StO2_U']:.6f}"
              f"  StO2_D {c['sqrt_crlb_psi']['StO2_D']:.6f}"
              f"  S_combined {c['targets']['S_combined']['sqrt_crlb']:.6f}")
    print(f"  S_combined nominal {branches[primary]['crlb']['targets']['S_combined']['value']:.6f}"
          f"  analytic gradient vs finite difference "
          f"{branches[primary]['crlb']['targets']['S_combined']['analytic_gradient_vs_finite_difference']:.3e}")
    print(f"  wrote {os.path.relpath(a.out, ROOT)} and {os.path.relpath(tsv, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
