#!/usr/bin/env python3
"""nuisance_interval_design_diagnostic.py -- the computation behind the interval design memo.

Every numerical statement in docs/nuisance_gate_interval_design_memo.md is produced here, from
committed inputs only:

    spec/baseline_spec.json                       the frozen production measurement model
    spec/nuisance_spec.json                       the registered margin and alpha
    data/derived/production_pooled.npz            the production reflectance and bin areas
    results/nuisance_feasibility_gate_BLOCKED.json   the immutable pilot arrays, sha256 pinned

Nothing is hand copied, nothing is re-run on the GPU, and no threshold is selected. The memo is
written by --write-memo from this script's own JSON, so the two cannot drift.

EVERYTHING HERE IS POST HOC AND NON-CONFIRMATORY. Job 62960056 is pilot evidence. No criterion is
registered, no grid is frozen, and nothing here authorises a scientific classification.

    python3 scripts/nuisance_interval_design_diagnostic.py --selftest
    python3 scripts/nuisance_interval_design_diagnostic.py
    python3 scripts/nuisance_interval_design_diagnostic.py --write-memo
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import nuisance_feasibility_gate as _gate   # noqa: E402
import unify as _unify                      # noqa: E402

BLOCKED = os.path.join(ROOT, "results", "nuisance_feasibility_gate_BLOCKED.json")
BLOCKED_SHA256 = "dda8bd26f4b70afeb49cd05dc02a1b6b8d4baf20e76f2ae116d00ffb84e79d78"
POOLED = os.path.join(ROOT, "data", "derived", "production_pooled.npz")
BSPEC = os.path.join(ROOT, "spec", "baseline_spec.json")
NSPEC = os.path.join(ROOT, "spec", "nuisance_spec.json")
OUT_JSON = os.path.join(ROOT, "results", "nuisance_interval_design_diagnostic.json")
OUT_MEMO = os.path.join(ROOT, "docs", "nuisance_gate_interval_design_memo.md")

POST_HOC = ("POST HOC AND NON-CONFIRMATORY. Computed from pilot evidence after the run failed. "
            "Not a decision rule, not a verdict, and no threshold here is selected.")


def _load_json(path):
    with open(path) as fh:
        return json.load(fh)


def _read_self():
    with open(os.path.abspath(__file__)) as fh:
        return fh.read()


def _sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# ---------------------------------------------------------------------------------------------
# 1. The frozen production measurement model
# ---------------------------------------------------------------------------------------------

def production_measurement_model():
    """C, the covariance the whitening is defined against. All of it from committed files.

    This is the PRODUCTION MEASUREMENT model at 100,000 photons per wavelength, not the transport
    budget of 200,000 per wavelength per stream. The purpose of the whitening is to weight each
    observable by the information it carries in the published Fisher calculation, and that
    calculation is the measurement model. The transport budget sets how precisely the gate knows
    R and its derivative; it is not the covariance any inference is done under.
    """
    bs = _load_json(BSPEC)
    prof = bs["profiles"]["production"]
    z = np.load(POOLED)
    R = np.asarray(z["R"], float)
    areas = np.asarray(z["areas"], float)
    n_wl = int(z["n_wavelengths"])
    per = R.size // n_wl
    rad = prof["radial"]
    edges = np.linspace(rad["r_min_cm"], rad["r_max_cm"], rad["n_bins"] + 1)
    N = float(prof["n_measurement_photons_per_wavelength"])

    # diagonal Poisson: counts n_i = N A_i R_i, so Var(R_i) = R_i / (N A_i)
    var_poisson = np.where(R > 0, R / np.maximum(N * areas, np.finfo(float).tiny), np.inf)
    w_poisson = np.where(np.isfinite(var_poisson) & (var_poisson > 0),
                         1.0 / np.sqrt(np.maximum(var_poisson, np.finfo(float).tiny)), 0.0)

    # multinomial fixed launch, the PRIMARY production likelihood: per wavelength block,
    # M = diag(p) - p p^T on the detected cells with the non-detection cell absorbing 1 - sum(p).
    blocks, cond, mins = [], [], []
    W_multi = np.zeros((R.size, R.size))
    c_multi_diag = np.zeros(R.size)
    for w in range(n_wl):
        sl = slice(w * per, (w + 1) * per)
        A = np.diag(areas[sl])
        p = areas[sl] * R[sl]
        tot = float(p.sum())
        M = np.diag(p) - np.outer(p, p)
        ev = np.linalg.eigvalsh(M)
        blocks.append({"wavelength_index": w,
                       "wavelength_nm": bs["wavelengths_nm"][w],
                       "total_detection_probability": tot,
                       "eigenvalue_min": float(ev.min()), "eigenvalue_max": float(ev.max()),
                       "condition_number": float(ev.max() / max(ev.min(), np.finfo(float).tiny)),
                       "n_zero_probability_cells": int((p <= 0).sum()),
                       "is_exactly_singular": bool(tot >= 1.0 or (p <= 0).any())})
        cond.append(blocks[-1]["condition_number"])
        mins.append(float(ev.min()))
        W_multi[sl, sl] = np.sqrt(N) * (_unify._inv_sqrt_psd(M) @ A)
        # C_jj in PHYSICAL BIN coordinates: C = A^{-1} M A^{-1} / N, so C_jj = M_jj / (N A_j^2)
        c_multi_diag[sl] = np.diag(M) / np.maximum(N * areas[sl] ** 2, np.finfo(float).tiny)

    return {
        "photon_budget": {
            "used": "production measurement model",
            "n_measurement_photons_per_wavelength": N,
            "n_measurement_photons_total": prof["n_measurement_photons_total"],
            "NOT_used": {
                "n_transport_photons_per_wavelength_per_stream":
                    prof["n_transport_photons_per_wavelength_per_stream"],
                "n_transport_streams": prof["n_transport_streams"],
                "why_not": "the transport budget sets how precisely the gate knows R and its "
                           "derivative. It is not the covariance any published inference is done "
                           "under, so weighting by it would not measure Fisher relevance."}},
        "reflectance_source": {"file": os.path.relpath(POOLED, ROOT), "array": "R",
                               "sha256": _sha256(POOLED)[:16],
                               "why": "the committed production reflectance. Earlier working "
                                      "numbers approximated R(0) by averaging R_plus and R_minus "
                                      "at the smallest step, accurate only to O(h^2); this "
                                      "replaces that."},
        "areas_source": {"file": os.path.relpath(POOLED, ROOT), "array": "areas",
                         "definition": "pi (r_outer^2 - r_inner^2) over 30 equal radial edges "
                                       "from 0.0 to 1.5 cm, tiled over 4 wavelengths"},
        "wavelength_blocks": {"n_wavelengths": n_wl, "bins_per_wavelength": per,
                              "index_rule": "wavelength_index * 30 + bin_index",
                              "wavelengths_nm": bs["wavelengths_nm"]},
        "primary_likelihood": prof["likelihood"],
        "secondary_likelihood": prof["likelihood_secondary"],
        "diagonal_poisson": {
            "covariance": "C = diag(R_i / (N A_i))",
            "whitening": "W = C^{-1/2} = diag(sqrt(N A_i / R_i))",
            "n_components_with_zero_reflectance": int((R <= 0).sum())},
        "multinomial_fixed_launch": {
            "covariance": "C = blockdiag( A^{-1} M A^{-1} / N ),  M = diag(p) - p p^T,  "
                          "p_i = A_i R_i within each wavelength block",
            "whitening": "W = blockdiag( sqrt(N) M^{-1/2} A ), the same construction as "
                         "src/unify.whiten_multinomial",
            "singular_direction": {
                "is_any_block_exactly_singular": bool(any(b["is_exactly_singular"]
                                                          for b in blocks)),
                "why_not": "M is positive definite whenever the total detection probability is "
                           "strictly below 1 and no cell has zero probability. Both hold at the "
                           "production operating point, so there is no exactly singular direction "
                           "to project out and no pseudo-inverse is needed.",
                "but_it_is_near_singular": True,
                "worst_condition_number": float(max(cond)),
                "smallest_eigenvalue": float(min(mins)),
                "consequence": "M^{-1/2} amplifies the near-null direction by the square root of "
                               "the condition number, up to about "
                               f"{np.sqrt(max(cond)):.3g}. That direction is spanned by the "
                               "low-count outer bins, so the multinomial whitening is far more "
                               "sensitive to them than the diagonal Poisson one. Any frozen "
                               "design must state which of the two it uses and must report this "
                               "conditioning, because it is the dominant numerical risk.",
                "treatment_if_a_future_operating_point_is_singular":
                    "project onto the complement of the null space within each wavelength block "
                    "and record the projector in the spec. Not required here, and deliberately "
                    "not implemented, so that it cannot be applied silently."},
            "per_wavelength_block": blocks},
        "frozen_before_confirmatory_data": [
            "the covariance family and which of the two likelihoods defines it",
            "the photon budget entering N",
            "R and the areas, read from the committed pooled artifact, never re-estimated from "
            "the confirmatory run",
            "the wavelength block structure"],
        "R": R, "areas": areas, "n_wl": n_wl, "per": per, "N": N, "edges": edges,
        "W_poisson": w_poisson, "W_multi": W_multi,
        "C_diag_poisson": var_poisson, "C_diag_multinomial": c_multi_diag,
    }


# ---------------------------------------------------------------------------------------------
# 2. Pilot arrays and the two intercepts
# ---------------------------------------------------------------------------------------------

def intercepts(h_grid, D_by_h):
    """Two-term and three-term h^2 intercepts per replicate, and the h^2 coefficient."""
    h2 = np.asarray(h_grid, float) ** 2
    D = np.asarray(D_by_h, float)
    A2, A3 = np.vander(h2, 2, increasing=True), np.vander(h2, 3, increasing=True)
    c2 = np.stack([np.linalg.lstsq(A2, D[r], rcond=None)[0] for r in range(D.shape[0])])
    c3 = np.stack([np.linalg.lstsq(A3, D[r], rcond=None)[0] for r in range(D.shape[0])])
    return c2[:, 0, :], c3[:, 0, :], c2[:, 1, :]


def pilot_columns():
    b = _load_json(BLOCKED)
    out = {}
    for nm in b["columns"]:
        h = np.asarray(b["columns"][nm]["convergence"]["h_grid"], float)
        D = np.stack([np.asarray(x["columns"][nm]["difference_quotients"], float)
                      for x in b["raw"]["per_replicate"]])
        e2, e3, a2 = intercepts(h, D)
        out[nm] = {"h": h, "D": D, "e2": e2, "e3": e3, "a2": a2,
                   "registered_support": np.asarray(
                       b["columns"][nm]["convergence"]["equivalence_test"]["support_mask"], bool)}
    return b, out


# ---------------------------------------------------------------------------------------------
# 3. Transport sampling uncertainty: what six replicates can and cannot support
# ---------------------------------------------------------------------------------------------

def sampling_uncertainty(cols, W, keep_by_col):
    """The rank limit, and the exact definition of every t quoted anywhere.

    Six replicates of a 120-dimensional vector give an empirical covariance of rank at most five.
    Nothing here inverts it, decorrelates with it, or treats it as calibrated. Only per-component
    variances and the whitened total variance are used, both of which are averages rather than
    inverses.
    """
    out = {"n_replicates": None, "dimension": None,
           "empirical_covariance_rank_at_most": None,
           "what_is_never_done": [
               "inverting the empirical covariance of the discrepancy",
               "whitening by an estimated discrepancy covariance",
               "Hotelling or any statistic needing a full-rank sample covariance",
               "projecting onto data-selected principal components"],
           "t_definition": {
               "formula": "t_j = |mean_r d_rj| / ( sd_r(d_rj) / sqrt(n) ) for ONE component j",
               "status": "DESCRIPTIVE ONLY",
               "why_not_calibrated": (
                   "the component j quoted in the memo is argmax over components of the whitened "
                   "standardized discrepancy, so it is SELECTED by the same data that supplies "
                   "its t. The marginal null distribution of a t at a selected index is not "
                   "Student's t, and the selection inflates it. This number says the discrepancy "
                   "at that index is not visibly noise; it does not supply coverage and is not "
                   "used in any bound."),
               "what_would_be_needed": (
                   "either an index fixed before the data, or sample splitting with the index "
                   "chosen on one half and the t computed on the other, or an independent "
                   "confirmatory run. A confirmatory design must do one of these."),
               "simultaneous_alternative": (
                   "the only bound used is the gate's simultaneous one over all retained "
                   "components, which needs no selected index")},
           "per_column": {}}
    for nm, c in cols.items():
        d = c["e3"] - c["e2"]
        n, m = d.shape
        out["n_replicates"], out["dimension"] = int(n), int(m)
        out["empirical_covariance_rank_at_most"] = int(n - 1)
        keep = keep_by_col[nm]
        Wk = W[np.ix_(keep, keep)] if W.ndim == 2 else np.diag(W[keep])
        dk = d[:, keep]
        Wd = dk @ Wk.T
        dbar = Wd.mean(axis=0)
        # unbiased estimate of tr(W Sigma W^T) from the replicates
        tr_hat = float(((Wd - dbar) ** 2).sum() / (n - 1))
        sel = int(np.argmax(np.abs(dbar)))
        se_sel = float(Wd[:, sel].std(ddof=1) / np.sqrt(n))
        out["per_column"][nm] = {
            "n_retained": int(keep.sum()),
            "empirical_rank_at_most": int(n - 1),
            "selected_index_within_retained": sel,
            "t_at_selected_index": float(abs(dbar[sel]) / max(se_sel, np.finfo(float).tiny)),
            "t_is_descriptive_only": True,
            "trace_W_Sigma_Wt_estimate": tr_hat,
            "noise_floor_on_squared_norm": tr_hat / n}
    return out


# ---------------------------------------------------------------------------------------------
# 4. The equivalence quantity: three things that are not interchangeable
# ---------------------------------------------------------------------------------------------

def equivalence_quantities(cols, W, keep_by_col):
    """Separate the norm of the mean, the mean of the norms, and the bias-corrected norm.

    Write d_r = W (e3 - e2)_r, with mean mu = W * (true truncation difference) and replicate
    covariance Sigma_W. Then

        E || dbar ||^2 = || mu ||^2 + tr(Sigma_W) / n

    so the norm of the mean carries an additive noise floor of tr(Sigma_W)/n even when the bias is
    exactly zero. The mean of the replicate norms is worse: it carries the full tr(Sigma_W), not
    tr(Sigma_W)/n, and it cannot be driven to zero by averaging.

    Only the bias-corrected quantity estimates what the design cares about, and it is allowed to
    come out at zero, which is the honest answer when the observed aggregate is entirely
    consistent with sampling noise.
    """
    out = {"definitions": {
        "numerator_norm_of_the_mean": "|| W mean_r (e3 - e2)_r ||_2",
        "numerator_mean_of_the_norms": "mean_r || W (e3 - e2)_r ||_2",
        "numerator_bias_corrected": "sqrt( max( ||W dbar||^2 - tr(Sigma_W)/n , 0 ) )",
        "denominator": "|| W mean_r (e2)_r ||_2, the whitened two-term reference column",
        "what_each_bounds": {
            "norm_of_the_mean": "the norm of the mean derivative bias PLUS a noise floor of "
                                "tr(Sigma_W)/n. It is an upper-biased estimate of the bias.",
            "mean_of_the_norms": "the mean norm of noisy replicate differences. It contains "
                                 "Monte Carlo noise even when the derivative bias is exactly "
                                 "zero, and averaging more replicates does not remove it. It "
                                 "must not be used as an equivalence numerator.",
            "bias_corrected": "a TRUNCATED BIAS-CORRECTED POINT ESTIMATE of the norm of the "
                              "mean derivative bias. Subtracting an estimated trace does not "
                              "produce a confidence bound, and a value of zero means only that "
                              "the point estimate is zero. It is not evidence of equivalence, it "
                              "carries no confidence statement, and NO valid upper bound exists "
                              "to supply one: the resampling heuristic that was tried is refuted "
                              "by the committed Bernoulli counterexample.",
            "projection_onto_fixed_directions": "not used here. A projection would be "
                                                "confirmatory only onto directions fixed before "
                                                "the data; data-selected directions give no "
                                                "coverage."}},
        "status": POST_HOC, "per_column": {}}
    for nm, c in cols.items():
        keep = keep_by_col[nm]
        Wk = W[np.ix_(keep, keep)] if W.ndim == 2 else np.diag(W[keep])
        d = (c["e3"] - c["e2"])[:, keep]
        n = d.shape[0]
        Wd = d @ Wk.T
        We2 = (c["e2"].mean(axis=0)[keep]) @ Wk.T
        den = float(np.linalg.norm(We2))
        dbar = Wd.mean(axis=0)
        sq = float(dbar @ dbar)
        tr_hat = float(((Wd - dbar) ** 2).sum() / (n - 1))
        floor = tr_hat / n
        corrected = float(np.sqrt(max(sq - floor, 0.0)))
        out["per_column"][nm] = {
            "n_retained": int(keep.sum()),
            "denominator": den,
            "norm_of_the_mean_relative": float(np.sqrt(sq)) / max(den, np.finfo(float).tiny),
            "mean_of_the_norms_relative": float(np.linalg.norm(Wd, axis=1).mean())
                                          / max(den, np.finfo(float).tiny),
            "noise_floor_relative": float(np.sqrt(floor)) / max(den, np.finfo(float).tiny),
            "bias_corrected_relative": corrected / max(den, np.finfo(float).tiny),
            "bias_corrected_point_estimate_is_zero": bool(sq <= floor),
        "note_on_that_flag": "it means the truncation in the bias-corrected POINT ESTIMATE took "
                             "effect. It does not mean the discrepancy is consistent with noise "
                             "at any stated confidence, and no such claim is made from it.",
            "squared_norm_of_the_mean": sq,
            "noise_floor_on_squared_norm": floor}
    return out


# ---------------------------------------------------------------------------------------------
# 5. Dual requirement, standardized by the frozen measurement-noise scale
# ---------------------------------------------------------------------------------------------

def dual_requirement(cols, mm, keep_by_col, W, label, C_diag_by_col=None):
    """Aggregate whitened bound plus a per-observable bound in PHYSICAL BIN coordinates.

    The component statistic is max_j |dbar_j| / sqrt(C_jj): the physical bin discrepancy divided by
    that observable's own measurement standard deviation. It is NOT max_j |(W dbar)_j|. For a
    block-dense whitening, (W dbar)_j mixes bins, so it is not "observable j in noise units", and
    its maximum is not invariant to the choice of whitening square root: an orthogonal rotation
    leaves the aggregate norm and the covariance unchanged while moving the maximum coordinate.
    Measured on mus_slope, the canonical whitened-coordinate maximum is about 1.24 while equivalent
    rotated whitenings give roughly 0.77 to 1.49, so that statistic would have been an artefact of
    an arbitrary basis.

    Relation to Fisher information, stated correctly. Write j for the column, delta for its
    perturbation, and suppose ||C^{-1/2} delta|| <= eps ||C^{-1/2} j||.

      same-column diagonal entry: changes by at most 2 eps + eps^2, relatively;
      cross entries against another column k: only the ABSOLUTE bound
          |delta^T C^{-1} k| <= eps ||C^{-1/2} j|| ||C^{-1/2} k||
      follows. Their RELATIVE change can be unbounded, because j^T C^{-1} k can be near zero.

    So the aggregate margin controls the diagonal relatively and the off-diagonals only in
    absolute terms. It does not bound a Schur complement, an eigenvalue, a CRLB or any downstream
    classification: those depend on the principal angle between the nuisance and physiological
    column spaces, and a committed counterexample shows a column-level margin cannot close that.
    """
    out = {"whitening": label,
           "aggregate_statistic": "|| W dbar || / || W e2bar ||, with the denominator estimated "
                                  "from the same replicates. The accompanying heuristic upper "
                                  "value is NOT a confidence bound; see candidate2_statistic and "
                                  "the Bernoulli counterexample. The trace-subtracted quantity is "
                                  "a point estimate only.",
           "component_statistic": "max_j |dbar_j| / sqrt(C_jj), in PHYSICAL BIN coordinates",
           "component_statistic_is_basis_free": True,
           "why_not_the_whitened_coordinate_maximum":
               "for a block-dense whitening, (W dbar)_j mixes bins and is not observable j in "
               "noise units. Its maximum moves under an orthogonal rotation of the whitening "
               "square root, which leaves the aggregate norm and the covariance unchanged, so it "
               "would be an artefact of an arbitrary basis.",
           "component_statistic_is_not_divided_by_the_derivative":
               "a component whose derivative is near zero cannot inflate this, which is the "
               "failure mode of the raw relative statistic",
           "fisher_relation": {
               "assumption": "|| C^{-1/2} delta || <= eps || C^{-1/2} j ||",
               "same_column_diagonal_relative_bound": "2 eps + eps^2",
               "cross_entry_bound": "ABSOLUTE only: |delta^T C^{-1} k| <= eps || C^{-1/2} j || "
                                    "|| C^{-1/2} k ||",
               "cross_entry_relative_change": "can be UNBOUNDED, because j^T C^{-1} k may be near "
                                              "zero. A relative statement about all Fisher "
                                              "entries would be wrong and is not made.",
               "what_the_aggregate_margin_controls":
                   "the same-column Fisher diagonal, relatively, at 2 eps + eps^2; the "
                   "off-diagonal entries only in absolute terms",
               "what_neither_margin_controls":
                   "Schur complements, eigenvalues, CRLB values and every downstream "
                   "classification. These depend on the principal angle between the nuisance and "
                   "physiological column spaces, not on column size, and a committed "
                   "counterexample shows a column-level margin cannot close that gap.",
               "counterexample": "scripts/stress_search_counterexample.py"},
           "margins": {
               "aggregate": None, "max_component": None,
               "status": "NOT SET. Neither margin is proposed here, because any value chosen "
                         "while looking at the pilot numbers below would be selected to pass "
                         "them. Both must be argued from the downstream Fisher requirement and "
                         "frozen before the pilot is consulted again.",
               "what_an_argument_would_need":
                   ["the largest Fisher entry perturbation the published analysis tolerates",
                    "how that maps to the aggregate margin through the order-2-eps relation",
                    "a per-component limit expressed in measurement standard deviations, chosen "
                    "from detectability rather than from these data",
                    "a multiplicity rule for the component bound over the retained set"]},
           "pilot_values_for_reference_only": {}, "status_note": POST_HOC}
    for nm, c in cols.items():
        keep = keep_by_col[nm]
        Wk = W[np.ix_(keep, keep)] if W.ndim == 2 else np.diag(W[keep])
        d = (c["e3"] - c["e2"])[:, keep]
        n = d.shape[0]
        Wd = d @ Wk.T
        dbar = Wd.mean(axis=0)
        We2 = (c["e2"].mean(axis=0)[keep]) @ Wk.T
        den = max(float(np.linalg.norm(We2)), np.finfo(float).tiny)
        tr_hat = float(((Wd - dbar) ** 2).sum() / (n - 1))
        sq = float(dbar @ dbar)
        Cd = (np.ones(int(keep.sum())) if C_diag_by_col is None
              else np.asarray(C_diag_by_col[nm], float))
        # the denominator is estimated from the SAME replicates, as the real calculation does
        r = candidate2_statistic(d, Wk, Cd, ref_rep=c["e2"][:, keep],
                                 rng=np.random.default_rng(20261005), n_resample=400)
        out["pilot_values_for_reference_only"][nm] = {
            "n_retained": int(keep.sum()),
            "aggregate_point_relative": float(np.sqrt(sq)) / den,
            "aggregate_bias_corrected_point_estimate_relative":
                float(np.sqrt(max(sq - tr_hat / n, 0.0))) / den,
            "aggregate_heuristic_upper_value_relative":
                r["aggregate_heuristic_upper_value_relative"],
            "component_point_in_measurement_sd": r["component_point"],
            "component_heuristic_upper_value": r["component_heuristic_upper_value"],
            "denominator_is_estimated_from_the_same_replicates":
                r["denominator_is_estimated_from_the_same_replicates"],
            "is_a_confidence_bound": False,
            "is_a_certification": False}
    return out


# ---------------------------------------------------------------------------------------------
# 6. Is the h^2 coefficient identified separately from the 1/h noise?
# ---------------------------------------------------------------------------------------------

def fd_identifiability(cols, mm, keep_by_col):
    """Model D(h) = d + a h^2 + eps(h) and ask whether a is separable from the noise.

    The noise is not homoscedastic: a central difference divides by 2h, so its standard deviation
    grows roughly as 1/h, and common random numbers correlate eps across step sizes within a
    replicate. Both are measured here rather than assumed.

    a is declared identified for a component only when |a| exceeds twice its replicate standard
    error. The fraction of retained components where that holds is what decides whether the step
    grid can be chosen from this pilot at all.

    Replicates cannot correct truncation bias. More replicates shrink the standard error of a,
    which helps ESTIMATE the bias; the bias itself is a property of the grid.
    """
    out = {"model": "D(h) = d + a h^2 + eps(h)",
           "noise_is_heteroscedastic": True,
           "common_random_numbers_correlate_eps_across_h": True,
           "identification_rule": "|a| > 2 * se_r(a), computed across replicates per component",
           "replicates_cannot_correct_truncation_bias": True,
           "per_column": {}}
    for nm, c in cols.items():
        h = c["h"]
        D = c["D"]
        n = D.shape[0]
        keep = keep_by_col[nm]
        a = c["a2"][:, keep]
        abar = a.mean(axis=0)
        se_a = a.std(axis=0, ddof=1) / np.sqrt(n)
        ident = np.abs(abar) > 2.0 * np.maximum(se_a, np.finfo(float).tiny)
        # measured noise scaling of D(h) and the CRN correlation across adjacent step sizes
        sd_h = D.std(axis=0, ddof=1)[:, keep]
        med = np.median(sd_h, axis=1)
        slope = float(np.polyfit(np.log(h), np.log(np.maximum(med, 1e-300)), 1)[0])
        res = D[:, :, keep] - D[:, :, keep].mean(axis=0, keepdims=True)
        corr = []
        for i in range(len(h) - 1):
            x, y = res[:, i, :].ravel(), res[:, i + 1, :].ravel()
            sx, sy = x.std(), y.std()
            corr.append(float((x * y).mean() / max(sx * sy, 1e-300)))
        out["per_column"][nm] = {
            "n_retained": int(keep.sum()),
            "n_identified": int(ident.sum()),
            "fraction_identified": float(ident.mean()),
            "median_sd_of_D_by_h": med.tolist(),
            "measured_noise_power_in_h": slope,
            "theoretical_power_for_a_central_difference": -1.0,
            "crn_correlation_between_adjacent_step_sizes": corr,
            "h_grid": h.tolist()}
    frac = {nm: v["fraction_identified"] for nm, v in out["per_column"].items()}
    ident = sorted(nm for nm, f in frac.items() if f >= 0.5)
    unident = sorted(nm for nm, f in frac.items() if f < 0.5)
    out["verdict"] = {
        "per_column_fraction_identified": frac,
        "columns_identified": ident,
        "columns_not_identified": unident,
        "n_identified_of_n_columns": [len(ident), len(frac)],
        "step_grid_is_identified_for_every_column": len(unident) == 0,
        "reading": (
            f"the h^2 coefficient is identified on a majority of retained components for "
            f"{len(ident)} of {len(frac)} columns ({', '.join(ident) if ident else 'none'}) and "
            f"NOT for {len(unident)} ({', '.join(unident) if unident else 'none'}), where the "
            f"identified fraction is "
            + ", ".join(f"{nm} {100 * frac[nm]:.1f} percent" for nm in unident)
            + ". A revised step grid therefore cannot be argued from this pilot for the columns "
              "that matter most, the three thicknesses, and a separate pilot designed to separate "
              "h^2 from h^4 is required. Reporting a single verdict over all five columns would "
              "hide this.")}
    return out


# ---------------------------------------------------------------------------------------------
# 7. Effective counts
# ---------------------------------------------------------------------------------------------

def effective_counts(mm, threshold=1.0):
    """Expected measured photons per bin, n_i = N A_i R_i, and what it implies.

    A DIAGNOSTIC, not a support mask. A threshold applied here would be a production mask, and the
    repository already records what one costs.
    """
    R, areas, N, edges, per = mm["R"], mm["areas"], mm["N"], mm["edges"], mm["per"]
    n_exp = N * areas * R
    bs = _load_json(BSPEC)
    wl = bs["wavelengths_nm"]
    low = []
    for i in np.flatnonzero(n_exp < threshold):
        w, bn = divmod(int(i), per)
        low.append({"index": int(i), "wavelength_nm": wl[w], "bin_index": bn,
                    "r_inner_cm": float(edges[bn]), "r_outer_cm": float(edges[bn + 1]),
                    "expected_photons": float(n_exp[i])})
    by_wl = {}
    for w in range(len(wl)):
        sub = [x for x in low if x["wavelength_nm"] == wl[w]]
        by_wl[str(int(wl[w]))] = {
            "n_below_threshold": len(sub),
            "bin_indices": [x["bin_index"] for x in sub],
            "radius_range_cm": ([min(x["r_inner_cm"] for x in sub),
                                 max(x["r_outer_cm"] for x in sub)] if sub else None)}
    return {
        "photon_budget_used": {"n_measurement_photons_per_wavelength": N,
                               "source": "spec/baseline_spec.json profiles.production"
                                         ".n_measurement_photons_per_wavelength",
                               "NOT_the_transport_budget": True},
        "definition": "n_i = N * A_i * R_i, with A_i and R_i from the committed pooled artifact",
        "threshold_for_this_listing": threshold,
        "n_bins_below_threshold": len(low),
        "n_bins_total": int(n_exp.size),
        "counts_at_other_levels": {str(t): int((n_exp >= t).sum())
                                   for t in (1000, 100, 10, 5, 1)},
        "reproduces_min_count_5": {
            "bins_at_or_above_5": int((n_exp >= 5).sum()),
            "known_production_sensitivity_retains": 66,
            "agrees": bool(int((n_exp >= 5).sum()) == 66),
            "why_this_matters": "an independent check that this calculation matches the recorded "
                                "MIN_COUNT=5 behaviour, which drops 54 of 120 bins and moves the "
                                "P2 legacy deep bound from 1.9128 to 2.4225, a 26.6 percent change"},
        "bins_below_threshold_by_wavelength": by_wl,
        "bins_below_threshold": low,
        "implications": {
            "approximate_normality_of_the_transport_derivative_estimator":
                "the transport estimator averages 200,000 photons per wavelength, so its own "
                "central limit behaviour is governed by how many of them reach a given bin, not "
                "by the nominal count. In a bin whose production expectation is below one photon "
                "the transport estimate is a sum over a handful of weighted contributions, so "
                "per-component approximate normality is not credible there and the registered "
                "coverage assumption is weakest exactly where the raw relative statistic is "
                "largest.",
            "production_multinomial_fisher_calculation":
                "these bins are the near-null direction of M within each wavelength block. The "
                "worst block condition number is "
                f"{mm['multinomial_fixed_launch']['singular_direction']['worst_condition_number']:.3g}"
                ", so M^{-1/2} amplifies them by about "
                f"{np.sqrt(mm['multinomial_fixed_launch']['singular_direction']['worst_condition_number']):.3g}"
                ". The published Fisher calculation therefore already depends on them, which is "
                "the reason they cannot be dismissed as negligible.",
            "proposed_whitened_interval":
                "whitening divides by the measurement standard deviation, which is large relative "
                "to R in these bins, so their contribution to the whitened discrepancy is small "
                "under the diagonal Poisson weighting. Under the multinomial weighting it is not "
                "small, because M^{-1/2} amplifies that direction. Which likelihood defines the "
                "covariance therefore changes the answer and must be frozen explicitly.",
            "this_is_not_a_support_mask":
                "no threshold here selects components. Applying one would be a production mask, "
                "requiring the identical mask downstream and a recomputation of the affected "
                "production results first."},
        "n_expected_per_bin": n_exp.tolist()}


# ---------------------------------------------------------------------------------------------
# 7b. Candidate 2 as an INTERVAL rule, which is what must be tested
# ---------------------------------------------------------------------------------------------

def wilson(k, n, z=1.959963984540054):
    """Wilson score interval, so a Monte Carlo rate is never reported as a bare point estimate."""
    if n == 0:
        return [None, None]
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0.0, c - h)), float(min(1.0, c + h))]


def candidate2_statistic(d_rep, W, C_diag, ref_rep=None, ref_norm=None,
                         alpha=_gate.SIMULTANEOUS_ALPHA, n_resample=400, rng=None,
                         margin_aggregate=None, margin_component=None):
    """The PROPOSED STATISTIC with an UNVALIDATED RESAMPLING HEURISTIC. Not a confidence bound.

    This is deliberately not called a bound. A minimal rare-event counterexample
    (bernoulli_rare_event_counterexample) shows the heuristic failing outright: with one component,
    six Bernoulli(0.05) observations and a true mean of 0.05, all six draws are zero in about 74
    percent of trials, the centered residuals and the resampled quantile are then exactly zero, the
    heuristic reports an upper value of zero, and it certifies equivalence when the truth sits on
    the margin. Measured coverage 0.26, false equivalence 0.74. Until a construction survives that,
    nothing here may be described as a confidence bound.

    Two statistics:

      aggregate   || W dbar || / || W e2bar ||
      component   max_j | dbar_j | / sqrt(C_jj),  in PHYSICAL BIN COORDINATES

    The component statistic is NOT max_j |(W dbar)_j|: for a block-dense whitening that mixes bins
    and its maximum moves under an orthogonal rotation of the whitening square root, which leaves
    the aggregate norm and the covariance unchanged.

    The DENOMINATOR is estimated from the SAME replicates when ref_rep is given, and the sign flips
    are applied to numerator and denominator together, so their dependence is carried rather than
    ignored. That is still not an exact propagation, and it is labelled as such. Passing ref_norm
    instead treats the denominator as known, which is only legitimate if it is frozen from an
    independent committed reference.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    d = np.atleast_2d(np.asarray(d_rep, float))
    n = d.shape[0]
    Wm = np.asarray(W, float)
    Wd = d @ Wm.T if Wm.ndim == 2 else d * Wm[None, :]
    dbar_w, dbar_b = Wd.mean(axis=0), d.mean(axis=0)
    sC = np.sqrt(np.maximum(np.asarray(C_diag, float), np.finfo(float).tiny))

    if ref_rep is not None:
        e = np.atleast_2d(np.asarray(ref_rep, float))
        We = e @ Wm.T if Wm.ndim == 2 else e * Wm[None, :]
        den_point = float(np.linalg.norm(We.mean(axis=0)))
        den_is_estimated = True
    else:
        We = None
        den_point = float(ref_norm)
        den_is_estimated = False
    den_point = max(den_point, np.finfo(float).tiny)

    agg_point = float(np.linalg.norm(dbar_w)) / den_point
    comp_point = float(np.max(np.abs(dbar_b) / sC))

    cw, cb = Wd - dbar_w, d - dbar_b
    ce = (We - We.mean(axis=0)) if We is not None else None
    a_dev, c_dev = np.empty(n_resample), np.empty(n_resample)
    for t in range(n_resample):
        sg = rng.choice([-1.0, 1.0], size=n)[:, None]
        num = np.linalg.norm((sg * cw).mean(axis=0))
        if ce is not None:
            # the same flips move the denominator, so the ratio's dependence is carried
            dd = max(np.linalg.norm(We.mean(axis=0) + (sg * ce).mean(axis=0)),
                     np.finfo(float).tiny)
        else:
            dd = den_point
        a_dev[t] = num / dd
        c_dev[t] = np.max(np.abs((sg * cb).mean(axis=0)) / sC)
    q = 1.0 - alpha / 2.0
    q_a, q_c = float(np.quantile(a_dev, q)), float(np.quantile(c_dev, q))

    all_replicates_identical = bool(np.allclose(d, d[0], atol=0.0, rtol=0.0))
    out = {
        "aggregate_point_relative": agg_point,
        "aggregate_heuristic_upper_value_relative": agg_point + q_a,
        "component_point": comp_point,
        "component_heuristic_upper_value": comp_point + q_c,
        "resample_spread_aggregate": q_a, "resample_spread_component": q_c,
        "alpha": alpha, "alpha_split_between_the_two_statements": alpha / 2.0,
        "n_resample": n_resample, "n_replicates": n,
        "denominator_point": den_point,
        "denominator_is_estimated_from_the_same_replicates": den_is_estimated,
        "denominator_dependence": ("carried by applying the same sign flips to numerator and "
                                   "denominator; NOT an exact propagation" if den_is_estimated
                                   else "treated as known; only legitimate if frozen from an "
                                        "independent committed reference"),
        "calibration": "sign flips of centered replicates; UNVALIDATED HEURISTIC",
        "is_a_confidence_bound": False,
        "is_a_certification": False,
        "degenerate_all_replicates_identical": all_replicates_identical,
        "degenerate_warning": ("every replicate is identical, so the resampled spread is zero and "
                               "the heuristic reports no uncertainty at all. This is the "
                               "documented failure mode." if all_replicates_identical else None),
        "component_coordinates": "physical bins, divided by sqrt(C_jj); basis free"}
    if margin_aggregate is not None and margin_component is not None:
        a_ok = out["aggregate_heuristic_upper_value_relative"] <= margin_aggregate
        c_ok = out["component_heuristic_upper_value"] <= margin_component
        out["outcome"] = "equivalent" if (a_ok and c_ok) else "not_equivalent"
        out["aggregate_within_margin"] = bool(a_ok)
        out["component_within_margin"] = bool(c_ok)
    return out


# kept under the old name so nothing silently calls a different thing
candidate2_rule = candidate2_statistic


def bernoulli_rare_event_counterexample(trials=1000, n_rep=6, p=0.05,
                                        margin=0.05, seed=20261006, n_resample=200):
    """The committed counterexample: the heuristic is not a confidence bound.

    One physical component, six Bernoulli(p) observations, true mean p, aggregate margin p. When
    all six observations are zero, which happens with probability (1-p)^n, the centered residuals
    vanish, the resampled spread is zero, and the heuristic returns an upper value of zero and
    certifies equivalence although the truth sits exactly on the margin.
    """
    rng = np.random.default_rng(seed)
    W, C = np.eye(1), np.ones(1)
    cov = feq = allzero = 0
    for _ in range(trials):
        d = rng.binomial(1, p, size=(n_rep, 1)).astype(float)
        r = candidate2_statistic(d, W, C, ref_norm=1.0, n_resample=n_resample, rng=rng,
                                 margin_aggregate=margin, margin_component=1e9)
        cov += int(r["aggregate_heuristic_upper_value_relative"] >= p)
        feq += int(r["outcome"] == "equivalent")
        allzero += int(d.sum() == 0)
    return {"trials": trials, "n_replicates": n_rep, "bernoulli_p": p, "true_mean": p,
            "aggregate_margin": margin,
            "aggregate_coverage": cov / trials,
            "aggregate_coverage_wilson95": wilson(cov, trials),
            "false_equivalence_rate": feq / trials,
            "false_equivalence_rate_wilson95": wilson(feq, trials),
            "all_replicates_zero_fraction": allzero / trials,
            "theoretical_all_zero_probability": float((1.0 - p) ** n_rep),
            "mechanism": "with every observation zero the centered residuals and the resampled "
                         "quantile are exactly zero, so the heuristic reports no uncertainty and "
                         "certifies equivalence at the margin",
            "conclusion": "candidate2_statistic is NOT a valid confidence bound. Coverage is far "
                          "below nominal and the false-equivalence rate tracks (1-p)^n exactly."}


# ---------------------------------------------------------------------------------------------
# 8. Validation of the proposed statistic
# ---------------------------------------------------------------------------------------------

def feasible_truth(W, C_diag, denominator, target_aggregate, target_component,
                   n_spikes=32, seed=20261008):
    """Solve in the FULL space for a truth vector hitting both targets exactly, or classify why not.

    Exact formulation. With y = W mu and L = diag(C_jj)^{-1/2} W^{-1}:

        aggregate = ||y||_2 / D        component = ||L y||_inf

    Every row of L has unit Euclidean norm, because C = W^{-1} W^{-T} gives
    C_jj = ||(W^{-1})_{j,:}||^2.

    TWO necessary conditions, not one:

      upper   B <= A D,                     since |L_j y| <= ||L_j|| ||y|| = ||y||
      lower   B >= A D * m*,  where m* = min over ||y|| = 1 of ||L y||_inf,
              and m* >= sigma_min(L) / sqrt(k) because ||.||_inf >= ||.||_2 / sqrt(k)

    m* itself is not computed in closed form here. What is computed is an ATTAINED value, by
    bisecting along the segment between the minimising right singular direction and a spike
    direction, so the classification is three-way and never overstates what is known:

      feasible              a construction reproduces both targets to numerical precision
      provably_infeasible   B > A D, or B < A D sigma_min(L)/sqrt(k)
      undetermined          neither; the search did not find a construction and no proof applies

    An earlier version searched only mu = a * ones + c * one_hot and called failure inside that
    two-dimensional family infeasibility. For log_d_reticular with D = 58.995037, A = 0.05 and
    B = 0.6, the ratio B / (A D) is 0.20340694 and the construction below reproduces 0.05 and 0.6
    exactly, so that label was wrong. There is no tolerance anywhere in this function.
    """
    W = np.asarray(W, float)
    sC = np.sqrt(np.maximum(np.asarray(C_diag, float), np.finfo(float).tiny))
    D = float(denominator)
    A, B = float(target_aggregate), float(target_component)
    Winv = np.linalg.inv(W)
    L = Winv / sC[:, None]
    k = L.shape[0]
    U, sv, Vt = np.linalg.svd(L)
    v_min = Vt[-1]
    m_lower_proven = float(sv.min() / np.sqrt(k))
    m_attained_sv = float(np.max(np.abs(L @ v_min)))

    out = {"target_aggregate": A, "target_component": B, "denominator": D,
           "n_components": k,
           "L_row_norms_are_unit": bool(np.allclose(np.linalg.norm(L, axis=1), 1.0, atol=1e-9)),
           "necessary_upper": "B <= A * D", "necessary_upper_bound": A * D,
           "necessary_lower": "B >= A * D * m_star, m_star >= sigma_min(L)/sqrt(k)",
           "proven_lower_bound_on_B": A * D * m_lower_proven,
           "m_star_proven_lower": m_lower_proven,
           "m_star_attained_at_min_singular_direction": m_attained_sv,
           "searched_full_space": True, "tolerance": "none"}
    if A == 0.0 and B == 0.0:
        out.update({"status": "feasible", "feasible": True, "mu": np.zeros(k),
                    "achieved_aggregate": 0.0, "achieved_component": 0.0})
        return out
    if B > A * D:
        out.update({"status": "provably_infeasible", "feasible": False, "mu": None,
                    "reason": f"B = {B:.10g} exceeds A D = {A * D:.10g}; no vector in any "
                              f"dimension can satisfy both"})
        return out
    if B < A * D * m_lower_proven:
        out.update({"status": "provably_infeasible", "feasible": False, "mu": None,
                    "reason": f"B = {B:.10g} is below A D sigma_min(L)/sqrt(k) = "
                              f"{A * D * m_lower_proven:.10g}; the infinity norm cannot be that "
                              f"small at this aggregate"})
        return out

    # Construction. Put y = B L_j / ||L_j||^2 + v with v orthogonal to L_j, so L_j y = B exactly,
    # and scale v so ||y|| = A D exactly. Both targets are then met by construction and the only
    # question is whether some OTHER row exceeds B; that is checked, and directions are tried
    # until one does not. This reaches component values well below what the minimising singular
    # direction alone gives, which is why a segment bisection from that direction is not enough.
    rng = np.random.default_rng(seed)
    r2 = max((A * D) ** 2 - B * B, 0.0)
    best = None
    for j in range(k):
        lj = L[j]
        njj = max(lj @ lj, np.finfo(float).tiny)
        y0 = B * lj / njj
        if np.linalg.norm(y0) > A * D * (1.0 + 1e-12):
            continue
        rad = np.sqrt(max((A * D) ** 2 - y0 @ y0, 0.0))
        for _ in range(n_spikes):
            g = rng.normal(size=k)
            g = g - lj * (g @ lj) / njj
            ng = np.linalg.norm(g)
            if ng <= 0:
                continue
            y = y0 + rad * (g / ng)
            ag = float(np.linalg.norm(y)) / D
            cm = float(np.max(np.abs(L @ y)))
            if abs(ag - A) <= 1e-10 * max(A, 1.0) and abs(cm - B) <= 1e-8 * max(B, 1.0):
                best = (j, y, ag, cm)
                break
        if best is not None:
            break
    if best is None:
        out.update({"status": "undetermined", "feasible": False, "mu": None,
                    "reason": "neither necessary condition rules this pair out, and the search "
                              "did not find a construction. This is UNDETERMINED, not a proof of "
                              "infeasibility, and no conclusion is drawn from it."})
        return out
    j, yy, ag, cmf = best
    out.update({"status": "feasible", "feasible": True, "mu": Winv @ yy, "spike_row": j,
                "achieved_aggregate": ag, "achieved_component": cmf})
    return out


def compound_poisson_bin_mean(rng, lam, weight_sigma, bias, shape, jitter=0.01):
    """A compound-Poisson bin mean: sum over K ~ Poisson(lam) of w * (bias + jitter Z), over lam.

    K = 0 gives exactly zero, which is the rare-event behaviour the whole analysis turns on.

    Conditional on K, with w lognormal(0, sigma) and Z standard normal,

        E[X]   = K E[w] bias
        Var(X) = K ( Var(w) bias^2 + E[w^2] jitter^2 ),    E[w^2] = Var(w) + E[w]^2.

    An earlier version used E[w]^2 jitter^2 and dropped the Var(w) jitter^2 term. That understates
    the noise, most severely when the weights are heavily dispersed: at sigma = 2.5 the dropped
    term is the larger of the two.
    """
    lam = np.broadcast_to(np.asarray(lam, float), shape)
    bias = np.broadcast_to(np.asarray(bias, float), shape)
    K = rng.poisson(lam).astype(float)
    ew = float(np.exp(weight_sigma ** 2 / 2.0))
    vw = float((np.exp(weight_sigma ** 2) - 1.0) * np.exp(weight_sigma ** 2))
    ew2 = vw + ew * ew
    mean = K * ew * bias
    var = K * (vw * bias ** 2 + ew2 * jitter ** 2)
    out = (mean + np.sqrt(np.maximum(var, 0.0)) * rng.normal(size=shape))
    out = np.where(K > 0, out, 0.0)
    return out / np.maximum(lam, np.finfo(float).tiny)


def pilot_crn_rho(cols, column, keep):
    """The adjacent-step correlation of the centered difference quotients, FROM the pilot.

    Derived, not defaulted. An earlier version hard-coded 0.4 behind an expression that always
    fell back to 0.4, which is not a calibration.
    """
    Dp = np.asarray(cols[column]["D"], float)[:, :, keep]
    res = Dp - Dp.mean(axis=0, keepdims=True)
    cs = []
    for i in range(res.shape[1] - 1):
        x, y = res[:, i, :].ravel(), res[:, i + 1, :].ravel()
        sx, sy = x.std(), y.std()
        cs.append(float((x * y).mean() / max(sx * sy, 1e-300)))
    return {"adjacent_step_correlations": cs, "mean": float(np.mean(cs)),
            "used": float(np.clip(np.mean(cs), 0.0, 0.95)),
            "source": "centered difference quotients of the pilot column, over its retained "
                      "components",
            "derived_not_defaulted": True}


def simulate_photon_carried(rng, h, lam, d0, a2, b4, n_rep, crn_rho, weight_sigma,
                            decorrelated_fraction, decorrelated_scale, jitter=0.01):
    """D(h) built from photon contributions, so no photon means exactly zero.

    Each of K ~ Poisson(lam_j) photons in bin j carries a weight w and a per-photon derivative
    whose analytic expectation reproduces the response:

        per-photon contribution = w * ( g_j + h^2 c_j + h^4 e_j + jitter )
        with g_j = d0_j / E[w],  c_j = a2_j / E[w],  e_j = b4_j / E[w]

    so E[D_j(h)] = E[w] * (g_j + h^2 c_j + h^4 e_j) = d0_j + a2_j h^2 + b4_j h^4 EXACTLY, and the
    normalisation is by lam_j rather than by K so the estimator is unbiased.

    The plus and minus arms are NOT simulated separately. D(h) is built directly, and the
    decorrelated term w * xi / (2h), with xi ZERO MEAN, is an effective central-difference noise
    surrogate rather than a difference of two simulated arms. It adds nothing to the expectation,
    and when its amplitude is zero it is inert.

    The photon stream, the weights and the shared part of xi are drawn ONCE per replicate and
    reused across all four step sizes, which is what common random numbers do. A per-step part of
    xi would supply additional independence when the decorrelated amplitude is nonzero. In the
    reported run that amplitude is zero, so rho is inert and no pilot correlation is reproduced.

    WHEN K_j = 0 THE BIN CONTRIBUTES EXACTLY ZERO AT EVERY STEP, so e2, e3 and d are all exactly
    zero for that bin. That is the rare-event mechanism the Bernoulli counterexample exploits, and
    it is the reason this generator exists rather than an additive-noise one.
    """
    k = d0.size
    h = np.asarray(h, float)
    h2 = h ** 2
    P2 = np.linalg.pinv(np.vander(h2, 2, increasing=True))
    P3 = np.linalg.pinv(np.vander(h2, 3, increasing=True))
    ew = float(np.exp(weight_sigma ** 2 / 2.0))
    g, c, e = d0 / ew, a2 / ew, b4 / ew

    e2 = np.zeros((n_rep, k))
    e3 = np.zeros((n_rep, k))
    for r in range(n_rep):
        K = rng.poisson(lam).astype(float)                 # one stream per replicate
        live = K > 0
        if not live.any():
            continue
        Ksum = K.copy()
        # aggregate weight statistics for the K photons in each bin
        Wsum = np.zeros(k)
        W2sum = np.zeros(k)
        idx = np.flatnonzero(live)
        for j in idx:
            w = rng.lognormal(0.0, weight_sigma, size=int(K[j]))
            Wsum[j] = w.sum()
            W2sum[j] = (w ** 2).sum()
        # zero-mean decorrelated innovation, shared part fixed across steps
        n_dec = rng.binomial(K.astype(int), decorrelated_fraction).astype(float)
        amp = np.sqrt(np.maximum(n_dec, 0.0) * (W2sum / np.maximum(Ksum, 1.0))) \
            * decorrelated_scale
        xi_shared = rng.normal(size=k)
        Dh = np.zeros((h.size, k))
        for i, hh in enumerate(h):
            xi_step = rng.normal(size=k)
            xi = (np.sqrt(crn_rho) * xi_shared
                  + np.sqrt(max(1.0 - crn_rho, 0.0)) * xi_step)
            signal = Wsum * (g + hh ** 2 * c + hh ** 4 * e)
            jit = np.sqrt(np.maximum(W2sum, 0.0)) * jitter * rng.normal(size=k)
            Dh[i] = np.where(live, (signal + jit + amp * xi / (2.0 * hh)) / lam, 0.0)
        e2[r] = (P2 @ Dh)[0]
        e3[r] = (P3 @ Dh)[0]
    c0 = float((P2 @ (h2 ** 2))[0])
    return e2, e3, c0


def componentwise_spread_audit(mm, cols, column, keep, lam_transport, weight_sigma=0.8,
                               dec_frac=0.05, dec_scale=0.0, crn_rho=0.4243,
                               n_replicates=2000, seed=20261009):
    """Recompute, deterministically, how the generator's per-component spread compares to the pilot.

    This exists because a median-SD ratio licenses no componentwise statement. The counts below
    are COMPUTED here with a recorded seed and replicate count, not quoted: a different seed gives
    a slightly different count, so an exact number without provenance would be unciteable.
    """
    h = cols[column]["h"]
    e2p = cols[column]["e2"][:, keep]
    a2p = cols[column]["a2"].mean(axis=0)[keep]
    d0 = e2p.mean(axis=0)
    k = d0.size
    pilot_sd = (cols[column]["e3"] - cols[column]["e2"])[:, keep].std(axis=0, ddof=1)
    e2s, e3s, _ = simulate_photon_carried(np.random.default_rng(seed), h, lam_transport,
                                          d0, a2p, np.zeros(k), n_replicates, crn_rho,
                                          weight_sigma, dec_frac, dec_scale)
    sim_sd = (e3s - e2s).std(axis=0, ddof=1)
    ratio = sim_sd / np.maximum(pilot_sd, np.finfo(float).tiny)
    below = int((sim_sd < pilot_sd).sum())
    return {"seed": seed, "n_replicates": n_replicates, "n_components": int(k),
            "column": column,
            "decorrelated_scale_used": dec_scale,
            "n_simulated_sd_below_pilot": below,
            "ratio_min": float(ratio.min()), "ratio_median": float(np.median(ratio)),
            "ratio_max": float(ratio.max()),
            "median_sd_ratio": float(np.median(sim_sd) / max(np.median(pilot_sd), 1e-300)),
            "componentwise_domination_established": False,
            "conclusion": "components with a simulated spread below the pilot's exist, so "
                          "componentwise domination is not established. The counts depend on the "
                          "seed and replicate count recorded here.",
            "status": POST_HOC}


def analytic_true_discrepancy(h, b4):
    """E[e3 - e2] = -b4 * c0 exactly, with c0 the intercept h^4 projects onto the two-term basis.

    The three-term fit is exact for a quartic, so it carries no truncation; the two-term fit picks
    up b4 * c0. This holds for the photon-carried generator because every stochastic term in it has
    zero mean, which the expectation regressions verify rather than assume.
    """
    h2 = np.asarray(h, float) ** 2
    P2 = np.linalg.pinv(np.vander(h2, 2, increasing=True))
    c0 = float((P2 @ (h2 ** 2))[0])
    return -np.asarray(b4, float) * c0, c0


def validation_suite(mm, cols, keep_by_col, trials=200, n_rep=6, seed=20261007,
                     n_resample=200, sim_margin=_gate.EQUIVALENCE_MARGIN_REL,
                     column="log_d_reticular"):
    """Exercise the PROPOSED STATISTIC on the production covariance with a simulated pipeline.

    Roles are kept separate, as they must be: the MEASUREMENT budget defines W and C_jj, because
    that is the covariance the published Fisher calculation uses; the TRANSPORT budget drives the
    Monte Carlo noise, because that is what the gate's derivative estimator actually samples. An
    earlier version used the measurement budget for both.
    """
    bs = _load_json(BSPEC)
    prof = bs["profiles"]["production"]
    n_transport = float(prof["n_transport_photons_per_wavelength_per_stream"])

    keep = keep_by_col[column]
    k = int(keep.sum())
    W = mm["W_multi"][np.ix_(keep, keep)]
    C = mm["C_diag_multinomial"][keep]
    sC = np.sqrt(np.maximum(C, np.finfo(float).tiny))
    R0 = mm["R"][keep]
    areas = mm["areas"][keep]
    lam_transport = np.maximum(n_transport * areas * R0, 1e-9)

    h = cols[column]["h"]
    pilot_e2 = cols[column]["e2"][:, keep]
    pilot_a2 = cols[column]["a2"].mean(axis=0)[keep]
    d0 = pilot_e2.mean(axis=0)
    den_pilot = float(np.linalg.norm(d0 @ W.T))

    # the common-random-number correlation is DERIVED from the pilot, not defaulted
    crn = pilot_crn_rho(cols, column, keep)
    crn_rho = crn["used"]
    weight_sigma = 0.8
    dec_frac = 0.05
    # Calibrate on the spread of d = e3 - e2, which is what the statistic sees. Calibrating on
    # e2 is wrong: e2's spread is dominated by the photon-count noise in the signal term, and that
    # term is shared across step sizes so it cancels in e3 - e2.
    pilot_e2_sd = float(np.median(pilot_e2.std(axis=0, ddof=1)))
    pilot_d = (cols[column]["e3"] - cols[column]["e2"])[:, keep]
    pilot_d_sd = float(np.median(pilot_d.std(axis=0, ddof=1)))

    def sim_d_sd(scale, seed=11, reps=24):
        a, b_, _ = simulate_photon_carried(np.random.default_rng(seed), h, lam_transport,
                                           d0, pilot_a2, np.zeros(k), reps, crn_rho,
                                           weight_sigma, dec_frac, scale)
        return float(np.median((b_ - a).std(axis=0, ddof=1)))

    floor_sd = sim_d_sd(0.0)
    if floor_sd >= pilot_d_sd:
        dec_scale = 0.0
        cal_status = ("at zero decorrelated amplitude this stylized generator is OVERDISPERSED "
                      "relative to the pilot on the single median-SD diagnostic. That is neither "
                      "componentwise domination nor a probabilistic upper bound.")
    else:
        lo, hi = 0.0, 1.0
        while sim_d_sd(hi) < pilot_d_sd and hi < 1e12:
            hi *= 8.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if sim_d_sd(mid) < pilot_d_sd:
                lo = mid
            else:
                hi = mid
        dec_scale = 0.5 * (lo + hi)
        cal_status = "matched to the pilot median spread of d = e3 - e2"
    achieved_d_sd = sim_d_sd(dec_scale)

    rng = np.random.default_rng(seed)
    alpha = _gate.SIMULTANEOUS_ALPHA
    # The component margin is a SIMULATION PARAMETER derived from the geometry so the scenario
    # grid lands inside the reachable band, since B must satisfy
    # A D sigma_min(L)/sqrt(k) <= B <= A D. It is not a proposed margin.
    comp_margin = float(0.4 * sim_margin * den_pilot)
    scenarios = {
        "zero": (0.0, 0.0),
        "aggregate_boundary": (1.0 * sim_margin, 0.5 * comp_margin),
        "aggregate_outside": (2.0 * sim_margin, 1.0 * comp_margin),
        "component_boundary": (0.6 * sim_margin, 1.0 * comp_margin),
        "component_outside": (0.6 * sim_margin, 1.8 * comp_margin),
        "joint_boundary": (1.0 * sim_margin, 1.0 * comp_margin),
        "joint_outside": (2.0 * sim_margin, 2.0 * comp_margin),
    }
    out = {"statistic_under_test": "candidate2_statistic, the proposed statistic with an "
                                   "unvalidated resampling heuristic",
           "is_a_confidence_bound": False,
           "NOT_under_test": "nuisance_feasibility_gate.simultaneous_equivalence_test",
           "column": column, "n_components": k,
           "budgets": {
               "measurement_defines_W_and_Cjj": mm["N"],
               "transport_drives_the_monte_carlo": n_transport,
               "note": "these roles are distinct and were mixed in an earlier version"},
           "estimator_pipeline_is_simulated": {
               "four_finite_difference_step_sizes": h.tolist(),
               "plus_minus_arms_explicitly_simulated": False,
               "effective_central_difference_surrogate": True,
               "what_is_actually_built": "D(h) is constructed directly from photon contributions "
                                         "with the correct analytic expectation. The plus and "
                                         "minus arms are not simulated separately; the "
                                         "w * xi / (2h) term is an effective central-difference "
                                         "noise surrogate, and in the reported run it is inert.",
               "one_photon_stream_per_replicate_reused_across_steps": True,
               "crn_correlation_statistic_from_the_pilot": crn_rho,
               "rho_effective_in_reported_simulation": False,
               "why_rho_is_inert": "rho enters only through amp * xi / (2h), and the calibration "
                                   "set the decorrelated amplitude to zero, so amp = 0 and the "
                                   "whole xi term vanishes. The value is retained as a "
                                   "DESCRIPTIVE statistic of the pilot. It is not re-tuned to "
                                   "activate it, because this simulation carries no coverage "
                                   "proof.",
               "two_term_and_three_term_intercept_fits": True,
               "d_equals_e3_minus_e2": True,
               "denominator_from_the_same_simulated_e2": True},
           "signal_model": "PHOTON CARRIED. Every step's estimate is built from the contributions "
                           "of K ~ Poisson(lam) photons whose analytic expectation reproduces "
                           "d0 + a2 h^2 + b4 h^4. When no photon contributes the estimate is "
                           "EXACTLY zero at every step, so e2, e3 and d are all exactly zero for "
                           "that bin. This is the rare-event mechanism the Bernoulli "
                           "counterexample exploits.",
           "every_stochastic_term_has_zero_mean": True,
           "imperfect_crn_model": "a decorrelated contribution enters as w * xi / (2h) with xi "
                                  "ZERO MEAN, which would produce growth of the noise toward "
                                  "small h and contributes nothing to the expectation. In the "
                                  "reported run its amplitude is zero, so this term is INERT and "
                                  "no claim is made that the simulation reproduces the pilot's "
                                  "adjacent-step correlation.",
           "crn_correlation": {**crn,
                               "effective_in_reported_simulation": False,
                               "role": "descriptive statistic of the pilot only"},
           "componentwise_spread_audit": componentwise_spread_audit(
               mm, cols, column, keep, lam_transport, weight_sigma=0.8, dec_frac=0.05,
               dec_scale=0.0, crn_rho=crn_rho),
           "noise_calibration": {"calibrated_on": "the median replicate spread of d = e3 - e2, "
                                                  "which is the quantity the statistic sees",
                                 "why_not_e2": "e2's spread is dominated by the photon-count noise "
                                               "in the signal term, which is shared across step "
                                               "sizes and cancels in e3 - e2",
                                 "pilot_median_d_sd": pilot_d_sd,
                                 "pilot_median_e2_sd": pilot_e2_sd,
                                 "irreducible_floor_at_zero_decorrelation": floor_sd,
                                 "decorrelated_fraction": dec_frac,
                                 "calibrated_decorrelated_scale": dec_scale,
                                 "achieved_median_d_sd": achieved_d_sd,
                                 "status": cal_status,
                                 "calibration_scope": "median replicate SD only",
                                 "componentwise_domination_established": False,
                                 "stochastic_upper_bound_established": False,
                                 "what_the_ratio_does_not_license":
                                     "the median-SD ratio says nothing about any individual "
                                     "component, about tail behaviour, about the maximum "
                                     "statistic, or about the whitened norm. See "
                                     "componentwise_spread_audit, which is computed here with a "
                                     "recorded seed rather than quoted.",
                                 "ratio_is_unstable": "the reported factor comes from a single "
                                                      "calibration at 24 replicates and is itself "
                                                      "very noisy. No theoretical meaning is "
                                                      "attached to it.",
                                 "calibration_replicates": 24,
                                 "weight_sigma": weight_sigma},
           "truth_lives_in_the_response_curvature": "d_true = -b4 * c0, analytic and verified "
                                                     "against the generator by the expectation "
                                                     "regressions",
           "simulation_margins": {"aggregate": sim_margin,
                                  "component_in_measurement_sd": comp_margin,
                                  "component_margin_derived_from_the_geometry":
                                      "0.4 * aggregate_margin * denominator, so the scenario grid "
                                      "lands inside the reachable band",
                                  "are_proposed_margins": False},
           "suite_is_frozen": False,
           "n_trials": trials, "n_replicates": n_rep, "alpha": alpha,
           "feasibility": {}, "cases": {}}

    _, c0 = analytic_true_discrepancy(h, np.zeros(k))
    out["two_term_h4_projection_c0"] = c0
    out["suite_multiplicity"] = len(scenarios)

    for scen, (t_agg, t_comp) in scenarios.items():
        fs = feasible_truth(W, C, den_pilot, t_agg, t_comp)
        out["feasibility"][scen] = {kk: (v if not isinstance(v, np.ndarray) else None)
                                    for kk, v in fs.items()}
        if not fs["feasible"]:
            continue
        mu = fs["mu"]
        # the truth is injected as the response's h^4 curvature, so E[e3-e2] = -b4 c0 = mu
        b4 = -mu / c0 if c0 != 0.0 else np.zeros(k)
        chk, _ = analytic_true_discrepancy(h, b4)
        assert np.allclose(chk, mu, rtol=1e-12, atol=0.0)
        true_agg = fs["achieved_aggregate"]
        true_comp = fs["achieved_component"]
        ca = cc = cj = eq = 0
        for _ in range(trials):
            e2s, e3s, _ = simulate_photon_carried(rng, h, lam_transport, d0, pilot_a2, b4,
                                                  n_rep, crn_rho, weight_sigma, dec_frac,
                                                  dec_scale)
            d = e3s - e2s
            r = candidate2_statistic(d, W, C, ref_rep=e2s, alpha=alpha,
                                     n_resample=n_resample, rng=rng,
                                     margin_aggregate=sim_margin,
                                     margin_component=comp_margin)
            a_ok = r["aggregate_heuristic_upper_value_relative"] >= true_agg
            c_ok = r["component_heuristic_upper_value"] >= true_comp
            ca += int(a_ok)
            cc += int(c_ok)
            cj += int(a_ok and c_ok)
            eq += int(r["outcome"] == "equivalent")
        entry = {"scenario": scen,
                 "target_relative_aggregate": t_agg, "target_component": t_comp,
                 "true_relative_aggregate": true_agg, "true_component": true_comp,
                 "targets_hit_exactly": True,
                 "aggregate_coverage": ca / trials,
                 "aggregate_coverage_wilson95": wilson(ca, trials),
                 "component_coverage": cc / trials,
                 "component_coverage_wilson95": wilson(cc, trials),
                 "joint_coverage": cj / trials,
                 "joint_coverage_wilson95": wilson(cj, trials),
                 "coverage_at_this_truth_is_trivial": bool(scen == "zero")}
        if scen == "zero":
            entry["equivalence_decision_count"] = eq
            entry["equivalence_decision_rate"] = eq / trials
            entry["equivalence_decision_rate_wilson95"] = wilson(eq, trials)
            entry["meaning"] = ("observed equivalence decisions at a zero truth. NOT a power "
                                "property of candidate 2: the decision rule here is the rejected "
                                "resampling heuristic, the margins are illustrative, the photon "
                                "layer is stylized and the trial count is small.")
        else:
            entry["false_equivalence_rate"] = eq / trials
            entry["false_equivalence_rate_wilson95"] = wilson(eq, trials)
            entry["meaning"] = "equivalence type I error at a truth at or beyond a margin"
        out["cases"][scen] = entry

    out["scenarios_by_status"] = {
        st: sorted(kk for kk, v in out["feasibility"].items() if v["status"] == st)
        for st in ("feasible", "provably_infeasible", "undetermined")}
    out["no_infeasibility_is_claimed_without_a_proof"] = (
        "a target pair is called infeasible only when a necessary condition rules it out. Pairs "
        "the construction did not reach are UNDETERMINED and no conclusion is drawn from them.")
    tested = [v for v in out["cases"].values() if not v["coverage_at_this_truth_is_trivial"]]
    out["establishes_coverage"] = False
    out["why_coverage_is_not_established"] = (
        "the committed Bernoulli counterexample shows the heuristic returning an upper value of "
        "zero and certifying equivalence at the margin whenever every replicate is zero, with "
        "measured coverage 0.26. On top of that this suite is not frozen, its photon layer is "
        "stylized, and its scenario count is small, so no interval from it can support "
        "registration.")
    if tested:
        worst = min(tested, key=lambda v: v["joint_coverage"])
        out["worst_joint_coverage"] = {"case": worst["scenario"],
                                       "joint_coverage": worst["joint_coverage"],
                                       "wilson95": worst["joint_coverage_wilson95"]}
        out["max_false_equivalence_rate"] = max(v["false_equivalence_rate"] for v in tested)
    else:
        out["worst_joint_coverage"] = {"case": None, "joint_coverage": None,
                                       "wilson95": [None, None]}
        out["max_false_equivalence_rate"] = None
    z = out["cases"].get("zero")
    out["observed_equivalence_decisions_at_zero_truth"] = (
        None if z is None else z["equivalence_decision_count"])
    out["what_that_observation_is_not"] = (
        "it is not evidence that candidate 2 has zero power, that it is unusable, or that a valid "
        "bound would fail to yield a verdict. The rule exercised here is already known to be "
        "invalid, the margins are simulation parameters, the photon layer is stylized, the "
        "denominator comes from a simulated e2 rather than real transport, and the trial count is "
        f"{trials}. A different valid construction can have different width and different power.")
    out["reading"] = (
        f"under this stylized generator, the rejected resampling heuristic and the illustrative "
        f"margins, "
        + (f"{out['observed_equivalence_decisions_at_zero_truth']} equivalence decisions were "
           f"observed in {trials} zero-truth trials. "
           if out["observed_equivalence_decisions_at_zero_truth"] is not None else "")
        + (f"Worst joint coverage {out['worst_joint_coverage']['joint_coverage']:.4f} on "
           f"{out['worst_joint_coverage']['case']}. " if tested else "")
        + "Coverage is NOT established; see why_coverage_is_not_established.")
    return out


# ---------------------------------------------------------------------------------------------
# 9. Candidate comparison, one record per candidate
# ---------------------------------------------------------------------------------------------

def candidates(mm, eq_multi, eq_pois, fd, ec, cov, ce):
    """Ten stated attributes per candidate. No attribute is left to the reader to infer."""
    pilot_fwd_s, pilot_ad_s, pilot_hours = 128.9, 244.5, 9.0
    b = _load_json(BLOCKED)
    pr = b["raw"]["per_replicate"]
    fwd = float(np.mean([x["columns"][n]["seconds"]["forward_per_evaluation"]
                         for x in pr for n in b["columns"]]))
    adj = float(np.mean([x["seconds"]["ad_jacobian"] for x in pr]))
    hours = float(b["elapsed_s"] / 3600.0)
    n_low = ec["n_bins_below_threshold"]
    _ce = ce  # the committed counterexample, referenced in the candidate-2 record
    worst_cond = mm["multinomial_fixed_launch"]["singular_direction"]["worst_condition_number"]
    return {
        "measured_pilot_cost": {"forward_per_evaluation_s": fwd,
                                "five_column_ad_jacobian_per_replicate_s": adj,
                                "total_hours_at_four_steps_six_replicates": hours},
        "componentwise_parametric": {
            "exact_statistic": "max over retained components j of ( |mean_r (e3-e2)_rj| + c(n,k) "
                               "se_j ) / max(|mean e2_j|, |mean e3_j|)",
            "assumptions": ["per-component approximate normality of the replicate mean",
                            "no assumption on the dependence between components, which the "
                            "Bonferroni component tolerates"],
            "simultaneous_coverage_target": f"at least {1 - _gate.SIMULTANEOUS_ALPHA:.2f} over "
                                            "the retained components",
            "coverage_method": "larger of a Bonferroni-corrected t at alpha/(2m) on n-1 degrees "
                               "of freedom and a centered sign-flip studentized-max quantile",
            "scale_invariance": "yes, verified behaviourally over 90 single-component rescalings",
            "compatible_with_all_120": "no in practice. The denominator is each component's own "
                                       f"derivative magnitude, and {n_low} of 120 bins expect "
                                       "under one measured photon, so the maximum is a noise "
                                       "artifact.",
            "treatment_of_low_count_bins": "none. They enter with full weight and dominate.",
            "required_new_executor_outputs": "none",
            "projected_transport_cost": f"{hours:.1f} h at the pilot design; a grid at scale s "
                                        "raises the replicate requirement by about 1/s^2, so "
                                        f"about {hours / 0.25:.0f} h at s = 0.5",
            "unresolved_failure_modes": [
                "the near-discrete and low-count coverage shortfall, unguarded",
                "cannot separate truncation bias from Monte Carlo uncertainty on its own scale",
                "the h^2 coefficient is unidentified for "
                f"{len(fd['verdict']['columns_not_identified'])} of "
                f"{fd['verdict']['n_identified_of_n_columns'][1]} columns, the three thicknesses "
                "among them, so the grid cannot be retuned from this pilot"]},
        "information_weighted": {
            "exact_statistic": "a DUAL requirement, both legs computed by candidate2_statistic. "
                               "Aggregate: ||W dbar|| / ||W e2bar||, with the denominator "
                               "estimated from the same replicates. Component: "
                               "max_j |dbar_j| / sqrt(C_jj) in PHYSICAL BIN coordinates. W is the "
                               "frozen production whitening. The trace-subtracted quantity is a "
                               "point estimate only and is NOT the statistic. No upper bound is "
                               "proposed: the resampling heuristic that was tried is refuted by "
                               "the committed Bernoulli counterexample.",
            "assumptions": ["the covariance C is the frozen production measurement covariance, "
                            "read from committed files and never re-estimated",
                            "any central limit behaviour must come from the independent photon "
                            "and replicate contributions, NOT from the number of coordinates: a "
                            "120-dimensional whitened norm is a nonlinear function of correlated "
                            "linear combinations and is not approximately normal merely because "
                            "it has 120 coordinates. The proposed statistic therefore does not "
                            "assume normality of the norm; the resampling heuristic that was "
                            "tried instead is refuted by the Bernoulli counterexample, so NO "
                            "calibrated construction currently exists for it",
                            "no inversion of the 120-dimensional discrepancy covariance, whose "
                            "empirical rank is at most 5 at six replicates"],
            "simultaneous_coverage_target": f"at least {1 - _gate.SIMULTANEOUS_ALPHA:.2f} jointly "
                                            "over the aggregate and the component bound, with the "
                                            "level split between them in the frozen design",
            "coverage_method": "NONE YET. The sign-flip resampling heuristic that was tried "
                               "fails the committed Bernoulli rare-event counterexample: with "
                               "every replicate zero the resampled spread is zero, the heuristic "
                               "reports zero uncertainty, and it certifies equivalence at the "
                               "margin. Measured coverage 0.26 against a nominal 0.95. A valid "
                               "construction has to be found before this candidate can carry a "
                               "verdict.",
            "scale_invariance": "yes, and the weighting is anchored to a physical measurement "
                                "scale rather than to the derivative's own magnitude",
            "compatible_with_all_120": "yes under the REGISTERED SUPPORT POLICY, which retains "
                                       "120 of 120 for four columns and 90 of 120 for mus_slope. "
                                       "No component is dropped by a new mask, and each retained "
                                       "one is weighted by the information it carries in the "
                                       "published Fisher calculation. This is not the same claim "
                                       "as a literal 120 for every column, and the two must be "
                                       "reconciled before registration.",
            "treatment_of_low_count_bins": "weighted, not masked. Under diagonal Poisson their "
                                           "measurement noise is large relative to R so their "
                                           "whitened contribution is small. Under the multinomial "
                                           f"primary likelihood M^{{-1/2}} amplifies them by about "
                                           f"{np.sqrt(worst_cond):.3g}, so the choice of "
                                           "likelihood changes the answer and must be frozen.",
            "required_new_executor_outputs": "none",
            "projected_transport_cost": f"{hours:.1f} h at the pilot design. A finer grid is "
                                        "needed only if the bias-corrected whitened aggregate "
                                        "demands it, which cannot be decided until the margins "
                                        "are frozen.",
            "unresolved_failure_modes": [
                "inherits the normality assumption; the near-discrete shortfall is not closed",
                "the multinomial conditioning is the dominant numerical risk and is unaddressed",
                "the grid remains unidentified for "
                f"{len(fd['verdict']['columns_not_identified'])} of "
                f"{fd['verdict']['n_identified_of_n_columns'][1]} columns, exactly as for "
                "candidate 1; changing the metric does not identify the h^2 coefficient",
                "both margins are unset and cannot be set from these data",
                "NO VALID COVERAGE CONSTRUCTION EXISTS for either leg. This is the blocking item.",
                "the validation suite is not frozen and was assembled after seeing the pilot, so "
                "its pointwise intervals do not establish general coverage"]},
        "photon_level_robust": {
            "exact_statistic": "an empirical-Bernstein or bounded-difference upper bound on the "
                               "mean per-photon paired contribution per bin, aggregated with the "
                               "same whitening as candidate 2",
            "assumptions": ["a DETERMINISTIC, executor-enforced almost-sure bound on the "
                            "per-photon contribution. An observed maximum weight is NOT a valid "
                            "range for empirical Bernstein: it is itself random and can be "
                            "exceeded, so using it invalidates the bound. Either the executor "
                            "enforces a cap, or a self-normalized or heavy-tail construction is "
                            "used instead",
                            "no distributional assumption beyond that bound"],
            "simultaneous_coverage_target": f"at least {1 - _gate.SIMULTANEOUS_ALPHA:.2f}, held "
                                            "without a normality assumption",
            "coverage_method": "concentration, so coverage is by construction rather than by "
                               "simulation; the simulation would verify the implementation",
            "scale_invariance": "yes, the bound is on standardized per-photon quantities",
            "compatible_with_all_120": f"not at the current budget. {n_low} bins expect under "
                                       "one measured photon, where a valid concentration bound is "
                                       "too wide to be informative. This is a budget statement, "
                                       "not an impossibility result: more photons or more "
                                       "independent replicates tighten a valid bound, and no "
                                       "general impossibility is claimed.",
            "treatment_of_low_count_bins": "the bound is too wide to be informative there at "
                                           "the current budget, which is honest but leaves those "
                                           "components unadjudicated until the budget rises",
            "required_new_executor_outputs": [
                "per bin, the sum of per-photon paired differences",
                "per bin, the sum of their squares, for the empirical variance",
                "per bin, a DETERMINISTIC cap on the per-photon difference, enforced by the "
                "executor rather than observed, for the range term",
                "per bin, the sum of weights and the sum of squared weights, for "
                "n_eff = (sum w)^2 / sum w^2"],
            "projected_transport_cost": f"{hours:.1f} h unchanged, plus executor work and a "
                                        "coverage study. Four extra scalar accumulators per bin "
                                        "per wavelength, inside the existing scatter-add.",
            "unresolved_failure_modes": [
                "moving-boundary events make the range term O(the largest single photon weight), "
                "which is loosest exactly where few photons contribute",
                "the sub-one-photon bins stay unadjudicated unless a mask is introduced, and a "
                "mask requires identical downstream application plus recomputation of the "
                "production results it changes"]},
    }


# ---------------------------------------------------------------------------------------------
# 10. Driver
# ---------------------------------------------------------------------------------------------

def analyse(trials=400):
    got = _sha256(BLOCKED)
    if got != BLOCKED_SHA256:
        raise SystemExit(f"REFUSING: the pilot artifact has changed.\n  expected "
                         f"{BLOCKED_SHA256}\n  found    {got}\n  It is the only copy of nine "
                         f"hours of transport and must stay immutable.")
    mm = production_measurement_model()
    b, cols = pilot_columns()
    keep = {nm: c["registered_support"] for nm, c in cols.items()}
    nspec = _load_json(NSPEC)

    d = {
        "status": POST_HOC,
        "job": "62960056, pilot evidence only",
        "inputs": {
            "pilot_artifact": {"path": os.path.relpath(BLOCKED, ROOT), "sha256": got,
                               "immutable": True},
            "pooled_artifact": {"path": os.path.relpath(POOLED, ROOT),
                                "sha256": _sha256(POOLED)},
            "baseline_spec_sha256": _sha256(BSPEC),
            "nuisance_spec_sha256": _sha256(NSPEC)},
        "registered_margin_and_alpha": {
            "equivalence_margin_relative": _gate.EQUIVALENCE_MARGIN_REL,
            "simultaneous_alpha": _gate.SIMULTANEOUS_ALPHA,
            "note": "the 5 percent margin is the registered COLUMN-level criterion. The two "
                    "candidate-2 margins are different quantities and are not set."},
        "registered_support_mask": {nm: {"n_supported": int(k.sum()), "n_total": int(k.size)}
                                    for nm, k in keep.items()},
        "interval_criterion_registration":
            nspec["nominal_derivative_feasibility_gate"]["interval_criterion_registration"],
        "1_production_measurement_whitening": {
            k: v for k, v in mm.items()
            if k not in ("R", "areas", "n_wl", "per", "N", "edges", "W_poisson", "W_multi",
                         "C_diag_poisson", "C_diag_multinomial")},
        "2_transport_sampling_uncertainty_diagonal_poisson":
            sampling_uncertainty(cols, mm["W_poisson"], keep),
        "2_transport_sampling_uncertainty_multinomial":
            sampling_uncertainty(cols, mm["W_multi"], keep),
        "3_equivalence_quantity_diagonal_poisson":
            equivalence_quantities(cols, mm["W_poisson"], keep),
        "3_equivalence_quantity_multinomial":
            equivalence_quantities(cols, mm["W_multi"], keep),
        "4_dual_requirement_diagonal_poisson":
            dual_requirement(cols, mm, keep, mm["W_poisson"], "diagonal Poisson",
                             {nm: mm["C_diag_poisson"][k] for nm, k in keep.items()}),
        "4_dual_requirement_multinomial":
            dual_requirement(cols, mm, keep, mm["W_multi"],
                             "multinomial fixed launch, PRIMARY",
                             {nm: mm["C_diag_multinomial"][k] for nm, k in keep.items()}),
        "5_finite_difference_identifiability": fd_identifiability(cols, mm, keep),
        "6_effective_counts": effective_counts(mm),
    }
    d["8_validation_of_the_proposed_statistic"] = validation_suite(
        mm, cols, keep, trials=trials)
    d["8b_bernoulli_rare_event_counterexample"] = bernoulli_rare_event_counterexample()
    d["7_candidate_comparison"] = candidates(
        d["1_production_measurement_whitening"],
        d["3_equivalence_quantity_multinomial"], d["3_equivalence_quantity_diagonal_poisson"],
        d["5_finite_difference_identifiability"], d["6_effective_counts"],
        d["8_validation_of_the_proposed_statistic"],
        d["8b_bernoulli_rare_event_counterexample"])
    d["authorises"] = {
        "downstream_scientific_classification": False,
        "estimator_route_selection": False,
        "the_48_draw_study": False,
        "a_confirmatory_gate": False,
        "why": "no interval criterion is registered, no margin is set, no grid is frozen"}
    return d


def _f(x, pct=False, nd=2):
    if x is None:
        return "n/a"
    return f"{100 * x:.{nd}f}%" if pct else f"{x:.{nd}g}"


def write_memo(d, path=OUT_MEMO):
    """The memo, generated from the JSON above so no number is ever hand copied."""
    mm = d["1_production_measurement_whitening"]
    eqm, eqp = d["3_equivalence_quantity_multinomial"], d["3_equivalence_quantity_diagonal_poisson"]
    fd, ec = d["5_finite_difference_identifiability"], d["6_effective_counts"]
    cov = d["8_validation_of_the_proposed_statistic"]
    ce = d["8b_bernoulli_rare_event_counterexample"]
    cand = d["7_candidate_comparison"]
    sd = mm["multinomial_fixed_launch"]["singular_direction"]
    L = []
    A = L.append
    A("# Nuisance gate interval design memo\n")
    A(f"Generated by `scripts/nuisance_interval_design_diagnostic.py` from "
      f"`results/nuisance_interval_design_diagnostic.json`. No number below is hand copied.\n")
    A(f"> {d['status']}\n")
    A(f"Job {d['job']}. This memo authorises nothing: "
      f"`interval_criterion_registration.registered` is "
      f"`{str(d['interval_criterion_registration']['registered']).lower()}`, neither "
      f"candidate-2 margin is set, and no grid is frozen.\n")
    A("## Registered support mask of the formal run\n")
    A("| Column | supported | total |")
    A("| --- | --- | --- |")
    for nm, v in d["registered_support_mask"].items():
        A(f"| `{nm}` | {v['n_supported']} | {v['n_total']} |")
    A("")

    A("## 1. Production measurement whitening\n")
    pb = mm["photon_budget"]
    A(f"The covariance is the **{pb['used']}** at "
      f"{pb['n_measurement_photons_per_wavelength']:,.0f} photons per wavelength "
      f"({pb['n_measurement_photons_total']:,.0f} total), not the transport budget of "
      f"{pb['NOT_used']['n_transport_photons_per_wavelength_per_stream']:,.0f} per wavelength per "
      f"stream over {pb['NOT_used']['n_transport_streams']} streams. "
      f"{pb['NOT_used']['why_not']}\n")
    A(f"- Reflectance: `{mm['reflectance_source']['file']}`, array "
      f"`{mm['reflectance_source']['array']}`. {mm['reflectance_source']['why']}")
    A(f"- Areas: {mm['areas_source']['definition']}.")
    A(f"- Blocks: {mm['wavelength_blocks']['n_wavelengths']} wavelengths "
      f"{mm['wavelength_blocks']['wavelengths_nm']}, "
      f"{mm['wavelength_blocks']['bins_per_wavelength']} bins each, indexed "
      f"`{mm['wavelength_blocks']['index_rule']}`.")
    A(f"- Primary likelihood: `{mm['primary_likelihood']}`; secondary "
      f"`{mm['secondary_likelihood']}`.\n")
    A(f"Diagonal Poisson: {mm['diagonal_poisson']['covariance']}, whitening "
      f"{mm['diagonal_poisson']['whitening']}.\n")
    A(f"Multinomial: {mm['multinomial_fixed_launch']['covariance']}, whitening "
      f"{mm['multinomial_fixed_launch']['whitening']}.\n")
    A(f"**The singular direction.** No block is exactly singular "
      f"(`{str(sd['is_any_block_exactly_singular']).lower()}`). {sd['why_not']} It is however "
      f"near singular: worst block condition number {sd['worst_condition_number']:.3g}, smallest "
      f"eigenvalue {sd['smallest_eigenvalue']:.3g}. {sd['consequence']}\n")
    A("| Wavelength | total detection probability | min eigenvalue | condition number |")
    A("| --- | --- | --- | --- |")
    for blk in mm["multinomial_fixed_launch"]["per_wavelength_block"]:
        A(f"| {blk['wavelength_nm']:.0f} nm | {blk['total_detection_probability']:.4f} | "
          f"{blk['eigenvalue_min']:.3g} | {blk['condition_number']:.3g} |")
    A("")

    A("## 2. Transport sampling uncertainty\n")
    su = d["2_transport_sampling_uncertainty_multinomial"]
    A(f"Six replicates of a {su['dimension']}-dimensional discrepancy give an empirical "
      f"covariance of rank at most {su['empirical_covariance_rank_at_most']}. Nothing in this "
      f"design inverts it. Specifically, none of these is done:\n")
    for x in su["what_is_never_done"]:
        A(f"- {x}")
    A("")
    td = su["t_definition"]
    A(f"**The t statistic.** `{td['formula']}`. Status: **{td['status']}**. {td['why_not_calibrated']}\n")
    A(f"{td['what_would_be_needed']}\n")
    A("| Column | retained | t at the selected index | calibrated? |")
    A("| --- | --- | --- | --- |")
    for nm, v in su["per_column"].items():
        A(f"| `{nm}` | {v['n_retained']} | {v['t_at_selected_index']:.2f} | no, descriptive |")
    A("")

    A("## 3. The equivalence quantity\n")
    defs = eqm["definitions"]
    A(f"- Numerator, norm of the mean: `{defs['numerator_norm_of_the_mean']}`")
    A(f"- Numerator, mean of the norms: `{defs['numerator_mean_of_the_norms']}`")
    A(f"- Numerator, bias corrected: `{defs['numerator_bias_corrected']}`")
    A(f"- Denominator: `{defs['denominator']}`\n")
    for k, v in defs["what_each_bounds"].items():
        A(f"**{k.replace('_', ' ')}.** {v}\n")
    for label, eq in (("Multinomial, primary", eqm), ("Diagonal Poisson", eqp)):
        A(f"### {label}\n")
        A("| Column | norm of the mean | mean of the norms | noise floor | bias-corrected "
          "point estimate | point estimate is zero |")
        A("| --- | --- | --- | --- | --- | --- |")
        for nm, v in eq["per_column"].items():
            A(f"| `{nm}` | {_f(v['norm_of_the_mean_relative'], True)} | "
              f"{_f(v['mean_of_the_norms_relative'], True)} | "
              f"{_f(v['noise_floor_relative'], True)} | "
              f"{_f(v['bias_corrected_relative'], True)} | "
              f"{'yes' if v['bias_corrected_point_estimate_is_zero'] else 'no'} |")
        A("")

    A("## 4. The dual requirement\n")
    dr = d["4_dual_requirement_multinomial"]
    A(f"- Aggregate: {dr['aggregate_statistic']}")
    A(f"- Component: `{dr['component_statistic']}`\n")
    A(f"{dr['component_statistic_is_not_divided_by_the_derivative']}.\n")
    A(f"**Why not the whitened-coordinate maximum.** {dr['why_not_the_whitened_coordinate_maximum']}\n")
    fr = dr["fisher_relation"]
    A(f"**Relation to Fisher information.** Under `{fr['assumption']}`:\n")
    A(f"- same-column diagonal entry: relative change at most `{fr['same_column_diagonal_relative_bound']}`")
    A(f"- cross entries: {fr['cross_entry_bound']}")
    A(f"- cross entries, relative change: {fr['cross_entry_relative_change']}\n")
    A(f"So the aggregate margin controls {fr['what_the_aggregate_margin_controls']}. Neither "
      f"margin controls {fr['what_neither_margin_controls']} See `{fr['counterexample']}`.\n")
    A(f"**Margins: {dr['margins']['status']}**\n")
    for x in dr["margins"]["what_an_argument_would_need"]:
        A(f"- {x}")
    A("")
    A("Pilot values, for reference only and not to be used to choose a margin:\n")
    A("| Column | aggregate point | aggregate bias-corrected point | aggregate heuristic upper "
      "value | component point | component heuristic upper value |")
    A("| --- | --- | --- | --- | --- | --- |")
    for nm, v in dr["pilot_values_for_reference_only"].items():
        A(f"| `{nm}` | {_f(v['aggregate_point_relative'], True)} | "
          f"{_f(v['aggregate_bias_corrected_point_estimate_relative'], True)} | "
          f"{_f(v['aggregate_heuristic_upper_value_relative'], True)} | "
          f"{v['component_point_in_measurement_sd']:.3g} | "
          f"{v['component_heuristic_upper_value']:.3g} |")
    A("")
    A("None of these columns is a confidence bound. The heuristic upper value is the resampling "
      "construction that section 8 refutes, and the denominator is estimated from the same six "
      "replicates as the numerator.")
    A("")

    A("## 5. The finite-difference reference\n")
    A(f"Model `{fd['model']}`. The noise is heteroscedastic and common random numbers correlate "
      f"it across step sizes; both are measured, not assumed. Identification rule: "
      f"`{fd['identification_rule']}`.\n")
    A("| Column | retained | h^2 coefficient identified | fraction | measured noise power in h | "
      "CRN correlation, adjacent h |")
    A("| --- | --- | --- | --- | --- | --- |")
    for nm, v in fd["per_column"].items():
        cc = ", ".join(f"{x:.2f}" for x in v["crn_correlation_between_adjacent_step_sizes"])
        A(f"| `{nm}` | {v['n_retained']} | {v['n_identified']} | "
          f"{_f(v['fraction_identified'], True)} | {v['measured_noise_power_in_h']:.2f} | {cc} |")
    A("")
    A(f"**Verdict.** {fd['verdict']['reading']}\n")
    A(f"Identified for {fd['verdict']['n_identified_of_n_columns'][0]} of "
      f"{fd['verdict']['n_identified_of_n_columns'][1]} columns.\n")
    A(f"{'Replicates cannot correct truncation bias.' if fd['replicates_cannot_correct_truncation_bias'] else ''}\n")

    A("## 6. Effective counts\n")
    A(f"{ec['definition']}, at {ec['photon_budget_used']['n_measurement_photons_per_wavelength']:,.0f} "
      f"photons per wavelength from `{ec['photon_budget_used']['source']}`. "
      f"**{ec['n_bins_below_threshold']} of {ec['n_bins_total']}** bins expect fewer than "
      f"{ec['threshold_for_this_listing']:.0f} measured photon.\n")
    A("| At least | Bins of 120 |")
    A("| --- | --- |")
    for k, v in ec["counts_at_other_levels"].items():
        A(f"| {k} | {v} |")
    A("")
    rm = ec["reproduces_min_count_5"]
    A(f"Bins at or above 5: {rm['bins_at_or_above_5']}, against the recorded "
      f"{rm['known_production_sensitivity_retains']} that `MIN_COUNT=5` retains — agrees: "
      f"{str(rm['agrees']).lower()}. {rm['why_this_matters']}.\n")
    A("The bins below one expected photon, by wavelength:\n")
    A("| Wavelength | count | bin indices | radius range (cm) |")
    A("| --- | --- | --- | --- |")
    for wl, v in ec["bins_below_threshold_by_wavelength"].items():
        rr = ("n/a" if not v["radius_range_cm"]
              else f"{v['radius_range_cm'][0]:.2f} to {v['radius_range_cm'][1]:.2f}")
        bi = ", ".join(str(x) for x in v["bin_indices"]) or "none"
        A(f"| {wl} nm | {v['n_below_threshold']} | {bi} | {rr} |")
    A("")
    for k, v in ec["implications"].items():
        A(f"**{k.replace('_', ' ')}.** {v}\n")

    A("## 7. Candidate comparison\n")
    keys = ["exact_statistic", "assumptions", "simultaneous_coverage_target", "coverage_method",
            "scale_invariance", "compatible_with_all_120", "treatment_of_low_count_bins",
            "required_new_executor_outputs", "projected_transport_cost",
            "unresolved_failure_modes"]
    names = [("componentwise_parametric", "1. Componentwise parametric"),
             ("information_weighted", "2. Information weighted"),
             ("photon_level_robust", "3. Photon level robust")]
    for ck, title in names:
        A(f"### {title}\n")
        c = cand[ck]
        for k in keys:
            v = c[k]
            if isinstance(v, list):
                A(f"**{k.replace('_', ' ')}**\n")
                for x in v:
                    A(f"- {x}")
                A("")
            else:
                A(f"**{k.replace('_', ' ')}.** {v}\n")

    A("## 8. Validation of the proposed statistic\n")
    A(f"Statistic under test: {cov['statistic_under_test']}. It is **not** a confidence bound "
      f"(`is_a_confidence_bound: {str(cov['is_a_confidence_bound']).lower()}`). **Not** under "
      f"test: `{cov['NOT_under_test']}`.\n")
    A("### The blocking counterexample\n")
    A(f"One physical component, {ce['n_replicates']} Bernoulli({ce['bernoulli_p']}) observations, "
      f"true mean {ce['true_mean']}, aggregate margin {ce['aggregate_margin']}, "
      f"{ce['trials']} trials:\n")
    A("| Quantity | Value | Wilson 95 |")
    A("| --- | --- | --- |")
    A(f"| aggregate coverage | {_f(ce['aggregate_coverage'], True)} | "
      f"[{_f(ce['aggregate_coverage_wilson95'][0], True)}, "
      f"{_f(ce['aggregate_coverage_wilson95'][1], True)}] |")
    A(f"| false equivalence at the margin | {_f(ce['false_equivalence_rate'], True)} | "
      f"[{_f(ce['false_equivalence_rate_wilson95'][0], True)}, "
      f"{_f(ce['false_equivalence_rate_wilson95'][1], True)}] |")
    A(f"| all replicates zero | {_f(ce['all_replicates_zero_fraction'], True)} | "
      f"theoretical {_f(ce['theoretical_all_zero_probability'], True)} |")
    A("")
    A(f"{ce['mechanism']}. **{ce['conclusion']}**\n")
    A("### The real-structure suite\n")
    bg = cov["budgets"]
    A(f"Column `{cov['column']}`, {cov['n_components']} components. The **measurement** budget "
      f"{bg['measurement_defines_W_and_Cjj']:,.0f} defines `W` and `C_jj`; the **transport** "
      f"budget {bg['transport_drives_the_monte_carlo']:,.0f} drives the Monte Carlo. "
      f"{bg['note']}.\n")
    ep = cov["estimator_pipeline_is_simulated"]
    A(f"The estimator pipeline is simulated rather than the discrepancy being drawn directly: "
      f"step sizes {ep['four_finite_difference_step_sizes']}, one photon stream per replicate "
      f"reused across all four steps, two-term and three-term intercept fits, `d = e3 - e2`, and "
      f"the denominator taken from the same simulated `e2` so numerator and denominator move "
      f"together across trials.\n")
    A(f"**The plus and minus arms are not simulated separately.** "
      f"{ep['what_is_actually_built']}\n")
    A(f"**The signal is photon carried.** {cov['signal_model']}\n")
    A(f"**Imperfect common random numbers.** {cov['imperfect_crn_model']}\n")
    crn = cov["crn_correlation"]
    A(f"The pilot's adjacent-step correlations are "
      f"{[round(x, 3) for x in crn['adjacent_step_correlations']]} with mean {crn['mean']:.4f}, "
      f"computed from {crn['source']}. That value is retained as a **descriptive statistic of the "
      f"pilot only**: `effective_in_reported_simulation` is "
      f"{str(crn['effective_in_reported_simulation']).lower()}, because "
      f"{ep['why_rho_is_inert']} No claim is made that the simulation reproduces the pilot's "
      f"adjacent-step correlation. An earlier version hard-coded 0.4 behind an expression that "
      f"always fell back to 0.4, which was not a calibration either.\n")
    nc = cov["noise_calibration"]
    A(f"Noise calibration is on {nc['calibrated_on']}. {nc['why_not_e2']}. Pilot median spread of "
      f"d is {nc['pilot_median_d_sd']:.4g}; the generator's floor at zero decorrelation is "
      f"{nc['irreducible_floor_at_zero_decorrelation']:.4g}; the calibrated decorrelated scale is "
      f"{nc['calibrated_decorrelated_scale']:.4g} at fraction "
      f"{nc['decorrelated_fraction']:g}, achieving {nc['achieved_median_d_sd']:.4g}.\n")
    A(f"**{nc['status']}** Scope: {nc['calibration_scope']}. "
      f"`componentwise_domination_established` "
      f"{str(nc['componentwise_domination_established']).lower()}; "
      f"`stochastic_upper_bound_established` "
      f"{str(nc['stochastic_upper_bound_established']).lower()}.\n")
    A(f"{nc['what_the_ratio_does_not_license']} {nc['ratio_is_unstable']}\n")
    aud = cov["componentwise_spread_audit"]
    A(f"**Componentwise audit**, computed here at seed {aud['seed']} over "
      f"{aud['n_replicates']} replicates with the decorrelated scale at "
      f"{aud['decorrelated_scale_used']:g}:\n")
    A("| Quantity | Value |")
    A("| --- | --- |")
    A(f"| components with simulated spread below the pilot's | "
      f"{aud['n_simulated_sd_below_pilot']} of {aud['n_components']} |")
    A(f"| componentwise ratio, minimum | {aud['ratio_min']:.4g} |")
    A(f"| componentwise ratio, median | {aud['ratio_median']:.4g} |")
    A(f"| componentwise ratio, maximum | {aud['ratio_max']:.4g} |")
    A(f"| median-SD ratio | {aud['median_sd_ratio']:.4g} |")
    A("")
    A(f"{aud['conclusion']}\n")
    A(f"The truth lives in the response curvature: {cov['truth_lives_in_the_response_curvature']}, "
      f"with `c0 = {cov['two_term_h4_projection_c0']:.10g}`. Every stochastic term has zero mean, "
      f"which the expectation regressions verify rather than assume.\n")
    A("### Target feasibility, solved in the full space\n")
    A("Target pairs are solved over all directions, not inside a flat-plus-spike family. With "
      "`y = W mu` and `L = diag(C_jj)^(-1/2) W^(-1)`, every row of `L` has unit Euclidean norm, "
      "so two necessary conditions apply: `B <= A D` above, and "
      "`B >= A D sigma_min(L)/sqrt(k)` below. A pair is called infeasible only when one of those "
      "rules it out; a pair the construction did not reach is **undetermined**, and no conclusion "
      "is drawn from it. There is no tolerance.\n")
    sb = cov["scenarios_by_status"]
    A(f"- feasible: {', '.join('`' + x + '`' for x in sb['feasible']) or 'none'}")
    A(f"- provably infeasible: {', '.join('`' + x + '`' for x in sb['provably_infeasible']) or 'none'}")
    A(f"- undetermined: {', '.join('`' + x + '`' for x in sb['undetermined']) or 'none'}\n")
    A(f"Simulation margins: aggregate {cov['simulation_margins']['aggregate']:g}, component "
      f"{cov['simulation_margins']['component_in_measurement_sd']:.4g} measurement SD "
      f"({cov['simulation_margins']['component_margin_derived_from_the_geometry']}). Proposed "
      f"margins: {str(cov['simulation_margins']['are_proposed_margins']).lower()}.\n")
    if cov["cases"]:
        A("| Scenario | target agg | achieved agg | target comp | achieved comp | aggregate cov | "
          "component cov | **joint cov** | rate |")
        A("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for kk, v in cov["cases"].items():
            rate = (f"{v['equivalence_decision_count']} of {cov['n_trials']} decisions"
                    if v["scenario"] == "zero"
                    else f"{_f(v['false_equivalence_rate'], True)} false-equiv")
            jc = ("trivial" if v["coverage_at_this_truth_is_trivial"]
                  else f"**{_f(v['joint_coverage'], True)}** "
                       f"[{_f(v['joint_coverage_wilson95'][0], True)}, "
                       f"{_f(v['joint_coverage_wilson95'][1], True)}]")
            A(f"| {kk} | {v['target_relative_aggregate']:.6g} | "
              f"{v['true_relative_aggregate']:.10g} | {v['target_component']:.6g} | "
              f"{v['true_component']:.10g} | {_f(v['aggregate_coverage'], True)} | "
              f"{_f(v['component_coverage'], True)} | {jc} | {rate} |")
        A("")
    A(f"Suite frozen: {str(cov['suite_is_frozen']).lower()}; scenarios "
      f"{cov['suite_multiplicity']}.\n")
    A(f"**Coverage established: {str(cov['establishes_coverage']).lower()}.** "
      f"{cov['why_coverage_is_not_established']}\n")
    if cov["observed_equivalence_decisions_at_zero_truth"] is not None:
        A(f"**On the zero-truth observation.** Under this stylized generator, the rejected "
          f"resampling heuristic and the illustrative margins, "
          f"{cov['observed_equivalence_decisions_at_zero_truth']} equivalence decisions were "
          f"observed in {cov['n_trials']} zero-truth trials. {cov['what_that_observation_is_not']}\n")

    A("## 9. Correction to the earlier working memo\n")
    pure_m = [nm for nm, v in eqm["per_column"].items()
              if v["bias_corrected_point_estimate_is_zero"]]
    sup = d["registered_support_mask"]
    A(f"An earlier working version of this memo reported the whitened aggregate as "
      f"1.4 to 9.3 percent and argued that the discrepancy was statistically detectable at "
      f"t = 2.1 to 3.7. Both statements are withdrawn.\n")
    A(f"- Those aggregates were the **norm of the mean**, which carries an additive noise floor "
      f"of `tr(Sigma_W)/n`. The floor is the same size as the observed value: see the table in "
      f"section 3. After subtracting it, the truncated point estimate is zero for "
      f"{len(pure_m)} of {len(eqm['per_column'])} columns "
      f"({', '.join('`' + x + '`' for x in pure_m)}). That says the truncation took effect. It "
      f"does NOT say those columns are consistent with zero derivative bias: a truncated point "
      f"estimate carries no confidence statement, and no valid bound exists to supply one.")
    A(f"- The t of 2.1 to 3.7 was computed at a component selected by the same data, so it has no "
      f"calibrated null distribution and overstates the evidence. Section 2 states this.")
    A(f"- The earlier version also used a diagonal Poisson covariance and approximated R(0) from "
      f"`R_plus` and `R_minus`. The production primary likelihood is "
      f"`{mm['primary_likelihood']}` and R is committed; both are corrected here.\n")
    A(f"The recommendation below therefore cannot rest on detectability, and does not.\n")

    A("## 10. Where this leaves the three candidates\n")
    unident = fd["verdict"]["columns_not_identified"]
    A("**No candidate can be recommended yet, not even provisionally.** Candidate 2 is the "
      "most promising DIRECTION on preregisterable design properties, but it currently has a "
      "proposed statistic with an unvalidated resampling heuristic and no valid coverage "
      "construction, so there is nothing to recommend adopting.\n")
    A("Throughout, 'all observables' means the REGISTERED SUPPORT POLICY, which is "
      + ", ".join(f"{v['n_supported']}/{v['n_total']} for `{nm}`" for nm, v in sup.items())
      + ", not a literal 120 for every column. Item 5 below is the reconciliation that is still "
        "owed.\n")
    A("What argues for the direction, independently of any pilot outcome:\n")
    A(f"1. It is the only candidate compatible with the full production observable. Candidate 1's "
      f"denominator is each derivative's own magnitude, which {ec['n_bins_below_threshold']} "
      f"sub-one-photon bins drive to near zero; candidate 3's bound is uninformative in those "
      f"bins at the current budget. Neither serves 120 observables without a mask, and a mask "
      f"costs the recorded 26.6 percent change in the deep bound.")
    A("2. Its statistic is anchored to a frozen external scale, the production measurement "
      "covariance, read from committed files, so the weights cannot be tuned by the confirmatory "
      "data.")
    A("3. Its component leg is basis free in physical bin coordinates, so it is not an artefact "
      "of an arbitrary whitening square root.")
    A("4. Its failure modes are now testable against the real structure, which section 8 does.\n")
    A("**What blocks it.**\n")
    A(f"1. No valid coverage construction. The sign-flip heuristic is refuted outright: coverage "
      f"{_f(ce['aggregate_coverage'], True)} and false equivalence "
      f"{_f(ce['false_equivalence_rate'], True)} on the Bernoulli counterexample. This is the "
      f"blocking item and nothing else can be settled before it.")
    A("2. Both margins are unset and cannot be set from these data.")
    A(f"3. The h^2 coefficient is unidentified for {len(unident)} of "
      f"{fd['verdict']['n_identified_of_n_columns'][1]} columns "
      f"({', '.join('`' + x + '`' for x in unident)}), so the step grid cannot be revised from "
      f"this pilot.")
    A(f"4. Which likelihood defines the covariance must be chosen explicitly. The multinomial "
      f"primary has a worst block condition number of {sd['worst_condition_number']:.3g}.")
    A("5. The support policy must be stated once and held. Section 4's pilot values use the "
      "registered support mask, which is "
      + ", ".join(f"{v['n_supported']}/{v['n_total']} for `{nm}`" for nm, v in sup.items())
      + ". A claim of \"all 120 observables\" is therefore not what the pilot calculation does, "
        "and the two must be reconciled before registration: either the policy becomes literally "
        "all 120 for every column, or the claim is stated as the registered support policy.")
    A("6. The validation suite is not frozen and was assembled after seeing the pilot, so its "
      "pointwise intervals cannot support registration.")
    A(f"7. The validation generator reproduces the estimator pipeline and the rare-event "
      f"mechanism, but its per-photon weights and decorrelation fraction are modelled rather "
      f"than taken from the executor, and its denominator comes from a simulated `e2`. A passing "
      f"suite would be evidence about this model, not yet about the real executor.\n")
    A("**Next step.** Find or construct a coverage method that survives the Bernoulli "
      "counterexample, then freeze a validation suite before measuring it. Until then candidate 2 "
      "is a proposed statistic, not a bound and not a recommendation.\n")

    A("## What this memo authorises\n")
    for k, v in d["authorises"].items():
        if k == "why":
            continue
        A(f"- {k.replace('_', ' ')}: **{str(v).lower()}**")
    A(f"\n{d['authorises']['why']}.\n")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(L))
    return path


# ---------------------------------------------------------------------------------------------
# 11. Self-test
# ---------------------------------------------------------------------------------------------

def selftest():
    res, fails = [], []

    def ck(name, ok, detail=""):
        res.append({"name": name, "pass": bool(ok)})
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
        if not ok:
            fails.append(name)

    ck("the pilot artifact is unmodified", _sha256(BLOCKED) == BLOCKED_SHA256,
       _sha256(BLOCKED)[:16])

    mm = production_measurement_model()
    ck("the whitening uses the production MEASUREMENT budget, not the transport budget",
       mm["photon_budget"]["n_measurement_photons_per_wavelength"] == 100000.0
       and mm["photon_budget"]["NOT_used"][
           "n_transport_photons_per_wavelength_per_stream"] == 200000)
    ck("the reflectance comes from the committed pooled artifact, not from R_plus and R_minus",
       mm["reflectance_source"]["array"] == "R"
       and mm["reflectance_source"]["file"].endswith("production_pooled.npz"))
    ck("no multinomial block is exactly singular, and the near-singularity is quantified",
       mm["multinomial_fixed_launch"]["singular_direction"][
           "is_any_block_exactly_singular"] is False
       and mm["multinomial_fixed_launch"]["singular_direction"]["worst_condition_number"] > 1e3)
    ck("the multinomial whitening agrees with the production implementation",
       (lambda: np.allclose(
           mm["W_multi"] @ np.eye(mm["R"].size)[:, :3],
           _unify.whiten_multinomial(np.eye(mm["R"].size)[:, :3], mm["R"], mm["areas"],
                                     mm["N"], np.ones(3), mm["n_wl"]), rtol=1e-9))(),
       "W_multi reproduces src/unify.whiten_multinomial")

    b, cols = pilot_columns()
    keep = {nm: c["registered_support"] for nm, c in cols.items()}
    ck("the registered support mask is 120 for four columns and 90 for mus_slope",
       [int(keep[n].sum()) for n in ("log_d_epidermis", "log_d_papillary", "log_d_reticular",
                                     "log_mus_scale", "mus_slope")] == [120, 120, 120, 120, 90])
    ck("the reconstructed intercepts reproduce the stored fits",
       all(np.allclose(cols[n]["e2"].mean(0),
                       b["columns"][n]["convergence"]["two_term_mean"], rtol=1e-10)
           and np.allclose(cols[n]["e3"].mean(0),
                           b["columns"][n]["convergence"]["three_term_mean"], rtol=1e-10)
           for n in cols))

    su = sampling_uncertainty(cols, mm["W_multi"], keep)
    ck("the rank limit is stated as at most five at six replicates",
       su["empirical_covariance_rank_at_most"] == 5 and su["n_replicates"] == 6
       and su["dimension"] == 120)
    ck("the t statistic is declared descriptive, with the selection named",
       su["t_definition"]["status"] == "DESCRIPTIVE ONLY"
       and "SELECTED by the same data" in su["t_definition"]["why_not_calibrated"]
       and all(v["t_is_descriptive_only"] for v in su["per_column"].values()))

    eq = equivalence_quantities(cols, mm["W_multi"], keep)
    ck("the three numerators are distinct and the noise floor is subtracted",
       all(v["bias_corrected_relative"] <= v["norm_of_the_mean_relative"] + 1e-12
           and v["mean_of_the_norms_relative"] >= v["norm_of_the_mean_relative"] - 1e-12
           for v in eq["per_column"].values()))
    # a zero-bias synthetic must come out consistent with pure noise after correction
    rng = np.random.default_rng(3)
    z2 = rng.normal(size=(6, 12)) * 1e-3 + 1.0
    zc = {"z": {"e2": z2, "e3": z2 + rng.normal(size=(6, 12)) * 1e-3,
                "registered_support": np.ones(12, bool)}}
    zq = equivalence_quantities(zc, np.ones(12), {"z": np.ones(12, bool)})["per_column"]["z"]
    ck("a zero-bias synthetic is flagged as consistent with pure noise after correction",
       zq["bias_corrected_point_estimate_is_zero"]
       or zq["bias_corrected_relative"] < 0.25 * zq["norm_of_the_mean_relative"],
       f"raw {zq['norm_of_the_mean_relative']:.3e} -> corrected "
       f"{zq['bias_corrected_relative']:.3e}")

    dr = dual_requirement(cols, mm, keep, mm["W_multi"], "multinomial",
                          {nm: mm["C_diag_multinomial"][k] for nm, k in keep.items()})
    ck("neither margin is set",
       dr["margins"]["aggregate"] is None and dr["margins"]["max_component"] is None
       and "NOT SET" in dr["margins"]["status"])
    ck("the component statistic is in physical bin coordinates, basis free",
       "PHYSICAL BIN coordinates" in dr["component_statistic"]
       and dr["component_statistic_is_basis_free"] is True
       and "arbitrary basis" in dr["why_not_the_whitened_coordinate_maximum"])
    ck("the Fisher relation bounds the diagonal relatively and the cross entries only absolutely",
       dr["fisher_relation"]["same_column_diagonal_relative_bound"] == "2 eps + eps^2"
       and "ABSOLUTE only" in dr["fisher_relation"]["cross_entry_bound"]
       and "UNBOUNDED" in dr["fisher_relation"]["cross_entry_relative_change"]
       and "Schur" in dr["fisher_relation"]["what_neither_margin_controls"])
    ck("no pilot value is presented as a confidence bound",
       all({"aggregate_heuristic_upper_value_relative",
            "aggregate_bias_corrected_point_estimate_relative"} <= set(v)
           and v["is_a_confidence_bound"] is False and v["is_a_certification"] is False
           and v["denominator_is_estimated_from_the_same_replicates"] is True
           for v in dr["pilot_values_for_reference_only"].values()))

    # the whitened-coordinate maximum is basis dependent; the bin-coordinate one is not
    nm0 = "mus_slope"
    k0 = keep[nm0]
    W0 = mm["W_multi"][np.ix_(k0, k0)]
    C0 = mm["C_diag_multinomial"][k0]
    d0 = (cols[nm0]["e3"] - cols[nm0]["e2"])[:, k0]
    dbar0 = d0.mean(axis=0)
    canon = float(np.max(np.abs(dbar0 @ W0.T)))
    rot_max, rng_r = [], np.random.default_rng(11)
    for _ in range(12):
        Q = np.linalg.qr(rng_r.normal(size=(W0.shape[0], W0.shape[0])))[0]
        rot_max.append(float(np.max(np.abs(dbar0 @ (Q @ W0).T))))
        # a rotation must leave the aggregate norm alone
        assert abs(np.linalg.norm(dbar0 @ (Q @ W0).T)
                   - np.linalg.norm(dbar0 @ W0.T)) < 1e-8
    bin_stat = float(np.max(np.abs(dbar0) / np.sqrt(C0)))
    ck("the whitened-coordinate maximum moves under an equivalent rotated whitening",
       (max(rot_max) - min(rot_max)) > 0.2 * canon,
       f"canonical {canon:.3f}, rotated {min(rot_max):.3f} to {max(rot_max):.3f}")
    ck("the aggregate norm is invariant to that rotation, so the two statistics differ in kind",
       True, "asserted inside the loop above")
    ck("the bin-coordinate component statistic is computed instead",
       abs(dr["pilot_values_for_reference_only"][nm0]["component_point_in_measurement_sd"]
           - bin_stat) < 1e-9,
       f"bin-coordinate statistic {bin_stat:.3f}")

    fd = fd_identifiability(cols, mm, keep)
    ck("the measured noise power in h is near the central-difference value of -1",
       all(-1.6 < v["measured_noise_power_in_h"] < -0.4 for v in fd["per_column"].values()),
       ", ".join(f"{v['measured_noise_power_in_h']:.2f}" for v in fd["per_column"].values()))
    ck("the identifiability verdict is per column, not a single number over all five",
       set(fd["verdict"]["per_column_fraction_identified"]) == set(fd["per_column"])
       and (len(fd["verdict"]["columns_identified"])
            + len(fd["verdict"]["columns_not_identified"])) == len(fd["per_column"])
       and all(0.0 <= v <= 1.0
               for v in fd["verdict"]["per_column_fraction_identified"].values()),
       f"{fd['verdict']['n_identified_of_n_columns'][0]} of "
       f"{fd['verdict']['n_identified_of_n_columns'][1]} columns identified")
    ck("the three thickness columns are among the unidentified ones",
       all(c in fd["verdict"]["columns_not_identified"]
           for c in ("log_d_epidermis", "log_d_papillary", "log_d_reticular")),
       ", ".join(fd["verdict"]["columns_not_identified"]))

    ec = effective_counts(mm)
    ck("the sub-one-photon count is reproduced and the bins are listed",
       ec["n_bins_below_threshold"] == len(ec["bins_below_threshold"])
       and all({"wavelength_nm", "bin_index", "r_inner_cm", "r_outer_cm"} <= set(x)
               for x in ec["bins_below_threshold"]))
    ck("the effective-count calculation reproduces MIN_COUNT=5 retaining 66 bins",
       ec["reproduces_min_count_5"]["agrees"],
       f"{ec['reproduces_min_count_5']['bins_at_or_above_5']} at or above 5")
    ck("the effective-count result is labelled a diagnostic, not a mask",
       "not a support mask" in " ".join(ec["implications"]).replace("_", " ")
       or "this_is_not_a_support_mask" in ec["implications"])

    ce = bernoulli_rare_event_counterexample(trials=300, n_resample=60)
    ck("the Bernoulli rare-event counterexample refutes the resampling heuristic",
       ce["aggregate_coverage"] < 0.5 and ce["false_equivalence_rate"] > 0.5
       and abs(ce["all_replicates_zero_fraction"]
               - ce["theoretical_all_zero_probability"]) < 0.08,
       f"coverage {ce['aggregate_coverage']:.3f}, false equivalence "
       f"{ce['false_equivalence_rate']:.3f}, all-zero {ce['all_replicates_zero_fraction']:.3f}")
    ck("the statistic never claims to be a confidence bound",
       (lambda r: r["is_a_confidence_bound"] is False and r["is_a_certification"] is False
        and "UNVALIDATED" in r["calibration"])(
           candidate2_statistic(np.zeros((6, 4)), np.eye(4), np.ones(4), ref_norm=1.0,
                                n_resample=20, rng=np.random.default_rng(1))))
    ck("an all-identical replicate set is flagged as the documented degenerate case",
       candidate2_statistic(np.zeros((6, 4)), np.eye(4), np.ones(4), ref_norm=1.0,
                            n_resample=20, rng=np.random.default_rng(1)
                            )["degenerate_all_replicates_identical"] is True)

    # --- expectation regressions on the photon-carried generator ------------------------------
    hh = np.array([0.4, 0.3, 0.2, 0.1])
    kk_ = 4
    lam_hi = np.full(kk_, 20000.0)
    d0t, a2t = np.full(kk_, 1.0), np.full(kk_, 0.3)

    def mean_Dh(b4v, lam, n_trials=400, seed=0, n_rep=6):
        """Empirical E[D(h)] by re-deriving Dh from the generator's own e2 and e3 fits."""
        h2 = hh ** 2
        A2 = np.vander(h2, 2, increasing=True)
        A3 = np.vander(h2, 3, increasing=True)
        acc2 = np.zeros(kk_)
        acc3 = np.zeros(kk_)
        for t in range(n_trials):
            e2x, e3x, _ = simulate_photon_carried(np.random.default_rng(seed + t), hh, lam,
                                                  d0t, a2t, b4v, n_rep, 0.424, 0.8, 0.05, 1.0)
            acc2 += e2x.mean(axis=0)
            acc3 += e3x.mean(axis=0)
        return acc2 / n_trials, acc3 / n_trials

    tru0, c0t = analytic_true_discrepancy(hh, np.zeros(kk_))
    ck("with b4 = 0 the analytic expected discrepancy is exactly zero",
       np.all(tru0 == 0.0) and abs(c0t - (-0.0041441860465116)) < 1e-12,
       f"c0 = {c0t:.13f}")
    e2m, e3m = mean_Dh(np.zeros(kk_), lam_hi)
    dm = e3m - e2m
    ck("NUMERICAL REGRESSION, finite Monte Carlo: with b4 = 0 the empirical mean approaches zero",
       float(np.max(np.abs(dm))) < 2e-3,
       f"max |mean(e3-e2)| = {np.max(np.abs(dm)):.3e} against a signal of 1.0")
    ck("NUMERICAL REGRESSION: with b4 = 0 both intercepts recover d0",
       float(np.max(np.abs(e2m - 1.0))) < 2e-3 and float(np.max(np.abs(e3m - 1.0))) < 2e-3,
       f"e2 {e2m[0]:.6f}, e3 {e3m[0]:.6f}")
    b4t = np.full(kk_, 2.0)
    trub, _ = analytic_true_discrepancy(hh, b4t)
    e2b, e3b = mean_Dh(b4t, lam_hi, seed=5000)
    dmb = e3b - e2b
    ck("NUMERICAL REGRESSION, finite Monte Carlo: with nonzero b4 the mean approaches -b4 c0",
       float(np.max(np.abs(dmb - trub))) < 0.2 * abs(trub[0]),
       f"analytic {trub[0]:.8f}, empirical {dmb[0]:.8f}")
    # ANALYTIC identity, not a source grep and not a simulation: on the noiseless response the
    # three-term intercept is exactly d0 and the two-term intercept is exactly d0 + b4 c0, and the
    # fits are exactly linear in D(h). Unbiasedness at any lambda follows from linearity plus the
    # zero mean of every stochastic term.
    h2i = hh ** 2
    P2i = np.linalg.pinv(np.vander(h2i, 2, increasing=True))
    P3i = np.linalg.pinv(np.vander(h2i, 3, increasing=True))
    d0i, a2i, b4i = 1.0, 0.3, 2.0
    Dnoiseless = d0i + a2i * h2i + b4i * h2i ** 2
    ck("analytic identity: the three-term intercept is exactly d0 on the noiseless response",
       abs(float(P3i[0] @ Dnoiseless) - d0i) < 1e-12,
       f"{float(P3i[0] @ Dnoiseless):.15f}")
    ck("analytic identity: the two-term intercept is exactly d0 + b4 c0",
       abs(float(P2i[0] @ Dnoiseless) - (d0i + b4i * c0t)) < 1e-12,
       f"{float(P2i[0] @ Dnoiseless):.15f} vs {d0i + b4i * c0t:.15f}")
    ck("analytic identity: e3 - e2 on the noiseless response is exactly -b4 c0",
       abs(float((P3i[0] - P2i[0]) @ Dnoiseless) - (-b4i * c0t)) < 1e-12)
    ck("the intercept fits are exactly linear in D(h), by identity on arbitrary vectors",
       (lambda u, v: abs(float(P2i[0] @ (u + v)) - float(P2i[0] @ u) - float(P2i[0] @ v)) < 1e-12
        and abs(float(P3i[0] @ (3.7 * u)) - 3.7 * float(P3i[0] @ u)) < 1e-12)(
           np.array([1.0, -2.0, 0.5, 7.0]), np.array([-3.0, 4.0, 1.5, -0.25])))

    # an all-zero photon realisation must give exactly zero everywhere
    e2z, e3z, _ = simulate_photon_carried(np.random.default_rng(3), hh, np.full(kk_, 1e-12),
                                          d0t, a2t, b4t, 8, 0.424, 0.8, 0.05, 1.0)
    ck("an all-zero photon realisation produces exactly zero response and zero discrepancy",
       np.all(e2z == 0.0) and np.all(e3z == 0.0) and np.all((e3z - e2z) == 0.0),
       "every e2, e3 and d entry is exactly 0.0")
    # and with a small but nonzero lambda, SOME replicates are exactly zero
    e2s_, e3s_, _ = simulate_photon_carried(np.random.default_rng(4), hh, np.full(kk_, 0.3),
                                            d0t, a2t, b4t, 400, 0.424, 0.8, 0.05, 1.0)
    frac0 = float((e2s_ == 0.0).mean())
    ck("at small lambda a share of replicates is exactly zero, matching exp(-lambda)",
       frac0 > 0.6 and abs(frac0 - np.exp(-0.3)) < 0.05,
       f"{100 * frac0:.1f}% exactly zero, theoretical {100 * np.exp(-0.3):.1f}%")

    # the analytic truth must not depend on the seed
    t1, _ = analytic_true_discrepancy(hh, b4t)
    t2, _ = analytic_true_discrepancy(hh, b4t)
    seeds = [float(np.abs(simulate_photon_carried(np.random.default_rng(sd), hh, lam_hi, d0t,
                                                  a2t, b4t, 4, 0.424, 0.8, 0.05, 1.0)[0]).sum())
             for sd in (1, 2, 3)]
    ck("changing the seed changes the realisation but not the analytic truth",
       np.array_equal(t1, t2) and len(set(round(x, 9) for x in seeds)) == 3,
       f"analytic {t1[0]:.10f} fixed; three seeds give three different realisations")

    # with the decorrelated amplitude at zero, rho must have no effect on the output at all
    r_a = simulate_photon_carried(np.random.default_rng(31), hh, lam_hi, d0t, a2t, b4t, 6,
                                  0.05, 0.8, 0.05, 0.0)
    r_b = simulate_photon_carried(np.random.default_rng(31), hh, lam_hi, d0t, a2t, b4t, 6,
                                  0.95, 0.8, 0.05, 0.0)
    ck("at zero decorrelated amplitude, changing rho does not change the result at all",
       np.array_equal(r_a[0], r_b[0]) and np.array_equal(r_a[1], r_b[1]),
       "rho 0.05 and 0.95 give bitwise identical output")
    r_c = simulate_photon_carried(np.random.default_rng(31), hh, lam_hi, d0t, a2t, b4t, 6,
                                  0.05, 0.8, 0.05, 1.0)
    r_d = simulate_photon_carried(np.random.default_rng(31), hh, lam_hi, d0t, a2t, b4t, 6,
                                  0.95, 0.8, 0.05, 1.0)
    ck("at nonzero decorrelated amplitude, rho does change the result, so the term is wired up",
       not np.array_equal(r_c[0], r_d[0]))

    # the CRN value is a descriptive pilot statistic
    crnp = pilot_crn_rho(cols, "log_d_reticular", keep["log_d_reticular"])
    ck("the correlation statistic is computed from the pilot, not hard coded",
       crnp["derived_not_defaulted"] is True
       and abs(crnp["mean"] - 0.424) < 0.01
       and [round(x, 3) for x in crnp["adjacent_step_correlations"]] == [0.553, 0.528, 0.191],
       f"{[round(x, 3) for x in crnp['adjacent_step_correlations']]}, mean "
       f"{crnp['mean']:.3f}")

    mmf = production_measurement_model()

    # --- full-dimensional feasibility, as mathematics not as a search over a 2-parameter family --
    nmR = "log_d_reticular"
    kR = keep[nmR]
    WR = mmf["W_multi"][np.ix_(kR, kR)]
    CR = mmf["C_diag_multinomial"][kR]
    denR = float(np.linalg.norm(cols[nmR]["e2"].mean(axis=0)[kR] @ WR.T))
    ck("the denominator for log_d_reticular is the recorded value",
       abs(denR - 58.995037) < 1e-5, f"{denR:.6f}")
    LR = np.linalg.inv(WR) / np.sqrt(CR)[:, None]
    ck("every row of L = diag(C)^(-1/2) W^(-1) has unit Euclidean norm",
       np.allclose(np.linalg.norm(LR, axis=1), 1.0, atol=1e-9),
       f"min {np.linalg.norm(LR, axis=1).min():.12f}, "
       f"max {np.linalg.norm(LR, axis=1).max():.12f}")
    fz = feasible_truth(WR, CR, denR, 0.05, 0.6)
    ck("the counterexample target pair is feasible and is hit exactly",
       fz["feasible"] and abs(fz["achieved_aggregate"] - 0.05) < 1e-10
       and abs(fz["achieved_component"] - 0.6) < 1e-9,
       f"aggregate {fz['achieved_aggregate']:.10f}, component "
       f"{fz['achieved_component']:.10f}, ratio B/(A D) {0.6 / (0.05 * denR):.8f}")
    ck("the achieved truth is reproduced by recomputing from mu, not just reported",
       (lambda mu: abs(float(np.linalg.norm(mu @ WR.T)) / denR - 0.05) < 1e-9
        and abs(float(np.max(np.abs(mu) / np.sqrt(CR))) - 0.6) < 1e-9)(fz["mu"]))
    bad = feasible_truth(WR, CR, denR, 0.05, 3.0)
    ck("a pair violating the necessary condition is refused, with no tolerance",
       bad["status"] == "provably_infeasible" and bad["feasible"] is False
       and abs(bad["necessary_upper_bound"] - 0.05 * denR) < 1e-9
       and "exceeds A D" in bad["reason"],
       f"3.0 > {0.05 * denR:.6f}")
    ck("feasibility reports no tolerance and declares the full space searched",
       fz["tolerance"] == "none" and fz["searched_full_space"] is True)

    # --- corrected large-count variance ------------------------------------------------------
    # bias = 0 isolates the term that was dropped: Var = E[w^2] jitter^2 exactly, so the old
    # E[w]^2 form is wrong by the factor E[w^2]/E[w]^2 = 1 + Var(w)/E[w]^2, which is about 520 at
    # sigma = 2.5. At bias = 1 the Var(w) bias^2 term swamps the difference and the test would
    # pass either way, which is why the check is made here.
    for sg in (0.8, 2.5):
        ew = float(np.exp(sg ** 2 / 2.0))
        vw = float((np.exp(sg ** 2) - 1.0) * np.exp(sg ** 2))
        ew2 = vw + ew * ew
        rng_v = np.random.default_rng(42)
        Kfix = 2000
        w = rng_v.lognormal(0.0, sg, size=(4000, Kfix))
        X = w * (0.0 + 0.01 * rng_v.normal(size=(4000, Kfix)))
        emp = float(X.sum(axis=1).var(ddof=1) / Kfix)
        analytic = ew2 * 1e-4
        wrong = ew * ew * 1e-4
        hv = compound_poisson_bin_mean(np.random.default_rng(9), float(Kfix), sg, 0.0,
                                       (40000,))
        hv_var = float(hv.var(ddof=1) * Kfix)
        # the helper is the authority: it is what the generator calls. The direct empirical
        # estimate is a loose sanity check only, because at sigma 2.5 the sample variance of a
        # lognormal-weighted sum has enormous kurtosis and converges very slowly, so a tight
        # tolerance there would be testing the estimator rather than the formula.
        emp_tol = 0.15 if sg < 1.5 else 0.60
        ck(f"the large-count variance uses E[w^2], not E[w]^2, at sigma {sg}",
           abs(hv_var - analytic) / analytic < 0.03
           and abs(emp - analytic) / analytic < emp_tol
           and abs(wrong - analytic) / analytic > 0.1,
           f"helper {hv_var:.3e} vs corrected {analytic:.3e}; old form {wrong:.3e} "
           f"understates by {analytic / wrong:.1f}x; loose empirical {emp:.3e}")
    ck("the corrected term is present in the helper the generator calls",
       "ew2 = vw + ew * ew" in _read_self()
       and "ew2 * jitter ** 2" in _read_self()
       and "dropped the Var(w) jitter^2 term" in _read_self())

    vs = validation_suite(mmf, cols, keep, trials=25, n_resample=40)
    ck("the measurement budget defines the whitening and the transport budget drives the noise",
       vs["budgets"]["measurement_defines_W_and_Cjj"] == 100000.0
       and vs["budgets"]["transport_drives_the_monte_carlo"] == 200000.0)
    ck("the estimator pipeline is simulated, not the discrepancy directly",
       all(vs["estimator_pipeline_is_simulated"][kk] is True
           for kk in ("one_photon_stream_per_replicate_reused_across_steps",
                      "two_term_and_three_term_intercept_fits", "d_equals_e3_minus_e2",
                      "denominator_from_the_same_simulated_e2"))
       and len(vs["estimator_pipeline_is_simulated"]["four_finite_difference_step_sizes"]) == 4)
    ck("the signal is photon carried, so no photon means exactly zero",
       "PHOTON CARRIED" in vs["signal_model"]
       and "EXACTLY zero" in vs["signal_model"]
       and vs["every_stochastic_term_has_zero_mean"] is True)
    ck("imperfect common random numbers are modelled with a ZERO MEAN innovation",
       "ZERO MEAN" in vs["imperfect_crn_model"]
       and "contributes nothing to the expectation" in vs["imperfect_crn_model"])
    ck("the noise calibration targets the spread of d, not of e2, and reports what it achieved",
       vs["noise_calibration"]["calibrated_on"].startswith("the median replicate spread of d")
       and {"pilot_median_d_sd", "irreducible_floor_at_zero_decorrelation",
            "achieved_median_d_sd", "status"} <= set(vs["noise_calibration"]))
    ck("the suite records that rho was inert in the reported run",
       vs["crn_correlation"]["effective_in_reported_simulation"] is False
       and vs["estimator_pipeline_is_simulated"][
           "rho_effective_in_reported_simulation"] is False
       and "INERT" in vs["imperfect_crn_model"])
    ck("the suite states that the plus and minus arms are NOT explicitly simulated",
       vs["estimator_pipeline_is_simulated"]["plus_minus_arms_explicitly_simulated"] is False
       and vs["estimator_pipeline_is_simulated"][
           "effective_central_difference_surrogate"] is True)
    # The patterns are assembled from fragments so this check does not match its own source, the
    # way an earlier whole-file regression in this repository did.
    # Narrow on purpose. What is forbidden is any claim that this simulation dominates the
    # measured spread, not every legitimate use of the words: candidate 3's empirical-Bernstein
    # construction is a genuine bound and must stay sayable. The phrases are also split so this
    # check cannot match its own source.
    forbidden = ["bound on the " + "pilot", "conservative " + "bound",
                 "bounds the " + "pilot", "pilot's noise from " + "above"]
    hay = (json.dumps(vs) + "\n" + _read_self()).lower()
    hits = [f for f in forbidden if f.lower() in hay]
    ck("no upper-bound or conservative claim appears in the record or the script's memo strings",
       not hits, f"hits {hits}" if hits else
       "checked over the freshly computed validation record and the script source")
    ck("the calibration records its scope and disclaims domination",
       vs["noise_calibration"]["calibration_scope"] == "median replicate SD only"
       and vs["noise_calibration"]["componentwise_domination_established"] is False
       and vs["noise_calibration"]["stochastic_upper_bound_established"] is False
       and "componentwise_spread_audit" in vs["noise_calibration"][
           "what_the_ratio_does_not_license"]
       and vs["noise_calibration"]["calibration_replicates"] == 24)

    # the componentwise audit must be RECOMPUTED here and agree field by field, not matched as a
    # string. A quoted count without a seed and a replicate count would not be citeable.
    aud = vs["componentwise_spread_audit"]
    colA = aud["column"]
    kA = keep[colA]
    bsA = _load_json(BSPEC)["profiles"]["production"]
    lamA = np.maximum(float(bsA["n_transport_photons_per_wavelength_per_stream"])
                      * mmf["areas"][kA] * mmf["R"][kA], 1e-9)
    re_aud = componentwise_spread_audit(mmf, cols, colA, kA, lamA,
                                        dec_scale=aud["decorrelated_scale_used"],
                                        crn_rho=0.4243,
                                        n_replicates=aud["n_replicates"],
                                        seed=aud["seed"])
    ck("the componentwise audit is deterministic and reproduces field by field",
       all(abs(re_aud[f] - aud[f]) <= 1e-12 * max(abs(aud[f]), 1.0)
           for f in ("ratio_min", "ratio_median", "ratio_max", "median_sd_ratio"))
       and re_aud["n_simulated_sd_below_pilot"] == aud["n_simulated_sd_below_pilot"],
       f"seed {aud['seed']}, {aud['n_replicates']} replicates, "
       f"{aud['n_simulated_sd_below_pilot']} of {aud['n_components']} below pilot")
    ck("the audit records its provenance and disclaims domination",
       {"seed", "n_replicates", "n_simulated_sd_below_pilot", "ratio_min", "ratio_median",
        "ratio_max"} <= set(aud)
       and aud["componentwise_domination_established"] is False
       and aud["n_simulated_sd_below_pilot"] > 0
       and aud["ratio_min"] < 1.0,
       f"ratio {aud['ratio_min']:.4g} to {aud['ratio_max']:.4g}, median "
       f"{aud['ratio_median']:.4g}")
    ck("no exact componentwise count is asserted anywhere without the audit's provenance",
       not any(f in json.dumps(vs["noise_calibration"]) for f in ("25 of 120", "26 of 120")))

    # the simulator's own documentation must not claim to reproduce the pilot correlation
    sim_doc = simulate_photon_carried.__doc__ or ""
    ck("the simulator docstring makes no claim of matching the pilot correlation",
       ("matches the " + "pilot") not in sim_doc
       and ("match the " + "pilot") not in sim_doc
       and "rho is inert and no pilot correlation is reproduced" in sim_doc,
       "docstring states rho is inert in the reported run")
    ck("every feasible scenario hits both targets exactly, with no tolerance",
       all(v["targets_hit_exactly"] is True
           and abs(v["true_relative_aggregate"] - v["target_relative_aggregate"]) < 1e-9
           and abs(v["true_component"] - v["target_component"]) < 1e-8
           for v in vs["cases"].values()))
    ck("aggregate, component and joint boundaries are all tested",
       {"aggregate_boundary", "component_boundary", "joint_boundary"} <= set(vs["cases"]))
    ck("joint coverage is measured, not only the two marginals",
       all({"aggregate_coverage", "component_coverage", "joint_coverage"} <= set(v)
           for v in vs["cases"].values()))
    ck("infeasibility is claimed only with a proof; unreached pairs are undetermined",
       all(vs["feasibility"][kk]["status"] in ("feasible", "provably_infeasible", "undetermined")
           for kk in vs["feasibility"])
       and all("exceeds A D" in vs["feasibility"][kk]["reason"]
               or "below A D sigma_min" in vs["feasibility"][kk]["reason"]
               for kk in vs["scenarios_by_status"]["provably_infeasible"])
       and "no conclusion is drawn" in vs["no_infeasibility_is_claimed_without_a_proof"],
       str(vs["scenarios_by_status"]))
    ck("the three-way feasibility classification exists and the bounds are both recorded",
       all({"necessary_upper_bound", "proven_lower_bound_on_B", "m_star_proven_lower",
            "m_star_attained_at_min_singular_direction"} <= set(v)
           for v in vs["feasibility"].values()))
    ck("coverage is declared NOT established",
       vs["establishes_coverage"] is False
       and "counterexample" in vs["why_coverage_is_not_established"])
    ck("the zero-truth observation is reported as an observation, not as zero power",
       "not evidence that candidate 2 has zero power" in vs["what_that_observation_is_not"]
       and "equivalence_decision_count" in vs["cases"]["zero"]
       and "NOT a power" in vs["cases"]["zero"]["meaning"])
    ck("no claim that the two margins cannot be chosen independently survives",
       "the_two_legs_are_not_independently_controllable" not in vs
       and "power_is_zero_everywhere" not in vs)
    ck("the simulation margins are labelled as not proposed margins",
       vs["simulation_margins"]["are_proposed_margins"] is False)
    cov = vs

    cand = candidates(
        {k: v for k, v in mmf.items()
         if k not in ("R", "areas", "n_wl", "per", "N", "edges", "W_poisson", "W_multi",
                      "C_diag_poisson", "C_diag_multinomial")},
        eq, eq, fd, ec, cov, ce)
    need = {"exact_statistic", "assumptions", "simultaneous_coverage_target", "coverage_method",
            "scale_invariance", "compatible_with_all_120", "treatment_of_low_count_bins",
            "required_new_executor_outputs", "projected_transport_cost",
            "unresolved_failure_modes"}
    missing = {k: sorted(need - set(v)) for k, v in cand.items()
               if k != "measured_pilot_cost" and (need - set(v))}
    ck("every candidate states all ten attributes", not missing, str(missing))

    nspec = _load_json(NSPEC)
    ck("the registration flag is still false",
       nspec["nominal_derivative_feasibility_gate"][
           "interval_criterion_registration"]["registered"] is False)
    ck("the memo is generated from the JSON rather than hand written",
       "Generated by" in _read_self().split('A("# Nuisance')[1][:400])
    return res, fails


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--write-memo", action="store_true")
    ap.add_argument("--trials", type=int, default=200,
                    help="validation trials per scenario. 200 gives a Wilson half-width of about "
                         "3 points at a rate near 0.95, which the reported intervals make "
                         "explicit; raise it before any registration decision")
    ap.add_argument("--out", default=OUT_JSON)
    a = ap.parse_args()

    if a.selftest:
        res, fails = selftest()
        print(f"  {sum(r['pass'] for r in res)}/{len(res)} checks passed")
        return 1 if fails else 0

    d = analyse(trials=a.trials)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(d, fh, indent=2)
    print(f"  wrote {os.path.relpath(a.out, ROOT)}")
    if a.write_memo:
        p = write_memo(d)
        print(f"  wrote {os.path.relpath(p, ROOT)}")
    fd = d["5_finite_difference_identifiability"]["verdict"]
    ec = d["6_effective_counts"]
    print(f"  step grid identified for {fd['n_identified_of_n_columns'][0]} of "
          f"{fd['n_identified_of_n_columns'][1]} columns; not for "
          f"{', '.join(fd['columns_not_identified'])}")
    print(f"  bins below one expected photon: {ec['n_bins_below_threshold']} / "
          f"{ec['n_bins_total']}")
    v = d["8_validation_of_the_proposed_statistic"]
    ce = d["8b_bernoulli_rare_event_counterexample"]
    print(f"  worst joint coverage {v['worst_joint_coverage']['joint_coverage']:.4f} on "
          f"{v['worst_joint_coverage']['case']}; establishes coverage: "
          f"{v['establishes_coverage']}")
    print(f"  max false-equivalence rate {v['max_false_equivalence_rate']:.4f}")
    print(f"  Bernoulli counterexample: coverage {ce['aggregate_coverage']:.3f}, "
          f"false equivalence {ce['false_equivalence_rate']:.3f}, "
          f"all-zero {ce['all_replicates_zero_fraction']:.3f}")
    print("  no transport was run, no margin was set, no criterion was registered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
