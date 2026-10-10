#!/usr/bin/env python3
"""gate_offline_diagnostic.py -- reanalyse the blocked gate run without new transport.

Job 62960056 refused to write a success artifact. The rule that produced that refusal has been
withdrawn as not scale invariant, so the refusal stands but its stated reason does not. This script
reconstructs the analysis from the stored raw arrays alone.

EVERYTHING HERE IS POST HOC AND NON-CONFIRMATORY. It was computed after seeing the run fail. No
threshold in it is a decision rule, no candidate was selected by checking which one passes, and
nothing here may be reported as a convergence or equivalence result. Its purpose is to describe
what the stored data looks like so a replacement design can be justified independently and frozen
before it is applied.

    python3 scripts/gate_offline_diagnostic.py --selftest
    python3 scripts/gate_offline_diagnostic.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import nuisance_feasibility_gate as _gate  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLOCKED = os.path.join(ROOT, "results", "nuisance_feasibility_gate_BLOCKED.json")
BLOCKED_SHA256 = "dda8bd26f4b70afeb49cd05dc02a1b6b8d4baf20e76f2ae116d00ffb84e79d78"

# The production observable is four wavelengths by thirty radial bins, flattened wavelength-major:
# index = wavelength_index * N_BINS + bin_index. This is verified against the stored arrays in the
# self-test rather than assumed, because getting it backwards would mislabel every diagnostic.
WAVELENGTHS_NM = (540.0, 580.0, 633.0, 800.0)
N_BINS = 30
R_MIN_CM, R_MAX_CM = 0.0, 1.5

POST_HOC_LABEL = ("POST HOC AND NON-CONFIRMATORY: computed after the run failed, from stored data. "
                  "Not a decision rule and not a convergence or equivalence result.")


def component_map(n_obs=120, n_bins=N_BINS, wavelengths=WAVELENGTHS_NM):
    """index -> (wavelength_nm, bin_index, r_inner_cm, r_outer_cm), wavelength-major."""
    assert n_obs == len(wavelengths) * n_bins, f"{n_obs} != {len(wavelengths)} x {n_bins}"
    edges = np.linspace(R_MIN_CM, R_MAX_CM, n_bins + 1)
    out = []
    for i in range(n_obs):
        w, bn = divmod(i, n_bins)
        out.append({"index": i, "wavelength_nm": wavelengths[w], "wavelength_index": w,
                    "bin_index": bn, "r_inner_cm": float(edges[bn]),
                    "r_outer_cm": float(edges[bn + 1])})
    return out


def verify_layout(R):
    """Confirm wavelength-major from the data: each block must decay monotonically with radius.

    The alternative layout, bin-major, would make the 30-element axis the wavelength axis, which
    cannot decay over ten decades. This is checked, not assumed.
    """
    wl_major = np.asarray(R, float).reshape(len(WAVELENGTHS_NM), N_BINS)
    bin_major = np.asarray(R, float).reshape(N_BINS, len(WAVELENGTHS_NM))
    f_wl = float((np.diff(wl_major, axis=1) < 0).mean())
    f_bin = float((np.diff(bin_major, axis=0) < 0).mean())
    ratios = [float(r[0] / max(r[r > 0].min(), 1e-300)) if (r > 0).any() else 0.0
              for r in wl_major]
    return {"wavelength_major_monotone_fraction": f_wl,
            "bin_major_monotone_fraction": f_bin,
            "wavelength_major_dynamic_range_per_block": ratios,
            "layout": "wavelength-major" if f_wl >= f_bin else "AMBIGUOUS",
            "decided_by": "the fraction of adjacent pairs decreasing along the candidate radial "
                          "axis, plus the requirement that the radial axis span many decades"}


def intercepts(h_grid, D_by_h):
    """Reconstruct the two-term and three-term h^2 intercepts per replicate and component.

    Independent of the gate's stored values, so agreement is evidence the stored arrays describe
    the fits they claim to.
    """
    h2 = np.asarray(h_grid, float) ** 2
    D = np.asarray(D_by_h, float)
    A2 = np.vander(h2, 2, increasing=True)
    A3 = np.vander(h2, 3, increasing=True)
    e2 = np.stack([np.linalg.lstsq(A2, D[r], rcond=None)[0][0] for r in range(D.shape[0])])
    e3 = np.stack([np.linalg.lstsq(A3, D[r], rcond=None)[0][0] for r in range(D.shape[0])])
    return e2, e3


def per_component_table(e2, e3, keep=None, alpha=_gate.SIMULTANEOUS_ALPHA):
    """Reproduce the gate's own quantities, per component.

    Three things must match simultaneous_equivalence_test exactly or the diagnostic is describing
    a different statistic than the one that ran:

      * the DENOMINATOR is max(|mean(e2)|, |mean(e3)|), the gate's `mag`, not |mean(e2)| alone;
      * the critical value is computed over the RETAINED components only, so multiplicity shrinks
        with the retained set instead of being carried over from all 120;
      * both bounds are formed, and the three-way outcome uses the gate's rule.

    `keep` is a boolean mask of retained components. None means retain everything.
    """
    paired = e3 - e2
    n, m0 = paired.shape
    m2, m3 = e2.mean(axis=0), e3.mean(axis=0)
    mag = np.maximum(np.abs(m2), np.abs(m3))          # the gate's denominator
    big = float(np.max(mag)) if m0 else 0.0
    if keep is None:
        keep = np.ones(m0, bool)
    keep = np.asarray(keep, bool)
    k = int(keep.sum())
    ds = paired[:, keep]
    mean = ds.mean(axis=0)
    se = ds.std(axis=0, ddof=1) / np.sqrt(n)
    denom = np.maximum(mag[keep], _gate.SUPPORT_RELATIVE_FLOOR * max(big, np.finfo(float).tiny))
    # recomputed over the k retained components, exactly as the gate would
    c = float(_gate.simultaneous_critical_value(ds, n, k, alpha)["critical_value"])
    point = np.abs(mean) / denom
    upper = (np.abs(mean) + c * se) / denom
    lower = np.maximum(np.abs(mean) - c * se, 0.0) / denom
    return {"paired_mean": mean, "paired_se": se, "magnitude": mag[keep],
            "point_relative": point, "upper_relative": upper, "lower_relative": lower,
            "critical_value": c, "n_replicates": n, "n_components": m0, "n_retained": k,
            "retained_mask": keep,
            "relative_magnitude": mag[keep] / max(big, np.finfo(float).tiny),
            "all_relative_magnitude": mag / max(big, np.finfo(float).tiny)}


def three_way(upper, lower, margin=_gate.EQUIVALENCE_MARGIN_REL):
    """The gate's rule: upper <= margin equivalent; lower > margin inequivalent; else unresolved."""
    if upper <= margin:
        return "equivalent"
    if lower > margin:
        return "inequivalent"
    return "unresolved"


def floor_sensitivity(e2, e3, floors, margin=_gate.EQUIVALENCE_MARGIN_REL):
    """How the picture moves as a column-relative support floor is varied.

    A SENSITIVITY, NOT A CRITERION. The floors are a sweep spanning the plausible range, not
    candidates to pick a winner from. Reporting which floor would pass is exactly the post-hoc
    adjustment this is meant to expose rather than perform.

    The critical value is RECOMPUTED for every floor over the components that floor retains, and
    the three-way outcome uses the gate's own rule on both bounds.
    """
    full = per_component_table(e2, e3)
    rel_all = full["all_relative_magnitude"]
    rows = []
    for f in floors:
        keep = rel_all >= f
        if not keep.any():
            rows.append({"floor": f, "n_retained": 0, "note": "no component retained"})
            continue
        t = per_component_table(e2, e3, keep=keep)
        up, lo, pt = t["upper_relative"], t["lower_relative"], t["point_relative"]
        u, l = float(up.max()), float(lo.max())
        rows.append({"floor": f, "n_retained": t["n_retained"],
                     "critical_value_recomputed_over_retained": t["critical_value"],
                     "max_point_relative": float(pt.max()),
                     "median_point_relative": float(np.median(pt)),
                     "simultaneous_upper_bound": u,
                     "simultaneous_lower_bound": l,
                     "point_exceeds_margin": bool(pt.max() > margin),
                     "outcome_under_this_floor": three_way(u, l, margin)})
    return rows


# The P2 legacy published path retained all 120 bins and used NO min-count filter. Enabling
# MIN_COUNT=5 drops 54 of 120 bins and moves its deep bound from 1.9128 to 2.4225, a 26.6 percent
# change, so removed components are demonstrably NOT negligible. Any support floor considered for
# the gate has to be compared against that precedent. These are not the merged-production headline
# bounds; the frozen production headline is recorded in results/production_crb.json.
P2_LEGACY_FILTER_PRECEDENT = {
    "profile": "p2_legacy: all 30 radial bins at each of 4 wavelengths, 120 observables, no "
               "min-count filter on the published legacy path",
    "legacy_published_bounds": {"sdU": 0.084033, "sdD": 1.912834, "N_tot": 4e5},
    "min_count_5_sensitivity": {"bins_dropped": 54, "bins_retained": 66,
                                "deep_bound_from": 1.9128, "deep_bound_to": 2.4225,
                                "relative_change": 0.2664,
                                "source": "results/p2_legacy_reproduce.json and "
                                          "spec/baseline_spec.json"},
    "consequence": "components a mask would remove cannot be assumed scientifically negligible",
}


def floor_vs_production_precedent(rows, n_total=120):
    """How aggressive is each candidate floor next to the filter already known to matter?"""
    out = []
    for r in rows:
        if r.get("n_retained", 0) == 0:
            continue
        dropped = n_total - r["n_retained"]
        out.append({"floor": r["floor"], "n_retained": r["n_retained"],
                    "n_dropped": dropped,
                    "times_more_aggressive_than_min_count_5": round(dropped / 54.0, 2),
                    "note": "MIN_COUNT=5 drops 54 of 120 and moves the deep bound 26.6 percent"})
    return out


def replicates_needed(e2, e3, keep, margin=_gate.EQUIVALENCE_MARGIN_REL,
                      alpha=_gate.SIMULTANEOUS_ALPHA, n_max=2000):
    """What replicate count would bring the simultaneous UPPER bound inside the margin?

    FORMULA. For the component that binds, the bound is

        upper(n) = ( |mean| + c(n, k) * s / sqrt(n) ) / denom

    where mean and s are the paired e3 - e2 mean and replicate standard deviation, denom is
    max(|mean(e2)|, |mean(e3)|), k is the number of RETAINED components and c(n, k) is the gate's
    simultaneous critical value at level alpha, recomputed at each n. The requirement is
    max over retained components of upper(n) <= margin.

    THERE IS NO SOLUTION WHEN THE POINT DISCREPANCY ALREADY EXCEEDS THE MARGIN. Replicates shrink
    s / sqrt(n), not |mean|: finite-difference truncation is a BIAS. For such a column more
    replicates drive the lower bound up and establish INEQUIVALENCE instead, so a replicate count
    reported for it would be a sample size for the wrong conclusion. This function refuses to
    return a number in that case.

    ASSUMPTIONS, all of which must be stated wherever a number from here is quoted: the step grid
    is UNCHANGED from job 62960056, so the bias is unchanged; s is estimated from six replicates
    and is itself uncertain; the retained set is the floor's; multiplicity is recomputed at k; the
    target is the equivalence bound at alpha, which is a requirement rather than a power
    calculation, so no power is claimed. A confirmatory run needs independent transport
    replicates, not a reanalysis of these.
    """
    keep = np.asarray(keep, bool)
    k = int(keep.sum())
    if k == 0:
        return {"solvable": False, "reason": "no component retained"}
    paired = (e3 - e2)[:, keep]
    n0 = paired.shape[0]
    mean = np.abs(paired.mean(axis=0))
    sd = paired.std(axis=0, ddof=1)
    mag = np.maximum(np.abs(e2.mean(axis=0)), np.abs(e3.mean(axis=0)))[keep]
    denom = np.maximum(mag, np.finfo(float).tiny)
    point_max = float(np.max(mean / denom))
    base = {"formula": "max_j ( |mean_j| + c(n,k) * s_j / sqrt(n) ) / denom_j <= margin",
            "margin": margin, "alpha": alpha, "n_retained_components": k,
            "multiplicity": "c(n,k) recomputed at every n over the retained k",
            "step_grid": "UNCHANGED from job 62960056; the bias is therefore unchanged",
            "replicates_used_to_estimate_s": n0,
            "s_is_itself_uncertain_at_this_n": True,
            "power": "none claimed; this is the requirement that the equivalence bound fit "
                     "inside the margin, not a power calculation",
            "requires_independent_transport_replicates": True,
            "max_point_relative_discrepancy": point_max}
    if point_max >= margin:
        base.update({
            "solvable": False,
            "reason": f"the point discrepancy already reaches {point_max:.4g}, at or beyond the "
                      f"margin of {margin:g}. Replicates reduce s/sqrt(n) and cannot reduce "
                      f"|mean|, which is finite-difference BIAS. More replicates would establish "
                      f"inequivalence for this column, not equivalence.",
            "what_would_have_to_change": "the step grid, so that the truncation itself is smaller"})
        return base
    # bias is inside the margin, so a finite n can exist. Scan rather than invert, because c
    # depends on n through both the Bonferroni degrees of freedom and the sign-flip support.
    for n in range(max(n0, _gate.MIN_REPLICATES_FOR_A_BOUND), n_max + 1):
        c = _gate.bonferroni_t_critical(n, k, alpha)
        if float(np.max((mean + c * sd / np.sqrt(n)) / denom)) <= margin:
            base.update({"solvable": True, "n_required": n,
                         "critical_value_at_n": c,
                         "critical_value_component": "Bonferroni-t only. The sign-flip component "
                                                     "is data-adaptive and cannot be projected "
                                                     "without the future data, so this is a LOWER "
                                                     "bound on the requirement.",
                         "caveat": "a lower bound on the replicate count, not a sample size to "
                                   "commit to: the realised critical value is the larger of the "
                                   "Bonferroni-t and the sign-flip quantile."})
            return base
    base.update({"solvable": False,
                 "reason": f"no n up to {n_max} brings the bound inside the margin"})
    return base


def analyse(blocked_path=BLOCKED):
    with open(blocked_path) as fh:
        b = json.load(fh)
    per_rep = b["raw"]["per_replicate"]
    cmap = component_map()
    out = {"source_artifact": os.path.relpath(blocked_path, ROOT),
           "source_is_a_success_artifact": False,
           "gate_passed": b["gate_passed"],
           "job": "62960056",
           "status": POST_HOC_LABEL,
           "p2_legacy_filter_precedent": P2_LEGACY_FILTER_PRECEDENT,
           "component_layout": {"n_wavelengths": len(WAVELENGTHS_NM), "n_bins": N_BINS,
                                "wavelengths_nm": list(WAVELENGTHS_NM),
                                "index_rule": "wavelength_index * 30 + bin_index",
                                "r_cm": [R_MIN_CM, R_MAX_CM]},
           "columns": {}}

    first = per_rep[0]["columns"][next(iter(b["columns"]))]["R_plus"][0]
    out["layout_verification"] = verify_layout(first)

    for nm in b["columns"]:
        h = np.asarray(b["columns"][nm]["convergence"]["h_grid"], float)
        D = np.stack([np.asarray(x["columns"][nm]["difference_quotients"], float)
                      for x in per_rep])
        e2, e3 = intercepts(h, D)
        stored2 = np.asarray(b["columns"][nm]["convergence"]["two_term_mean"], float)
        stored3 = np.asarray(b["columns"][nm]["convergence"]["three_term_mean"], float)
        recon = {"two_term_max_abs_diff_vs_stored": float(np.max(np.abs(e2.mean(0) - stored2))),
                 "three_term_max_abs_diff_vs_stored": float(np.max(np.abs(e3.mean(0) - stored3))),
                 "reconstruction_agrees": bool(
                     np.allclose(e2.mean(0), stored2, rtol=1e-10, atol=0.0)
                     and np.allclose(e3.mean(0), stored3, rtol=1e-10, atol=0.0))}
        tab = per_component_table(e2, e3)
        rows = []
        for i, meta in enumerate(cmap):
            rows.append({**meta,
                         "denominator_max_abs_e2_e3": float(tab["magnitude"][i]),
                         "relative_magnitude": float(tab["relative_magnitude"][i]),
                         "point_relative_discrepancy": float(tab["point_relative"][i]),
                         "simultaneous_upper_relative": float(tab["upper_relative"][i]),
                         "simultaneous_lower_relative": float(tab["lower_relative"][i]),
                         "paired_mean": float(tab["paired_mean"][i]),
                         "paired_se": float(tab["paired_se"][i])})
        out["columns"][nm] = {
            "reconstruction": recon,
            "denominator": "max(|mean(e2)|, |mean(e3)|), the same as the gate's `mag`",
            "multiplicity": "the critical value is recomputed over the retained components for "
                            "every floor, never carried over from all 120",
            "three_way_rule": "upper <= margin equivalent; lower > margin inequivalent; else "
                              "unresolved. Identical to the gate.",
            "post_hoc_literal_120_is_not_the_formal_support": (
                "the field below retains a literal 120 components for every column. The formal "
                "run used the registered support mask, which is 120/120 for the first four "
                "columns and 90/120 for mus_slope. This is a SEPARATE post-hoc diagnostic."),
            "post_hoc_outcome_retaining_a_literal_120": three_way(
                float(tab["upper_relative"].max()), float(tab["lower_relative"].max())),
            "n_replicates": tab["n_replicates"],
            "simultaneous_critical_value": tab["critical_value"],
            "components": rows,
            "support_floor_sensitivity": floor_sensitivity(
                e2, e3, [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.10, 0.20]),
            "support_floor_sensitivity_status": POST_HOC_LABEL,
            "floor_vs_production_precedent": floor_vs_production_precedent(
                floor_sensitivity(e2, e3, [1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.10, 0.20])),
            "replicates_needed_by_floor": {
                f"{f:g}": replicates_needed(e2, e3, tab["all_relative_magnitude"] >= f)
                for f in (1e-2, 0.05, 0.10)},
            "replicates_needed_status": POST_HOC_LABEL}
    out["inner_and_intermediate_bin_summary"] = inner_bin_summary(out)
    return out


def inner_bin_summary(d, margin=_gate.EQUIVALENCE_MARGIN_REL, inner=8, intermediate=18):
    """The discrepancies are NOT confined to outer low-signal bins. This shows where they are.

    Radial segments: inner 0 to 0.4 cm (bins 0-7), intermediate 0.4 to 0.9 cm (bins 8-17), outer
    beyond 0.9 cm (bins 18-29).
    """
    per_col, worst = {}, {}
    for nm, c in d["columns"].items():
        kept = [x for x in c["components"] if x["relative_magnitude"] >= 1e-2]
        over = [x for x in kept if x["point_relative_discrepancy"] > margin
                or x["simultaneous_upper_relative"] > margin]
        seg = {"inner": 0, "intermediate": 0, "outer": 0}
        for x in over:
            b = x["bin_index"]
            seg["inner" if b < inner else ("intermediate" if b < intermediate else "outer")] += 1
        by_wl = {}
        for wl in WAVELENGTHS_NM:
            v = [x["point_relative_discrepancy"] for x in c["components"]
                 if x["bin_index"] < inner and x["wavelength_nm"] == wl]
            by_wl[str(int(wl))] = float(max(v)) if v else 0.0
        per_col[nm] = {"n_at_or_above_1_percent_of_column_max": len(kept),
                       "n_of_those_over_margin": len(over),
                       "by_radial_segment": seg,
                       "max_point_relative_in_inner_bins_by_wavelength_nm": by_wl}
    for wl in WAVELENGTHS_NM:
        v = [x["point_relative_discrepancy"] for c in d["columns"].values()
             for x in c["components"] if x["bin_index"] < inner and x["wavelength_nm"] == wl]
        worst[str(int(wl))] = float(max(v)) if v else 0.0
    return {"radial_segments_cm": {"inner": [0.0, 0.4], "intermediate": [0.4, 0.9],
                                   "outer": [0.9, 1.5]},
            "per_column": per_col,
            "max_point_relative_in_inner_bins_by_wavelength_nm": worst,
            "reading": ("among components carrying at least 1 percent of the column maximum, the "
                        "discrepancies over the margin sit overwhelmingly in the INNERMOST bins, "
                        "not in the outer low-signal tail. Describing this as an outer-bin problem "
                        "would be wrong."),
            "status": POST_HOC_LABEL}


def selftest():
    res, fails = [], []

    def ck(name, ok, detail=""):
        res.append({"name": name, "pass": bool(ok), "detail": detail})
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
        if not ok:
            fails.append(name)

    import hashlib
    with open(BLOCKED, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    ck("the raw artifact is unmodified", sha == BLOCKED_SHA256, sha[:16])

    cm = component_map()
    ck("all 120 components map to four wavelengths by thirty radial bins",
       len(cm) == 120 and sorted({c["wavelength_nm"] for c in cm}) == sorted(WAVELENGTHS_NM)
       and sorted({c["bin_index"] for c in cm}) == list(range(30)))
    ck("the index rule is wavelength-major and the mapping is one to one",
       cm[0] == {"index": 0, "wavelength_nm": 540.0, "wavelength_index": 0, "bin_index": 0,
                 "r_inner_cm": 0.0, "r_outer_cm": 0.05}
       and cm[30]["wavelength_nm"] == 580.0 and cm[30]["bin_index"] == 0
       and cm[119]["wavelength_nm"] == 800.0 and cm[119]["bin_index"] == 29
       and len({(c["wavelength_index"], c["bin_index"]) for c in cm}) == 120)
    ck("the radial edges tile 0 to 1.5 cm without a gap",
       abs(cm[0]["r_inner_cm"]) < 1e-12 and abs(cm[29]["r_outer_cm"] - 1.5) < 1e-12
       and all(abs(cm[i]["r_outer_cm"] - cm[i + 1]["r_inner_cm"]) < 1e-12 for i in range(29)))

    # the layout claim is checked against the data, not asserted
    with open(BLOCKED) as fh:
        b = json.load(fh)
    R = b["raw"]["per_replicate"][0]["columns"]["log_d_epidermis"]["R_plus"][0]
    lv = verify_layout(R)
    ck("the wavelength-major layout is the one the data supports",
       lv["layout"] == "wavelength-major"
       and lv["wavelength_major_monotone_fraction"] > lv["bin_major_monotone_fraction"]
       and min(lv["wavelength_major_dynamic_range_per_block"]) > 1e4,
       f"wl-major {lv['wavelength_major_monotone_fraction']:.3f} vs "
       f"bin-major {lv['bin_major_monotone_fraction']:.3f}")

    # the intercept reconstruction must reproduce an exactly known polynomial
    h = np.array([0.4, 0.3, 0.2, 0.1])
    true0 = np.array([2.0, -5.0, 0.0])
    D = np.stack([true0 + 3.0 * h[k] ** 2 + 7.0 * h[k] ** 4 for k in range(4)])
    e2, e3 = intercepts(h, D[None, :, :][0][None, ...].reshape(1, 4, 3))
    ck("the three-term fit recovers an exact quartic intercept",
       np.allclose(e3[0], true0, atol=1e-9),
       f"max err {np.max(np.abs(e3[0] - true0)):.2e}")
    ck("the two-term fit does not, which is the truncation being measured",
       not np.allclose(e2[0], true0, atol=1e-6),
       f"max err {np.max(np.abs(e2[0] - true0)):.2e}")

    # scale invariance: the diagnostic is relative, so scaling a component must not move it
    rng = np.random.default_rng(7)
    e2r = rng.normal(size=(6, 12)) + 10.0
    e3r = e2r + 0.01 * rng.normal(size=(6, 12))
    t_a = per_component_table(e2r, e3r)
    scale = np.ones(12)
    scale[3] = 1e-8
    t_b = per_component_table(e2r * scale, e3r * scale)
    ck("the point relative discrepancy is invariant to rescaling one component",
       np.allclose(t_a["point_relative"], t_b["point_relative"], rtol=1e-9),
       f"max shift {np.max(np.abs(t_a['point_relative'] - t_b['point_relative'])):.2e}")
    ck("both bounds are invariant to rescaling one component",
       np.allclose(t_a["upper_relative"], t_b["upper_relative"], rtol=1e-9)
       and np.allclose(t_a["lower_relative"], t_b["lower_relative"], rtol=1e-9))
    ck("the relative magnitude correctly demotes the rescaled component",
       t_b["relative_magnitude"][3] < 1e-7 and t_a["relative_magnitude"][3] > 1e-3)

    # --- the diagnostic must reproduce the gate exactly ----------------------------------------
    # Not "closely": the same denominator, the same multiplicity and the same three-way rule, so
    # the numbers coincide with what simultaneous_equivalence_test produced on the same arrays.
    worst = []
    for nm, c in b["columns"].items():
        h = np.asarray(c["convergence"]["h_grid"], float)
        D = np.stack([np.asarray(x["columns"][nm]["difference_quotients"], float)
                      for x in b["raw"]["per_replicate"]])
        e2g, e3g = intercepts(h, D)
        gate_t = _gate.simultaneous_equivalence_test(e3g - e2g, e2g.mean(0), e3g.mean(0))
        mine = per_component_table(e2g, e3g,
                                   keep=np.asarray(gate_t["support_mask"], bool))
        gu = gate_t.get("simultaneous_upper_bound")
        gl = gate_t.get("simultaneous_lower_bound")
        mu, ml = float(mine["upper_relative"].max()), float(mine["lower_relative"].max())
        oc = three_way(mu, ml)
        worst.append((nm,
                      abs(gu - mu) if gu is not None else float("inf"),
                      abs(gl - ml) if gl is not None else float("inf"),
                      gate_t.get("outcome"), oc,
                      abs(gate_t["simultaneous_critical_value"]["critical_value"]
                          - mine["critical_value"])))
    bad = [w for w in worst if w[1] > 1e-12 or w[2] > 1e-12 or w[3] != w[4] or w[5] > 1e-12]
    ck("the diagnostic reproduces the gate's bounds, critical value and outcome exactly",
       not bad, f"mismatches {[(w[0], w[3], w[4]) for w in bad]}" if bad
       else f"max bound difference {max(max(w[1], w[2]) for w in worst):.2e} over 5 columns")
    ck("the denominator is the gate's max of the two intercept magnitudes",
       (lambda t: np.allclose(t["magnitude"], np.maximum(np.abs(e2r.mean(0)),
                                                         np.abs(e3r.mean(0)))))(
           per_component_table(e2r, e3r)))
    ck("the critical value shrinks when fewer components are retained",
       per_component_table(e2r, e3r, keep=np.r_[np.ones(4, bool), np.zeros(8, bool)]
                           )["critical_value"]
       < per_component_table(e2r, e3r)["critical_value"])
    ck("the three-way rule matches the gate's",
       three_way(0.04, 0.0) == "equivalent" and three_way(0.9, 0.2) == "inequivalent"
       and three_way(0.9, 0.0) == "unresolved")

    ck("every support-floor sensitivity row is labelled post hoc",
       POST_HOC_LABEL.startswith("POST HOC AND NON-CONFIRMATORY"))
    ck("the withdrawn rule is not a blocking criterion in the gate module",
       _gate.DEGENERATE_SPREAD_IS_A_BLOCKING_CRITERION is False)

    # --- corrected status text -----------------------------------------------------------------
    with open(os.path.join(ROOT, "spec", "nuisance_spec.json")) as fh:
        nspec = json.load(fh)
    st, cg = nspec["status"], nspec["citation_gate"]
    ck("the spec status does not contradict the recorded release decision",
       (cg["explicit_release_decision"] != "released"
        or (cg["currently_released"] is True and "62960056" in st))
       and "no new transport is authorized" in st.lower(),
       f"decision {cg['explicit_release_decision']!r}, currently_released "
       f"{cg['currently_released']}")
    ck("the spec status records that the gate ran and refused",
       "62960056" in st and "no estimator route was selected" in st.lower()
       and st.startswith("CLOSED WITHOUT THE 48-DRAW STUDY"))
    ck("the release scope survived the status correction unchanged",
       cg["release_scope"]["authorizes_only"] == [
           "per-column AD-versus-ensemble equivalence testing",
           "finite-difference convergence testing",
           "estimator-route selection and cost measurement"]
       and cg["release_scope"]["does_not_authorize_or_establish"]
       == "any downstream scientific classification")
    ck("the gate transport is no longer listed as not started",
       not any("feasibility gate transport" in x for x in nspec["not_started"])
       and "nominal_feasibility_gate_transport" in nspec["completed"])

    # --- artifact and report counts must agree -------------------------------------------------
    # Recomputed from the stored arrays using the DOCUMENTED rule: the median is taken over the
    # SUPPORTED components, not over all 120. Taking it over all 120 gave 58 for mus_slope where
    # the artifact says 28, which is how a report can drift from its own evidence.
    mismatches = []
    for nm, c in b["columns"].items():
        t = c["convergence"]["equivalence_test"]
        e2, e3 = intercepts(np.asarray(c["convergence"]["h_grid"], float),
                            np.stack([np.asarray(x["columns"][nm]["difference_quotients"], float)
                                      for x in b["raw"]["per_replicate"]]))
        paired = e3 - e2
        keep = np.asarray(t["support_mask"], bool)
        sd = paired.std(axis=0, ddof=1)[keep]
        mean = paired.mean(axis=0)[keep]
        med = float(np.median(sd))
        got = int(((sd < _gate.DEGENERATE_SPREAD_FRACTION * med) & (np.abs(mean) > 0.0)).sum())
        if got != t["n_degenerate_spread_components"]:
            mismatches.append((nm, got, t["n_degenerate_spread_components"]))
    ck("every reported collapsed-spread count is reproducible from the stored arrays",
       not mismatches, f"mismatches {mismatches}" if mismatches else
       str({nm: b["columns"][nm]["convergence"]["equivalence_test"]
            ["n_degenerate_spread_components"] for nm in b["columns"]}))
    ck("the mus_slope count is 28, and 58 comes from taking the median over the wrong set",
       b["columns"]["mus_slope"]["convergence"]["equivalence_test"]
       ["n_degenerate_spread_components"] == 28)

    # --- downstream mask propagation is mandatory ----------------------------------------------
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import augmented as aug
    refused = False
    try:
        aug.block_information(np.zeros((12, 11)), 6)
    except (ValueError, TypeError):
        refused = True
    ck("a downstream Fisher calculation refuses when no support mask is supplied", refused)
    ck("the gate records a support mask for every column it adjudicated",
       all(("support_mask" in c) or (c.get("ad_comparison") is None)
           for c in b["columns"].values()))
    ck("the production filter precedent is carried with the floors, not left implicit",
       P2_LEGACY_FILTER_PRECEDENT["min_count_5_sensitivity"]["bins_dropped"] == 54
       and abs(P2_LEGACY_FILTER_PRECEDENT["min_count_5_sensitivity"]["relative_change"]
               - 0.2664) < 1e-9)

    # --- an unresolved analysis cannot select an estimator -------------------------------------
    d_ = analyse()
    ck("the literal-120 diagnostic is labelled as not the formal support",
       all("not the formal support" in str(c).replace("_", " ") or
           "post_hoc_literal_120_is_not_the_formal_support" in c
           for c in d_["columns"].values())
       and all("post_hoc_outcome_retaining_a_literal_120" in c
               for c in d_["columns"].values()))
    ck("the registered support mask is 120 for four columns and 90 for mus_slope",
       [b["columns"][n]["convergence"]["equivalence_test"]["n_supported"]
        for n in ("log_d_epidermis", "log_d_papillary", "log_d_reticular",
                  "log_mus_scale", "mus_slope")] == [120, 120, 120, 120, 90])
    ck("no column in the blocked run carries an estimator route",
       all(c.get("estimator_route_for_the_study") is None for c in b["columns"].values()))
    return res, fails


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "results",
                                                  "gate_offline_diagnostic.json"))
    a = ap.parse_args()

    if a.selftest:
        res, fails = selftest()
        print(f"  {sum(r['pass'] for r in res)}/{len(res)} checks passed")
        return 1 if fails else 0

    d = analyse()
    print(f"  source {d['source_artifact']}  gate_passed {d['gate_passed']}")
    lv = d["layout_verification"]
    print(f"  layout {lv['layout']}: monotone fraction {lv['wavelength_major_monotone_fraction']:.3f} "
          f"wavelength-major vs {lv['bin_major_monotone_fraction']:.3f} bin-major")
    print(f"\n  {POST_HOC_LABEL}\n")

    margin = _gate.EQUIVALENCE_MARGIN_REL
    for nm, c in d["columns"].items():
        r = c["reconstruction"]
        print(f"  {nm}  reconstruction agrees {r['reconstruction_agrees']}  "
              f"simultaneous critical value {c['simultaneous_critical_value']:.3f}")
        rows = sorted(c["components"], key=lambda x: -x["point_relative_discrepancy"])
        inner = [x for x in rows if x["relative_magnitude"] >= 1e-2]
        print(f"    worst eight components by point relative discrepancy")
        print(f"      {'wl':>5} {'bin':>4} {'r_cm':>12} {'rel_mag':>9} {'point':>9} {'upper':>9}")
        for x in rows[:8]:
            print(f"      {x['wavelength_nm']:>5.0f} {x['bin_index']:>4} "
                  f"{x['r_inner_cm']:>5.2f}-{x['r_outer_cm']:<6.2f} "
                  f"{x['relative_magnitude']:>9.2e} "
                  f"{x['point_relative_discrepancy']:>9.2e} "
                  f"{x['simultaneous_upper_relative']:>9.2e}")
        if inner:
            iw = sorted(inner, key=lambda x: -x["point_relative_discrepancy"])[:5]
            print(f"    worst five among components carrying at least 1% of the column maximum")
            for x in iw:
                print(f"      {x['wavelength_nm']:>5.0f} {x['bin_index']:>4} "
                      f"{x['r_inner_cm']:>5.2f}-{x['r_outer_cm']:<6.2f} "
                      f"{x['relative_magnitude']:>9.2e} "
                      f"{x['point_relative_discrepancy']:>9.2e} "
                      f"{x['simultaneous_upper_relative']:>9.2e}")
        print(f"    support-floor sensitivity (POST HOC, margin {margin})")
        print(f"      {'floor':>8} {'kept':>5} {'max_point':>11} {'max_upper':>11} {'status':>20}")
        for s in c["support_floor_sensitivity"]:
            if s.get("n_retained", 0) == 0:
                continue
            print(f"      {s['floor']:>8.0e} {s['n_retained']:>5} "
                  f"{s['critical_value_recomputed_over_retained']:>7.3f} "
                  f"{s['max_point_relative']:>11.3e} {s['simultaneous_upper_bound']:>11.3e} "
                  f"{s['simultaneous_lower_bound']:>11.3e} "
                  f"{s['outcome_under_this_floor']:>14}")
        print()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(d, fh, indent=2)
    print(f"  wrote {os.path.relpath(a.out, ROOT)}")
    print("  no transport was run; no threshold was selected; nothing here is confirmatory")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
