#!/usr/bin/env python3
"""Read-only alternative-design audit for Candidate 3 Phase 3B2.

The current literal-all-120 / 0.5-measurement-SD formulation is infeasible because its
deterministic empirical-Bernstein range term requires more than ten thousand transport
replicates.  This program quantifies two ways one might reduce that lower bound:

* a data-independent support contract based on nominal expected measurement counts; and
* a detector-resolution floor in the component scale.

Both are sensitivity analyses, not approved replacements.  The first changes the estimand by
discarding bins; the second changes the component margin.  The literature review in the output
also records why a tighter bounded-mean interval is not treated as an automatic drop-in fix.

This program runs no transport or coverage experiment and cannot emit a Candidate 3 verdict.

    <python> -W error scripts/candidate3_alternative_design_analysis.py --selftest
    <python> -W error scripts/candidate3_alternative_design_analysis.py --run --out PATH
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REGISTRATION_ANALYSIS = os.path.join(
    ROOT, "results", "candidate3_registration_analysis.json")

STATUS = "ALTERNATIVE DESIGN AUDIT; NOT A CANDIDATE 3 VERDICT"
SUPPORT_THRESHOLDS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 2.0, 5.0, 10.0,
                      20.0, 50.0, 100.0)
COUNT_FLOORS = (0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def canonical_digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def minimum_replicates(range_width, component_scale, photons_per_replicate,
                       alpha_component):
    """Range-term-only lower bound for simultaneous component intervals."""
    r = np.asarray(range_width, float)
    s = np.asarray(component_scale, float)
    if r.shape != s.shape or r.size == 0:
        raise ValueError("range width and component scale must be non-empty and conformable")
    if np.any(~np.isfinite(r)) or np.any(~np.isfinite(s)):
        raise ValueError("range width and component scale must be finite")
    if np.any(r <= 0.0) or np.any(s <= 0.0):
        raise ValueError("range width and component scale must be strictly positive")
    if int(photons_per_replicate) <= 0 or not (0.0 < alpha_component < 1.0):
        raise ValueError("invalid photon count or component alpha")
    alpha_bin = float(alpha_component) / int(r.size)
    required = 1.0 + 7.0 * r * math.log(4.0 / alpha_bin) / (3.0 * s)
    worst = int(np.argmax(required))
    photons = int(math.ceil(float(required[worst])))
    return {
        "minimum_replicates": int(math.ceil(photons / int(photons_per_replicate))),
        "minimum_total_photons_at_worst_component": photons,
        "worst_local_index": worst,
    }


def _inputs():
    sys.path.insert(0, HERE)
    import candidate3_grid_analysis as grid_module
    import candidate3_registration_analysis as registration_module

    registration = load_json(REGISTRATION_ANALYSIS)
    if registration.get("candidate3_verdict") is not None:
        raise ValueError("registration analysis unexpectedly contains a Candidate 3 verdict")
    if registration.get("coverage_claimed") is not False:
        raise ValueError("registration analysis must record coverage_claimed=false")
    if registration.get("registration_recommendation", {}).get(
            "freeze_current_design") is not False:
        raise ValueError("the infeasible source design must remain unfrozen")

    decisions = registration["fixed_decisions"]
    if decisions.get("support") != registration_module.SUPPORT:
        raise ValueError("the source analysis is not the literal-all-120 design")
    if tuple(decisions.get("common_grid", ())) != registration_module.SELECTED_GRID:
        raise ValueError("the source analysis does not use the selected common grid")

    _, areas, reflectance = grid_module.whitening_matrix()
    base = load_json(grid_module.BASELINE_SPEC)
    profile = base["profiles"]["production"]
    n_measurement = int(profile["n_measurement_photons_per_wavelength"])
    photons_per_replicate = int(
        decisions["photons_per_transport_replicate_per_wavelength"])
    expected_counts = n_measurement * areas * reflectance
    covariance_diagonal = (
        reflectance * (1.0 - areas * reflectance) / (n_measurement * areas))
    fit = registration_module.fit_rows(registration_module.SELECTED_GRID)
    range_width = registration_module.combined_range_width(
        registration_module.SELECTED_GRID, areas, fit["qd"])
    if any(x.shape != (registration_module.N_COMPONENTS,)
           for x in (areas, reflectance, expected_counts,
                     covariance_diagonal, range_width)):
        raise ValueError("production inputs must contain exactly 120 components")
    if np.any(expected_counts <= 0.0) or np.any(covariance_diagonal <= 0.0):
        raise ValueError("production expected counts and covariance must be positive")
    return (registration, registration_module, areas, expected_counts,
            covariance_diagonal, range_width, n_measurement,
            photons_per_replicate)


def analyze():
    (registration, rm, areas, expected_counts, covariance_diagonal,
     range_width, n_measurement, photons_per_replicate) = _inputs()
    measurement_sd = np.sqrt(covariance_diagonal)

    support_rows = []
    for threshold in SUPPORT_THRESHOLDS:
        retained = expected_counts >= threshold
        bound = minimum_replicates(
            range_width[retained],
            rm.COMPONENT_MARGIN * measurement_sd[retained],
            photons_per_replicate, rm.ALPHA_COMPONENT)
        global_indices = np.flatnonzero(retained)
        worst = int(global_indices[bound.pop("worst_local_index")])
        support_rows.append({
            "nominal_expected_measurement_count_threshold": threshold,
            "retained_components": int(retained.sum()),
            "removed_components": int(retained.size - retained.sum()),
            "minimum_replicates_range_term_only": bound["minimum_replicates"],
            "minimum_total_photons_at_worst_component": bound[
                "minimum_total_photons_at_worst_component"],
            "worst_component_index_wavelength_major": worst,
            "worst_component_nominal_expected_measurement_count": float(
                expected_counts[worst]),
        })

    floor_rows = []
    for floor_count in COUNT_FLOORS:
        scale = measurement_sd.copy()
        if floor_count > 0.0:
            scale = np.maximum(scale, floor_count / (n_measurement * areas))
        bound = minimum_replicates(
            range_width, rm.COMPONENT_MARGIN * scale,
            photons_per_replicate, rm.ALPHA_COMPONENT)
        worst = bound.pop("worst_local_index")
        floor_rows.append({
            "detector_count_resolution_floor": floor_count,
            "retained_components": 120,
            "minimum_replicates_range_term_only": bound["minimum_replicates"],
            "minimum_total_photons_at_worst_component": bound[
                "minimum_total_photons_at_worst_component"],
            "worst_component_index_wavelength_major": worst,
        })

    original = registration["range_only_hard_lower_bound"]
    if support_rows[0]["minimum_replicates_range_term_only"] != int(
            original["minimum_replicates"]):
        raise ValueError("threshold-zero sensitivity does not reproduce the source result")
    if floor_rows[0]["minimum_replicates_range_term_only"] != int(
            original["minimum_replicates"]):
        raise ValueError("zero-floor sensitivity does not reproduce the source result")

    return {
        "schema_version": "candidate3-alternative-design-audit-v1",
        "status": STATUS,
        "analysis_only": True,
        "transport_run_by_this_analysis": False,
        "coverage_run": False,
        "candidate3_verdict": None,
        "coverage_claimed": False,
        "inputs": {
            "registration_analysis_digest": canonical_digest(registration),
            "selected_common_grid": list(rm.SELECTED_GRID),
            "source_support": rm.SUPPORT,
            "source_component_margin_measurement_sd": rm.COMPONENT_MARGIN,
            "photons_per_transport_replicate_per_wavelength": photons_per_replicate,
            "production_measurement_photons_per_wavelength": n_measurement,
        },
        "support_sensitivity": {
            "rule": "retain a wavelength-major component iff its nominal production expected measurement count is at least the predeclared threshold",
            "data_independent_inputs": "committed production reflectance, annular area, and measurement photon budget",
            "rows": support_rows,
            "interpretation": "this can lower cost only by changing the estimand; the old threshold 5 rule retains 66/120 components and still requires 55 replicates from the range term alone",
        },
        "component_scale_sensitivity": {
            "rule": "replace each measurement SD by max(measurement SD, floor_count/(N_measurement*area)) before applying the unchanged 0.5 multiplier",
            "rows": floor_rows,
            "interpretation": "this retains all bins but changes the component equivalence scale; no floor is adopted because no external detector-resolution basis has been supplied",
        },
        "bounded_mean_method_review": {
            "scope": "qualitative method audit only; no substitute interval is implemented or registered",
            "sources": [
                {
                    "method": "empirical Bernstein",
                    "citation": "Maurer and Pontil (2009), Empirical Bernstein Bounds and Sample Variance Penalization",
                    "url": "https://arxiv.org/abs/0907.3740",
                },
                {
                    "method": "Bentkus bounded-sum inequality",
                    "citation": "Bentkus (2004), On Hoeffding's inequalities",
                    "url": "https://arxiv.org/abs/math/0410159",
                },
                {
                    "method": "betting confidence intervals and sequences",
                    "citation": "Waudby-Smith and Ramdas, Estimating means of bounded random variables by betting",
                    "url": "https://arxiv.org/abs/2010.09686",
                },
                {
                    "method": "near-optimal bounded-variable confidence sequences",
                    "citation": "Kuchibhotla and Zheng (2021), Near-Optimal Confidence Sequences for Bounded Random Variables",
                    "url": "https://proceedings.mlr.press/v139/kuchibhotla21a.html",
                },
            ],
            "finding": "the reviewed methods can improve constants and finite-sample width, but they still require bounded support or another justified tail/variance assumption; none is an audited drop-in change that removes the three-orders-of-magnitude resource gap while preserving literal all 120 and the 0.5-SD component estimand",
            "pilot_variance_warning": "using the six grid-pilot replicates to set a formal variance cap would make those trials part of inference and would require a new error allocation and registration; they may not be silently reused",
        },
        "decision": {
            "automatic_replacement_selected": False,
            "freeze_alternative_design": False,
            "run_gate_wiring_authorized": False,
            "candidate3_current_formulation": "retire_as_operationally_infeasible",
            "reason": "support trimming changes which physical bins are judged, a count floor changes the component equivalence scale, and the reviewed bounded-mean methods do not supply an assumption-free orders-of-magnitude rescue",
            "future_reopening_condition": "a new independently justified scientific support contract, detector-resolution scale, or stochastic assumption with its own coverage audit",
        },
    }


def selftest():
    checks = []

    def ck(name, value):
        checks.append((name, bool(value)))
        print(f"  {'PASS' if value else 'FAIL'}  {name}")

    r = np.array([2.0, 1.0])
    s = np.array([0.5, 0.5])
    loose = minimum_replicates(r, s, 100, 0.01)
    tight = minimum_replicates(r, 2.0 * s, 100, 0.01)
    ck("a wider component scale cannot increase the range-only count",
       tight["minimum_replicates"] <= loose["minimum_replicates"])
    ck("the worst component is returned", loose["worst_local_index"] == 0)
    try:
        minimum_replicates(np.array([]), np.array([]), 100, 0.01)
        empty_rejected = False
    except ValueError:
        empty_rejected = True
    ck("empty support is rejected", empty_rejected)
    artifact = analyze()
    support = artifact["support_sensitivity"]["rows"]
    floors = artifact["component_scale_sensitivity"]["rows"]
    ck("zero-threshold and zero-floor reproduce 10803 replicates",
       support[0]["minimum_replicates_range_term_only"] == 10803
       and floors[0]["minimum_replicates_range_term_only"] == 10803)
    row5 = next(x for x in support
                if x["nominal_expected_measurement_count_threshold"] == 5.0)
    ck("the old count-five support retains 66 components and needs 55 replicates",
       row5["retained_components"] == 66
       and row5["minimum_replicates_range_term_only"] == 55)
    ck("support sensitivity is monotone over the audited thresholds",
       all(a["retained_components"] >= b["retained_components"]
           and a["minimum_replicates_range_term_only"] >=
           b["minimum_replicates_range_term_only"]
           for a, b in zip(support, support[1:])))
    ck("the audit selects no replacement, verdict, coverage, or gate wiring",
       artifact["decision"]["automatic_replacement_selected"] is False
       and artifact["candidate3_verdict"] is None
       and artifact["coverage_claimed"] is False
       and artifact["decision"]["run_gate_wiring_authorized"] is False)
    failed = [name for name, ok in checks if not ok]
    print(f"  {len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--selftest", action="store_true")
    group.add_argument("--run", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.selftest:
        if args.out:
            ap.error("--out is valid only with --run")
        return selftest()
    if not args.out:
        ap.error("--run requires --out")
    out = os.path.abspath(args.out)
    if os.path.exists(out):
        raise SystemExit(f"REFUSING to overwrite existing analysis artifact: {out}")
    artifact = analyze()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "x") as fh:
        json.dump(artifact, fh, indent=2, sort_keys=True)
        fh.write("\n")
    row5 = next(x for x in artifact["support_sensitivity"]["rows"]
                if x["nominal_expected_measurement_count_threshold"] == 5.0)
    print(json.dumps({
        "out": out,
        "old_count_five_retained_components": row5["retained_components"],
        "old_count_five_range_only_replicates": row5[
            "minimum_replicates_range_term_only"],
        "automatic_replacement_selected": False,
        "candidate3_verdict": None,
        "coverage_claimed": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
