#!/usr/bin/env python3
"""Read-only disposition audit for the current Candidate 1 estimator route.

Candidate 1 is the componentwise relative interval used by the historical nominal feasibility
gate.  It assumes approximate normality of each replicate value and divides each component by its
own derivative magnitude.  This audit binds the committed diagnostic, gate design and nuisance
spec, reproduces the low-support counts, and records the existing offline reanalysis.

It runs no transport or coverage simulation, changes no support mask, registers no interval, and
cannot select an estimator route or authorize the 48-draw study.

    <python> -W error scripts/candidate1_route_audit.py --selftest
    <python> -W error scripts/candidate1_route_audit.py --run --out PATH
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DIAGNOSTIC = os.path.join(ROOT, "results", "nuisance_interval_design_diagnostic.json")
GATE_DESIGN = os.path.join(ROOT, "results", "nuisance_feasibility_gate_design.json")
BLOCKED_RUN = os.path.join(ROOT, "results", "nuisance_feasibility_gate_BLOCKED.json")
NUISANCE_SPEC = os.path.join(ROOT, "spec", "nuisance_spec.json")
PRODUCTION_POOL = os.path.join(ROOT, "data", "derived", "production_pooled.npz")

STATUS = "CANDIDATE 1 ROUTE AUDIT; NOT AN ESTIMATOR VERDICT"
EXPECTED_CANONICAL_DIGESTS = {
    "diagnostic": "78bf19e403b97d00fb71786c6a8f54665d50bd7a89b0068d85e3aca0b56d86e9",
    "gate_design": "2212f9696b63f13d58ab2bf8132efc2b8b505d9cdcc6d856af1e33870a772162",
    "nuisance_spec": "b46740e7e0a03b8d0556538f5918b5fb14e7f43c256eb399b779c3eb02b1ab1f",
}


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def canonical_digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _offline_bounds(text):
    names = ("log_d_epidermis", "log_d_papillary", "log_d_reticular",
             "log_mus_scale", "mus_slope")
    out = {}
    for name in names:
        match = re.search(rf"{re.escape(name)}\s+([0-9]+(?:\.[0-9]+)?)", text)
        if match is None:
            raise ValueError(f"offline reanalysis is missing the bound for {name}")
        out[name] = float(match.group(1))
    return out


def analyze():
    diagnostic = load_json(DIAGNOSTIC)
    gate_design = load_json(GATE_DESIGN)
    nuisance = load_json(NUISANCE_SPEC)
    blocked = load_json(BLOCKED_RUN)
    digests = {
        "diagnostic": canonical_digest(diagnostic),
        "gate_design": canonical_digest(gate_design),
        "nuisance_spec": canonical_digest(nuisance),
    }
    if digests != EXPECTED_CANONICAL_DIGESTS:
        raise ValueError(f"a bound Candidate 1 input changed: {digests}")
    if file_digest(BLOCKED_RUN) != diagnostic["inputs"]["pilot_artifact"]["sha256"]:
        raise ValueError("the historical gate artifact does not match the diagnostic binding")

    gate = nuisance["nominal_derivative_feasibility_gate"]
    registration = gate["interval_criterion_registration"]
    if registration.get("registered") is not False:
        raise ValueError("the Candidate 1 interval criterion is unexpectedly registered")
    candidate = diagnostic["7_candidate_comparison"]["componentwise_parametric"]
    if "per-component approximate normality" not in " ".join(candidate["assumptions"]):
        raise ValueError("the Candidate 1 coverage assumption changed")
    if blocked.get("gate_passed") is not False or blocked.get(
            "downstream_study_authorized") is not False:
        raise ValueError("the historical run boundary changed")

    run_design = blocked["frozen_run_design"]
    n_replicates = int(run_design["n_replicates"])
    photons = int(run_design["n_photons_per_wavelength"])
    with np.load(PRODUCTION_POOL) as pool:
        reflectance = np.asarray(pool["R"], float)
        areas = np.asarray(pool["areas"], float)
    if reflectance.shape != (120,) or areas.shape != (120,):
        raise ValueError("production support must contain 120 components")
    measurement_n = int(diagnostic["6_effective_counts"]["photon_budget_used"][
        "n_measurement_photons_per_wavelength"])
    measurement_counts = measurement_n * areas * reflectance
    per_replicate_transport_counts = photons * areas * reflectance
    all_six_transport_counts = n_replicates * per_replicate_transport_counts

    effective = diagnostic["6_effective_counts"]
    if int((measurement_counts < 1.0).sum()) != int(effective["n_bins_below_threshold"]):
        raise ValueError("the production low-count diagnostic no longer reproduces")
    historical = gate["historical_run_62960056"]
    offline_text = historical["offline_reanalysis"]
    bounds = _offline_bounds(offline_text)
    margin = float(diagnostic["registered_margin_and_alpha"][
        "equivalence_margin_relative"])
    if "Every lower bound is 0" not in offline_text:
        raise ValueError("the offline lower-bound record changed")

    coverage = gate_design["equivalence_rules"][
        "measured_coverage_by_error_distribution"]
    support = historical["registered_support_mask_in_the_formal_run"]
    return {
        "schema_version": "candidate1-route-audit-v1",
        "status": STATUS,
        "analysis_only": True,
        "transport_run_by_this_audit": False,
        "coverage_simulation_run_by_this_audit": False,
        "estimator_route_selected": False,
        "candidate1_verdict": None,
        "coverage_claimed": False,
        "interval_criterion_registered": False,
        "inputs": {
            "canonical_digests": digests,
            "historical_gate_artifact_sha256": file_digest(BLOCKED_RUN),
            "production_pool_sha256": diagnostic["inputs"]["pooled_artifact"]["sha256"],
        },
        "current_formulation": {
            "statistic": candidate["exact_statistic"],
            "coverage_assumption": gate["equivalence_design"]["coverage_assumption"],
            "coverage_method": candidate["coverage_method"],
            "registered": False,
            "why_not_registered": nuisance["estimator_route_disposition"][
                "candidate1"]["reason"],
        },
        "low_support_audit": {
            "measurement_photons_per_wavelength": measurement_n,
            "transport_photons_per_wavelength_per_replicate": photons,
            "transport_replicates": n_replicates,
            "measurement_components_below_one_expected_count": int(
                (measurement_counts < 1.0).sum()),
            "one_transport_replicate_components_below_one_expected_hit": int(
                (per_replicate_transport_counts < 1.0).sum()),
            "all_six_transport_replicates_components_below_one_expected_hit": int(
                (all_six_transport_counts < 1.0).sum()),
            "minimum_nominal_expected_hits_across_all_six_transport_replicates": float(
                all_six_transport_counts.min()),
            "interpretation": "nominal support remains sparse in the outer bins even after pooling the six historical replicates; this is a support diagnostic, not a proof of any derivative error law",
        },
        "coverage_evidence": {
            "registered_assumption": "per-component approximate normality",
            "measured_simultaneous_coverage": {
                "gaussian": float(coverage["gaussian"]),
                "laplace": float(coverage["laplace"]),
                "t3": float(coverage["t3"]),
                "near_discrete_rademacher_mixture": float(
                    coverage["rademacher_plus_small_normal"]),
            },
            "near_discrete_component_coverage_recorded_in_note": 0.0283,
            "withdrawn_guard": gate["withdrawn_degenerate_spread_rule"],
            "finding": "the only behavioural guard for the near-discrete failure was withdrawn as non-scale-invariant, so the conditional normality assumption is unguarded in the low-support regime",
        },
        "historical_offline_reanalysis": {
            "support": support,
            "relative_margin": margin,
            "simultaneous_upper_bounds_as_margin_multiples": bounds,
            "all_lower_bounds": 0.0,
            "all_columns_outcome": "unresolved",
            "smallest_upper_multiple": min(bounds.values()),
            "largest_upper_multiple": max(bounds.values()),
        },
        "decision": {
            "current_candidate1_route_selected": False,
            "current_candidate1_formulation": "retire_as_unsupported_and_uninformative",
            "authorizes_transport": False,
            "authorizes_48_draw_study": False,
            "reason": "the route relies on an unguarded approximate-normality assumption exactly where nominal support is sparse, is not compatible with the literal full observable without a consequential support change, and all five historical columns remain unresolved even under that assumption",
            "future_reopening_condition": "an independently justified error model or exact low-count construction, a predeclared support contract propagated through downstream Fisher analyses, and a newly frozen validation design",
        },
    }


def selftest():
    checks = []

    def ck(name, value):
        checks.append((name, bool(value)))
        print(f"  {'PASS' if value else 'FAIL'}  {name}")

    parsed = _offline_bounds(
        "log_d_epidermis 51.9, log_d_papillary 205.3, log_d_reticular 10.8, "
        "log_mus_scale 9.3, mus_slope 35.0")
    ck("offline bound parsing is name-stable", parsed["log_mus_scale"] == 9.3
       and parsed["log_d_papillary"] == 205.3)
    artifact = analyze()
    low = artifact["low_support_audit"]
    ck("the committed measurement diagnostic has 43 sub-one-count components",
       low["measurement_components_below_one_expected_count"] == 43)
    ck("even all six transport replicates leave 25 components below one nominal hit",
       low["all_six_transport_replicates_components_below_one_expected_hit"] == 25)
    cov = artifact["coverage_evidence"]
    ck("the near-discrete coverage shortfall and withdrawn guard are explicit",
       cov["measured_simultaneous_coverage"]["near_discrete_rademacher_mixture"] == 0.9283
       and cov["near_discrete_component_coverage_recorded_in_note"] == 0.0283
       and cov["withdrawn_guard"]["behavioural_effect_now"].startswith("none"))
    off = artifact["historical_offline_reanalysis"]
    ck("all five offline columns are unresolved far above the 0.05 margin",
       off["all_columns_outcome"] == "unresolved"
       and len(off["simultaneous_upper_bounds_as_margin_multiples"]) == 5
       and off["smallest_upper_multiple"] == 9.3
       and off["largest_upper_multiple"] == 205.3)
    ck("the audit selects no route, registers nothing, and authorizes no run",
       artifact["estimator_route_selected"] is False
       and artifact["interval_criterion_registered"] is False
       and artifact["candidate1_verdict"] is None
       and artifact["decision"]["authorizes_transport"] is False
       and artifact["decision"]["authorizes_48_draw_study"] is False)
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
    print(json.dumps({
        "out": out,
        "measurement_components_below_one_count": artifact[
            "low_support_audit"]["measurement_components_below_one_expected_count"],
        "all_six_transport_components_below_one_hit": artifact[
            "low_support_audit"][
                "all_six_transport_replicates_components_below_one_expected_hit"],
        "smallest_historical_upper_multiple": artifact[
            "historical_offline_reanalysis"]["smallest_upper_multiple"],
        "route_selected": False,
        "candidate1_verdict": None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
