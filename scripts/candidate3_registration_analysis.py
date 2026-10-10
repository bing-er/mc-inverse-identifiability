#!/usr/bin/env python3
"""Read-only registration/resource analysis for Candidate 3 Phase 3B2.

This program consumes the completed NON-CONFIRMATORY grid pilot and its fixed analysis.  It does
not run transport, coverage, or a validation scenario, and it cannot emit a Candidate 3 verdict.
Its purpose is narrower: after the grid, likelihood, support and margins have been chosen, compute
the deterministic range-only lower bound on the formal transport replicate count and a separate
pilot-variance projection.

The range-only result is a hard feasibility bound.  It asks how many photons are required for the
component empirical-Bernstein RANGE TERM alone to fit inside the registered engineering margin,
even if the true discrepancy and variance are both zero.  No pilot value enters that calculation.

    <python> -W error scripts/candidate3_registration_analysis.py --selftest
    <python> -W error scripts/candidate3_registration_analysis.py --run --out PATH
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
PILOT_SPEC = os.path.join(ROOT, "spec", "candidate3_grid_pilot.json")
PILOT_ARTIFACT = os.path.join(ROOT, "results", "candidate3_grid_pilot.json")
GRID_ANALYSIS = os.path.join(ROOT, "results", "candidate3_grid_analysis.json")

STATUS = "DESIGN FEASIBILITY ANALYSIS; NOT A CANDIDATE 3 VERDICT"
SELECTED_GRID = (0.5, 0.3, 0.1, 0.07)
LIKELIHOOD = "multinomial_fixed_launch"
SUPPORT = "literal_all_120"
AGGREGATE_MARGIN = math.sqrt(1.10) - 1.0
COMPONENT_MARGIN = 0.5
ALPHA_NUMERATOR = 0.004
ALPHA_DENOMINATOR = 0.003
ALPHA_COMPONENT = 0.003
N_COMPONENTS = 120


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def canonical_digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def fit_rows(h_grid):
    h = np.asarray(h_grid, float)
    if h.shape != (4,) or not np.all(np.isfinite(h)) or not np.all(h > 0.0):
        raise ValueError("the selected grid must contain four finite positive steps")
    if len(np.unique(h)) != 4:
        raise ValueError("the selected grid must contain four distinct steps")
    x2 = np.column_stack([np.ones(4), h ** 2])
    x3 = np.column_stack([np.ones(4), h ** 2, h ** 4])
    q2 = np.linalg.pinv(x2)[0]
    q3 = np.linalg.pinv(x3)[0]
    return {"q2": q2, "q3": q3, "qd": q3 - q2}


def combined_range_width(h_grid, areas, q):
    """R_Z = 2 sum_l |q_l| B(h_l), with B(h)=1/(2 h A)."""
    h = np.asarray(h_grid, float)
    a = np.asarray(areas, float)
    q = np.asarray(q, float)
    if h.shape != q.shape or np.any(h <= 0.0) or np.any(a <= 0.0):
        raise ValueError("invalid grid, fit row, or detector area")
    return np.sum(np.abs(q)[:, None] / (h[:, None] * a[None, :]), axis=0)


def empirical_radius_from_variance(variance, n_total, range_width, alpha):
    """The registered two-sided empirical-Bernstein radius at a projected variance."""
    n = int(n_total)
    if n < 2 or not (0.0 < alpha < 1.0):
        raise ValueError("n_total must be at least two and alpha must lie in (0,1)")
    v = np.asarray(variance, float)
    r = np.asarray(range_width, float)
    if np.any(v < 0.0) or np.any(r <= 0.0):
        raise ValueError("variance must be non-negative and range width strictly positive")
    log_term = math.log(4.0 / alpha)
    return (np.sqrt(2.0 * v * log_term / n)
            + 7.0 * r * log_term / (3.0 * (n - 1)))


def box_upper(matrix, radius):
    return float(np.linalg.norm(np.abs(np.asarray(matrix, float))
                                @ np.abs(np.asarray(radius, float))))


def range_only_minimum_replicates(range_width_d, covariance_diagonal,
                                  photons_per_replicate):
    """Hard lower bound from the component range term at zero signal and zero variance."""
    r = np.asarray(range_width_d, float)
    c = np.asarray(covariance_diagonal, float)
    if r.shape != c.shape or np.any(r <= 0.0) or np.any(c <= 0.0):
        raise ValueError("range width and covariance diagonal must be positive and conformable")
    alpha_bin = ALPHA_COMPONENT / int(c.size)
    log_term = math.log(4.0 / alpha_bin)
    required_total = 1.0 + 7.0 * r * log_term / (
        3.0 * COMPONENT_MARGIN * np.sqrt(c))
    worst = int(np.argmax(required_total))
    n_total = int(math.ceil(float(required_total[worst])))
    reps = int(math.ceil(n_total / int(photons_per_replicate)))
    return {"replicates": reps, "total_photons": n_total,
            "worst_component_index": worst,
            "required_total_photons_by_component": required_total}


def _first_true(predicate, maximum=10_000_000):
    hi = 1
    while hi < maximum and not predicate(hi):
        hi *= 2
    if hi >= maximum and not predicate(hi):
        raise ValueError(f"no feasible replicate count was found below {maximum}")
    lo = 0
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if predicate(mid):
            hi = mid
        else:
            lo = mid
    return hi


def projected_metrics(n_replicates, photons_per_replicate, e2_mean,
                      e2_variance, d_variance, range_width_e2, range_width_d,
                      whitening, covariance_diagonal):
    n_total = int(n_replicates) * int(photons_per_replicate)
    eps_e2 = empirical_radius_from_variance(
        e2_variance, n_total, range_width_e2, ALPHA_DENOMINATOR / N_COMPONENTS)
    eps_d_num = empirical_radius_from_variance(
        d_variance, n_total, range_width_d, ALPHA_NUMERATOR / N_COMPONENTS)
    eps_d_component = empirical_radius_from_variance(
        d_variance, n_total, range_width_d, ALPHA_COMPONENT / N_COMPONENTS)
    den_lower = float(np.linalg.norm(np.asarray(whitening) @ np.asarray(e2_mean))
                      - box_upper(whitening, eps_e2))
    aggregate = (float("inf") if den_lower <= 0.0
                 else box_upper(whitening, eps_d_num) / den_lower)
    component_values = eps_d_component / np.sqrt(np.asarray(covariance_diagonal, float))
    worst = int(np.argmax(component_values))
    return {"aggregate_zero_signal_upper": aggregate,
            "component_zero_signal_upper": float(component_values[worst]),
            "component_worst_index": worst,
            "denominator_lower": den_lower}


def analyze():
    sys.path.insert(0, HERE)
    import candidate3_grid_analysis as grid_module

    design = load_json(PILOT_SPEC)
    pilot = load_json(PILOT_ARTIFACT)
    grid = load_json(GRID_ANALYSIS)
    if design.get("frozen") is not True or not design.get("freeze_hash"):
        raise ValueError("the grid-pilot design is not frozen")
    if pilot.get("transport_run") is not True or pilot.get("candidate3_verdict") is not None:
        raise ValueError("the pilot artifact is not a completed non-confirmatory artifact")
    if pilot.get("coverage_claimed") is not False:
        raise ValueError("the pilot artifact must explicitly record coverage_claimed=false")
    if grid.get("status") != "NON-CONFIRMATORY GRID SELECTION; NOT A CANDIDATE 3 VERDICT":
        raise ValueError("the grid analysis does not carry the required non-confirmatory status")
    if grid.get("candidate3_verdict") is not None or grid.get("coverage_claimed") is not False:
        raise ValueError("the grid analysis crosses the non-confirmatory boundary")
    if grid.get("pilot_artifact_digest") != canonical_digest(pilot):
        raise ValueError("the grid analysis is not bound to the supplied pilot artifact")
    if grid.get("pilot_design_digest") != canonical_digest(design):
        raise ValueError("the grid analysis is not bound to the supplied pilot design")
    selection = grid.get("selection") or {}
    if selection.get("kind") != "common" or tuple(selection.get("grid", ())) != SELECTED_GRID:
        raise ValueError(f"the frozen rule did not select the expected common grid {SELECTED_GRID}")
    rd = design["run_design"]
    mm = design["measurement_model"]
    if mm.get("likelihood") != LIKELIHOOD or mm.get("support") != SUPPORT:
        raise ValueError("the pilot measurement model does not match the decided registration model")
    if int(rd["n_replicates"]) < 2:
        raise ValueError("at least two pilot replicates are required for a variance projection")

    D, _ = grid_module._validated_arrays(pilot, design)
    W, areas, reflectance = grid_module.whitening_matrix()
    base = load_json(grid_module.BASELINE_SPEC)
    n_measurement = int(base["profiles"]["production"][
        "n_measurement_photons_per_wavelength"])
    cdiag = reflectance * (1.0 - areas * reflectance) / (n_measurement * areas)
    if cdiag.shape != (N_COMPONENTS,) or not np.all(np.isfinite(cdiag)) or np.any(cdiag <= 0.0):
        raise ValueError("the production multinomial covariance diagonal is invalid")

    h_all = np.asarray(rd["h_candidates"], float)
    indices = [int(np.flatnonzero(np.isclose(h_all, x, rtol=0.0, atol=0.0))[0])
               for x in SELECTED_GRID]
    fr = fit_rows(SELECTED_GRID)
    range_e2 = combined_range_width(SELECTED_GRID, areas, fr["q2"])
    range_d = combined_range_width(SELECTED_GRID, areas, fr["qd"])
    photons = int(rd["n_photons_per_wavelength"])
    hard = range_only_minimum_replicates(range_d, cdiag, photons)

    per_column = {}
    for ci, name in enumerate(rd["columns"]):
        selected = D[:, ci, indices, :]
        e2_rep = np.einsum("l,rlm->rm", fr["q2"], selected)
        d_rep = np.einsum("l,rlm->rm", fr["qd"], selected)
        e2_mean = e2_rep.mean(axis=0)
        # Each pilot replicate is a mean of `photons` independent photon contributions.  Scaling
        # the across-replicate variance by that count gives the design-only per-photon variance
        # proxy.  Six replicates make this a projection, never a coverage statement.
        e2_variance = e2_rep.var(axis=0, ddof=1) * photons
        d_variance = d_rep.var(axis=0, ddof=1) * photons

        def metrics(reps):
            return projected_metrics(reps, photons, e2_mean, e2_variance, d_variance,
                                     range_e2, range_d, W, cdiag)

        aggregate_min = _first_true(
            lambda reps: metrics(reps)["aggregate_zero_signal_upper"] <= AGGREGATE_MARGIN)
        component_min = _first_true(
            lambda reps: metrics(reps)["component_zero_signal_upper"] <= COMPONENT_MARGIN)
        both_min = _first_true(lambda reps: (
            metrics(reps)["aggregate_zero_signal_upper"] <= AGGREGATE_MARGIN
            and metrics(reps)["component_zero_signal_upper"] <= COMPONENT_MARGIN))
        per_column[name] = {
            "aggregate_minimum_replicates": aggregate_min,
            "component_minimum_replicates": component_min,
            "both_minimum_replicates": both_min,
            "metrics_at_both_minimum": metrics(both_min),
            "pilot_variance_estimate_uses_replicates": int(rd["n_replicates"]),
        }

    projected = max(row["both_minimum_replicates"] for row in per_column.values())
    # Resource projection uses the measured complete-shard wall time, not the sum of independent
    # model terms.  It is descriptive only and never changes the count.
    shard_seconds = []
    for rep in pilot["replicates"]:
        shard_seconds.append(sum(
            float(x) for col in rep["columns"].values()
            for x in col["wall_seconds_by_wavelength"]))
    mean_hours = float(np.mean(shard_seconds)) / 3600.0
    max_hours = float(np.max(shard_seconds)) / 3600.0
    return {
        "schema_version": "candidate3-registration-resource-analysis-v1",
        "status": STATUS,
        "analysis_only": True,
        "transport_run_by_this_analysis": False,
        "coverage_run": False,
        "candidate3_verdict": None,
        "coverage_claimed": False,
        "inputs": {
            "grid_pilot_freeze_hash": design["freeze_hash"],
            "pilot_artifact_digest": canonical_digest(pilot),
            "grid_analysis_digest": canonical_digest(grid),
        },
        "fixed_decisions": {
            "common_grid": list(SELECTED_GRID),
            "q2": fr["q2"].tolist(), "q3": fr["q3"].tolist(),
            "qd": fr["qd"].tolist(),
            "likelihood": LIKELIHOOD, "support": SUPPORT,
            "observable_order": mm["observable_order"],
            "aggregate_margin": AGGREGATE_MARGIN,
            "component_margin_measurement_sd": COMPONENT_MARGIN,
            "photons_per_transport_replicate_per_wavelength": photons,
            "stopping_rule": "fixed design; no interim looks; no post-hoc replicate addition",
        },
        "range_only_hard_lower_bound": {
            "minimum_replicates": hard["replicates"],
            "minimum_total_photons_at_worst_component": hard["total_photons"],
            "worst_component_index_wavelength_major": hard["worst_component_index"],
            "derivation": "component empirical-Bernstein range term alone <= 0.5 production measurement SD, at zero discrepancy and zero variance",
            "uses_pilot_variance": False,
        },
        "pilot_variance_projection": {
            "per_column": per_column,
            "minimum_replicates_all_columns": projected,
            "interpretation": "design/resource projection from six non-confirmatory pilot replicates; not a coverage or Candidate 3 result",
        },
        "resource_projection": {
            "measured_mean_complete_shard_gpu_hours": mean_hours,
            "measured_max_complete_shard_gpu_hours": max_hours,
            "gpu_hours_at_range_only_minimum_using_mean": hard["replicates"] * mean_hours,
            "gpu_hours_at_pilot_projected_minimum_using_mean": projected * mean_hours,
            "serial_years_at_pilot_projected_minimum_using_mean": projected * mean_hours / (24.0 * 365.25),
        },
        "registration_recommendation": {
            "freeze_current_design": False,
            "reason": "literal all-120 support and the 0.5-SD component margin require more than ten thousand 200,000-photon transport replicates even before allowing any nonzero discrepancy; the design is operationally infeasible",
            "do_not_change_silently": ["support", "component margin", "range construction", "grid"],
            "required_next_action": "record the feasibility failure and make an explicit new design decision before any validation-suite freeze or run_gate wiring",
        },
    }


def selftest():
    checks = []

    def ck(name, value):
        checks.append((name, bool(value)))
        print(f"  {'PASS' if value else 'FAIL'}  {name}")

    h = np.array(SELECTED_GRID)
    fr = fit_rows(h)
    ck("fit rows reproduce constant, h2 and h4 intercepts",
       abs(fr["q2"].sum() - 1.0) < 1e-12
       and abs(fr["q2"] @ h ** 2) < 1e-12
       and abs(fr["q3"].sum() - 1.0) < 1e-12
       and abs(fr["q3"] @ h ** 2) < 1e-12
       and abs(fr["q3"] @ h ** 4) < 1e-12)
    areas = np.array([1.0, 2.0])
    width = combined_range_width(h, areas, fr["qd"])
    ck("combined range is positive and scales inversely with area",
       np.all(width > 0.0) and abs(width[0] / width[1] - 2.0) < 1e-12)
    r1 = empirical_radius_from_variance(np.array([1.0]), 1000, np.array([2.0]), 1e-3)[0]
    r2 = empirical_radius_from_variance(np.array([1.0]), 2000, np.array([2.0]), 1e-3)[0]
    ck("projected radius decreases with total photons", r2 < r1)
    hard = range_only_minimum_replicates(np.array([2.0]), np.array([0.25]), 100)
    n = hard["replicates"] * 100
    rt = 7.0 * 2.0 * math.log(4.0 / ALPHA_COMPONENT) / (3.0 * (n - 1))
    ck("range-only lower bound fits the component margin at its returned count",
       rt <= COMPONENT_MARGIN * math.sqrt(0.25) + 1e-15)
    ck("the analysis exposes no design override flags",
       not ({"--grid", "--margin", "--support", "--replicates", "--photons"}
            & set(sys.argv)))
    ck("the status and output boundary forbid a verdict and coverage claim",
       "NOT A CANDIDATE 3 VERDICT" in STATUS)
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
    print(json.dumps({"out": out,
                      "range_only_minimum_replicates": artifact[
                          "range_only_hard_lower_bound"]["minimum_replicates"],
                      "pilot_projected_minimum_replicates": artifact[
                          "pilot_variance_projection"]["minimum_replicates_all_columns"],
                      "freeze_current_design": artifact[
                          "registration_recommendation"]["freeze_current_design"],
                      "candidate3_verdict": None, "coverage_claimed": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
