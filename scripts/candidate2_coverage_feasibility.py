#!/usr/bin/env python3
"""Read-only necessary-condition audit for Candidate 2 coverage.

The committed Candidate 2 diagnostic contains a Bernoulli rare-event counterexample: six zero
observations make the rejected centered-sign-flip heuristic report zero uncertainty.  This program
turns that counterexample into an analytic lower bound that applies to any non-randomized one-sided
upper confidence bound over the Bernoulli class.

If n Bernoulli observations are all zero, a level-(1-alpha) upper bound U(0) must satisfy

    U(0) >= 1 - alpha**(1/n).

Otherwise choose p just above U(0): the all-zero event alone has probability greater than alpha
and causes undercoverage.  Equivalently, no valid equivalence rule can place the upper endpoint
strictly below a margin m after an all-zero sample unless

    n >= log(alpha) / log(1-m).

This is a necessary condition for the rare-event class, not a sufficient Candidate 2 design and
not a universal lower bound under stronger registered distributional assumptions.  The program
runs no transport or coverage simulation and cannot register a criterion or emit a verdict.

    <python> -W error scripts/candidate2_coverage_feasibility.py --selftest
    <python> -W error scripts/candidate2_coverage_feasibility.py --run --out PATH
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SOURCE = os.path.join(ROOT, "results", "nuisance_interval_design_diagnostic.json")
STATUS = "CANDIDATE 2 COVERAGE FEASIBILITY AUDIT; NOT A VERDICT"

ILLUSTRATIVE_MARGIN = 0.05
ALPHA_SCENARIOS = (
    ("single_one_sided_statement", 0.05),
    ("equal_split_one_of_two_legs", 0.025),
    ("candidate3_component_reference_only", 0.003 / 120.0),
)
MARGIN_GRID = (0.10, 0.05, 0.02, 0.01)


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def canonical_digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def all_zero_upper(n_replicates, alpha):
    if isinstance(n_replicates, bool):
        raise ValueError("n_replicates must be a positive integer")
    n = int(n_replicates)
    if n != n_replicates or n <= 0:
        raise ValueError("n_replicates must be a positive integer")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must lie strictly between zero and one")
    return 1.0 - float(alpha) ** (1.0 / n)


def minimum_replicates_for_all_zero_upper(margin, alpha):
    m = float(margin)
    a = float(alpha)
    if not (0.0 < m < 1.0) or not (0.0 < a < 1.0):
        raise ValueError("margin and alpha must lie strictly between zero and one")
    # Strict equivalence requires upper < margin.  This integer is the first n for which the
    # all-zero event has probability <= alpha at truth=margin; exact equality is still a boundary
    # case and must not be called equivalent.
    n = int(math.ceil(math.log(a) / math.log1p(-m)))
    while (1.0 - m) ** n > a:
        n += 1
    return n


def analyze():
    source = load_json(SOURCE)
    ce = source.get("8b_bernoulli_rare_event_counterexample") or {}
    if int(ce.get("n_replicates", -1)) != 6:
        raise ValueError("the committed counterexample no longer has six replicates")
    if float(ce.get("bernoulli_p", -1.0)) != ILLUSTRATIVE_MARGIN:
        raise ValueError("the committed counterexample no longer sits at p=0.05")
    if float(ce.get("aggregate_margin", -1.0)) != ILLUSTRATIVE_MARGIN:
        raise ValueError("the committed counterexample no longer uses margin=0.05")
    theoretical = (1.0 - ILLUSTRATIVE_MARGIN) ** 6
    if abs(float(ce.get("theoretical_all_zero_probability", -1.0)) - theoretical) > 1e-15:
        raise ValueError("the committed counterexample probability is inconsistent")
    if source.get("interval_criterion_registration", {}).get("registered") is not False:
        raise ValueError("Candidate 2 unexpectedly has a registered interval criterion")

    cost = source["7_candidate_comparison"]["measured_pilot_cost"]
    hours_per_replicate = float(cost["total_hours_at_four_steps_six_replicates"]) / 6.0
    scenarios = []
    for name, alpha in ALPHA_SCENARIOS:
        rows = []
        for margin in MARGIN_GRID:
            n = minimum_replicates_for_all_zero_upper(margin, alpha)
            rows.append({
                "margin": margin,
                "minimum_replicates_necessary": n,
                "all_zero_upper_at_minimum": all_zero_upper(n, alpha),
                "descriptive_gpu_hours_at_measured_pilot_rate": n * hours_per_replicate,
            })
        scenarios.append({
            "name": name,
            "alpha": alpha,
            "all_zero_upper_at_six_replicates": all_zero_upper(6, alpha),
            "rows": rows,
            "registration_status": ("reference_only_not_a_candidate2_allocation"
                                    if "reference_only" in name else "illustrative_not_frozen"),
        })

    return {
        "schema_version": "candidate2-coverage-feasibility-v1",
        "status": STATUS,
        "analysis_only": True,
        "transport_run": False,
        "coverage_simulation_run": False,
        "candidate2_verdict": None,
        "coverage_claimed": False,
        "criterion_registered": False,
        "inputs": {
            "source_artifact_digest": canonical_digest(source),
            "counterexample_replicates": 6,
            "counterexample_truth_and_illustrative_margin": ILLUSTRATIVE_MARGIN,
            "counterexample_theoretical_all_zero_probability": theoretical,
            "counterexample_measured_false_equivalence_rate": float(
                ce["false_equivalence_rate"]),
            "measured_pilot_gpu_hours_for_six_replicates": float(
                cost["total_hours_at_four_steps_six_replicates"]),
        },
        "analytic_result": {
            "all_zero_upper_formula": "1 - alpha**(1/n)",
            "necessary_replicate_formula": "ceil(log(alpha)/log(1-margin))",
            "scope": "necessary for a non-randomized one-sided bound uniformly valid over the Bernoulli rare-event class; not sufficient for Candidate 2 and not a bound under a stronger separately registered model",
            "scenarios": scenarios,
        },
        "six_replicate_finding": {
            "single_statement_alpha_0p05_all_zero_upper": all_zero_upper(6, 0.05),
            "equal_split_alpha_0p025_all_zero_upper": all_zero_upper(6, 0.025),
            "illustrative_margin": ILLUSTRATIVE_MARGIN,
            "current_design_can_certify_below_margin_after_all_zero": False,
            "why": "a valid upper endpoint must be far above 0.05 after six all-zero observations; the rejected heuristic's endpoint of zero is the source of its measured undercoverage",
        },
        "decision": {
            "current_six_replicate_candidate2_registerable": False,
            "automatic_replacement_selected": False,
            "authorizes_transport": False,
            "authorizes_48_draw_study": False,
            "required_next_action": "choose and justify Candidate 2 margins and a stochastic assumption or increase the frozen replicate design, then implement and independently validate a bound before registration",
            "do_not_interpret_as": [
                "a Candidate 2 verdict",
                "a sufficient sample-size calculation",
                "a registered alpha split",
                "permission to reuse pilot trials",
            ],
        },
    }


def selftest():
    checks = []

    def ck(name, value):
        checks.append((name, bool(value)))
        print(f"  {'PASS' if value else 'FAIL'}  {name}")

    ck("the committed six-replicate one-sided upper endpoint is exact",
       abs(all_zero_upper(6, 0.05) - 0.39303776899708276) < 1e-15)
    ck("the 5-percent margin needs at least 59 replicates at alpha 0.05",
       minimum_replicates_for_all_zero_upper(0.05, 0.05) == 59)
    ck("splitting alpha to 0.025 raises that necessary count to 72",
       minimum_replicates_for_all_zero_upper(0.05, 0.025) == 72)
    for bad in (0, -1, 2.5, True):
        try:
            all_zero_upper(bad, 0.05)
            rejected = False
        except ValueError:
            rejected = True
        ck(f"invalid replicate count {bad!r} is rejected", rejected)
    artifact = analyze()
    ck("the committed counterexample is hash-bound and reproduced",
       artifact["inputs"]["source_artifact_digest"] ==
       "78bf19e403b97d00fb71786c6a8f54665d50bd7a89b0068d85e3aca0b56d86e9"
       and abs(artifact["inputs"]["counterexample_theoretical_all_zero_probability"]
               - 0.7350918906249998) < 1e-15)
    ck("the audit registers nothing and authorizes no run",
       artifact["criterion_registered"] is False
       and artifact["candidate2_verdict"] is None
       and artifact["coverage_claimed"] is False
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
        "single_statement_all_zero_upper_at_six": artifact[
            "six_replicate_finding"]["single_statement_alpha_0p05_all_zero_upper"],
        "minimum_replicates_at_margin_0p05_alpha_0p05": artifact[
            "analytic_result"]["scenarios"][0]["rows"][1][
                "minimum_replicates_necessary"],
        "criterion_registered": False,
        "candidate2_verdict": None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
