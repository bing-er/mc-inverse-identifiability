#!/usr/bin/env python3
"""check_radial_pilot.py -- verify the radial pilot artifact, then apply the frozen criteria.

Runs no transport. It answers, in order:

  1. are the raw arrays present and complete for every replicate of every target
  2. does each replicate's recorded array_sha256 recompute from the stored arrays
  3. are the shapes the expected 120 components, with no non-finite value
  4. do the seeds match the frozen sequence
  5. what do the aggregates and the PC-subspace multiplier say, read together

It does NOT decide from the best bin. best_multiplier is a minimum over many components and is
optimistic by construction: the smallest of 120 noisy multipliers is biased low whether or not any
real signal is present. The frozen rule requires the aggregates and the PC subspace, and requires a
cheap radial group to be corroborated by its neighbour.

    python3 scripts/check_radial_pilot.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jacobian_validation import (array_digest, derive_observables, load,  # noqa: E402
                                 resource_verdict, OBSERVABLE_ORDER)

ARRAY_KEYS_2D = ("R_plus_h", "R_minus_h", "difference_quotients_120")
ARRAY_KEYS_1D = ("reference_two_term_120", "reference_three_term_120", "ad_120", "pmc_120")


def verify(payload, vspec):
    rnd = vspec["random_number_design"]
    n_exp = payload["n_replicates"]
    n_h = len(vspec["step_size_rule"]["h_relative_grid"])
    n_obs = (len(vspec["operating_point"]["wavelengths_nm"])
             * vspec["operating_point"]["radial_grid"]["n_bins"])
    checks, fails = [], []

    def ck(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})
        if not ok:
            fails.append(name)

    raw = payload["raw"]
    ck("both targets present", sorted(raw) == sorted(vspec["pilot"]["columns"]), str(sorted(raw)))
    for target, e in raw.items():
        reps = e["raw_per_replicate"]
        ck(f"{target}: all {n_exp} replicates present", len(reps) == n_exp, f"{len(reps)} found")
        seeds_exp = [rnd["seed_base"] + rnd["stride"] * r for r in range(n_exp)]
        ck(f"{target}: seeds match the frozen sequence",
           [r["seed"] for r in reps] == seeds_exp, str([r["seed"] for r in reps]))
        shapes_ok, finite_ok, digest_ok, flag_ok = True, True, True, True
        for r in reps:
            for k in ARRAY_KEYS_2D:
                a = np.asarray(r[k], float)
                shapes_ok &= a.shape == (n_h, n_obs)
                finite_ok &= bool(np.all(np.isfinite(a)))
            for k in ARRAY_KEYS_1D:
                a = np.asarray(r[k], float)
                shapes_ok &= a.shape == (n_obs,)
                finite_ok &= bool(np.all(np.isfinite(a)))
            d = array_digest(r["R_plus_h"], r["R_minus_h"], r["reference_two_term_120"],
                             r["reference_three_term_120"], r["ad_120"], r["pmc_120"])
            digest_ok &= (d == r["array_sha256"])
            flag_ok &= (r["stored_in_full_120_component_radial_coordinates"] is True
                        and r["reduced_before_saving"] is False)
        ck(f"{target}: every stored array has the expected shape", shapes_ok,
           f"expected ({n_h}, {n_obs}) and ({n_obs},)")
        ck(f"{target}: no non-finite value in any stored array", finite_ok)
        ck(f"{target}: array_sha256 recomputes for every replicate", digest_ok)
        ck(f"{target}: arrays are flagged as unreduced full radial output", flag_ok)
        # the difference quotients must follow from the stored plus/minus vectors
        q_ok = True
        for r in reps:
            Rp, Rm = np.asarray(r["R_plus_h"], float), np.asarray(r["R_minus_h"], float)
            hs = np.asarray(r["h_absolute"], float)
            q_ok &= np.allclose((Rp - Rm) / (2.0 * hs)[:, None],
                                np.asarray(r["difference_quotients_120"], float),
                                rtol=0, atol=0)
        ck(f"{target}: difference quotients follow exactly from the stored plus/minus vectors",
           q_ok)
    return checks, fails


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", default=os.path.join(
        ROOT, "results", "jacobian_validation_pilot_radial.json"))
    ap.add_argument("--write-back", action="store_true",
                    help="rewrite the artifact with the renamed field and the resource verdict")
    a = ap.parse_args()
    vspec = load(os.path.join(ROOT, "spec", "validation_spec.json"))
    payload = load(a.pilot)

    checks, fails = verify(payload, vspec)
    for c in checks:
        print(f"  {'PASS' if c['pass'] else 'FAIL'}  {c['check']}"
              + (f"  [{c['detail']}]" if c["detail"] else ""))
    print(f"  {sum(c['pass'] for c in checks)}/{len(checks)} artifact checks passed")
    if fails:
        print(f"  FAILED: {fails}\n  the artifact is not trustworthy; no verdict is reported")
        return 1

    derived = derive_observables(payload["raw"], vspec)
    verdict = resource_verdict(derived, vspec)
    key = "budget_multiplier_for_reference_resolution_at_ad_pmc_gap_scale"

    print(f"\n  resource criteria frozen before results were seen: "
          f"{verdict['criteria_frozen_before_results_were_seen']}")
    for t, e in derived.items():
        print(f"\n  ===== {t} =====")
        print(f"  {'observable':<30}{'adm':>9}{'best':>12}{'at':>14}{'PC-subspace':>14}")
        for nm in OBSERVABLE_ORDER:
            i = e["by_observable"][nm]
            b, ad_ = i[key], i["component_admissibility"]
            print(f"  {nm:<30}{ad_['n_admissible']:>4}/{ad_['n_components']:<4}"
                  f"{str(b['best_multiplier']):>12}{str(b['best_component_label']):>14}"
                  f"{str(b['pc_subspace_multiplier']):>14}")
        v = verdict["per_target"][t]
        adj = v["adjacency"]
        print(f"  adjacency: {len(adj.get('cheap_groups', []))} cheap group(s), "
              f"corroborated={adj.get('any_corroborated')}")
        for r in adj.get("cheap_groups", []):
            print(f"    {r['label']} at {r['multiplier']}, neighbours "
                  f"{r['neighbour_multipliers']}, corroborated={r['corroborated_by_a_neighbour']}")
        print(f"  BAND: {v['band']}  -> {v['action']}")
        print(f"  reason: {v['reason']}")

    if a.write_back:
        payload["resource_verdict"] = verdict
        payload["targets"] = derived
        payload["field_rename_applied"] = {
            "from": "budget_multiplier_to_resolve_ad_vs_pmc",
            "to": key,
            "why": "AD and pMC are already separated at this budget; the quantity estimates the "
                   "budget needed for the REFERENCE to resolve a discrepancy the size of the "
                   "AD-minus-pMC gap",
            "transport_rerun": False}
        payload["artifact_checks"] = checks
        with open(a.pilot, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\n  rewrote {os.path.relpath(a.pilot, ROOT)} with the renamed field and the "
              f"verdict; no transport was run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
