#!/usr/bin/env python3
"""Read-only analysis for the non-confirmatory Candidate 3 grid pilot.

This module never imports JAX, never runs transport and never produces a Candidate 3 verdict. It
validates a completed pilot artifact against spec/candidate3_grid_pilot.json, evaluates every
pre-declared four-point subset and applies the frozen common-versus-per-column selection rule.

    <python> -W error scripts/candidate3_grid_analysis.py --selftest
    <python> -W error scripts/candidate3_grid_analysis.py --artifact PATH --out PATH
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PILOT_SPEC = os.path.join(ROOT, "spec", "candidate3_grid_pilot.json")
BASELINE_SPEC = os.path.join(ROOT, "spec", "baseline_spec.json")
PRODUCTION_POOL = os.path.join(ROOT, "data", "derived", "production_pooled.npz")


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def canonical_digest(value):
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=False).encode("utf-8"))


def fit_rows(h_grid):
    """Return the deterministic intercept rows q2, q3 and qd for a four-point grid."""
    h = np.asarray(h_grid, float)
    if h.shape != (4,) or not np.all(np.isfinite(h)) or not np.all(h > 0):
        raise ValueError(f"a candidate grid must contain four finite positive steps; got {h}")
    if len(np.unique(h)) != 4:
        raise ValueError("a candidate grid must contain four distinct steps")
    x2 = np.column_stack([np.ones(4), h ** 2])
    x3 = np.column_stack([np.ones(4), h ** 2, h ** 4])
    q2 = np.linalg.pinv(x2)[0]
    q3 = np.linalg.pinv(x3)[0]
    return {"q2": q2, "q3": q3, "qd": q3 - q2}


def candidate_subsets(design):
    h = tuple(float(x) for x in design["run_design"]["h_candidates"])
    k = int(design["selection_rule"]["subset_size"])
    hi = float(design["selection_rule"]["must_include_h_at_least"])
    lo = float(design["selection_rule"]["must_include_h_at_most"])
    out = [tuple(c) for c in itertools.combinations(h, k)
           if max(c) >= hi and min(c) <= lo]
    if not out:
        raise ValueError("the frozen subset constraints admit no candidate grid")
    return tuple(out)


def whitening_matrix():
    """The committed production multinomial whitening in wavelength-major order."""
    import sys
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import unify

    bs = load_json(BASELINE_SPEC)
    p = bs["profiles"]["production"]
    n_wl = len(bs["wavelengths_nm"])
    n_bins = int(p["radial"]["n_bins"])
    edges = np.linspace(p["radial"]["r_min_cm"], p["radial"]["r_max_cm"], n_bins + 1)
    areas_one = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    areas = np.tile(areas_one, n_wl)
    with np.load(PRODUCTION_POOL) as z:
        R = np.asarray(z["R"], float)
    eye = np.eye(R.size)
    W = unify.whiten_multinomial(
        eye, R, areas, p["n_measurement_photons_per_wavelength"],
        np.ones(R.size), n_wl)
    if W.shape != (120, 120) or not np.all(np.isfinite(W)):
        raise ValueError(f"production whitening has invalid shape or values: {W.shape}")
    return W, areas, R


def _validated_arrays(artifact, design, require_frozen=True):
    rd = design["run_design"]
    h = np.asarray(rd["h_candidates"], float)
    Rn = int(rd["n_replicates"])
    Cn = len(rd["columns"])
    Wn = len(rd["wavelengths_nm"])
    Ln = h.size
    if artifact.get("schema_version") != "candidate3-grid-pilot-artifact-v1":
        raise ValueError("artifact schema_version is not candidate3-grid-pilot-artifact-v1")
    if require_frozen:
        if design.get("frozen") is not True or not design.get("freeze_hash"):
            raise ValueError("pilot design is not frozen; a real artifact may not be analyzed")
        if artifact.get("freeze_hash") != design["freeze_hash"]:
            raise ValueError("artifact freeze_hash does not match the frozen pilot design")
        if artifact.get("normative_input_hashes") != design.get("normative_input_hashes"):
            raise ValueError("artifact normative_input_hashes do not match the frozen design")
    if artifact.get("run_design_digest") != canonical_digest(rd):
        raise ValueError("artifact run_design_digest does not match the frozen run design")
    if artifact.get("transport_run") is not True or artifact.get("candidate3_verdict") is not None:
        raise ValueError("artifact must record transport_run=true and candidate3_verdict=null")
    reps = artifact.get("replicates") or []
    if len(reps) != Rn:
        raise ValueError(f"artifact has {len(reps)} replicates, expected {Rn}")
    D = np.empty((Rn, Cn, Ln, Wn * 30), float)
    V = np.empty_like(D)
    for r, (rep, seed) in enumerate(zip(reps, rd["replicate_seeds"])):
        if rep.get("replicate") != r or rep.get("seed") != seed:
            raise ValueError(f"replicate {r} index/seed mismatch")
        if not rep.get("checks", {}).get("all_passed"):
            raise ValueError(f"replicate {r} did not pass all structural checks")
        for c, name in enumerate(rd["columns"]):
            col = (rep.get("columns") or {}).get(name)
            if col is None:
                raise ValueError(f"replicate {r} is missing column {name}")
            sy = np.asarray(col.get("sum_Y"), float)
            sy2 = np.asarray(col.get("sum_Y2"), float)
            if sy.shape != (Ln, Wn * 30) or sy2.shape != sy.shape:
                raise ValueError(f"replicate {r} column {name} moment shape is invalid")
            if not np.all(np.isfinite(sy)) or not np.all(np.isfinite(sy2)):
                raise ValueError(f"replicate {r} column {name} contains non-finite moments")
            n = int(rd["n_photons_per_wavelength"])
            D[r, c] = sy / n
            V[r, c] = np.maximum((sy2 - sy * sy / n) / max(n - 1, 1), 0.0)
    return D, V


def _subset_metrics(Dc, h_all, subset, W, areas):
    """Metrics for one column. Dc is (replicate, h, 120)."""
    idx = [int(np.flatnonzero(np.isclose(h_all, x, rtol=0.0, atol=0.0))[0]) for x in subset]
    held = [i for i in range(len(h_all)) if i not in idx]
    h = h_all[idx]
    X = np.column_stack([np.ones(4), h ** 2, h ** 4])
    P = np.linalg.pinv(X)
    coef = np.einsum("kl,rlm->rkm", P, Dc[:, idx, :])
    errs = []
    for i in held:
        row = np.array([1.0, h_all[i] ** 2, h_all[i] ** 4])
        pred = np.einsum("k,rkm->rm", row, coef)
        err = pred - Dc[:, i, :]
        errs.append(np.einsum("ij,rj->ri", W, err))
    abs_err = np.abs(np.stack(errs, axis=1))
    objective = float(np.quantile(abs_err, 0.95, method="linear"))
    intercept = coef[:, 0, :]
    centered = intercept - intercept.mean(axis=0, keepdims=True)
    whitened_centered = np.einsum("ij,rj->ri", W, centered)
    intercept_variance = float(np.mean(whitened_centered ** 2))
    fr = fit_rows(subset)
    areas_one = areas[:30]
    B = 1.0 / (2.0 * h[:, None] * areas_one[None, :])
    range_coefficient = float(np.max(2.0 * (np.abs(fr["qd"])[:, None] * B).sum(axis=0)))
    return {"grid": list(subset), "objective": objective,
            "combined_range_coefficient_max": range_coefficient,
            "whitened_intercept_variance": intercept_variance,
            "q2": fr["q2"].tolist(), "q3": fr["q3"].tolist(),
            "qd": fr["qd"].tolist()}


def _metric_sort_key(m, tol):
    # Quantise only for tie handling; the original objective remains in the record.
    bucket = round(m["objective"] / tol) if tol > 0 else m["objective"]
    lex_desc = tuple(-x for x in m["grid"])
    return (bucket, m["combined_range_coefficient_max"],
            m["whitened_intercept_variance"], lex_desc)


def analyze(artifact, design, require_frozen=True):
    D, _ = _validated_arrays(artifact, design, require_frozen=require_frozen)
    W, areas, _ = whitening_matrix()
    rd, sr = design["run_design"], design["selection_rule"]
    h_all = np.asarray(rd["h_candidates"], float)
    subsets = candidate_subsets(design)
    tol = float(sr["tie_absolute_tolerance"])
    by_column = {}
    all_metrics = {}
    for c, name in enumerate(rd["columns"]):
        rows = [_subset_metrics(D[:, c], h_all, s, W, areas) for s in subsets]
        rows.sort(key=lambda m: _metric_sort_key(m, tol))
        by_column[name] = rows[0]
        all_metrics[name] = rows

    budget = float(sr["common_grid_ratio_budget"])
    common = []
    for s in subsets:
        per = {name: next(m for m in all_metrics[name] if tuple(m["grid"]) == s)
               for name in rd["columns"]}
        ratios = {name: (per[name]["objective"] / by_column[name]["objective"]
                         if by_column[name]["objective"] > 0
                         else (1.0 if per[name]["objective"] == 0 else float("inf")))
                  for name in rd["columns"]}
        common.append({"grid": list(s), "worst_ratio": max(ratios.values()),
                       "ratios": ratios,
                       "combined_range_coefficient_max": max(
                           per[n]["combined_range_coefficient_max"] for n in rd["columns"]),
                       "whitened_intercept_variance": max(
                           per[n]["whitened_intercept_variance"] for n in rd["columns"])})
    feasible = [x for x in common if x["worst_ratio"] <= budget]
    if feasible:
        feasible.sort(key=lambda x: (x["worst_ratio"],
                                     x["combined_range_coefficient_max"],
                                     x["whitened_intercept_variance"],
                                     tuple(-v for v in x["grid"])))
        choice = {"kind": "common", "grid": feasible[0]["grid"],
                  "worst_ratio": feasible[0]["worst_ratio"], "budget": budget}
    else:
        choice = {"kind": "per_column",
                  "grids": {n: by_column[n]["grid"] for n in rd["columns"]},
                  "best_common_worst_ratio": min(x["worst_ratio"] for x in common),
                  "budget": budget}
    return {"schema_version": "candidate3-grid-analysis-v1",
            "status": "NON-CONFIRMATORY GRID SELECTION; NOT A CANDIDATE 3 VERDICT",
            "pilot_artifact_digest": canonical_digest(artifact),
            "pilot_design_digest": canonical_digest(design),
            "selection": choice, "per_column_optima": by_column,
            "all_candidates": all_metrics,
            "candidate3_verdict": None, "coverage_claimed": False}


def selftest():
    design = load_json(PILOT_SPEC)
    subsets = candidate_subsets(design)
    checks = []

    def ck(name, ok):
        checks.append((name, bool(ok)))

    ck("candidate subsets are four distinct declared steps",
       all(len(s) == 4 and len(set(s)) == 4 for s in subsets))
    ck("every subset spans a coarse and fine step",
       all(max(s) >= 0.3 and min(s) <= 0.1 for s in subsets))
    fr = fit_rows([0.4, 0.3, 0.2, 0.1])
    h = np.array([0.4, 0.3, 0.2, 0.1])
    for power, key in ((0, "q3"), (2, "q3"), (4, "q3")):
        target = 1.0 if power == 0 else 0.0
        ck(f"q3 reproduces h^{power} intercept", np.allclose(fr[key] @ h ** power, target,
                                                               rtol=0.0, atol=2e-14))
    ck("qd is q3 minus q2", np.allclose(fr["qd"], fr["q3"] - fr["q2"],
                                         rtol=0.0, atol=0.0))
    ck("design carries an internally consistent explicit freeze state",
       isinstance(design.get("frozen"), bool)
       and ((design["frozen"] and isinstance(design.get("freeze_hash"), str))
            or (not design["frozen"] and design.get("freeze_hash") is None)))
    ck("analysis module has no run or transport option",
       set(vars(_build_parser().parse_args(["--selftest"])))
       == {"selftest", "artifact", "out"})

    # End-to-end with a deterministic synthetic artifact. This exercises schema validation,
    # wavelength-major shapes, every subset, multinomial whitening and the common-grid rule without
    # reading or fabricating any transport result.
    rd = design["run_design"]
    n, hs = int(rd["n_photons_per_wavelength"]), np.asarray(rd["h_candidates"], float)
    reps = []
    coords = np.linspace(0.5, 1.5, 120)
    for r, seed in enumerate(rd["replicate_seeds"]):
        cols = {}
        for c, name in enumerate(rd["columns"]):
            D = np.stack([(0.01 * (c + 1) * coords + 0.002 * coords * x ** 2
                           + 0.0004 * coords * x ** 4 + 1e-5 * (r - 2.5) * coords)
                          for x in hs])
            sy = n * D
            sy2 = sy * sy / n + (n - 1) * 1e-4
            cols[name] = {"sum_Y": sy.tolist(), "sum_Y2": sy2.tolist()}
        reps.append({"replicate": r, "seed": seed, "columns": cols,
                     "checks": {"all_passed": True}})
    artifact = {"schema_version": "candidate3-grid-pilot-artifact-v1",
                "run_design_digest": canonical_digest(rd), "transport_run": True,
                "candidate3_verdict": None, "replicates": reps}
    result = analyze(artifact, design, require_frozen=False)
    ck("synthetic artifact reaches exactly one predeclared selection",
       result["selection"]["kind"] in ("common", "per_column")
       and len(result["per_column_optima"]) == len(rd["columns"]))
    ck("analysis cannot emit a Candidate 3 verdict or coverage claim",
       result["candidate3_verdict"] is None and result["coverage_claimed"] is False)
    draft_design = json.loads(json.dumps(design))
    draft_design.update({"frozen": False, "freeze_hash": None,
                         "normative_input_hashes": {}})
    try:
        analyze(artifact, draft_design)
        unfrozen_refused = False
    except ValueError:
        unfrozen_refused = True
    ck("real analysis refuses an artifact under a draft design", unfrozen_refused)
    bad = json.loads(json.dumps(artifact)); bad["run_design_digest"] = "0" * 64
    try:
        analyze(bad, design, require_frozen=False)
        refused = False
    except ValueError:
        refused = True
    ck("a stale run-design digest is refused", refused)
    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"  {len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


def _build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--artifact")
    ap.add_argument("--out")
    return ap


def main():
    a = _build_parser().parse_args()
    if a.selftest:
        if a.artifact or a.out:
            raise SystemExit("--selftest cannot be combined with --artifact or --out")
        return selftest()
    if not a.artifact or not a.out:
        raise SystemExit("analysis requires both --artifact and --out")
    design = load_json(PILOT_SPEC)
    result = analyze(load_json(a.artifact), design)
    with open(a.out, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"  wrote {os.path.relpath(a.out, ROOT)}")
    print(f"  selection: {result['selection']}")
    print("  NON-CONFIRMATORY; no Candidate 3 verdict or coverage claim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
