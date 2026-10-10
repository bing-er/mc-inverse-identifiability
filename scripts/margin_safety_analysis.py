#!/usr/bin/env python3
"""margin_safety_analysis.py -- is a 5 percent column error in the nuisance Jacobian safe?

The margin was justified in words: the classifications it feeds are thresholds with two to three
fold headroom, so a five percent error in a nuisance column cannot move them. That argument is not
good enough. The classifications are read from a Schur complement of a near-singular Fisher matrix,
where a small perturbation of the nuisance block can move the psi bound by far more than five
percent, and the near-null direction is exactly the quantity a near-singular matrix reports
unstably.

So this measures it, on the frozen production J_psi, over a randomized family of nuisance blocks
spanning weak to strong coupling, perturbing every eta column by the margin in the worst direction
the analysis can find and recording whether any pre-registered classification flips.

It runs no transport. The real eta block is unknown until the gate runs, so the question asked here
is whether the margin is safe for the FAMILY of blocks that could turn up. A margin that is unsafe
for some plausible block is not a safe margin.

    python3 scripts/margin_safety_analysis.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import augmented as aug  # noqa: E402
import unify  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unify_spectrum import column_scale  # noqa: E402

MARGINS = (0.05, 0.02, 0.01, 0.005, 0.002)
N_BLOCKS_PER_FAMILY = 40
N_PERTURBATION_DIRECTIONS = 60
COUPLING_LEVELS = (0.05, 0.2, 0.5, 1.0, 2.0)
SEED = 20261005

# The families the first version of this analysis missed. It built only well-conditioned nuisance
# blocks nearly orthogonal to psi, which is precisely the safe regime, and then reported the margin
# as safe. The dangerous regimes are a nearly singular F_ee, where the Schur complement divides by
# something small, and eta columns nearly inside the psi column span, where F_psi_eff itself goes
# near-singular. A family that excludes those cannot establish anything.
BLOCK_FAMILIES = ("generic", "internally_collinear", "nearly_inside_psi_span",
                  "tiny_relative_to_psi", "one_column_nearly_degenerate")

# The pre-registered classifications, exactly as the nuisance spec states them.
THRESHOLDS = {"deep_StO2_not_tightly_constrained": ("StO2_D", "ge", 1.0),
              "upper_StO2_well_conditioned": ("StO2_U", "lt", 0.2)}
NEAR_NULL_COS_FLOOR = 0.90


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def classify(F_psi_eff, D, labels):
    """The pre-registered classifications from an effective information matrix."""
    est = aug.estimability(F_psi_eff, scale=D, labels=labels)
    nn = aug.near_null_report(F_psi_eff, scale=D, labels=labels)
    out = {"rank": est["rank"], "nullity": est["nullity"]}
    for name, (param, op, thr) in THRESHOLDS.items():
        e = est["per_parameter"][param]
        if not e["individually_estimable"]:
            out[name] = (op == "ge")        # an infinite bound satisfies "at least", not "below"
        else:
            out[name] = (e["sqrt_crlb"] >= thr) if op == "ge" else (e["sqrt_crlb"] < thr)
    out["single_near_null_direction_reported"] = nn["single_direction_reported"]
    out["near_null_tangent"] = (nn.get("tangent_psi") if nn["single_direction_reported"]
                                else None)
    out["near_null_dominant"] = nn.get("dominant_component")
    return out


def flips(base, pert):
    """Which classifications changed, treating a lost near-null direction as a change."""
    ch = []
    for k in list(THRESHOLDS) + ["rank", "single_near_null_direction_reported",
                                 "near_null_dominant"]:
        if base.get(k) != pert.get(k):
            ch.append(k)
    a, b = base.get("near_null_tangent"), pert.get("near_null_tangent")
    if a is not None and b is not None:
        cos = aug.direction_agreement(a, b)
        if cos < NEAR_NULL_COS_FLOOR:
            ch.append(f"near_null_direction_cos_{cos:.3f}")
    return ch


def main():
    pooled = os.path.join(ROOT, "data", "derived", "production_pooled.npz")
    spec = json.load(open(os.path.join(ROOT, "spec", "baseline_spec.json")))
    d = np.load(pooled)
    J_psi = np.asarray(d["J_psi"], float)
    R = np.asarray(d["R"], float)
    areas = np.asarray(d["areas"], float)
    psi = np.asarray(d["psi_nominal"], float)
    n_wl = int(d["n_wavelengths"])
    N = spec["profiles"]["production"]["n_measurement_photons_per_wavelength"]
    labels = list(unify.PSI_LABELS)
    D_psi = column_scale(psi)
    n_obs, n_p = J_psi.shape
    n_eta = 5

    # the whitened psi block, exactly as the headline CRB uses it
    Jw_psi = unify.whiten_multinomial(J_psi, R, areas, N, D_psi, n_wl)
    s_psi = np.linalg.svd(Jw_psi, compute_uv=False)
    rng = np.random.default_rng(SEED)
    # This is a synthetic analysis over constructed nuisance blocks, so every observable row is
    # intentionally in play. That is stated with an explicit all-true mask rather than left to a
    # default, because a Fisher matrix built from an implicit mask is the thing the contract
    # forbids.
    ALL_ROWS = np.ones(n_obs, dtype=bool)
    ALL_ROWS_RATIONALE = ("synthetic nuisance blocks: all rows intentionally used, declared "
                          "explicitly rather than relying on a default")

    results = {m: {"n_blocks": 0, "n_flipped": 0, "flipped_examples": [],
                   "worst_rel_change": {k: 0.0 for k in THRESHOLDS},
                   "worst_near_null_cos": 1.0} for m in MARGINS}

    Q_psi = np.linalg.svd(Jw_psi, full_matrices=False)[0]

    def make_block(family, level):
        """A nuisance block from one of the families, including the dangerous ones."""
        A = rng.normal(size=(n_obs, n_eta))
        if family == "internally_collinear":
            # three independent directions spanning five columns: F_ee is singular
            B = rng.normal(size=(n_obs, 3))
            A = np.hstack([B, B @ rng.normal(size=(3, 2))])
            A += rng.normal(scale=1e-6, size=A.shape)
        elif family == "nearly_inside_psi_span":
            # the eta columns sit almost entirely inside the psi column space, which drives
            # F_psi_eff toward singular
            A = Q_psi @ rng.normal(size=(Q_psi.shape[1], n_eta))
            A += rng.normal(scale=1e-4, size=A.shape) * float(np.linalg.norm(A, 2))
        elif family == "tiny_relative_to_psi":
            A *= 1e-6
        elif family == "one_column_nearly_degenerate":
            A[:, 0] = A[:, 1] * (1.0 + 1e-8)
        A = A / max(float(np.linalg.norm(A, 2)), 1e-300)
        return A * (level * float(s_psi[0]))

    def worst_direction_perturbation(Jw_eta, m, base_est):
        """Search for the perturbation at the margin that moves the deep bound the most.

        Random directions in a 120 by 5 space essentially never find the worst case, and reporting
        a margin safe against random perturbations would be reporting the wrong thing.
        """
        best, best_change = None, -1.0
        cols = np.linalg.norm(Jw_eta, axis=0, keepdims=True)
        # the left singular directions of the psi block are where a perturbation couples hardest
        U_psi = Q_psi[:, :min(6, Q_psi.shape[1])]
        cands = [U_psi[:, i % U_psi.shape[1]][:, None] * np.ones((1, n_eta))
                 for i in range(U_psi.shape[1])]
        cands += [rng.normal(size=Jw_eta.shape) for _ in range(N_PERTURBATION_DIRECTIONS)]
        for P0 in cands:
            P = P0 / np.maximum(np.linalg.norm(P0, axis=0, keepdims=True), 1e-300)
            P = P * m * cols
            bi = aug.block_information(np.hstack([Jw_psi, Jw_eta + P]), n_p,
                                       support_mask=ALL_ROWS, require_mask=True)
            ep = aug.estimability(bi["F_psi_eff"], scale=D_psi, labels=labels)
            a = base_est["per_parameter"]["StO2_D"]["sqrt_crlb"]
            c = ep["per_parameter"]["StO2_D"]["sqrt_crlb"]
            chg = (abs(c - a) / a if (a and c) else float("inf"))
            if chg > best_change:
                best, best_change = (P, bi, ep), chg
        return best

    blocks = [(fam, lvl) for fam in BLOCK_FAMILIES for lvl in COUPLING_LEVELS
              for _ in range(max(N_BLOCKS_PER_FAMILY // len(COUPLING_LEVELS), 1))]
    for b, (family, level) in enumerate(blocks):
        Jw_eta = make_block(family, level)
        Jw = np.hstack([Jw_psi, Jw_eta])
        base_bi = aug.block_information(Jw, n_p, support_mask=ALL_ROWS, require_mask=True)
        base = classify(base_bi["F_psi_eff"], D_psi, labels)
        eb = aug.estimability(base_bi["F_psi_eff"], scale=D_psi, labels=labels)

        for m in MARGINS:
            P, bi, ep = worst_direction_perturbation(Jw_eta, m, eb)
            pert = classify(bi["F_psi_eff"], D_psi, labels)
            ch = flips(base, pert)
            for k, (param, _, _) in THRESHOLDS.items():
                a = eb["per_parameter"][param]["sqrt_crlb"]
                c = ep["per_parameter"][param]["sqrt_crlb"]
                if a and c:
                    results[m]["worst_rel_change"][k] = max(
                        results[m]["worst_rel_change"][k], abs(c - a) / a)
                elif bool(a) != bool(c):
                    results[m]["worst_rel_change"][k] = float("inf")
            if base.get("near_null_tangent") and pert.get("near_null_tangent"):
                results[m]["worst_near_null_cos"] = min(
                    results[m]["worst_near_null_cos"],
                    aug.direction_agreement(base["near_null_tangent"],
                                            pert["near_null_tangent"]))
            results[m]["n_blocks"] += 1
            if ch:
                results[m]["n_flipped"] += 1
                if len(results[m]["flipped_examples"]) < 8:
                    results[m]["flipped_examples"].append(
                        {"family": family, "coupling_level": level, "flipped": ch,
                         "block_index": b})
            results[m].setdefault("per_family", {}).setdefault(
                family, {"blocks": 0, "flipped": 0})
            results[m]["per_family"][family]["blocks"] += 1
            if ch:
                results[m]["per_family"][family]["flipped"] += 1

    out = {
        "analysis": "does a column perturbation at the equivalence margin flip any pre-registered "
                    "classification?",
        "runs_transport": False,
        "support_mask": {"all_rows": True, "n_rows": int(n_obs),
                         "rationale": ALL_ROWS_RATIONALE,
                         "passed_explicitly": True},
        "why":
            "the margin was justified in words, by the two to three fold headroom of the final "
            "thresholds. That is not sufficient: the classifications come from a Schur complement "
            "of a near-singular Fisher matrix, where a small perturbation of the nuisance block "
            "can move a psi bound by much more than the perturbation, and the near-null direction "
            "is exactly what a near-singular matrix reports unstably.",
        "psi_block": {"source": "data/derived/production_pooled.npz",
                      "sha256": sha256(pooled),
                      "whitening": "multinomial_fixed_launch, as the headline CRB uses",
                      "singular_values": s_psi.tolist(),
                      "condition_number": float(s_psi[0] / s_psi[-1])},
        "eta_blocks": {"n_blocks": len(blocks), "n_eta": n_eta,
                       "families": list(BLOCK_FAMILIES),
                       "families_include_the_dangerous_regimes":
                           "internally_collinear gives a singular F_ee so the Schur complement "
                           "divides by something small; nearly_inside_psi_span drives F_psi_eff "
                           "itself toward singular; tiny_relative_to_psi and "
                           "one_column_nearly_degenerate probe the ill-conditioned extremes. The "
                           "first version of this analysis used only the generic family, which is "
                           "the safe regime, and reported the margin as safe on that basis.",
                       "coupling_levels_relative_to_the_psi_block": list(COUPLING_LEVELS),
                       "coupling_levels_note": "each family is built at every coupling level",
                       "why_a_family": "the real eta block is unknown until the gate runs, so the "
                                       "question is whether the margin is safe for the blocks "
                                       "that could turn up. A margin unsafe for some plausible "
                                       "block is not a safe margin."},
        "perturbation": {"per_column_relative_size": "the margin",
                         "directions_searched_per_block": N_PERTURBATION_DIRECTIONS
                                                          + min(6, n_p),
                         "search": "the direction that moves the deep StO2 bound the most is "
                                   "searched for, over the leading left singular directions of "
                                   "the psi block plus random directions. Random directions in a "
                                   "120 by 5 space essentially never find the worst case, so a "
                                   "margin safe against random perturbations would be the wrong "
                                   "thing to report."},
        "classifications": {k: f"{v[0]} {v[1]} {v[2]}" for k, v in THRESHOLDS.items()},
        "near_null_cos_floor": NEAR_NULL_COS_FLOOR,
        "seed": SEED,
        "by_margin": {}}
    for m in MARGINS:
        r = results[m]
        out["by_margin"][str(m)] = {
            "margin": m, "n_blocks": r["n_blocks"], "n_blocks_with_a_flip": r["n_flipped"],
            "flip_rate": r["n_flipped"] / max(r["n_blocks"], 1),
            "worst_relative_bound_change": {k: round(v, 6)
                                            for k, v in r["worst_rel_change"].items()},
            "worst_near_null_cosine": round(r["worst_near_null_cos"], 6),
            "per_family": r.get("per_family", {}),
            "examples": r["flipped_examples"],
            "safe": bool(r["n_flipped"] == 0)}
    safe = [m for m in MARGINS if out["by_margin"][str(m)]["safe"]]
    out["largest_safe_margin"] = (max(safe) if safe else None)
    out["verdict"] = (
        f"no classification flips at a margin of {max(safe)}, so that margin is safe for this "
        f"family of nuisance blocks" if safe else
        "every margin tested flips at least one classification for some block in the family; the "
        "margin cannot be justified this way and the criterion must change")
    out["five_percent_is_safe"] = bool(out["by_margin"]["0.05"]["safe"])
    dst = os.path.join(ROOT, "results", "margin_safety_analysis.json")
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"  psi block condition number {out['psi_block']['condition_number']:.3e}")
    print(f"  {len(blocks)} nuisance blocks over {len(BLOCK_FAMILIES)} families, worst-direction "
          f"search\n")
    print(f"  {'margin':>8}{'blocks with a flip':>20}{'flip rate':>11}"
          f"{'worst deep bound change':>25}{'worst near-null cos':>21}")
    for m in MARGINS:
        r = out["by_margin"][str(m)]
        print(f"  {m:>8}{r['n_blocks_with_a_flip']:>20}{r['flip_rate']:>11.3f}"
              f"{r['worst_relative_bound_change']['deep_StO2_not_tightly_constrained']:>25.4f}"
              f"{r['worst_near_null_cosine']:>21.4f}")
    print(f"\n  per family, at the five percent margin:")
    for fam, v in out["by_margin"]["0.05"].get("per_family", {}).items():
        print(f"    {fam:<32} {v['flipped']:>3}/{v['blocks']:<3} blocks flipped")
    print(f"\n  largest safe margin: {out['largest_safe_margin']}")
    print(f"  five percent is safe: {out['five_percent_is_safe']}")
    print(f"  {out['verdict']}")
    print(f"  wrote {os.path.relpath(dst, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
