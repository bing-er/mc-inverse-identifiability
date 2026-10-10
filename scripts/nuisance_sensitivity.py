#!/usr/bin/env python3
"""nuisance_sensitivity.py -- MERGE_CHECKS item 2: does the conclusion survive the prior?

The design is in spec/nuisance_spec.json and this script carries no defaults of its own.

Three questions, all pre-registered before anything was computed.

First, and this is what version 1 of the spec got wrong: sweeping nuisance values and recomputing
the six-parameter bound at each one treats the nuisance as KNOWN inside every draw. That is
conditional sensitivity, how the conclusion moves across plausible operating points. The reviewer
asked what the method is worth when the thickness and the scattering are NOT known, which is a
joint estimation problem whose answer is the nuisance-marginalized effective information,

    F_eff = F_pp - F_pe pinv(F_ee) F_ep

the Schur complement of the block Fisher matrix of theta = [psi, eta]. Both are reported, the
marginalized one as primary and the conditional one as supplementary.

Second, estimator choice. Item 1 closed without adjudicating AD against perturbation Monte Carlo:
they differ resolvably from each other and the ensemble reference could not say which is right
within acceptable resources. Every identifiability conclusion is therefore computed with an
estimator whose correctness is not established, and the remaining honest defence is to show the
conclusions do not depend on which one is used. Agreement does not make either correct, and the
script says so in its output.

One thing this spec cannot evaluate is recorded rather than worked around: the production parameter
vector contains no scattering parameter, so there is no scattering column whose estimability could
be classified, and the phantom scattering claim has no home here until that is settled.

    python3 scripts/nuisance_sensitivity.py --selftest
    python3 scripts/nuisance_sensitivity.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import augmented as aug  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nuisance_feasibility_gate as _gate  # noqa: E402
import unify  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unify_spectrum import column_scale  # noqa: E402

NSPEC = os.path.join(ROOT, "spec", "nuisance_spec.json")
BSPEC = os.path.join(ROOT, "spec", "baseline_spec.json")
CRB = os.path.join(ROOT, "results", "production_crb.json")


def load(path):
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------------------------
# Draws, in eta coordinates
# ---------------------------------------------------------------------------------------------

def eta_draws(nspec):
    """Row 0 is the nominal model, eta = 0. Log coordinates make every draw feasible by
    construction: no thickness or scattering coefficient can be driven non-positive."""
    d = nspec["draw_design"]
    dims = nspec["nuisance_dimensions"]
    m = d["n_draws"]
    rng = np.random.default_rng(d["draw_seed"])
    out = np.zeros((m, len(dims)))
    for j, dim in enumerate(dims):
        out[:, j] = float(dim["prior_sd"]) * rng.standard_normal(m)
    if d["row_zero_is_nominal"]:
        out[0, :] = 0.0
    return out


def prior_sd(nspec):
    """From the declared field, never parsed from prose."""
    return np.array([float(d["prior_sd"]) for d in nspec["nuisance_dimensions"]])


def citation_gate_open(nspec):
    """The gate policy, reused rather than reimplemented.

    This file used to carry its own copy keyed on a flat citation_verified_against_source flag.
    The policy moved to prior_kind plus an explicit release decision and the copy did not, so the
    two drifted apart until one of them crashed. There is now one implementation.
    """
    g = _gate.citation_gate(nspec)
    return {"released": g["open"], "per_dimension": g["per_dimension"],
            "blocking": g["blocking"],
            "explicit_release_decision": g["explicit_release_decision"],
            "source": "scripts/nuisance_feasibility_gate.py citation_gate",
            "rule": g["rule"]}


def _refuses_without_mask():
    """block_information must fail closed when the gate-produced mask is omitted."""
    import numpy as _np
    try:
        aug.block_information(_np.zeros((12, 11)), 6)
        return False
    except ValueError as exc:
        return "support_mask is required" in str(exc)


def selftest(nspec, bspec):
    res, fail = [], []

    def ck(name, ok, detail=""):
        res.append({"check": name, "pass": bool(ok), "detail": detail})
        if not ok:
            fail.append(name)

    par = nspec["parameterization"]
    ck("production psi is unchanged and is the production spec's",
       par["production_psi"] == bspec["parameters"]["psi_labels"]
       and par["production_psi_unchanged"] is True)
    ck("the augmented vector is declared for this check only",
       par["augmented_vector_is_for_this_check_only"] is True
       and len(par["eta"]) == 5)
    ck("every eta entry is a log coordinate or an additive exponent shift",
       all(("log" in e["coordinate"]) or ("additive" in e["coordinate"]) for e in par["eta"]))
    ck("production artifacts are listed as not to be modified",
       "results/production_crb.json" in par["production_artifacts_must_not_be_modified"]
       and "spec/baseline_spec.json" in par["production_artifacts_must_not_be_modified"])

    # --- marginalization algebra and estimability, on cases whose answer is known ------------
    rng = np.random.default_rng(3)
    Dp = np.array([1.0, 0.0560139, 1.0, 0.0186713, 0.49, 0.45])
    ETA = [e["name"] for e in par["eta"]]
    p, e, n = 6, len(ETA), 240   # eta is five-dimensional since melanin was removed

    # 1. F = diag(1, 0): one parameter finite, one infinite
    r1 = aug.estimability(np.diag([1.0, 0.0]), labels=["a", "b"])
    ck("F = diag(1, 0): parameter a is finite and b is infinite",
       r1["per_parameter"]["a"]["bound"] == "finite"
       and abs(r1["per_parameter"]["a"]["sqrt_crlb"] - 1.0) < 1e-12
       and r1["per_parameter"]["b"]["bound"] == "infinite"
       and r1["per_parameter"]["b"]["sqrt_crlb"] is None
       and abs(r1["per_parameter"]["b"]["null_space_leakage"] - 1.0) < 1e-12,
       f"a finite, b leakage {r1['per_parameter']['b']['null_space_leakage']:.3g}")

    # 2. a rotated single null direction: decided per parameter by its null-space projection
    th = np.deg2rad(30.0)
    Q = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    r2 = aug.estimability(Q @ np.diag([1.0, 0.0]) @ Q.T, labels=["a", "b"])
    ck("a rotated null direction makes both parameters inestimable, by their projections",
       not r2["per_parameter"]["a"]["individually_estimable"]
       and not r2["per_parameter"]["b"]["individually_estimable"]
       and abs(r2["per_parameter"]["a"]["null_space_leakage"] - 0.5) < 1e-9
       and abs(r2["per_parameter"]["b"]["null_space_leakage"] - np.sin(np.deg2rad(60))) < 1e-9,
       f"leakages {r2['per_parameter']['a']['null_space_leakage']:.4f} and "
       f"{r2['per_parameter']['b']['null_space_leakage']:.4f}")
    r2b = aug.estimability(np.diag([2.0, 0.0, 3.0]), labels=["a", "b", "c"])
    ck("a lost direction does not erase the bounds of the parameters that survive",
       r2b["per_parameter"]["a"]["bound"] == "finite"
       and r2b["per_parameter"]["c"]["bound"] == "finite"
       and r2b["per_parameter"]["b"]["bound"] == "infinite"
       and r2b["n_individually_estimable"] == 2)

    # 3. a two-dimensional null space: no unique near-null vector, report the subspace
    nn3 = aug.near_null_report(np.diag([5.0, 1.0, 0.0, 0.0]), labels=list("abcd"))
    ck("a two-dimensional null space forbids a single near-null direction",
       nn3["single_direction_reported"] is False
       and nn3["null_space_dimension"] == 2
       and "dominant_component" not in nn3
       and "principal angles" in nn3["compare_with"],
       f"nullity {nn3['nullity']}, subspace reported instead")
    nn3b = aug.near_null_report(np.diag([5.0, 1.0, 0.99]), labels=list("abc"))
    nn3c = aug.near_null_report(np.diag([5.0, 1.0, 0.01]), labels=list("abc"))
    ck("full rank without eigenvalue separation also forbids a single direction",
       nn3b["single_direction_reported"] is False
       and nn3c["single_direction_reported"] is True
       and nn3c["dominant_component"] == "c",
       f"separation {nn3b['eigenvalue_separation_ratio']:.3f} refused, "
       f"{nn3c['eigenvalue_separation_ratio']:.0f} accepted")
    ck("the separation ratio is declared in the spec",
       nspec["estimability_rules"]["near_null_separation_ratio"] == 10.0
       and nspec["estimability_rules"]["separation_ratio_is_declared_in_advance"] is True)
    sd3 = aug.subspace_distance(nn3["null_space_basis_z"], nn3["null_space_basis_z"])
    sd4 = aug.subspace_distance(nn3["null_space_basis_z"], np.eye(4)[:2])
    ck("subspace comparison uses principal angles and a projector distance",
       sd3["projector_distance"] < 1e-12 and sd4["projector_distance"] > 0.9
       and len(sd3["principal_angles_deg"]) == 2,
       f"identical {sd3['projector_distance']:.1e}, different {sd4['projector_distance']:.3f}")

    # 4. both marginalizations, and the reverse projection route agreeing with Schur
    all_rows = np.ones(n, dtype=bool)   # synthetic: every row intentionally used, stated
    b4 = aug.block_information(np.hstack([rng.normal(size=(n, p)),
                                          rng.normal(size=(n, e))]), p,
                               support_mask=all_rows, require_mask=True)
    ck("both marginalizations agree with their Schur cross-checks", b4["routes_agree"],
       f"psi {b4['schur_vs_projection_psi']:.2e}, eta {b4['schur_vs_projection_eta']:.2e}")
    ck("neither effective information exceeds its block in the positive-semidefinite order",
       np.linalg.eigvalsh(b4["F_pp"] - b4["F_psi_eff"]).min() > -1e-10
       and np.linalg.eigvalsh(b4["F_ee"] - b4["F_eta_eff"]).min() > -1e-10)
    Q6, _ = np.linalg.qr(rng.normal(size=(n, p + e)))
    b4b = aug.block_information(np.hstack([Q6[:, :p], Q6[:, p:p + e]]), p,
                                support_mask=all_rows, require_mask=True)
    ck("nuisance orthogonal to psi costs neither side any information",
       np.max(np.abs(b4b["F_psi_eff"] - b4b["F_pp"])) / np.max(np.abs(b4b["F_pp"])) < 1e-12
       and np.max(np.abs(b4b["F_eta_eff"] - b4b["F_ee"])) / np.max(np.abs(b4b["F_ee"])) < 1e-12)

    # 5. bare F_ee says the scattering is measurable; marginalizing psi says it is not
    Jp5 = rng.normal(size=(n, p))
    Je5 = rng.normal(size=(n, e))
    Je5[:, 3] = Jp5 @ rng.normal(size=p)          # the log_mus_scale column lies inside span(psi)
    b5 = aug.block_information(np.hstack([Jp5, Je5]), p,
                               support_mask=all_rows, require_mask=True)
    bare = aug.estimability(b5["F_ee"], labels=ETA)["per_parameter"]["log_mus_scale"]
    marg = aug.estimability(b5["F_eta_eff"], labels=ETA)["per_parameter"]["log_mus_scale"]
    ck("bare F_ee would wrongly call the scattering nuisance measurable",
       bare["individually_estimable"] and bare["sqrt_crlb"] < 0.20
       and not marg["individually_estimable"] and marg["bound"] == "infinite",
       f"F_ee sqrt_crlb {bare['sqrt_crlb']:.5f} inside the 0.20 prior, "
       f"F_eta_eff leakage {marg['null_space_leakage']:.6f}")
    ck("the spec requires the scattering claim to come from F_eta_eff",
       "never from bare F_ee"
       in nspec["nuisance_marginalization"]["scattering_claim_must_use_F_eta_eff"])

    # 6. the prior-informed posterior is never wider than the prior, and is named separately
    sd0 = prior_sd(nspec)
    po = aug.eta_posterior(b5["F_eta_eff"], sd0, labels=ETA)
    ck("the prior-informed posterior is never wider than the prior",
       po["never_wider_than_the_prior"] is True)
    ck("the posterior is named separately from the data-only bound",
       po["name"] == "prior_informed_posterior" and po["distinct_from"] == "data_only_crlb"
       and "masquerade as a measurement"
       in nspec["nuisance_marginalization"]["prior_informed_posterior"]["why_separately"])
    ck("a nuisance with no data-only bound still has a finite posterior, which is why the names "
       "must differ",
       marg["sqrt_crlb"] is None
       and abs(po["posterior_over_prior_sd_ratio"]["log_mus_scale"] - 1.0) < 1e-9,
       f"posterior/prior {po['posterior_over_prior_sd_ratio']['log_mus_scale']:.6f}, so the data "
       f"narrowed nothing")
    po0 = aug.eta_posterior(np.zeros((e, e)), sd0, labels=ETA)
    ck("with zero information the posterior equals the prior exactly",
       all(abs(po0["posterior_over_prior_sd_ratio"][k] - 1.0) < 1e-12 for k in ETA))

    # psi-side bounds still behave through the estimability route
    b_c = aug.estimability(b4["F_pp"], scale=Dp, labels=list(aug.PSI_LABELS))
    b_m = aug.estimability(b4["F_psi_eff"], scale=Dp, labels=list(aug.PSI_LABELS))
    ck("marginalized bounds are never tighter than conditional ones",
       all(b_m["per_parameter"][k]["sqrt_crlb"] >= b_c["per_parameter"][k]["sqrt_crlb"] - 1e-12
           for k in aug.PSI_LABELS))

    # --- eta draws -----------------------------------------------------------------------------
    X = eta_draws(nspec)
    dims = nspec["nuisance_dimensions"]
    ck("eta draw matrix has the declared shape",
       X.shape == (nspec["draw_design"]["n_draws"], len(dims)), str(X.shape))
    ck("row zero is eta = 0, the nominal model", np.allclose(X[0], 0.0))
    ck("draws are reproducible from the recorded seed", np.array_equal(X, eta_draws(nspec)))
    sd = prior_sd(nspec)
    ok = all(abs(X[1:, j].std(ddof=1) / sd[j] - 1.0) < 0.35 for j in range(len(dims)))
    ck("each eta marginal has the declared spread", ok,
       f"sd {np.round(X[1:].std(axis=0, ddof=1), 3).tolist()} against {sd.tolist()}")

    # --- feasibility by construction, over the whole draw set ----------------------------------
    base = np.array([0.0020, 0.0100, 0.0250, 0.0330, 0.1830, 0.1910, 0.8000])
    B = np.stack([aug.boundaries_from_eta(base, *X[i, :3]) for i in range(X.shape[0])])
    ck("layer boundaries are strictly increasing for every draw",
       bool(np.all(np.diff(B, axis=1) > 0)))
    ck("layer boundaries are strictly positive for every draw", bool(np.all(B > 0)))
    wl = bspec["wavelengths_nm"]
    F = np.stack([aug.mus_factor_from_eta(X[i, 3], X[i, 4], wl, 633.0)
                  for i in range(X.shape[0])])
    ck("the reduced scattering factor is strictly positive for every draw and wavelength",
       bool(np.all(F > 0)), f"min {F.min():.4f}, max {F.max():.4f}")
    ck("eta carries no melanin column, so no melanosome fraction is derived",
       all(e["name"] != "log_f_melanin" for e in par["eta"]) and X.shape[1] == 5,
       "melanin was removed because the frozen production model does not expose it")
    # the log coordinate is what makes this automatic, not a filter
    extreme = aug.boundaries_from_eta(base, 6.0, -6.0, 6.0)
    ck("log coordinates keep the geometry feasible even at an absurd eta",
       bool(np.all(np.diff(extreme) > 0) and np.all(extreme > 0)),
       "no draw needs to be rejected")

    # --- column scale --------------------------------------------------------------------------
    Dth = aug.theta_column_scale(Dp, 6)
    ck("the eta column scale is unity and the psi part is the production one",
       np.allclose(Dth[:6], Dp) and np.allclose(Dth[6:], 1.0))

    # --- classification groups -----------------------------------------------------------------
    q = nspec["qualitative_classifications"]
    ck("classifications are split into conditional and marginalized groups",
       set(q) >= {"conditional", "marginalized"}
       and q["conditional"]["role"] == "supplementary only"
       and q["marginalized"]["role"] == "primary")
    ck("the reviewer's question is assigned to the marginalized group",
       "jointly unknown" in q["marginalized"]["answers"]
       and "FIXED" in q["conditional"]["answers"])
    ck("every conditional classification has a marginalized counterpart",
       {c["name"].replace("conditional_", "") for c in q["conditional"]["classifications"]}
       <= {c["name"].replace("marginalized_", "") for c in q["marginalized"]["classifications"]})
    ck("the scattering classification is named for what it measures",
       any(c["name"] == "scattering_nuisance_remains_weakly_constrained"
           for c in q["marginalized"]["classifications"]))
    ck("the verdict was defined before any draw was evaluated",
       q["defined_before_any_draw_was_evaluated"] is True
       and abs(q["stability_threshold_fraction"] - 0.90) < 1e-12)

    # --- phantom scattering --------------------------------------------------------------------
    ps = nspec["phantom_scattering"]
    ck("the phantom scattering claim is handled separately and labelled",
       ps["handled_separately"] is True
       and ps["mandatory_label"] == "not re-evaluated under the nuisance prior"
       and "section 3.4" in ps["p1_p2_empirical_result_stays"])
    # the phrase may appear while being prohibited; what must not happen is it appearing as an
    # assertion. So every field containing it must also mark it as prohibited.
    def fields_containing(node, needle, path="", hits=None):
        hits = [] if hits is None else hits
        if isinstance(node, dict):
            for k, v in node.items():
                fields_containing(v, needle, f"{path}.{k}", hits)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                fields_containing(v, needle, f"{path}[{i}]", hits)
        elif isinstance(node, str) and needle in node:
            hits.append((path, node))
        return hits
    bad = []
    for phrase in ps["forbidden_phrasing"]:
        for path, text in fields_containing(nspec, phrase):
            prohibited = ("forbidden" in path or "must NOT" in text
                          or "may and may not" in path)
            if not prohibited:
                bad.append(path)
    ck("the forbidden phantom scattering phrasings appear only where they are prohibited",
       not bad, f"offending fields {bad}" if bad else
       "every occurrence sits in a forbidden list or is marked must NOT")

    # --- estimator column support --------------------------------------------------------------
    sup = nspec["estimator_column_support"]
    ck("pMC carries no eta column this round, with the mathematical reason recorded",
       sup["pmc_not_used_this_round"] is True and sup["pmc_supports_this_round"] == []
       and "neither geometry nor scattering derivatives" in sup["why"])
    ck("pMC is still recorded as the right tool for the psi absorption columns",
       "psi" in sup["pmc_remains_available_for_psi"])
    ck("every downstream Fisher call passes an explicit support mask",
       (lambda: _refuses_without_mask())(),
       "omitting the mask fails closed on every production path")

    # --- feasibility gate ----------------------------------------------------------------------
    g = nspec["nominal_derivative_feasibility_gate"]
    ck("the feasibility gate runs no draws",
       g["n_draws_in_this_gate"] == 0 and g["must_pass_before_any_draw"] is True)
    ck("the gate requires a demonstration for AD on geometry columns",
       any("pathwise derivative" in c for c in g["checks"])
       and "MOVE with the parameter" in g["why_the_geometry_columns_are_suspect"])
    ck("the cost re-estimate is deferred until after the gate",
       "must be produced after the gate" in g["consequence_if_AD_fails_on_a_column"])

    # --- citation gate -------------------------------------------------------------------------
    cg = citation_gate_open(nspec)
    ck("every dimension carries a citation, a prior_kind and a numeric prior width",
       all("citation" in d_ and "prior_kind" in d_
           and isinstance(d_.get("prior_sd"), (int, float)) for d_ in dims))
    ck("every dimension states a numeric range implied by its prior",
       all("numeric_range_implied" in d_ for d_ in dims))
    ck("the gate policy is shared with the feasibility gate rather than reimplemented",
       cg["source"].endswith("citation_gate")
       and set(cg["per_dimension"]) == {d_["eta_name"] for d_ in dims},
       "one implementation, so the two cannot drift apart again")
    # Released on 2026-10-04 for the feasibility gate only. The scope limits are asserted here
    # too, so a release that widened itself fails in both suites rather than only in one.
    ck("the citation gate is released, with every dimension passing its prior_kind policy",
       cg["released"] and cg["blocking"] == []
       and cg["explicit_release_decision"] == "released",
       f"decision {cg['explicit_release_decision']!r}, blocking {cg['blocking']}")
    rs = nspec["citation_gate"]["release_scope"]
    ck("the release is limited to the three engineering items",
       rs["authorizes_only"] == [
           "per-column AD-versus-ensemble equivalence testing",
           "finite-difference convergence testing",
           "estimator-route selection and cost measurement"]
       and rs["does_not_authorize_or_establish"]
       == "any downstream scientific classification")
    ck("the release does not authorize the 48-draw study",
       "must not be launched automatically" in rs["does_not_authorize_the_48_draw_study"])
    ck("the spec forbids submitting transport while the gate is closed",
       "No transport may be submitted while this gate is closed" in nspec["citation_gate"]["rule"])

    # --- melanin and provenance hygiene --------------------------------------------------------
    mn = nspec["melanin_not_evaluated"]
    ck("melanin is recorded as not evaluated, for a model-scope reason",
       "does not expose a separately perturbable melanin component" in mn["statement"]
       and "not of evidence" in mn["why_this_mattered"])
    ck("nothing claims melanin is known, unaffected or validated",
       {"that melanin is known", "that melanin has no effect",
        "that melanin has been validated"} <= set(mn["not_claimed"]))
    ck("draws and seeds are disjoint from P5 and from item 1",
       nspec["draw_design"]["n_draws"] != 20
       and nspec["draw_design"]["transport_seed"] not in
       [43000 + 1000 * k for k in range(32)] + [910000 + 10007 * r for r in range(64)])
    ck("estimator-choice sensitivity names item 1 and carries its caveat",
       "unresolved adjudication" in nspec["estimator_choice_sensitivity"]["why_required"]
       and "does not make either correct"
       in nspec["estimator_choice_sensitivity"]["what_this_does_not_establish"])
    ck("the spec records that it supersedes the conditional-only version",
       "treats the nuisance as KNOWN inside every draw" in nspec["supersedes"])
    ck("the downstream mask contract is recorded as enforced in code",
       nspec["nominal_derivative_feasibility_gate"]["downstream_mask_contract"]
       ["enforced_in_code"] is True)
    return res, fail


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    nspec, bspec = load(NSPEC), load(BSPEC)

    if a.selftest:
        res, fail = selftest(nspec, bspec)
        for r in res:
            print(f"  {'PASS' if r['pass'] else 'FAIL'}  {r['check']}"
                  + (f"  [{r['detail']}]" if r["detail"] else ""))
        print(f"  {sum(r['pass'] for r in res)}/{len(res)} checks passed")
        return 1 if fail else 0

    if not a.dry_run:
        ap.error("one of --selftest or --dry-run is required. The feasibility gate and the draw "
                 "run are not enabled: the citation gate is closed and the phantom scattering "
                 "decision is unsettled.")

    res, fail = selftest(nspec, bspec)
    if fail:
        raise SystemExit(f"self-tests must pass before a dry run is meaningful: {fail}")

    X = eta_draws(nspec)
    dims = nspec["nuisance_dimensions"]
    base = np.array([0.0020, 0.0100, 0.0250, 0.0330, 0.1830, 0.1910, 0.8000])
    wl = bspec["wavelengths_nm"]
    B = np.stack([aug.boundaries_from_eta(base, *X[i, :3]) for i in range(X.shape[0])])
    F = np.stack([aug.mus_factor_from_eta(X[i, 3], X[i, 4], wl, 633.0) for i in range(X.shape[0])])

    n_psi = len(nspec["parameterization"]["production_psi"])
    n_eta = len(nspec["nuisance_dimensions"])
    print(f"  theta = [psi({n_psi}), eta({n_eta})]; production psi unchanged\n")
    print(f"  {nspec['draw_design']['n_draws']} draws, seed "
          f"{nspec['draw_design']['draw_seed']}, row 0 is eta = 0\n")
    print(f"  {'eta':<18}{'prior sd':>10}{'min':>10}{'median':>10}{'max':>10}"
          f"{'implied range':>34}")
    for j, d_ in enumerate(dims):
        col = X[1:, j]
        rng_txt = d_["numeric_range_implied"][:32]
        print(f"  {d_['eta_name']:<18}{prior_sd(nspec)[j]:>10.2f}{col.min():>10.4f}"
              f"{np.median(col):>10.4f}{col.max():>10.4f}{rng_txt:>34}")

    print(f"\n  layer geometry across draws, cm (log coordinates, so no draw is infeasible)")
    print(f"  {'boundary below':<24}{'nominal':>10}{'min':>10}{'max':>10}")
    for k, nm in enumerate(["stratum corneum", "living epidermis", "papillary dermis",
                            "upper blood net plexus", "reticular dermis",
                            "deep blood net plexus", "subcutis"]):
        print(f"  {nm:<24}{base[k]:>10.4f}{B[:, k].min():>10.4f}{B[:, k].max():>10.4f}")
    print(f"  strictly increasing for every draw: "
          f"{bool(np.all(np.diff(B, axis=1) > 0))}")
    print(f"  reduced scattering factor strictly positive: {bool(np.all(F > 0))}  "
          f"range [{F.min():.4f}, {F.max():.4f}]")

    print(f"\n  estimator support by column")
    sup = nspec["estimator_column_support"]
    print(f"    pMC supports      {sup['pmc_supports_this_round']}  (none this round)")
    print(f"    reason            {sup['why'][:88]}...")
    print(f"    every eta column  AD against the ensemble central difference")

    print(f"\n  cost cannot be estimated yet")
    g = nspec["nominal_derivative_feasibility_gate"]
    print(f"    {g['consequence_if_AD_fails_on_a_column']}")
    print(f"    lower bound, if AD holds on every column: 48 draws x "
          f"(372 s AD + 164 s pMC) = {48 * 536 / 3600:.1f} h")
    print(f"    upper bound is set by how many of the five unsupported columns need an ensemble "
          f"central difference, which the gate measures")

    cg = citation_gate_open(nspec)
    print(f"\n  CITATION GATE: {'open' if cg['released'] else 'CLOSED'}")
    print(f"    {'dimension':<18}{'prior_kind':<42}passes")
    for nm, v in cg["per_dimension"].items():
        print(f"    {nm:<18}{v['prior_kind']:<42}{v['passes']}")
    print(f"    dimensions failing policy:  {cg['blocking']}")
    print(f"    explicit release decision:  {cg['explicit_release_decision']}")
    print(f"    policy source: {cg['source']}")
    print(f"\n  PHANTOM SCATTERING: {nspec['phantom_scattering']['mandatory_label']}")
    print(f"    this round may only claim: scattering_nuisance_remains_weakly_constrained")

    out = {"mode": "dry run, design freeze; nothing computed",
           "spec_version": nspec["spec_version"], "status": nspec["status"],
           "theta": f"[psi({n_psi}), eta({n_eta})]",
           "n_psi": n_psi, "n_eta": n_eta,
           "theta_dimension": n_psi + n_eta,
           "production_psi_unchanged": True,
           "n_draws": int(X.shape[0]), "eta_draws": X.tolist(),
           "eta_names": [d_["eta_name"] for d_ in dims],
           "layer_boundaries_per_draw_cm": B.tolist(),
           "mus_factor_per_draw": F.tolist(),
           "feasible_by_construction": {
               "boundaries_strictly_increasing_every_draw":
                   bool(np.all(np.diff(B, axis=1) > 0)),
               "scattering_strictly_positive_every_draw": bool(np.all(F > 0)),
               "why": "log coordinates for every positive quantity, so no draw is rejected"},
           "estimator_column_support": sup,
           "citation_gate": cg,
           "cost_estimate": {
               "status": "deferred until the feasibility gate reports per-column cost",
               "lower_bound_hours_if_AD_holds_on_every_column": round(48 * 536 / 3600, 2),
               "upper_bound": "set by how many of the five pMC-unsupported columns need an "
                              "ensemble central difference"},
           "selftest": {"checks": len(res), "passed": sum(r["pass"] for r in res)},
           "transport_run": False, "feasibility_gate_run": False}

    # the declared theta dimension must agree with the eta names and with EVERY eta draw, not just
    # with the spec field it was read from. A stale hard-coded count is exactly what this catches.
    n_eta_decl = len(nspec["parameterization"]["eta"])
    if n_eta_decl != n_eta:
        raise AssertionError(f"parameterization.eta lists {n_eta_decl} entries but "
                             f"nuisance_dimensions has {n_eta}")
    if f"{n_psi} + {n_eta} = {n_psi + n_eta}" not in nspec["parameterization"]["theta"]:
        raise AssertionError(f"the spec theta string {nspec['parameterization']['theta']!r} "
                             f"disagrees with {n_psi} + {n_eta}")
    if len(out["eta_names"]) != n_eta:
        raise AssertionError(f"eta_names has {len(out['eta_names'])} entries, n_eta is {n_eta}")
    if X.shape[1] != n_eta:
        raise AssertionError(f"eta draws have width {X.shape[1]}, n_eta is {n_eta}")
    bad = [i for i, row in enumerate(out["eta_draws"]) if len(row) != n_eta]
    if bad:
        raise AssertionError(f"eta draws {bad[:5]} do not have {n_eta} components")
    if out["theta_dimension"] != n_psi + len(out["eta_names"]):
        raise AssertionError("theta_dimension disagrees with psi labels plus eta names")
    if out["theta"] != f"[psi({n_psi}), eta({len(out['eta_names'])})]":
        raise AssertionError(f"the theta string {out['theta']!r} disagrees with the counts")
    print(f"  theta dimension {out['theta_dimension']} agrees with {n_psi} psi labels, "
          f"{len(out['eta_names'])} eta names and all {X.shape[0]} eta draws")

    dst = os.path.join(ROOT, "results", "nuisance_design.json")
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  wrote {os.path.relpath(dst, ROOT)}")
    print("  no transport was run; the feasibility gate was not run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
