#!/usr/bin/env python3
"""candidate3_design.py -- interface draft and DRAFT ADVERSARIAL SKELETON for candidate 3.

DESIGN ONLY. No executor change is implemented, no margin is set, no step grid is frozen, nothing
is registered, and no GPU work is authorized. The specification is
docs/planning/CANDIDATE3_DESIGN_SPEC.md and the acceptance criteria in its section 10 are fixed
before anything here is run.

What this file is for:

  * the executor interface candidate 3 would need under the chosen propagation, with its storage
    cost computed from the fields actually retained;
  * the interval construction, implemented against mock accumulators so the mathematics can be
    checked on a CPU without transport;
  * a DRAFT adversarial scenario skeleton: nine intentions with no generator, no seed, no trial
    count and no freeze hash. Asking for a freeze hash raises.
  * CPU tests of the parts that can be settled now: the deterministic range, the fail-closed paths,
    the alpha allocation, and the empirical-Bernstein radius against its own definition.

What this file deliberately does NOT do: claim coverage. Running the suite is a later step and its
verdict is governed by the written criteria, not by whatever comes out.

    python3 scripts/candidate3_design.py --selftest
    python3 scripts/candidate3_design.py --describe
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
SPEC_DOC = os.path.join(ROOT, "docs", "planning", "CANDIDATE3_DESIGN_SPEC.md")

DESIGN_ONLY = ("DESIGN ONLY. Nothing registered, no margin set, no grid frozen, no executor change "
               "implemented, no GPU work authorized, and no coverage claimed.")

# ---------------------------------------------------------------------------------------------
# PRE-SPECIFIED DRAFT PROPOSAL. Nothing here is registered: the live spec still has
# interval_criterion_registration.registered = false. The alpha allocation below is a proposal
# written down in advance of any result, which is not the same thing as a registration.
# ---------------------------------------------------------------------------------------------

ALPHA_TOTAL = 0.05
N_COLUMNS = 5
ALPHA_PER_COLUMN = ALPHA_TOTAL / N_COLUMNS          # 0.010
ALPHA_NUMERATOR = 0.004
ALPHA_DENOMINATOR = 0.003
ALPHA_COMPONENT = 0.003

AGGREGATE_MARGIN = None      # deliberately unset
COMPONENT_MARGIN = None      # deliberately unset
STEP_GRID = None             # deliberately unset
REPLICATES = None            # deliberately unset
SUPPORT_POLICY = None        # deliberately unset
STOPPING_RULE = None         # deliberately unset
WHITENING_LIKELIHOOD = None  # deliberately unset

# The global weight ceiling. The kernel launches at w0 = 1 - r_sp with r_sp the specular Fresnel
# reflectance, so w0 < 1 in general; 1 remains a valid ceiling because the weight is non-increasing
# from w0 <= 1. The ceiling is what the range rests on, NOT a claim that w0 equals 1.
WEIGHT_CEILING = 1.0


# ---------------------------------------------------------------------------------------------
# Executor interface the construction would need
# ---------------------------------------------------------------------------------------------

CROSS_STEP_PAIRING_INVARIANT = {
    "requirement": "within one replicate, ALL four step sizes and BOTH arms must be traced from "
                   "exactly the same photon key array, in exactly the same order, at the same "
                   "fixed launch count N",
    "why": "Z_p,i = sum_l q_l Y_p,i(h_l) is a per-photon combination. It is only defined if the "
           "index p refers to the same photon at every step and in both arms. A reordered or "
           "differently seeded key array silently combines different photons and the resulting "
           "second moment is meaningless.",
    "where_independence_lives": "ACROSS photon index p. The N contributions are independent of "
                                "one another, which is what the concentration bound uses.",
    "where_crn_reuse_lives": "WITHIN one photon, across its two arms and its four step sizes. "
                             "That reuse is deliberate and is what makes the paired difference "
                             "small; it does not couple different photons.",
    "fail_closed_on": ["a different key array at any step or arm",
                       "the same keys in a different order",
                       "a missing arm at any step",
                       "a missing step",
                       "a launch count that differs between arms or steps"],
}


def verify_pairing(key_arrays, expected_N=None):
    """Fail closed unless every (step, arm) used the identical key array in the identical order.

    `key_arrays` maps a label such as "h0.4+" to the key array that step and arm actually used.
    Equality is checked elementwise and in order: a permutation is a failure, not a pass.
    """
    if not key_arrays:
        return {"ok": False, "reason": "no key arrays were supplied, so pairing cannot be "
                                       "verified; option A requires the check to succeed"}
    labels = sorted(key_arrays)
    ref_label = labels[0]
    ref = np.asarray(key_arrays[ref_label])
    problems = []
    for lab in labels[1:]:
        a = np.asarray(key_arrays[lab])
        if a.shape != ref.shape:
            problems.append(f"{lab}: shape {a.shape} differs from {ref_label} {ref.shape}")
        elif not np.array_equal(a, ref):
            same_set = (np.array_equal(np.sort(a, axis=None), np.sort(ref, axis=None)))
            problems.append(f"{lab}: " + ("same keys in a DIFFERENT ORDER" if same_set
                                          else "a DIFFERENT key array"))
    if expected_N is not None and ref.shape[0] != int(expected_N):
        problems.append(f"launch count {ref.shape[0]} differs from the fixed N {int(expected_N)}")
    return {"ok": not problems, "n_labels": len(labels), "problems": problems,
            "reason": None if not problems else
            "REFUSE: the cross-step pairing invariant is violated, so a per-photon combination "
            "cannot be formed. " + "; ".join(problems[:4]),
            "invariant": CROSS_STEP_PAIRING_INVARIANT["requirement"]}


def required_labels(h_grid):
    """Every (step, arm) label option A needs present. A missing one is a refusal."""
    return tuple(f"h{float(h):g}{sign}" for h in np.asarray(h_grid, float) for sign in ("+", "-"))


def verify_coverage_of_labels(key_arrays, h_grid):
    """A missing arm or step fails closed, separately from the key-equality check."""
    need = set(required_labels(h_grid))
    have = set(key_arrays)
    missing = sorted(need - have)
    return {"ok": not missing, "missing": missing, "n_required": len(need),
            "reason": None if not missing else
            f"REFUSE: {len(missing)} of {len(need)} (step, arm) combinations are absent: "
            f"{missing[:6]}"}


def specular_reflectance(n_air, n_first_layer):
    """r_sp = ((n_air - n0) / (n_air + n0))^2, the production Fresnel specular reflectance."""
    n_air, n0 = float(n_air), float(n_first_layer)
    if not (n_air > 0.0 and n0 > 0.0):
        raise ValueError("both refractive indices must be strictly positive")
    return ((n_air - n0) / (n_air + n0)) ** 2


def launch_weight(n_air, n_first_layer):
    """w0 = 1 - r_sp. This is NOT 1: the production kernel removes the specular reflection."""
    return 1.0 - specular_reflectance(n_air, n_first_layer)


def kernel_weight_assertions():
    """What the executor must assert for the derived range to hold.

    The earlier draft asserted `w0 == 1 exactly`, which is false for the production kernel: it
    launches at w0 = 1 - r_sp with r_sp the specular Fresnel reflectance. A global ceiling of 1 is
    still valid, because the weight is non-increasing from w0 <= 1, but the claim that the weight
    equals 1 after entry is wrong and is withdrawn.
    """
    return [
        "r_sp is computed by the production Fresnel formula "
        "((n_air - n0)/(n_air + n0))^2, not supplied",
        "0 <= r_sp <= 1",
        "w0 == 1 - r_sp, checked to numerical precision against the kernel's own launch value",
        "0 <= w0 <= 1",
        "the weight is non-increasing thereafter, since w <- w exp(-mua d) with mua >= 0, d >= 0",
        "no splitting, no weight-raising roulette, no biasing anywhere in the kernel",
        "the range uses the GLOBAL CEILING of 1, which w0 <= 1 and non-increase together justify; "
        "it does not assume w0 equals 1",
    ]


def _sha256_file(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def executor_interface(n_bins=120, n_steps=4, n_columns=5, keep_per_step_max=True):
    """The accumulators candidate 3 needs UNDER THE CHOSEN PROPAGATION (option A).

    Option A forms the combined contribution per photon BEFORE squaring, so the accumulators are
    per `(column, bin)` for each of `Z_e2` and `Z_d`, not per `(column, step, bin)`. An earlier
    draft listed per-step `sum_Y`/`sum_Y2` and quoted 4*120*4*5 scalars, which contradicts the
    chosen option: per-step second moments cannot supply the variance of the combination.

    The per-step `max_abs_Y` is retained SEPARATELY and only as a kernel consistency check against
    the per-step absolute bound. It carries no variance information and is never used as one.
    """
    fields_combined = ("sum_Z", "sum_Z2", "max_abs_Z")
    n_combined = len(fields_combined) * 2 * n_bins * n_columns          # Z_e2 and Z_d
    n_perstep = (n_bins * n_steps * n_columns) if keep_per_step_max else 0
    total = n_combined + n_perstep
    return {
        "status": DESIGN_ONLY,
        "propagation_option": PROPAGATION_OPTION,
        "combined_accumulators_per_column_bin": {
            "sum_Z_e2": "sum over all N launched photons of Z_e2,p,i = sum_l q2_l Y_p,i(h_l)",
            "sum_Z2_e2": "sum of squares of the SAME combined contribution; this is what gives "
                         "e2 an empirical-Bernstein variance",
            "max_abs_Z_e2": "consistency check against B_Z_e2 = R_Z_e2 / 2; never a new range",
            "sum_Z_d": "sum over all N launched photons of Z_d,p,i = sum_l qd_l Y_p,i(h_l)",
            "sum_Z2_d": "sum of squares of the SAME combined contribution; this is what gives d "
                        "an empirical-Bernstein variance",
            "max_abs_Z_d": "consistency check against B_Z_d = R_Z_d / 2; never a new range"},
        "per_step_accumulator_kept_separately": {
            "max_abs_Y": "per (column, step, bin). KERNEL CONSISTENCY CHECK ONLY, compared against "
                         "B_i(h) = R_i(h)/2. It carries no variance information.",
            "retained": bool(keep_per_step_max)},
        "explicitly_not_accumulated": {
            "per_step_sum_Y2": "NOT accumulated, and if it were it would NOT provide the variance "
                               "of the combination: the per-photon cross-step terms are absent. "
                               "Option A exists precisely so that this is not relied on.",
            "n_contributing": "not a denominator and not needed; the mean is always over the fixed "
                              "launch count N"},
        "denominator_is_always_N": "the fixed launch count, deterministic and recorded. A photon "
                                   "reaching no bin contributes an exact zero rather than being "
                                   "omitted.",
        "kernel_assertions": kernel_weight_assertions(),
        "pairing_invariant": CROSS_STEP_PAIRING_INVARIANT,
        "storage": {
            "combined_scalars": n_combined,
            "per_step_consistency_scalars": n_perstep,
            "scalars_per_replicate": total,
            "bytes_per_replicate_float64": total * 8,
            "arithmetic": f"3 fields x 2 combinations x {n_bins} bins x {n_columns} columns = "
                          f"{n_combined}, plus {n_bins} x {n_steps} x {n_columns} = {n_perstep} "
                          f"per-step maxima",
            "note": f"{total} scalars is {total * 8 / 1024:.1f} KiB per replicate"},
        "photon_storage": {
            "persistent_per_photon_products_required": False,
            "but": "option A requires that, for a given photon index, the contributions at ALL "
                   "FOUR step sizes and BOTH arms be available simultaneously so that Z can be "
                   "formed before squaring. That is a live-coupling requirement on the execution "
                   "order, not a persistence requirement.",
            "if_the_architecture_cannot_do_that": "option A is NOT implementable as written and "
                                                  "the design must fall back to option B, paying "
                                                  "the explicit step multiplicity in alpha. This "
                                                  "is an open implementation question, not a "
                                                  "settled one.",
            "the_earlier_unconditional_claim_is_withdrawn":
                "an earlier draft said no per-photon storage is required, with no condition "
                "attached. The correct statement is the two entries above."},
    }
# ---------------------------------------------------------------------------------------------
# The deterministic range, and its verification
# ---------------------------------------------------------------------------------------------

def deterministic_range_width(h, areas, weight_ceiling=WEIGHT_CEILING):
    """The INTERVAL WIDTH R_i(h) = weight_ceiling / (h A_i), derived from the kernel.

    With a = w+ 1[b+ = i] and b = w- 1[b- = i], both in [0, weight_ceiling], the per-photon
    contribution Y = (a - b) / (2 h A_i) lies in

        [ -B_i(h), +B_i(h) ]   with   B_i(h) = weight_ceiling / (2 h A_i)

    so the ABSOLUTE BOUND is B and the INTERVAL WIDTH is R = 2B = weight_ceiling / (h A_i).

    The distinction matters because Maurer and Pontil's Theorem 4 is stated for variables in [0,1]
    and its 7/(3(n-1)) term scales with the WIDTH of the interval, not with the absolute bound. The
    two differ by a factor of two and conflating them halves the range term.

    This function returns the WIDTH. absolute_bound() returns B.
    """
    h = float(h)
    if not (h > 0.0):
        raise ValueError("the step size must be strictly positive")
    a = np.asarray(areas, float)
    if np.any(a <= 0.0):
        raise ValueError("every bin area must be strictly positive")
    return weight_ceiling / (h * a)


def absolute_bound(h, areas, weight_ceiling=WEIGHT_CEILING):
    """B_i(h) = R_i(h) / 2, the almost-sure bound on |Y|."""
    return 0.5 * deterministic_range_width(h, areas, weight_ceiling)


def verify_range(observed_max_abs_Y, range_width):
    """Check the observed max |Y| against the ABSOLUTE BOUND B = range_width / 2.

    The observed maximum is compared against B, not against the width: |Y| <= B is the almost-sure
    statement, and the width is what enters the Bernstein range term. A violation is a refusal; the
    range is never widened to accommodate the data.
    """
    o = np.asarray(observed_max_abs_Y, float)
    B = 0.5 * np.asarray(range_width, float)
    bad = np.flatnonzero(o > B * (1.0 + 1e-12))
    return {"violated": bool(bad.size), "n_violating_bins": int(bad.size),
            "violating_bins": bad.tolist()[:20],
            "compared_against": "the absolute bound B = range_width / 2",
            "worst_ratio": float(np.max(o / np.maximum(B, np.finfo(float).tiny)))
            if o.size else 0.0,
            "action_if_violated": "REFUSE. The deterministic range rests on a kernel invariant; a "
                                  "violation means the invariant is broken, so the bound does not "
                                  "apply, and the absolute bound B must not be re-derived from "
                                  "the data."}


# ---------------------------------------------------------------------------------------------
# Empirical Bernstein radius
# ---------------------------------------------------------------------------------------------

def empirical_bernstein_radius(sum_Y, sum_Y2, N, range_width, alpha):
    """Two-sided Maurer-Pontil radius on the mean, from accumulators only.

    radius = sqrt(2 Vhat ln(2/a)/N) + 7 R ln(2/a)/(3(N-1)),  a = alpha/2 per side,
    Vhat = (sum_Y2 - (sum_Y)^2/N)/(N-1),  R the INTERVAL WIDTH.

    R is the width of the almost-sure interval, not the absolute bound: Maurer and Pontil state
    Theorem 4 for [0,1] and the 7/(3(n-1)) term scales with the width under an affine change of
    variable. For Y in [-B, B] the width is 2B, so passing B here would halve the range term.

    Not distribution free: it assumes Y lies in an interval of width R almost surely, with R
    derived from the kernel, and assumes nothing about the shape within it.
    """
    N = int(N)
    if N < 2:
        raise ValueError("at least two launched photons are required for a variance")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie strictly between 0 and 1")
    sY = np.asarray(sum_Y, float)
    sY2 = np.asarray(sum_Y2, float)
    R = np.asarray(range_width, float)
    if np.any(R <= 0.0):
        raise ValueError("the interval width must be strictly positive")
    mean = sY / N
    vhat = np.maximum((sY2 - sY * sY / N) / (N - 1), 0.0)
    a = alpha / 2.0
    L = np.log(2.0 / a)
    return {"mean": mean, "variance_estimate": vhat,
            "radius": np.sqrt(2.0 * vhat * L / N) + 7.0 * R * L / (3.0 * (N - 1)),
            "variance_term": np.sqrt(2.0 * vhat * L / N),
            "range_term": 7.0 * R * L / (3.0 * (N - 1)),
            "range_width_used": R, "implied_absolute_bound": 0.5 * R,
            "alpha_two_sided": alpha, "N": N,
            "assumes": "Y lies in an interval of WIDTH range_width almost surely, equivalently "
                       "|Y| <= range_width/2, with the width derived from the kernel",
            "range_term_scales_with_the_width_not_the_absolute_bound": True,
            "is_distribution_free": False}


# ---------------------------------------------------------------------------------------------
# From the four D(h) to e2, e3 and d: the propagation, which must be frozen
# ---------------------------------------------------------------------------------------------

def fit_rows(h_grid):
    """q2, q3 and qd: the intercept rows of the two-term and three-term fits, and their difference.

    e2_i = sum_l q2_l D_i(h_l),  e3_i = sum_l q3_l D_i(h_l),  d_i = sum_l qd_l D_i(h_l),
    with qd = q3 - q2. These are FIXED by the step grid alone and contain no data.
    """
    h2 = np.asarray(h_grid, float) ** 2
    q2 = np.linalg.pinv(np.vander(h2, 2, increasing=True))[0]
    q3 = np.linalg.pinv(np.vander(h2, 3, increasing=True))[0]
    return {"q2": q2, "q3": q3, "qd": q3 - q2,
            "abs_sum_q2": float(np.abs(q2).sum()),
            "abs_sum_qd": float(np.abs(q3 - q2).sum()),
            "note": "fixed by the step grid; no data enters"}


PROPAGATION_OPTION = "A"
PROPAGATION_CHOICE = {
    "frozen_choice": "A",
    "A_direct_combined_contribution": {
        "what": "accumulate the per-photon COMBINED contribution directly: "
                "Z_e2,p,i = sum_l q2_l Y_p,i(h_l) and Z_d,p,i = sum_l qd_l Y_p,i(h_l), "
                "with sum_Z, sum_Z2 and max_abs_Z for each",
        "why_chosen": "it gives a genuine empirical-Bernstein interval on e2 and on d themselves. "
                      "The per-photon cross-step covariance is captured because the combination is "
                      "formed BEFORE squaring.",
        "derived_range_width": "W_Z,i = 2 * B_Z,i with B_Z,i <= sum_l |q_l| * B_i(h_l); the "
                               "triangle inequality on the almost-sure per-step bounds",
        "alpha_cost": "no extra step multiplicity: e2 and d are each one quantity per bin",
        "executor_cost": "three extra accumulators per bin per column for each of Z_e2 and Z_d, "
                         "replacing the four per-step sets rather than adding to them"},
    "B_per_step_then_propagate": {
        "what": "bound each of the four D(h) separately and propagate "
                "eps_e2,i = sum_l |q2_l| eps_i(h_l),  eps_d,i = sum_l |qd_l| eps_i(h_l)",
        "why_not_chosen": "valid but looser, and it must pay an explicit step multiplicity in the "
                          "alpha allocation: four per-step statements per bin instead of one",
        "alpha_cost": "the per-bin allocation must be divided by the number of step sizes",
        "kept_as": "the documented fallback if the executor cannot form the combination inline"},
    "what_the_current_accumulators_cannot_do": (
        "four independent sum_Y2(h) values CANNOT recover the sample variance of the combination, "
        "because the per-photon cross-step terms are missing. An empirical-Bernstein variance for "
        "d therefore does NOT follow from them, and the earlier draft's use of a d_radius without "
        "saying where it came from was unjustified."),
}


def combined_range_width(h_grid, areas, q, weight_ceiling=WEIGHT_CEILING):
    """Width of the almost-sure interval for Z = sum_l q_l Y(h_l), from the triangle inequality.

    |Z| <= sum_l |q_l| B_i(h_l), so the width is twice that.
    """
    q = np.asarray(q, float)
    acc = np.zeros_like(np.asarray(areas, float))
    for ql, hl in zip(q, np.asarray(h_grid, float)):
        acc = acc + abs(float(ql)) * absolute_bound(hl, areas, weight_ceiling)
    return 2.0 * acc


def propagate_option_B(h_grid, per_step_radii, q):
    """eps = sum_l |q_l| eps(h_l), the documented fallback. Requires step multiplicity in alpha."""
    q = np.asarray(q, float)
    E = np.asarray(per_step_radii, float)
    if E.shape[0] != q.size:
        raise ValueError("one radius vector per step size is required")
    return {"radius": np.abs(q)[:, None].T @ E if False else (np.abs(q)[:, None] * E).sum(axis=0),
            "alpha_must_include_step_multiplicity": True,
            "n_steps": int(q.size)}


# ---------------------------------------------------------------------------------------------
# The two legs, and the ratio with a random denominator
# ---------------------------------------------------------------------------------------------

def box_upper(W, eps):
    """sup over the box |z_i| <= eps_i of ||W z||_2, bounded by || |W| eps ||_2.

    The naive ||W eps||_2 is WRONG for a dense whitening with mixed signs, because the worst z is
    not eps itself: signs inside a row can cancel in W eps while reinforcing for some other z in
    the box. Two-dimensional counterexample, W = [[1,-1],[0,1]] and eps = [1,1]: the naive value is
    ||W eps|| = 1, while z = [1,-1] is in the box and gives ||W z|| = sqrt(5).

    The bound here is valid because |(W z)_i| = |sum_j W_ij z_j| <= sum_j |W_ij| eps_j, so
    ||W z||_2 <= || |W| eps ||_2 componentwise. It is attained in the example above. A tighter
    maximisation over the box is possible and is not required for validity.
    """
    W = np.asarray(W, float)
    e = np.abs(np.asarray(eps, float))
    if W.ndim == 1:
        return float(np.linalg.norm(np.abs(W) * e))
    return float(np.linalg.norm(np.abs(W) @ e))


def aggregate_ratio_bound(d_mean, d_radius, e2_mean, e2_radius, W):
    """rho_upper = (||W d|| + box(eps_d)) / (||W e2|| - box(eps_e2)), fail-closed if that is <= 0.

    A union bound over the numerator and the denominator, so their dependence does not need to be
    modelled. The radii enter through box_upper, NOT through ||W eps||, which does not cover the
    box for a mixed-sign dense whitening.
    """
    W = np.asarray(W, float)

    def nrm(v):
        v = np.asarray(v, float)
        return float(np.linalg.norm(v @ W.T if W.ndim == 2 else v * W))

    num_upper = nrm(d_mean) + box_upper(W, d_radius)
    den_lower = nrm(e2_mean) - box_upper(W, e2_radius)
    if den_lower <= 0.0:
        return {"reportable": False,
                "reason": "the simultaneous lower bound on the whitened reference norm is not "
                          "positive, so the ratio is not reportable. The point denominator is NOT "
                          "substituted.",
                "numerator_upper": num_upper, "denominator_lower": den_lower,
                "level": ALPHA_NUMERATOR + ALPHA_DENOMINATOR}
    return {"reportable": True, "numerator_upper": num_upper, "denominator_lower": den_lower,
            "ratio_upper": num_upper / den_lower,
            "level": ALPHA_NUMERATOR + ALPHA_DENOMINATOR,
            "construction": "union bound over numerator and denominator; dependence not modelled "
                            "because it does not need to be",
            "radius_enters_through": "box_upper(W, eps) = || |W| eps ||_2, which covers the box "
                                     "|z| <= eps. ||W eps||_2 does NOT and is not used."}


D_RADIUS_POLICY = {
    "choice": "SEPARATE radii, computed at each leg's own level",
    "aggregate_numerator": {"alpha": ALPHA_NUMERATOR,
                            "multiplicity": "divided over the retained components by union bound",
                            "symbol": "eps_d_num"},
    "component_leg": {"alpha": ALPHA_COMPONENT,
                      "multiplicity": "divided over the retained components by union bound",
                      "symbol": "eps_d_component"},
    "why_not_reuse_one": "a radius computed at 0.004 is not a valid 0.003-level statement, and "
                         "reusing it for the component leg would spend the numerator's budget "
                         "twice while claiming the component's. The two legs have separate "
                         "allocations precisely so each carries its own radius.",
    "the_alternative_considered": "a single shared family at the stricter of the two levels, "
                                  "min(0.004, 0.003) = 0.003, counted ONCE. That is also valid "
                                  "and slightly tighter in total, but it couples the two legs' "
                                  "budgets, so the separate form is chosen for auditability.",
    "eps_d_component_is_at_least_eps_d_num": "true, because 0.003 < 0.004 makes the component "
                                             "radius the wider of the two",
}


def d_radii(sum_Z_d, sum_Z2_d, N, range_width_Z_d, n_retained):
    """The TWO radii for d, one per leg, each at its own level with its own multiplicity."""
    m = max(int(n_retained), 1)
    num = empirical_bernstein_radius(sum_Z_d, sum_Z2_d, N, range_width_Z_d,
                                     ALPHA_NUMERATOR / m)
    comp = empirical_bernstein_radius(sum_Z_d, sum_Z2_d, N, range_width_Z_d,
                                      ALPHA_COMPONENT / m)
    return {"eps_d_num": num["radius"], "eps_d_component": comp["radius"],
            "alpha_num_per_component": ALPHA_NUMERATOR / m,
            "alpha_component_per_component": ALPHA_COMPONENT / m,
            "n_retained": m,
            "policy": D_RADIUS_POLICY["choice"],
            "never_reused_across_legs": True}


def component_bound(d_mean, d_radius, C_diag):
    """max_i (|d_i| + eps_i) / sqrt(C_ii), in PHYSICAL BIN coordinates.

    Not max_i |(W d)_i|: for a block-dense whitening that mixes bins, and its maximum moves under
    an orthogonal rotation of the whitening square root while the aggregate norm does not.
    """
    sC = np.sqrt(np.maximum(np.asarray(C_diag, float), np.finfo(float).tiny))
    v = (np.abs(np.asarray(d_mean, float)) + np.abs(np.asarray(d_radius, float))) / sC
    return {"component_upper": float(np.max(v)),
            "argmax_bin": int(np.argmax(v)),
            "coordinates": "physical bins divided by sqrt(C_ii); basis free",
            "level": ALPHA_COMPONENT}


# ---------------------------------------------------------------------------------------------
# The DRAFT adversarial suite skeleton
# ---------------------------------------------------------------------------------------------

DRAFT_SUITE_SKELETON = {
    "status": "DRAFT SKELETON, NOT FROZEN. This is a catalogue of nine scenario intentions. It has "
              "no executable generator, no fixed seed, no trial count, no margins, no step grid "
              "and no support policy, so it cannot be run and cannot be frozen. Calling it frozen "
              "earlier was wrong.",
    "not_runnable_until": [
        "each scenario has an executable generator with its parameters fixed",
        "each scenario has a recorded seed and trial count",
        "the aggregate and component margins are set",
        "the step grid, support policy and replicate count are set",
        "the stopping rule is set",
        "the four construction corrections are in place and tested"],
    "freeze_hash_withheld": "no hash is emitted for a skeleton. A hash over an unrunnable "
                            "catalogue would imply a commitment that does not exist.",
    "two_levels_that_must_not_be_confused": {
        "alpha": {"value": ALPHA_TOTAL,
                  "what_it_is": "the STATISTICAL TARGET of the construction. The claim is that "
                                "under every admissible distribution the JOINT coverage of both "
                                "legs is at least 1 - alpha. This target is the SAME in every "
                                "scenario and is never rewritten as alpha/S.",
                  "per_scenario_target": 1.0 - ALPHA_TOTAL},
        "beta": {"value": 0.05,
                 "what_it_is": "the VALIDATION CONFIDENCE error rate. Nine scenarios are checked "
                               "at once, so each scenario's EXACT one-sided Clopper-Pearson bound "
                               "is taken at error rate beta/S, and the union over the family gives "
                               "a genuine validation confidence of 1 - beta.",
                 "bound_family": "exact one-sided Clopper-Pearson",
                 "why_exactness_matters": "a union bound over approximate intervals gives only a "
                                          "nominal family-wise statement. Wilson is a normal "
                                          "approximation and is therefore diagnostic only."},
        "the_error_to_avoid": "writing the target coverage as 1 - alpha/S. That changes what the "
                              "construction claims. The multiplicity belongs to the VALIDATION "
                              "confidence, not to the statistical target."},
    "judged_on": "JOINT coverage of the aggregate and component legs together, never the two "
                 "marginals separately",
    "judged_by": "an EXACT one-sided Clopper-Pearson bound at error rate beta/S per scenario, "
                 "compared against the FIXED target 1 - alpha. Coverage uses the exact lower "
                 "bound; the equivalence rate uses the exact upper bound. The point estimate is "
                 "never the judge, and a Wilson interval is a descriptive diagnostic that may not "
                 "decide acceptance and is not a family-wise guarantee.",
    "scenario_intentions": [
        {"name": "bernoulli_rare_event",
         "why": "the mechanism that refuted candidate 2's heuristic: every replicate zero",
         "generator": "one component, Bernoulli(p) per observation, true mean p on the margin",
         "must_survive": "no exception may be carved out for all-zero samples"},
        {"name": "compound_poisson_with_K_zero",
         "why": "a bin with no contributing photon must give an exact zero and a valid wide "
                "interval, not a refusal and not a narrow one",
         "generator": "K ~ Poisson(lambda) with lambda well below 1, lognormal weights",
         "must_survive": "coverage holds at the same level as the high-count bins"},
        {"name": "lognormal_heavy_tail_weights",
         "why": "survival weights span orders of magnitude, so the variance term must carry it",
         "generator": "lognormal(0, sigma) weights with sigma up to 2.5"},
        {"name": "moving_boundary_events",
         "why": "a photon whose radial bin changes between the two arms contributes to two bins "
                "with opposite signs, which is the heaviest tail in the problem",
         "generator": "a registered flip probability per photon, with the contribution split "
                      "across the two bins exactly as section 1 defines it"},
        {"name": "imperfect_common_random_numbers",
         "why": "the two arms share photons only partially, which is where the noise growth "
                "toward small h comes from",
         "generator": "a decorrelated fraction whose amplitude is a scenario parameter"},
        {"name": "component_selection_and_random_denominator",
         "why": "the component leg takes a maximum over bins and the ratio has a random "
                "denominator; both are selection effects the bound must already cover",
         "generator": "truth placed at the bin the selection is most likely to pick, with the "
                      "denominator estimated from the same photons"},
        {"name": "equivalence_boundary_inside",
         "why": "power at a truth safely inside the margins",
         "generator": "known truth at a registered fraction of each margin"},
        {"name": "equivalence_boundary_on",
         "why": "type I error exactly on the margin, the hardest case for an equivalence claim",
         "generator": "known truth exactly on each margin, solved in the full space as the "
                      "feasibility construction does"},
        {"name": "equivalence_boundary_outside",
         "why": "type I error well outside, the error that would matter most",
         "generator": "known truth at twice each margin"},
    ],
    "acceptance_criteria_drafted": [
        "JOINT coverage target 1 - alpha in EVERY scenario, identical across scenarios; the "
        "scenario count enters only through the validation confidence, never through the target",
        "for each scenario, the EXACT one-sided Clopper-Pearson lower bound at error rate beta/S "
        "must reach 1 - alpha; exactness is what makes the union over S scenarios a genuine "
        "family-wise 1 - beta rather than a nominal one",
        "equivalence rate target at most alpha at every truth at or beyond a margin, judged by the "
        "EXACT one-sided Clopper-Pearson upper bound at error rate beta/S; alpha is the "
        "statistical target and beta/S is the validation confidence, and the two are never merged",
        "a Wilson interval may be reported as a descriptive diagnostic but may NOT decide "
        "acceptance and may not be called exact or family-wise",
        "no observation of abs(Y) above the derived absolute bound B_i(h), and none of abs(Z) "
        "above B_Z, in any scenario",
        "the rare-event scenario judged with no carve-out",
        "power reported per scenario and never used to justify relaxing the above",
        "a failure rejects the construction; no threshold, allocation or scenario may be changed "
        "after seeing a result",
    ],
    "what_passing_would_mean": "a candidate for registration. Not registration, and not "
                              "authorization of a gate, a margin, a grid or the downstream study.",
    "nothing_here_has_been_run": True,
}


VALIDATION_BETA = 0.05


def validation_bound_level(n_scenarios, beta=VALIDATION_BETA):
    """beta/S: the per-scenario ERROR RATE of the VALIDATION bound, not the statistical target.

    The statistical target stays 1 - alpha in every scenario. Only the confidence with which the
    validation claims to have checked it is split across the family, and the split is valid because
    the per-scenario bounds are EXACT one-sided Clopper-Pearson bounds: a union bound over S exact
    statements at beta/S each gives a genuine family-wise 1 - beta. The same split over Wilson
    intervals would give only a nominal statement, since Wilson is a normal approximation.
    """
    S = int(n_scenarios)
    if S < 1:
        raise ValueError("at least one scenario is required")
    return {"beta": float(beta), "n_scenarios": S,
            "per_scenario_error_rate": float(beta) / S,
            "bound_family": "exact one-sided Clopper-Pearson",
            "family_wise_validation_confidence": 1.0 - float(beta),
            "why_the_split_is_valid": "a union bound over S EXACT one-sided statements, each at "
                                      "error rate beta/S. Wilson intervals would make this "
                                      "nominal rather than guaranteed and may not decide "
                                      "acceptance.",
            "wilson_role": "descriptive diagnostic only; never the judge",
            "statistical_target_unchanged": 1.0 - ALPHA_TOTAL,
            "not_the_target": "the target coverage is 1 - alpha in every scenario and is NOT "
                              "1 - alpha/S"}


def _log_binom_tail(k, n, p, upper):
    """log of a binomial tail, in log space so large n does not overflow.

    upper=True  -> log P(X >= k);  upper=False -> log P(X <= k).
    """
    import math
    if p <= 0.0:
        return 0.0 if (not upper and k >= 0) or (upper and k <= 0) else -math.inf
    if p >= 1.0:
        return 0.0 if (upper and k <= n) or (not upper and k >= n) else -math.inf
    lp, lq = math.log(p), math.log1p(-p)
    rng_ = range(k, n + 1) if upper else range(0, k + 1)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
             + i * lp + (n - i) * lq for i in rng_]
    m = max(terms)
    return m + math.log(sum(math.exp(t - m) for t in terms))


def clopper_pearson_lower(k, n, error_rate):
    """EXACT one-sided lower confidence bound on a binomial proportion.

    p_L is the p solving P(X >= k | p) = error_rate, and 0 when k = 0. Exact in the
    Clopper-Pearson sense: the coverage of the one-sided statement is at least 1 - error_rate for
    every p, with no normal approximation anywhere. Solved by bisection on the binomial tail in log
    space rather than through an inverse Beta, so there is no library dependency.
    """
    import math
    k, n = int(k), int(n)
    if not (0 <= k <= n) or n < 1:
        raise ValueError("require 0 <= k <= n and n >= 1")
    if not (0.0 < error_rate < 1.0):
        raise ValueError("error_rate must lie strictly between 0 and 1")
    if k == 0:
        return 0.0
    target = math.log(error_rate)
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _log_binom_tail(k, n, mid, upper=True) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clopper_pearson_upper(k, n, error_rate):
    """EXACT one-sided upper confidence bound on a binomial proportion.

    p_U is the p solving P(X <= k | p) = error_rate, and 1 when k = n.
    """
    import math
    k, n = int(k), int(n)
    if not (0 <= k <= n) or n < 1:
        raise ValueError("require 0 <= k <= n and n >= 1")
    if not (0.0 < error_rate < 1.0):
        raise ValueError("error_rate must lie strictly between 0 and 1")
    if k == n:
        return 1.0
    target = math.log(error_rate)
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _log_binom_tail(k, n, mid, upper=False) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def wilson_bounds(k, n, error_rate):
    """Wilson score interval. DESCRIPTIVE DIAGNOSTIC ONLY.

    The Wilson interval is a normal approximation. Its coverage is nominal rather than guaranteed,
    so it may NOT decide acceptance and may not be described as exact or as supplying a family-wise
    guarantee. It is reported beside the Clopper-Pearson bounds because it is tighter and useful for
    reading a result, not for judging one.
    """
    import math
    n = int(n)
    if n <= 0:
        return (None, None)
    target = 1.0 - error_rate / 2.0
    lo_z, hi_z = 0.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo_z + hi_z)
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < target:
            lo_z = mid
        else:
            hi_z = mid
    z = 0.5 * (lo_z + hi_z)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, c - hw)), float(min(1.0, c + hw)))


def suite_freeze_hash():
    """Refuses. A skeleton is not frozen, so no freeze hash is issued.

    A hash is emitted only once every scenario carries a generator, parameters, a seed, a trial
    count and a stopping rule, and the margins, grid and support policy are set.
    """
    missing = [k for k in ("generator_callable", "seed", "trials")
               if not all(k in x for x in DRAFT_SUITE_SKELETON["scenario_intentions"])]
    raise RuntimeError(
        "REFUSING to issue a freeze hash: the suite is a DRAFT SKELETON. Missing from every "
        f"scenario: {missing}. See DRAFT_SUITE_SKELETON['not_runnable_until'].")


def describe():
    return {"status": DESIGN_ONLY,
            "specification": os.path.relpath(SPEC_DOC, ROOT),
            "specification_sha256": _sha256_file(SPEC_DOC),
            "executor_interface": executor_interface(),
            "alpha_allocation": {
                "status": "PRE-SPECIFIED DRAFT PROPOSAL, not registered",
                "total": ALPHA_TOTAL, "per_column": ALPHA_PER_COLUMN,
                "numerator": ALPHA_NUMERATOR, "denominator": ALPHA_DENOMINATOR,
                "component": ALPHA_COMPONENT,
                "within_column_sums_to_budget": abs(
                    ALPHA_NUMERATOR + ALPHA_DENOMINATOR + ALPHA_COMPONENT
                    - ALPHA_PER_COLUMN) < 1e-12,
                "written_before_any_result": True,
                "is_registered": False,
                "not_optimised_against_data": True},
            "unset_slots": {"aggregate_margin": AGGREGATE_MARGIN,
                            "component_margin": COMPONENT_MARGIN,
                            "step_grid": STEP_GRID, "replicates": REPLICATES,
                            "support_policy": SUPPORT_POLICY,
                            "stopping_rule": STOPPING_RULE,
                            "whitening_likelihood": WHITENING_LIKELIHOOD},
            "draft_suite_skeleton": DRAFT_SUITE_SKELETON,
            "validation_confidence": validation_bound_level(
                len(DRAFT_SUITE_SKELETON["scenario_intentions"])),
            "d_radius_policy": D_RADIUS_POLICY,
            "cross_step_pairing_invariant": CROSS_STEP_PAIRING_INVARIANT,
            "suite_is_frozen": False,
            "suite_freeze_hash": None,
            "propagation": PROPAGATION_CHOICE,
            "authorises": {"a_gate": False, "a_margin": False, "a_grid": False,
                           "the_48_draw_study": False, "gpu_work": False,
                           "registration": False}}


# ---------------------------------------------------------------------------------------------
# CPU tests of what can be settled now
# ---------------------------------------------------------------------------------------------

def duplicate_top_level_symbols(path=None):
    """AST scan for a top-level name defined twice in one file.

    A later definition silently shadows an earlier one, so the tests exercise only the last and a
    duplicate is invisible to them. This happened in this very file: eight names, including
    CROSS_STEP_PAIRING_INVARIANT and kernel_weight_assertions, were defined twice and every test
    still passed because it ran the second copy.
    """
    import ast
    import collections
    path = path or os.path.abspath(__file__)
    with open(path) as fh:
        tree = ast.parse(fh.read())
    seen = collections.defaultdict(list)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            seen[node.name].append(node.lineno)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id.isupper():
                    seen[tgt.id].append(node.lineno)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id.isupper():
                seen[node.target.id].append(node.lineno)
    return {name: lines for name, lines in seen.items() if len(lines) > 1}


def selftest():
    res, fails = [], []

    def ck(name, ok, detail=""):
        res.append({"name": name, "pass": bool(ok)})
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
        if not ok:
            fails.append(name)

    # --- no top-level name may be defined twice ------------------------------------------------
    dups = duplicate_top_level_symbols()
    ck("no top-level function, class or constant is defined twice in this file",
       not dups, f"duplicates {dups}" if dups else "AST scan over every top-level definition")
    with open(os.path.abspath(__file__)) as fh:
        _n_top = len([x for x in __import__("ast").parse(fh.read()).body
                      if isinstance(x, (__import__("ast").FunctionDef,
                                        __import__("ast").ClassDef))])
    ck("the duplicate scan actually sees the file's definitions, so a pass is not vacuous",
       _n_top >= 15, f"{_n_top} top-level functions and classes scanned")
    _probe = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_dup_probe_tmp.py")
    try:
        with open(_probe, "w") as fh:
            fh.write("A = 1\nA = 2\ndef f():\n    pass\ndef f():\n    pass\n")
        planted = duplicate_top_level_symbols(_probe)
    finally:
        if os.path.exists(_probe):
            os.remove(_probe)
    ck("the scanner catches a planted duplicate",
       planted == {"A": [1, 2], "f": [3, 5]},
       "a planted file with A twice and f twice is reported")
    ck("the probe file is removed afterwards", not os.path.exists(_probe))

    # --- the design is design only -------------------------------------------------------------
    d = describe()
    ck("nothing is registered and every production slot is unset",
       all(v is None for v in d["unset_slots"].values())
       and not any(d["authorises"].values()),
       f"{len(d['unset_slots'])} slots unset, nothing authorised")
    ck("the draft alpha allocation is pre-specified and sums to its proposed per-column budget",
       d["alpha_allocation"]["within_column_sums_to_budget"]
       and d["alpha_allocation"]["is_registered"] is False
       and d["alpha_allocation"]["status"].startswith("PRE-SPECIFIED DRAFT PROPOSAL")
       and abs(ALPHA_PER_COLUMN * N_COLUMNS - ALPHA_TOTAL) < 1e-12,
       f"{ALPHA_NUMERATOR} + {ALPHA_DENOMINATOR} + {ALPHA_COMPONENT} = {ALPHA_PER_COLUMN}, "
       f"not registered")
    ck("the specification document exists and is hashed",
       d["specification_sha256"] is not None, str(d["specification_sha256"])[:16])

    # --- the deterministic range ---------------------------------------------------------------
    edges = np.linspace(0.0, 1.5, 31)
    areas = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    R = deterministic_range_width(0.1, areas)
    ck("the width is 1/(h A_i) and spans the bin areas, not a constant",
       np.allclose(R, 1.0 / (0.1 * areas))
       and abs(R.max() / R.min() - areas.max() / areas.min()) < 1e-9,
       f"width from {R.min():.3g} to {R.max():.3g}, ratio {R.max() / R.min():.1f}")
    ck("halving the step doubles the width",
       np.allclose(deterministic_range_width(0.05, areas), 2.0 * R))
    for bad_h in (0.0, -0.1):
        try:
            deterministic_range_width(bad_h, areas)
            ok = False
        except ValueError:
            ok = True
        ck(f"a step size of {bad_h} is refused", ok)
    try:
        deterministic_range_width(0.1, np.array([1.0, 0.0]))
        ok = False
    except ValueError:
        ok = True
    ck("a zero bin area is refused rather than producing an infinite width", ok)

    # the range must never be taken from the data
    v = verify_range(R, R)
    ck("an observed value above the absolute bound is a refusal, not a new range",
       verify_range(R, R)["violated"]
       and "REFUSE" in v["action_if_violated"]
       and "must not be re-derived from the data" in v["action_if_violated"],
       f"{v['n_violating_bins']} bins violating at the full width")

    # --- the empirical Bernstein radius against its own definition ------------------------------
    rng = np.random.default_rng(20261010)
    N = 20000
    Y = rng.uniform(-1.0, 1.0, size=N)
    r = empirical_bernstein_radius(Y.sum(), (Y ** 2).sum(), N, 2.0, 0.05)
    vhat_direct = float(Y.var(ddof=1))
    ck("the variance estimate from accumulators matches the direct sample variance",
       abs(float(r["variance_estimate"]) - vhat_direct) < 1e-9 * max(vhat_direct, 1.0),
       f"{float(r['variance_estimate']):.10f} vs {vhat_direct:.10f}")
    L = np.log(2.0 / (0.05 / 2.0))
    ck("the radius equals its definition term by term",
       abs(float(r["variance_term"]) - np.sqrt(2.0 * vhat_direct * L / N)) < 1e-12
       and abs(float(r["range_term"]) - 7.0 * 2.0 * L / (3.0 * (N - 1))) < 1e-12
       and abs(float(r["radius"]) - float(r["variance_term"]) - float(r["range_term"])) < 1e-12)
    ck("the radius is not claimed to be distribution free",
       r["is_distribution_free"] is False and "almost surely" in r["assumes"])
    r_big = empirical_bernstein_radius(Y.sum(), (Y ** 2).sum(), N, 200.0, 0.05)
    ck("a wider range widens only the range term, leaving the variance term alone",
       abs(float(r_big["variance_term"]) - float(r["variance_term"])) < 1e-12
       and float(r_big["range_term"]) > 50.0 * float(r["range_term"]))
    r10 = empirical_bernstein_radius(Y.sum() * 10, (Y ** 2).sum() * 10, N * 10, 2.0, 0.05)
    ck("the range term decays as R/(N-1), so a large fixed launch count is what buys precision",
       abs(float(r10["range_term"]) - 7.0 * 2.0 * L / (3.0 * (10 * N - 1))) < 1e-15
       and 0.09 < float(r10["range_term"]) / float(r["range_term"]) < 0.11,
       f"ratio {float(r10['range_term']) / float(r['range_term']):.5f} at ten times N")
    for bad in ((1, 0.05), (N, 0.0), (N, 1.0)):
        try:
            empirical_bernstein_radius(0.0, 0.0, bad[0], 2.0, bad[1])
            ok = False
        except ValueError:
            ok = True
        ck(f"N={bad[0]}, alpha={bad[1]} is refused", ok)

    # K = 0 everywhere must give an exact zero mean and a valid, wide radius
    rz = empirical_bernstein_radius(0.0, 0.0, N, float(R.max()), 0.05)
    ck("an all-zero sample gives an exact zero mean and a strictly positive radius",
       float(rz["mean"]) == 0.0 and float(rz["variance_estimate"]) == 0.0
       and float(rz["radius"]) > 0.0,
       f"radius {float(rz['radius']):.6g} from the range term alone")
    ck("that radius comes entirely from the range term, which is why K=0 is safe here",
       abs(float(rz["radius"]) - float(rz["range_term"])) < 1e-15)

    # --- the ratio with a random denominator ---------------------------------------------------
    W = np.eye(6)
    ok_r = aggregate_ratio_bound(np.full(6, 0.1), np.full(6, 0.01),
                                 np.full(6, 10.0), np.full(6, 0.1), W)
    ck("the ratio bound is numerator-upper over denominator-lower",
       ok_r["reportable"]
       and abs(ok_r["ratio_upper"]
               - (np.linalg.norm(np.full(6, 0.1)) + np.linalg.norm(np.full(6, 0.01)))
               / (np.linalg.norm(np.full(6, 10.0)) - np.linalg.norm(np.full(6, 0.1)))) < 1e-12,
       f"{ok_r['ratio_upper']:.6f}")
    ck("its level is the sum of the numerator and denominator allocations",
       abs(ok_r["level"] - (ALPHA_NUMERATOR + ALPHA_DENOMINATOR)) < 1e-12)
    bad_r = aggregate_ratio_bound(np.full(6, 0.1), np.full(6, 0.01),
                                  np.full(6, 1.0), np.full(6, 10.0), W)
    ck("a non-positive denominator lower bound fails closed and substitutes nothing",
       bad_r["reportable"] is False
       and "NOT substituted" in bad_r["reason"]
       and "ratio_upper" not in bad_r,
       f"denominator lower bound {bad_r['denominator_lower']:.4f}")

    # --- the component leg is basis free -------------------------------------------------------
    C = np.array([1.0, 4.0, 0.25, 1.0, 9.0, 1.0])
    cb = component_bound(np.array([0.1, 0.2, 0.05, 0.0, 0.3, 0.1]), np.full(6, 0.01), C)
    expect = float(np.max((np.abs([0.1, 0.2, 0.05, 0.0, 0.3, 0.1]) + 0.01) / np.sqrt(C)))
    ck("the component leg is the max of |d| plus its radius over sqrt(C_ii)",
       abs(cb["component_upper"] - expect) < 1e-12
       and "basis free" in cb["coordinates"],
       f"{cb['component_upper']:.8f}, argmax bin {cb['argmax_bin']}")
    ck("the component leg does not use the whitened-coordinate maximum",
       "physical bins" in cb["coordinates"])

    # --- the box bound, with the committed counterexample --------------------------------------
    Wc = np.array([[1.0, -1.0], [0.0, 1.0]])
    epsc = np.array([1.0, 1.0])
    zc = np.array([1.0, -1.0])
    naive = float(np.linalg.norm(Wc @ epsc))
    actual = float(np.linalg.norm(Wc @ zc))
    boxb = box_upper(Wc, epsc)
    ck("the committed counterexample shows ||W eps|| failing to cover the box",
       abs(naive - 1.0) < 1e-12 and abs(actual - np.sqrt(5.0)) < 1e-12 and actual > naive,
       f"naive {naive:.6f} but a legal error gives {actual:.6f} = sqrt(5)")
    ck("box_upper covers it, at or above sqrt(5)",
       boxb >= actual - 1e-12 and abs(boxb - np.sqrt(5.0)) < 1e-12,
       f"box_upper {boxb:.10f}")
    rng_b = np.random.default_rng(7)
    worst_gap = 0.0
    for _ in range(300):
        Wr = rng_b.normal(size=(5, 5))
        er = np.abs(rng_b.normal(size=5)) + 0.05
        zr = er * rng_b.choice([-1.0, 1.0], size=5)
        worst_gap = max(worst_gap,
                        float(np.linalg.norm(Wr @ zr)) - box_upper(Wr, er))
    ck("box_upper dominates every sampled vertex of the box",
       worst_gap <= 1e-9, f"worst excess {worst_gap:.3e} over 300 random boxes")
    ck("the ratio bound routes its radii through box_upper, not through ||W eps||",
       "box_upper(W, eps)" in aggregate_ratio_bound(
           np.zeros(2), np.zeros(2), np.ones(2), np.zeros(2), Wc)["radius_enters_through"])

    # --- range width against absolute bound ---------------------------------------------------
    Rw = deterministic_range_width(0.1, areas)
    Bb = absolute_bound(0.1, areas)
    ck("the width is twice the absolute bound, and both are reported separately",
       np.allclose(Rw, 2.0 * Bb) and np.allclose(Bb, 1.0 / (2.0 * 0.1 * areas)),
       f"B {Bb.min():.4g}..{Bb.max():.4g}, R {Rw.min():.4g}..{Rw.max():.4g}")
    ck("verify_range compares the observed maximum against the ABSOLUTE BOUND",
       verify_range(Bb * 0.999, Rw)["violated"] is False
       and verify_range(Bb * 1.001, Rw)["violated"] is True
       and "absolute bound" in verify_range(Bb, Rw)["compared_against"])
    ru = empirical_bernstein_radius(Y.sum(), (Y ** 2).sum(), N, 2.0, 0.05)
    ck("a Uniform(-1,1) sample is passed a width of 2, not 1",
       abs(float(ru["range_term"]) - 7.0 * 2.0 * L / (3.0 * (N - 1))) < 1e-15
       and abs(float(ru["implied_absolute_bound"]) - 1.0) < 1e-15,
       f"range term {float(ru['range_term']):.8g} at width 2")
    ck("passing the absolute bound instead of the width would halve the range term",
       abs(float(empirical_bernstein_radius(Y.sum(), (Y ** 2).sum(), N, 1.0,
                                            0.05)["range_term"])
           - 0.5 * float(ru["range_term"])) < 1e-15)
    ck("the radius records that its range term scales with the width",
       ru["range_term_scales_with_the_width_not_the_absolute_bound"] is True
       and "WIDTH" in ru["assumes"])
    try:
        empirical_bernstein_radius(0.0, 0.0, 100, 0.0, 0.05)
        ok = False
    except ValueError:
        ok = True
    ck("a non-positive width is refused", ok)

    # --- the real launch weight ----------------------------------------------------------------
    n_air, n0 = 1.0, 1.4
    rsp = specular_reflectance(n_air, n0)
    w0 = launch_weight(n_air, n0)
    ck("the launch weight is 1 - r_sp with the production Fresnel formula, not 1",
       abs(rsp - ((n_air - n0) / (n_air + n0)) ** 2) < 1e-15
       and abs(w0 - (1.0 - rsp)) < 1e-15 and w0 < 1.0 and 0.0 <= rsp <= 1.0,
       f"r_sp {rsp:.8f}, w0 {w0:.8f}")
    ka = kernel_weight_assertions()
    ck("the assertions no longer claim the launch weight equals 1",
       not any("launch weight equals 1 exactly" in x for x in ka)
       and any("w0 == 1 - r_sp" in x for x in ka)
       and any("0 <= w0 <= 1" in x for x in ka)
       and any("non-increasing" in x for x in ka)
       and any("no splitting" in x for x in ka)
       and any("GLOBAL CEILING" in x for x in ka),
       f"{len(ka)} assertions")
    ck("the ceiling is justified by w0 <= 1 plus non-increase, not by w0 == 1",
       any("does not assume w0 equals 1" in x for x in ka))

    # --- the fit propagation -------------------------------------------------------------------
    hg = np.array([0.4, 0.3, 0.2, 0.1])
    fr = fit_rows(hg)
    h2g = hg ** 2
    ck("q2 and q3 are the intercept rows and reproduce the fits exactly",
       abs(float(fr["q2"] @ (1.0 + 0.3 * h2g)) - 1.0) < 1e-12
       and abs(float(fr["q3"] @ (1.0 + 0.3 * h2g + 2.0 * h2g ** 2)) - 1.0) < 1e-12,
       f"sum|q2| {fr['abs_sum_q2']:.4f}, sum|qd| {fr['abs_sum_qd']:.4f}")
    ck("qd = q3 - q2 and annihilates a pure quadratic",
       np.allclose(fr["qd"], fr["q3"] - fr["q2"])
       and abs(float(fr["qd"] @ (1.0 + 0.3 * h2g))) < 1e-12)
    cw = combined_range_width(hg, areas, fr["qd"])
    tri = 2.0 * sum(abs(float(q)) * absolute_bound(float(hh), areas)
                    for q, hh in zip(fr["qd"], hg))
    ck("the combined range width is the triangle inequality on the per-step bounds",
       np.allclose(cw, tri),
       f"width {cw.min():.4g}..{cw.max():.4g}")
    ck("option A is the frozen propagation choice, and B is kept as a documented fallback",
       PROPAGATION_CHOICE["frozen_choice"] == "A"
       and "no extra step multiplicity" in PROPAGATION_CHOICE[
           "A_direct_combined_contribution"]["alpha_cost"]
       and "divided by the number of step sizes" in PROPAGATION_CHOICE[
           "B_per_step_then_propagate"]["alpha_cost"])
    ck("the record states that four independent sum_Y2 cannot give the combined variance",
       "CANNOT recover the sample variance of the combination"
       in PROPAGATION_CHOICE["what_the_current_accumulators_cannot_do"]
       and "cross-step terms are missing" in PROPAGATION_CHOICE[
           "what_the_current_accumulators_cannot_do"])
    pb = propagate_option_B(hg, np.tile(np.full(30, 0.01), (4, 1)), fr["qd"])
    ck("option B propagates with sum |q_l| eps and demands step multiplicity in alpha",
       np.allclose(pb["radius"], fr["abs_sum_qd"] * 0.01)
       and pb["alpha_must_include_step_multiplicity"] is True and pb["n_steps"] == 4)

    # --- the suite is a draft skeleton, not frozen ---------------------------------------------
    sk = DRAFT_SUITE_SKELETON
    names = [x["name"] for x in sk["scenario_intentions"]]
    required = {"bernoulli_rare_event", "compound_poisson_with_K_zero",
                "lognormal_heavy_tail_weights", "moving_boundary_events",
                "imperfect_common_random_numbers",
                "component_selection_and_random_denominator",
                "equivalence_boundary_inside", "equivalence_boundary_on",
                "equivalence_boundary_outside"}
    ck("the skeleton lists every intended scenario", required <= set(names),
       f"{len(names)} intentions")
    ck("the skeleton declares itself not frozen and not runnable",
       sk["status"].startswith("DRAFT SKELETON, NOT FROZEN")
       and len(sk["not_runnable_until"]) >= 6
       and "Calling it frozen" in sk["status"])
    ck("no scenario carries a generator, a seed or a trial count yet",
       not any(({"generator_callable", "seed", "trials"} & set(x))
               for x in sk["scenario_intentions"]))
    try:
        suite_freeze_hash()
        ok = False
    except RuntimeError as exc:
        ok = "REFUSING to issue a freeze hash" in str(exc)
    ck("asking for a freeze hash is refused while the suite is a skeleton", ok)
    ck("the describe record reports no freeze hash",
       d["suite_is_frozen"] is False and d["suite_freeze_hash"] is None
       and "draft_suite_skeleton" in d)
    tl = sk["two_levels_that_must_not_be_confused"]
    ck("the target coverage is 1 - alpha in every scenario, never 1 - alpha/S",
       abs(tl["alpha"]["per_scenario_target"] - (1.0 - ALPHA_TOTAL)) < 1e-12
       and "never rewritten as alpha/S" in tl["alpha"]["what_it_is"]
       and "1 - alpha/S" in tl["the_error_to_avoid"]
       and "belongs to the VALIDATION" in tl["the_error_to_avoid"])
    vb = validation_bound_level(len(sk["scenario_intentions"]))
    ck("the scenario multiplicity enters the validation confidence, not the target",
       abs(vb["per_scenario_error_rate"] - 0.05 / 9) < 1e-15
       and abs(vb["family_wise_validation_confidence"] - 0.95) < 1e-12
       and abs(vb["statistical_target_unchanged"] - 0.95) < 1e-12
       and "NOT" in vb["not_the_target"],
       f"beta/S = {vb['per_scenario_error_rate']:.6f}, target "
       f"{vb['statistical_target_unchanged']}")
    ck("the validation bounds are declared EXACT Clopper-Pearson, not Wilson",
       vb["bound_family"] == "exact one-sided Clopper-Pearson"
       and "union bound over S EXACT" in vb["why_the_split_is_valid"]
       and "nominal rather than guaranteed" in vb["why_the_split_is_valid"]
       and "never the judge" in vb["wilson_role"])

    # --- the exact one-sided bounds -----------------------------------------------------------
    ck("the exact lower bound matches its closed form at k = n",
       abs(clopper_pearson_lower(10, 10, 0.05) - 0.05 ** 0.1) < 1e-9,
       f"{clopper_pearson_lower(10, 10, 0.05):.10f} vs {0.05 ** 0.1:.10f}")
    ck("the exact upper bound matches its closed form at k = 0",
       abs(clopper_pearson_upper(0, 10, 0.05) - (1.0 - 0.05 ** 0.1)) < 1e-9,
       f"{clopper_pearson_upper(0, 10, 0.05):.10f} vs {1.0 - 0.05 ** 0.1:.10f}")
    ck("the endpoints are handled: k = 0 gives a lower bound of 0, k = n an upper bound of 1",
       clopper_pearson_lower(0, 50, 0.01) == 0.0
       and clopper_pearson_upper(50, 50, 0.01) == 1.0)
    lo_seq = [clopper_pearson_lower(k, 20, 0.05) for k in range(21)]
    up_seq = [clopper_pearson_upper(k, 20, 0.05) for k in range(21)]
    ck("both bounds are monotone non-decreasing in k",
       all(lo_seq[i] <= lo_seq[i + 1] + 1e-12 for i in range(20))
       and all(up_seq[i] <= up_seq[i + 1] + 1e-12 for i in range(20)),
       f"lower {lo_seq[0]:.3f}..{lo_seq[-1]:.3f}, upper {up_seq[0]:.3f}..{up_seq[-1]:.3f}")
    ck("the bounds bracket the point estimate",
       all(lo_seq[k] - 1e-12 <= k / 20.0 and k / 20.0 <= up_seq[k] + 1e-12
           for k in range(21)))
    ck("a smaller error rate widens both bounds outward",
       clopper_pearson_lower(15, 20, 0.001) < clopper_pearson_lower(15, 20, 0.05)
       and clopper_pearson_upper(5, 20, 0.001) > clopper_pearson_upper(5, 20, 0.05))
    ck("the exact lower bound is at or below the Wilson lower end, as a conservative bound must be",
       clopper_pearson_lower(387, 400, 0.025)
       <= wilson_bounds(387, 400, 0.05)[0] + 1e-12,
       f"exact {clopper_pearson_lower(387, 400, 0.025):.6f} vs Wilson "
       f"{wilson_bounds(387, 400, 0.05)[0]:.6f}")
    for bad in ((-1, 10, 0.05), (11, 10, 0.05), (5, 0, 0.05), (5, 10, 0.0), (5, 10, 1.0)):
        for fn in (clopper_pearson_lower, clopper_pearson_upper):
            try:
                fn(*bad)
                ok = False
            except ValueError:
                ok = True
            if not ok:
                break
        ck(f"the bounds refuse k={bad[0]}, n={bad[1]}, error_rate={bad[2]}", ok)
    ck("the exact bounds take an error_rate, not an ambiguous level",
       "error_rate" in clopper_pearson_lower.__code__.co_varnames
       and "error_rate" in clopper_pearson_upper.__code__.co_varnames
       and "error_rate" in wilson_bounds.__code__.co_varnames)
    ck("Wilson is documented as a descriptive diagnostic that cannot decide acceptance",
       "DESCRIPTIVE DIAGNOSTIC ONLY" in (wilson_bounds.__doc__ or "")
       and "may NOT decide acceptance" in (wilson_bounds.__doc__ or "")
       and "not for judging one" in (wilson_bounds.__doc__ or ""))

    ck("the drafted criteria require JOINT coverage judged by an exact bound at beta/S",
       "JOINT coverage" in sk["judged_on"] and "beta/S" in sk["judged_by"]
       and "EXACT one-sided Clopper-Pearson" in sk["judged_by"]
       and "may not decide acceptance" in sk["judged_by"]
       and "point estimate is never the judge" in sk["judged_by"]
       and any("never through the target" in c for c in sk["acceptance_criteria_drafted"])
       and any("genuine" in c and "family-wise 1 - beta" in c
               for c in sk["acceptance_criteria_drafted"])
       and any("may NOT decide" in c for c in sk["acceptance_criteria_drafted"])
       and not any("1 - alpha/S must" in c for c in sk["acceptance_criteria_drafted"]))
    ck("the Wilson helper reproduces the recorded example, as a diagnostic",
       (lambda lo, hi: abs(lo - 0.9452) < 1e-3 and abs(hi - 0.9809) < 1e-3)(
           *wilson_bounds(387, 400, 0.05)),
       f"387/400 at 0.05 -> {[round(x, 4) for x in wilson_bounds(387, 400, 0.05)]}")
    ck("nothing in the skeleton has been run", sk["nothing_here_has_been_run"] is True)

    # --- the executor interface must match the chosen propagation ------------------------------
    ei = executor_interface()
    ck("the interface declares the propagation option it is built for",
       ei["propagation_option"] == PROPAGATION_OPTION == "A")
    ck("the combined accumulators are the two Z sets, per column and bin",
       set(ei["combined_accumulators_per_column_bin"]) == {
           "sum_Z_e2", "sum_Z2_e2", "max_abs_Z_e2", "sum_Z_d", "sum_Z2_d", "max_abs_Z_d"},
       str(sorted(ei["combined_accumulators_per_column_bin"])))
    ck("the second moments are of the COMBINED contribution, which is what gives the variance",
       "SAME combined contribution" in ei["combined_accumulators_per_column_bin"]["sum_Z2_e2"]
       and "SAME combined contribution" in ei["combined_accumulators_per_column_bin"]["sum_Z2_d"])
    ck("per-step sum_Y2 is explicitly not accumulated and not claimed to give the variance",
       "NOT accumulated" in ei["explicitly_not_accumulated"]["per_step_sum_Y2"]
       and "cross-step terms are absent" in ei["explicitly_not_accumulated"]["per_step_sum_Y2"])
    ck("the per-step maximum is listed separately as a kernel consistency check only",
       "KERNEL CONSISTENCY CHECK ONLY" in ei["per_step_accumulator_kept_separately"]["max_abs_Y"]
       and "B_i(h) = R_i(h)/2" in ei["per_step_accumulator_kept_separately"]["max_abs_Y"]
       and "no variance information" in ei["per_step_accumulator_kept_separately"]["max_abs_Y"])
    exp_comb = 3 * 2 * 120 * 5
    exp_step = 120 * 4 * 5
    ck("the storage is recomputed from the fields actually retained, not the old 4*120*4*5",
       ei["storage"]["combined_scalars"] == exp_comb
       and ei["storage"]["per_step_consistency_scalars"] == exp_step
       and ei["storage"]["scalars_per_replicate"] == exp_comb + exp_step
       and ei["storage"]["scalars_per_replicate"] != 4 * 120 * 4 * 5,
       f"{exp_comb} combined + {exp_step} per-step = {exp_comb + exp_step} scalars, "
       f"{(exp_comb + exp_step) * 8 / 1024:.1f} KiB")
    ei_nostep = executor_interface(keep_per_step_max=False)
    ck("dropping the per-step consistency maximum changes the storage accordingly",
       ei_nostep["storage"]["per_step_consistency_scalars"] == 0
       and ei_nostep["storage"]["scalars_per_replicate"] == exp_comb)
    ps = ei["photon_storage"]
    ck("the unconditional no-per-photon-storage claim is withdrawn and conditioned",
       ps["persistent_per_photon_products_required"] is False
       and "ALL FOUR step sizes" in " ".join(ps["but"].split())
       and "BOTH arms" in ps["but"]
       and "live-coupling requirement" in ps["but"]
       and "option A is NOT implementable as written" in ps["if_the_architecture_cannot_do_that"]
       and "the_earlier_unconditional_claim_is_withdrawn" in ps
       and "with no condition attached"
       in ps["the_earlier_unconditional_claim_is_withdrawn"])
    ck("the kernel assertions name the invariants the range rests on",
       any("w0 == 1 - r_sp" in x for x in ei["kernel_assertions"])
       and any("mua >= 0" in x for x in ei["kernel_assertions"])
       and any("no splitting" in x for x in ei["kernel_assertions"])
       and not any("launch weight equals 1 exactly" in x for x in ei["kernel_assertions"]))

    # --- the cross-step pairing invariant ------------------------------------------------------
    inv = CROSS_STEP_PAIRING_INVARIANT
    ck("the invariant demands identical keys, identical order and a fixed N",
       "exactly the same photon key array" in inv["requirement"]
       and "in exactly the same order" in " ".join(inv["requirement"].split())
       and "fixed launch count N" in inv["requirement"])
    ck("independence is across photon index and CRN reuse is within one photon",
       "ACROSS photon index" in inv["where_independence_lives"]
       and "WITHIN one photon" in inv["where_crn_reuse_lives"]
       and "does not couple different photons" in inv["where_crn_reuse_lives"])
    keys = np.arange(1000, dtype=np.uint32).reshape(500, 2)
    hgv = np.array([0.4, 0.3, 0.2, 0.1])
    labs = required_labels(hgv)
    ck("every step and both arms are required, eight labels for four steps",
       len(labs) == 8 and labs[0].endswith("+") and labs[1].endswith("-"), str(labs))
    good = {lab: keys for lab in labs}
    ck("identical key arrays in identical order pass", verify_pairing(good, 500)["ok"])
    perm = {lab: (keys if i else keys[::-1].copy()) for i, lab in enumerate(labs)}
    vp = verify_pairing(perm, 500)
    ck("the same keys in a different order are REFUSED, not accepted",
       (not vp["ok"]) and any("DIFFERENT ORDER" in p_ for p_ in vp["problems"])
       and "REFUSE" in vp["reason"],
       vp["problems"][0])
    diff = {lab: (keys if i else keys + 1) for i, lab in enumerate(labs)}
    vd = verify_pairing(diff, 500)
    ck("a different key array is REFUSED",
       (not vd["ok"]) and any("DIFFERENT key array" in p_ for p_ in vd["problems"]))
    short = {lab: (keys if i else keys[:400]) for i, lab in enumerate(labs)}
    ck("a key array of a different length is REFUSED", not verify_pairing(short, 500)["ok"])
    ck("a launch count differing from the fixed N is REFUSED",
       not verify_pairing(good, 499)["ok"])
    ck("no key arrays at all is REFUSED rather than silently passing",
       not verify_pairing({}, 500)["ok"])
    missing_arm = {lab: keys for lab in labs if lab != labs[3]}
    vc = verify_coverage_of_labels(missing_arm, hgv)
    ck("a missing arm is REFUSED by the coverage check",
       (not vc["ok"]) and vc["missing"] == [labs[3]] and "REFUSE" in vc["reason"],
       f"missing {vc['missing']}")
    missing_step = {lab: keys for lab in labs if not lab.startswith("h0.2")}
    ck("a missing step is REFUSED by the coverage check",
       len(verify_coverage_of_labels(missing_step, hgv)["missing"]) == 2)
    ck("a complete label set passes the coverage check",
       verify_coverage_of_labels(good, hgv)["ok"])

    # --- the two radii for d ------------------------------------------------------------------
    rr = d_radii(np.full(3, 5.0), np.full(3, 50.0), 20000, np.full(3, 2.0), 120)
    ck("d carries two radii, one per leg, at its own level",
       abs(rr["alpha_num_per_component"] - ALPHA_NUMERATOR / 120) < 1e-18
       and abs(rr["alpha_component_per_component"] - ALPHA_COMPONENT / 120) < 1e-18
       and rr["never_reused_across_legs"] is True,
       f"num {rr['alpha_num_per_component']:.3e}, component "
       f"{rr['alpha_component_per_component']:.3e}")
    ck("the component radius is strictly wider, since its level is stricter",
       float(np.min(rr["eps_d_component"] - rr["eps_d_num"])) > 0.0,
       f"component {float(rr['eps_d_component'][0]):.8f} > num "
       f"{float(rr['eps_d_num'][0]):.8f}")
    ck("the policy records why one radius is not reused for both legs",
       "spend the numerator's budget twice" in D_RADIUS_POLICY["why_not_reuse_one"]
       and "counted ONCE" in D_RADIUS_POLICY["the_alternative_considered"]
       and D_RADIUS_POLICY["choice"].startswith("SEPARATE radii"))

    # --- the specification document ------------------------------------------------------------
    with open(SPEC_DOC) as fh:
        doc_raw = fh.read()
    # every doc assertion runs on whitespace-normalised text, so a line wrap cannot decide a test
    doc = " ".join(doc_raw.split())
    ck("the specification refuses to call the construction distribution free",
       "NOT distribution free" in doc)
    ck("the specification derives the range from the kernel rather than from data",
       "may **not** be the observed maximum" in doc and "w <- w * exp(-mua * d_move)" in doc)
    ck("the specification writes the pairing invariant and the option-A accumulators",
       "## 8b. Cross-step pairing invariant" in doc
       and "## 9. Executor accumulators under option A" in doc
       and "ACROSS photon index" in doc and "WITHIN one photon" in doc
       and "6000 scalars, 46.9 KiB" in doc
       and "9600` scalars from a per-step interface" in doc
       and "that figure is withdrawn" in doc
       and "option A is not implementable as written" in doc)
    # a real check, not a string-presence one: no standalone M may appear as a bound symbol
    import re as _re
    # M-estimator is a standard statistical term, not a bound symbol, and is allowed
    scrubbed = doc.replace("M-estimator", "Mestimator")
    m_hits = [mo.group(0) for mo in _re.finditer(r"`[^`]*\bM\b[^`]*`", scrubbed)]
    m_hits += _re.findall(r"\bM_i\b|\|Y\| *<= *M\b|\babove the derived range M\b", scrubbed)
    m_bound_hits = m_hits
    ck("no standalone M survives as a bound symbol anywhere in the specification",
       not m_bound_hits, f"hits {m_bound_hits[:4]}")
    ck("the specification states the B and R convention and keeps the roles apart",
       "`B` is an absolute bound and `R = 2B` is an interval width" in doc
       and "No single symbol" in doc and "is used for both roles" in doc
       and "`w0 != 1`" not in doc)
    ck("the specification judges acceptance by exact Clopper-Pearson, with Wilson diagnostic only",
       "EXACT, one-sided Clopper-Pearson" in doc
       and "does **not** decide acceptance" in doc
       and "normal approximation" in doc)
    ck("the specification separates the two d radii by leg",
       "separate radii for the same `d`" in doc
       and "never reused across legs" in doc
       and "D_RADIUS_POLICY" in doc)
    ck("the specification lists every unset slot",
       all(f"**unset**" in doc for _ in (0,))
       and doc.count("**unset**") >= 7, f"{doc.count('**unset**')} unset slots")
    ck("the specification drafts the acceptance criteria before validation",
       "drafted before validation runs" in doc and "JOINT coverage" in doc)
    ck("the specification separates the statistical target from the validation confidence",
       "STATISTICAL TARGET" in doc and "VALIDATION CONFIDENCE" in doc
       and "never rewritten as" in doc
       and "multiplicity belongs to the validation confidence alone" in doc
       and "0.00556" in doc)
    ck("the specification declares the suite a draft skeleton with no freeze hash",
       "DRAFT SKELETON, NOT FROZEN" in doc and "no freeze hash is issued" in doc
       and "Nothing in this specification has been run" in doc)
    ck("the specification carries the box-bound counterexample with its numbers",
       "sqrt(5) = 2.2360679" in doc and "is not an upper bound on" in doc
       and "would make both the numerator upper bound and the" in doc)
    ck("the specification separates the absolute bound from the interval width",
       "absolute bound `B_i(h)`" in doc and "interval width `R_i(h)`" in doc
       and "would halve the range term" in doc)
    ck("the specification states the real launch weight and the ceiling argument",
       "w_0 \\;=\\; 1 - r_{sp}" in doc
       and "which is false for this kernel" in doc
       and "not** on" in doc)
    ck("the specification writes the fit propagation and freezes option A",
       "## 4b. From the four D(h) to e2, e3 and d" in doc
       and "Frozen choice: option A" in doc
       and "Documented fallback: option B" in doc
       and "CANNOT recover the" in doc
       and "divided by the number of steps" in doc)
    ck("the specification handles K=0 by formulation, not by a guard",
       "not** a refusal" in doc
       and "handled by the formulation" in doc
       and "rather than by a guard" in doc
       and "a radius that comes entirely from the range term" in doc)
    return res, fails


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--describe", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        res, fails = selftest()
        print(f"  {sum(r['pass'] for r in res)}/{len(res)} checks passed")
        return 1 if fails else 0
    if a.describe:
        print(json.dumps(describe(), indent=2))
        return 0
    ap.error("either --selftest or --describe is required")


if __name__ == "__main__":
    raise SystemExit(main())
