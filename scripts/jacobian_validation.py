#!/usr/bin/env python3
"""jacobian_validation.py -- validate the production Jacobian against an estimator-independent
reference. MERGE_CHECKS item 1.

The design lives in spec/validation_spec.json; this script reads it and must not carry implicit
defaults of its own. The short version of why it is built this way:

    with the random numbers held fixed, a central difference converges to the PATHWISE derivative
    as h -> 0 wherever the realised estimator is differentiable

so validating pathwise AD with small-h common-random-number finite differences is close to
circular. The expectation, however, is smooth even where each realisation is not, so a central
difference of the expectation expands in h^2 and can be extrapolated from MODERATE steps. That
limit is estimator independent, and it is what AD and perturbation Monte Carlo are both compared
against. The h -> 0 limit is deliberately never approached.

Step sizes are bounded by physics, not only by parameter magnitude. StO2_U sits at 0.95 against an
upper bound of 1.0, so a rule that scales h by |psi| alone would evaluate saturations above unity
and silently corrupt the extrapolation. Every parameter declares explicit bounds and every single
perturbed point is checked before any transport is launched.

Nothing here runs transport except --pilot and --final, and neither is started by --selftest or
--dry-run.

    python3 scripts/jacobian_validation.py --selftest
    python3 scripts/jacobian_validation.py --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPLAY_BOUNDARIES = [0.0020, 0.0100, 0.0250, 0.0330, 0.1830, 0.1910, 0.8000]
VSPEC = os.path.join(ROOT, "spec", "validation_spec.json")
BSPEC = os.path.join(ROOT, "spec", "baseline_spec.json")
CRB = os.path.join(ROOT, "results", "production_crb.json")


def load(path):
    with open(path) as fh:
        return json.load(fh)


def bspec_scale(vspec):
    """The column scale D, read from the CRB artifact so one definition serves both scripts."""
    return load(CRB)["column_scale_D"]


# ---------------------------------------------------------------------------------------------
# Step sizes. The whole point of this section is that it refuses to produce an infeasible point.
# ---------------------------------------------------------------------------------------------

def bounds_arrays(vspec):
    labs = vspec["operating_point"]["psi_labels"]
    b = vspec["parameter_bounds"]["bounds"]
    return (np.array([b[l]["lo"] for l in labs], float),
            np.array([b[l]["hi"] for l in labs], float))


def safe_scale(psi, u, lo, hi, h_rel, value_fraction, headroom_fraction, comp_tol=1e-6):
    """Largest scale for which every psi +/- h_rel * scale * u stays strictly inside the bounds.

    Two caps are applied componentwise and the tighter wins: a step may not move any component by
    more than `value_fraction` of its own magnitude, and may not consume more than
    `headroom_fraction` of that component's distance to its nearest bound.
    """
    u = np.asarray(u, float)
    act = np.abs(u) > comp_tol
    if not act.any():
        raise ValueError("direction has no active component")
    head = np.minimum(psi - lo, hi - psi)
    if np.any(head[act] <= 0):
        raise ValueError("nominal point is on or outside a bound")
    cap_value = value_fraction * np.abs(psi[act]) / np.abs(u[act])
    cap_head = headroom_fraction * head[act] / (np.abs(u[act]) * max(h_rel))
    return float(min(cap_value.min(), cap_head.min()))


def step_points(psi, u, scale, h_rel, lo, hi, comp_tol=1e-6):
    """Every evaluation point, with the bound check that gates the run.

    Two margins are reported. The one that matters for a step grid is the margin over the
    components the step actually moves; the margin over all components is also kept, because a
    component that never moves can sit close to a bound without that saying anything about this
    step grid.
    """
    act = np.abs(np.asarray(u, float)) > comp_tol
    out = []
    for hr in h_rel:
        h = hr * scale
        for sign in (+1, -1):
            p = psi + sign * h * np.asarray(u, float)
            m = np.minimum(p - lo, hi - p)
            out.append({"h_relative": hr, "h_absolute": h, "sign": sign, "point": p,
                        "inside_bounds": bool(np.all(p > lo) and np.all(p < hi)),
                        "margin_moving_components": float(np.min(m[act])),
                        "margin_all_components": float(np.min(m))})
    return out


def plan_for(name, psi, u, vspec, lo, hi):
    r = vspec["step_size_rule"]
    h_rel = r["h_relative_grid"]
    scale = safe_scale(psi, u, lo, hi, h_rel, r["value_fraction"], r["headroom_fraction"])
    pts = step_points(psi, u, scale, h_rel, lo, hi)
    bad = [p for p in pts if not p["inside_bounds"]]
    if bad:
        raise SystemExit(f"{name}: {len(bad)} perturbed point(s) outside the declared bounds; "
                         "the run is aborted rather than shortening the step grid")
    uu = np.asarray(u, float)
    act = np.abs(uu) > 1e-6
    hmax = max(p["h_absolute"] for p in pts)
    hi_pt, lo_pt = psi + hmax * uu, psi - hmax * uu
    return {"target": name, "direction": uu.tolist(), "scale": scale,
            "h_absolute": [p["h_absolute"] for p in pts[::2]],
            "n_evaluations": len(pts),
            "margin_moving_components": float(min(p["margin_moving_components"] for p in pts)),
            "margin_all_components": float(min(p["margin_all_components"] for p in pts)),
            "binding_cap": binding_cap(psi, u, lo, hi, h_rel, r),
            "moving_component_indices": [int(i) for i in np.flatnonzero(act)],
            "moving_component_range": {int(i): [float(min(lo_pt[i], hi_pt[i])),
                                                float(max(lo_pt[i], hi_pt[i]))]
                                       for i in np.flatnonzero(act)},
            "largest_component_displacement": float(np.max(np.abs(hmax * uu))),
            "extreme_points": {"max": hi_pt.tolist(), "min": lo_pt.tolist()}}


def binding_cap(psi, u, lo, hi, h_rel, rule):
    u = np.asarray(u, float)
    act = np.abs(u) > 1e-6
    head = np.minimum(psi - lo, hi - psi)
    cv = rule["value_fraction"] * np.abs(psi[act]) / np.abs(u[act])
    ch = rule["headroom_fraction"] * head[act] / (np.abs(u[act]) * max(h_rel))
    labs = np.array(range(psi.size))[act]
    if cv.min() <= ch.min():
        return {"cap": "value_fraction", "binding_component_index": int(labs[int(cv.argmin())])}
    return {"cap": "headroom_fraction", "binding_component_index": int(labs[int(ch.argmin())])}


# ---------------------------------------------------------------------------------------------
# Observables and statistics
# ---------------------------------------------------------------------------------------------

def bin_areas(vspec):
    g = vspec["operating_point"]["radial_grid"]
    e = np.linspace(g["r_min_cm"], g["r_max_cm"], g["n_bins"] + 1)
    return np.pi * (e[1:] ** 2 - e[:-1] ** 2), e


def observables(R, vspec):
    """R is the flat production vector, n_wavelengths blocks of n_bins."""
    areas, _ = bin_areas(vspec)
    n_wl = len(vspec["operating_point"]["wavelengths_nm"])
    per = vspec["operating_point"]["radial_grid"]["n_bins"]
    W = np.asarray(R, float).reshape(n_wl, per) * areas[None, :]
    edges = vspec["observables"]["radial_aggregates"]["bin_group_edges_index"]
    agg = np.stack([W[:, edges[i]:edges[i + 1]].sum(axis=1) for i in range(len(edges) - 1)], axis=1)
    return {"total_escape_per_wavelength": W.sum(axis=1),
            "radial_aggregates": agg.reshape(-1),
            "full_radial_vector": np.asarray(R, float)}


def extrapolate(h, D):
    """D(h) = a0 + a1 h^2 (+ a2 h^4). Two-term is primary; three-term is a truncation diagnostic."""
    h2 = np.asarray(h, float) ** 2
    D = np.atleast_2d(np.asarray(D, float))
    a2 = np.linalg.lstsq(np.vander(h2, 2, increasing=True), D, rcond=None)[0][0]
    a3 = np.linalg.lstsq(np.vander(h2, 3, increasing=True), D, rcond=None)[0][0]
    return a2, a3


def paired_se(delta):
    """SE of the replicate mean of a paired difference. delta is (n_replicates, n_components)."""
    delta = np.atleast_2d(np.asarray(delta, float))
    n = delta.shape[0]
    if n < 2:
        return np.full(delta.shape[1], np.nan)
    return delta.std(axis=0, ddof=1) / np.sqrt(n)


def paired_z(delta):
    """mean(delta) / SE(delta), componentwise. The denominator is this difference's own scatter."""
    delta = np.atleast_2d(np.asarray(delta, float))
    se = paired_se(delta)
    out = np.full(delta.shape[1], np.nan)
    good = se > 0
    out[good] = delta.mean(axis=0)[good] / se[good]
    return out


def paired_inference(ref_reps, ad_reps, pmc_reps, crit):
    """Compare AD and pMC against the reference using PAIRED replicate differences.

    Within a replicate the three evaluations share random numbers, so their errors are correlated.
    Each comparison therefore gets the standard error of its OWN paired difference; SE(reference)
    is reported as a descriptive scale for the reference and never used as a denominator here.

    pmc_reps is None when the replay kernel has not been shown equivalent to this production
    kernel. In that case no pMC number is produced and the relevant verdicts read 'unavailable'.
    """
    ref = np.atleast_2d(np.asarray(ref_reps, float))
    ad = np.atleast_2d(np.asarray(ad_reps, float))
    d_ad = ad - ref
    z_ad = paired_z(d_ad)
    thr, dom = crit["threshold_sigma"], crit["dominance_ratio"]

    out = {"n_replicates": int(ref.shape[0]),
           "se_reference_descriptive_only": paired_se(ref).tolist(),
           "delta_AD_mean": d_ad.mean(axis=0).tolist(),
           "delta_AD_se": paired_se(d_ad).tolist(),
           "z_AD": z_ad.tolist()}

    if pmc_reps is None:
        out.update({
            "pmc_available": False,
            "label": "pMC unavailable; comparison incomplete",
            "delta_pMC_mean": None, "delta_pMC_se": None, "z_pMC": None,
            "delta_AD_pMC_mean": None, "delta_AD_pMC_se": None, "separation_z": None,
            "verdict_ad_vs_reference": [
                "resolved disagreement" if abs(z) >= thr else "unresolved"
                for z in z_ad],
            "verdict_pmc_vs_reference": ["unavailable"] * z_ad.size,
            "verdict_ad_vs_pmc": ["unavailable"] * z_ad.size,
            "preference": ["no preference"] * z_ad.size,
            "preference_reason": ["pMC unavailable, so no estimator can be preferred"] * z_ad.size,
        })
        return out

    pmc = np.atleast_2d(np.asarray(pmc_reps, float))
    d_pmc, d_both = pmc - ref, ad - pmc
    z_pmc, z_sep = paired_z(d_pmc), paired_z(d_both)

    v_ad, v_pmc, v_sep, pref, why = [], [], [], [], []
    for i in range(z_ad.size):
        za, zp, zs = z_ad[i], z_pmc[i], z_sep[i]
        v_ad.append("resolved disagreement" if abs(za) >= thr else "unresolved")
        v_pmc.append("resolved disagreement" if abs(zp) >= thr else "unresolved")
        resolved = np.isfinite(zs) and abs(zs) >= thr
        v_sep.append("resolved separation" if resolved else "unresolved")
        if not resolved:
            pref.append("no preference")
            why.append("AD and pMC are not separated at this budget")
        elif abs(za) < thr <= abs(zp) and abs(za) < abs(zp) / dom:
            pref.append("favours AD")
            why.append("AD is consistent with the reference while pMC disagrees with it")
        elif abs(zp) < thr <= abs(za) and abs(zp) < abs(za) / dom:
            pref.append("favours pMC")
            why.append("pMC is consistent with the reference while AD disagrees with it")
        else:
            pref.append("no preference")
            why.append("AD and pMC are separated, but the paired differences against the "
                       "reference do not support preferring either")
    out.update({"pmc_available": True,
                "delta_pMC_mean": d_pmc.mean(axis=0).tolist(),
                "delta_pMC_se": paired_se(d_pmc).tolist(), "z_pMC": z_pmc.tolist(),
                "delta_AD_pMC_mean": d_both.mean(axis=0).tolist(),
                "delta_AD_pMC_se": paired_se(d_both).tolist(), "separation_z": z_sep.tolist(),
                "verdict_ad_vs_reference": v_ad, "verdict_pmc_vs_reference": v_pmc,
                "verdict_ad_vs_pmc": v_sep, "preference": pref, "preference_reason": why})
    return out


def pc_standardized_paired_mean_discrepancy(delta, k_max):
    """Standardized size of the mean PAIRED difference, in the leading PCs of its own scatter.

    Deliberately not called a chi-square and given no p-value: the covariance comes from the same
    small set of replicates as the mean, so the statistic has no calibrated null distribution here.
    The retained rank is at most n_replicates - 2, because the mean and the scatter are estimated
    from the same replicates and retaining all n-1 scatter directions lets the subspace absorb the
    mean.
    """
    delta = np.atleast_2d(np.asarray(delta, float))
    n = delta.shape[0]
    k = int(min(n - 2, k_max))
    if k < 1:
        return {"statistic": None, "retained_components": 0,
                "reason": f"{n} replicates allow at most {max(n - 2, 0)} retained components"}
    m = delta.mean(axis=0)
    X = delta - m
    _, sv, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt[:k].T
    se = sv[:k] / np.sqrt(n * (n - 1))
    good = se > 0
    if not good.any():
        return {"statistic": None, "retained_components": 0,
                "reason": "the replicate scatter is degenerate"}
    z = (m @ V)[good] / se[good]
    return {"statistic": float(np.sum(z ** 2)),
            "per_component_z": z.tolist(),
            "retained_components": int(good.sum()),
            "name": "PC-standardized paired-mean discrepancy",
            "not_a_chi_square": True, "p_value": None}


# ---------------------------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------------------------

def cost_estimate(vspec, n_photons, n_replicates, targets, measured, include_pmc):
    """Forward-evaluation count times a measured per-evaluation cost.

    The pMC term is counted only when pMC can actually run. Budgeting a replay kernel that is not
    vendored would overstate the plan that exists today, so the two are reported separately.
    """
    n_h = len(vspec["step_size_rule"]["h_relative_grid"])
    per_rep = 2 * n_h + measured["ad_jvp_in_forward_equivalents"]
    if include_pmc:
        per_rep += measured["pmc_replay_in_forward_equivalents"]
    total_fe = per_rep * n_replicates * targets
    unit = measured["forward_seconds_at_reference_photons"] * \
        (n_photons / measured["reference_photons_per_wavelength"])
    return {"includes_pmc": bool(include_pmc),
            "forward_equivalents_per_replicate_per_target": per_rep,
            "targets": targets, "replicates": n_replicates,
            "total_forward_equivalents": total_fe,
            "n_photons_per_wavelength_per_replicate": n_photons,
            "estimated_seconds_per_forward_evaluation": unit,
            "estimated_wall_clock_seconds": total_fe * unit,
            "estimated_wall_clock_hours": total_fe * unit / 3600.0,
            "basis": measured["basis"],
            "status": "estimate from the production pool timing; the pilot measures it directly"}


# ---------------------------------------------------------------------------------------------


def near_null_direction(vspec):
    """The psi-space tangent D v_z, UNNORMALIZED.

    The right singular vectors of J_w = C^{-1/2} J_psi D live in the scaled coordinate z. The
    artifact stores both v_z and its pushforward; this reads the pushforward, and leaves it
    unnormalized so a scalar step along it means a unit displacement in z.
    """
    nn = vspec["observables"]["near_null_directional_derivative"]
    node = load(os.path.join(ROOT, nn["direction_source_file"]))
    for part in nn["direction_source_pointer"]:
        node = node[part]
    labs = vspec["operating_point"]["psi_labels"]
    return np.array([node[l] for l in labs], float)


def near_null_z_vector(vspec):
    """v_z itself, for the test that distinguishes it from its pushforward."""
    nn = vspec["observables"]["near_null_directional_derivative"]
    node = load(os.path.join(ROOT, nn["direction_source_file"]))
    node = node["branches"]["multinomial_fixed_launch"]["geometry"][
        "near_null_direction_z_from_svd"]
    labs = vspec["operating_point"]["psi_labels"]
    return np.array([node[l] for l in labs], float)


def build_plans(vspec, which):
    psi = np.array(vspec["operating_point"]["nominal_psi"], float)
    labs = vspec["operating_point"]["psi_labels"]
    lo, hi = bounds_arrays(vspec)
    plans = []
    for name in which:
        if name == "near_null":
            u = near_null_direction(vspec)
        else:
            u = np.zeros(6)
            u[labs.index(name)] = 1.0
        plans.append(plan_for(name, psi, u, vspec, lo, hi))
    return plans


def replay_kernel_status():
    """pMC is available only if the gate artifact exists AND records a pass. Presence of the file
    is not enough: the question is whether the equivalence check passed, including its negative
    control."""
    src = os.path.join(ROOT, "src", "replay_kernel.py")
    # Bitwise agreement under one XLA backend does not carry over to another, so the gate that
    # counts is the one run on the backend this process will use. A CPU gate does not license a
    # GPU pilot.
    try:
        import jax as _jx
        backend = _jx.default_backend()
    except Exception:
        backend = "unknown"
    cand = ([os.path.join(ROOT, "results", "replay_equivalence_gpu.json")]
            if backend == "gpu" else []) + \
           [os.path.join(ROOT, "results", "replay_equivalence.json")]
    gate = next((c for c in cand if os.path.exists(c)), cand[0])
    st = {"kernel_path": "src/replay_kernel.py", "kernel_present": os.path.exists(src),
          "process_backend": backend,
          "gate_artifact": os.path.relpath(gate, ROOT),
          "gate_artifact_present": os.path.exists(gate),
          "precondition": "must reproduce the production kernel photon by photon, identical "
                          "detector bin assignment and identical escaping weight, before any "
                          "pMC number is reported"}
    if not st["gate_artifact_present"]:
        st.update({"vendored": False,
                   "blocking": "the equivalence gate has not been run; no pMC number may be "
                               "reported"})
        return st
    g = load(gate)
    sm = g["summary"]
    gate_backend = g.get("environment", {}).get("jax_backend", "unknown")
    backend_ok = (backend == "unknown") or (gate_backend == backend)
    ok = (bool(sm["all_passed"]) and bool(sm.get("negative_control_rejected"))
          and backend_ok)
    st.update({
        "vendored": ok,
        "gate_summary": {k: sm[k] for k in (
            "n_cases", "n_passed", "all_passed", "negative_control_rejected",
            "total_photon_comparisons", "det_id_mismatches_total",
            "escaped_flag_mismatches_total", "weight_bitwise_identical_in_all_cases",
            "det_id_bitwise_identical_in_all_cases",
            "worst_pathlength_identity_max_rel_diff") if k in sm},
        "gate_backend": gate_backend,
        "backend_matches_this_process": bool(backend_ok),
        "gate_verdict": g["verdict"],
        "blocking": None if ok else (
            f"the available gate was run on {gate_backend} but this process is on {backend}; "
            "bitwise agreement does not carry across XLA backends, so no pMC number may be "
            "reported until the gate is run here" if not backend_ok else
            "the equivalence gate did not pass; no pMC number may be reported")})
    return st


# ---------------------------------------------------------------------------------------------
# Pilot. Reference, AD and pMC are produced in ONE run on the SAME replicates, sharing one key
# array per replicate, and every raw per-replicate array is stored. That is what makes the three
# paired differences well defined rather than reconstructed after the fact from separate runs.
# ---------------------------------------------------------------------------------------------

def build_forward(vspec, jaxpad_root, n_photons):
    """The production forward model, plus its AD and pMC derivatives. Transport happens here."""
    import importlib.util
    sys.path.insert(0, os.path.abspath(jaxpad_root))
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from scripts.inverse.phase2_v12_runner import (Phase2RunConfig, scattering_for_transport,
                                                   _build_mu)
    import scripts.archive.inverse.multi_wl_v12_colab as v12
    from src.pad.mc_forward import forward_y_multi_layer as prod_forward

    spec = load(BSPEC)
    kappa = spec["blood_coordinate"]["legacy_to_physical_divisor"]
    op = vspec["operating_point"]
    max_steps, radial = op["max_steps"], op["radial_grid"]
    psi0 = np.array(op["nominal_psi"], float)
    n_wl = int(v12.N_WL)
    edges = jnp.linspace(radial["r_min_cm"], radial["r_max_cm"], radial["n_bins"] + 1,
                         dtype=jnp.float64)
    cfg = Phase2RunConfig(n_photons_final=n_photons, seed_keys=0, seed_noise=1,
                          sto2_ubnp_true=psi0[0], sto2_dbnp_true=psi0[2],
                          b_ubnp_true=psi0[1] * kappa, b_dbnp_true=psi0[3] * kappa,
                          max_steps=max_steps)
    bounds = jnp.array(REPLAY_BOUNDARIES, dtype=jnp.float64)
    mus_list = [scattering_for_transport(v12.MUSS_WL[k], v12.GS_WL[k], cfg.scattering_convention)
                for k in range(n_wl)]

    def psi_to_theta(psi):
        s_u, f_u, s_d, f_d, dermr, subq = psi
        c = jnp.stack([kappa * f_u * s_u, kappa * f_u * (1.0 - s_u),
                       kappa * f_d * s_d, kappa * f_d * (1.0 - s_d), dermr, subq])
        return jnp.log(c + v12.EPS)

    def opt(k):
        return dict(muss=mus_list[k], gs=v12.GS_WL[k], ns=v12.NS_WL[k],
                    layer_boundaries=bounds, detector_r_edges=edges, max_steps=max_steps)

    def refl(psi, keys):
        mu_list, _, _ = _build_mu(psi_to_theta(psi))
        return jnp.concatenate([prod_forward(muas=mu_list[k], keys=keys, **opt(k))
                                for k in range(n_wl)])

    refl_jit = jax.jit(refl)
    dmua_dpsi = jax.jit(jax.jacfwd(lambda psi, k: _build_mu(psi_to_theta(psi))[0][k],
                                   argnums=0), static_argnums=1)

    _rk_spec = importlib.util.spec_from_file_location(
        "mcii_replay_kernel", os.path.join(ROOT, "src", "replay_kernel.py"))
    _rk = importlib.util.module_from_spec(_rk_spec)
    _rk_spec.loader.exec_module(_rk)

    def pmc_jacobian_column(psi, keys, tangent):
        """Perturbation Monte Carlo directional derivative of the full reflectance vector.

        dR_i/dmua_j = -S[i, j], so the derivative along a psi tangent is -S @ (dmua/dpsi) @ t,
        taken blockwise per wavelength. The trajectories are the production ones, which is what
        the equivalence gate established.
        """
        mu_list, _, _ = _build_mu(psi_to_theta(psi))
        out = []
        for k in range(n_wl):
            _, S = _rk.replay_moments(muas=mu_list[k], keys=keys, **opt(k))
            out.append(-(np.asarray(S) @ np.asarray(dmua_dpsi(psi, k))) @ np.asarray(tangent))
        return np.concatenate(out)

    # keys and the tangent are TRACED ARGUMENTS, not closure constants. Closing over a per
    # replicate key array would retrace the jvp through a max_steps 4000 trajectory once per
    # replicate, which is the same mistake that was caught in the transport pool.
    @jax.jit
    def ad_jit(psi, keys, tangent):
        return jax.jvp(lambda p: refl(p, keys), (psi,), (tangent,))[1]

    def ad_directional(psi, keys, tangent):
        return np.asarray(ad_jit(jnp.asarray(psi), keys, jnp.asarray(tangent)))

    def warm_up(psi, keys, tangent):
        """Pay every compilation once, before the clock starts, so the per-replicate timings mean
        something. Each entry point takes keys as an argument, so one compile serves all seeds."""
        t = {}
        for label, fn in (("refl", lambda: refl_jit(jnp.asarray(psi), keys)),
                          ("ad", lambda: ad_jit(jnp.asarray(psi), keys, jnp.asarray(tangent))),
                          ("pmc", lambda: _rk.replay_moments(
                              muas=_build_mu(psi_to_theta(jnp.asarray(psi)))[0][0],
                              keys=keys, **opt(0)))):
            t0 = time.perf_counter()
            jax.block_until_ready(fn())
            t[label] = round(time.perf_counter() - t0, 2)
        return t

    return {"refl": lambda psi, keys: np.asarray(refl_jit(jnp.asarray(psi), keys)),
            "ad": ad_directional, "pmc": pmc_jacobian_column, "n_wl": n_wl,
            "warm_up": warm_up,
            "jax": jax, "split": lambda seed: jax.random.split(jax.random.PRNGKey(seed),
                                                               n_photons)}


def run_pilot(vspec, fwd, plans, n_replicates, pmc_available, reduce_to,
              observable_name):
    """One run, same replicates for all three estimators, raw arrays kept."""
    rnd = vspec["random_number_design"]
    h_rel = vspec["step_size_rule"]["h_relative_grid"]
    psi = np.array(vspec["operating_point"]["nominal_psi"], float)
    out = {}
    for pl in plans:
        u = np.array(pl["direction"], float)
        hs = np.array(pl["h_absolute"], float)
        ref_r, ad_r, pmc_r, raw = [], [], [], []
        for r in range(n_replicates):
            seed = rnd["seed_base"] + rnd["stride"] * r
            keys = fwd["split"](seed)                    # ONE key array per replicate, shared by
            D = []                                       # the forward pair, the AD jvp and pMC
            t_fwd = time.perf_counter()
            for h in hs:
                Rp = fwd["refl"](psi + h * u, keys)
                Rm = fwd["refl"](psi - h * u, keys)
                D.append((reduce_to(Rp) - reduce_to(Rm)) / (2.0 * h))
            t_fwd = time.perf_counter() - t_fwd
            D = np.stack(D)
            a2, a3 = extrapolate(hs, D)
            t_ad = time.perf_counter()
            ad = reduce_to(fwd["ad"](psi, keys, u))
            t_ad = time.perf_counter() - t_ad
            t_pmc = time.perf_counter()
            pm = reduce_to(fwd["pmc"](psi, keys, u)) if pmc_available else None
            t_pmc = time.perf_counter() - t_pmc
            ref_r.append(a2)
            ad_r.append(ad)
            if pmc_available:
                pmc_r.append(pm)
            raw.append({"replicate": r, "seed": seed,
                        "difference_quotients_by_h": D.tolist(),
                        "reference_two_term": a2.tolist(),
                        "reference_three_term": a3.tolist(),
                        "ad": ad.tolist(), "pmc": (pm.tolist() if pmc_available else None),
                        "seconds": {"forward_pairs_total": round(t_fwd, 2),
                                    "forward_per_evaluation": round(t_fwd / (2 * len(hs)), 2),
                                    "ad_jvp": round(t_ad, 2),
                                    "pmc_replay": round(t_pmc, 2) if pmc_available else None,
                                    "ad_in_forward_equivalents":
                                        round(t_ad / (t_fwd / (2 * len(hs))), 2) if t_fwd > 0
                                        else None,
                                    "pmc_in_forward_equivalents":
                                        round(t_pmc / (t_fwd / (2 * len(hs))), 2)
                                        if pmc_available and t_fwd > 0 else None}})
        ref = np.stack(ref_r)
        ad = np.stack(ad_r)
        pmc = np.stack(pmc_r) if pmc_available else None
        ref3 = np.stack([np.array(x["reference_three_term"]) for x in raw])
        trunc = np.abs(ref3.mean(0) - ref.mean(0)) / np.maximum(paired_se(ref), 1e-300)
        inf = paired_inference(ref, ad, pmc, vspec["acceptance_criteria"])
        inf["truncation_diagnostic_three_minus_two_over_se"] = trunc.tolist()
        # The statistic is named after the observable it was actually given. Calling it a
        # full-vector discrepancy while feeding it the 4-component total escape would misdescribe
        # it: the 120-component radial comparison the spec requires for the final validation has
        # not been run here.
        kmax = vspec["observables"]["full_radial_vector"]["declared_max_retained_components"]
        inf["paired_mean_discrepancy"] = {
            "observable": observable_name,
            "n_components": int(ref.shape[1]),
            "is_the_full_radial_vector":
                observable_name == "full_radial_vector",
            "ad": pc_standardized_paired_mean_discrepancy(ad - ref, kmax),
            "pmc": (pc_standardized_paired_mean_discrepancy(pmc - ref, kmax)
                    if pmc_available else None)}
        out[pl["target"]] = {"plan": pl, "inference": inf, "raw_per_replicate": raw,
                             "pmc_available": bool(pmc_available)}
    return out


def array_digest(*arrays):
    """A hash over the stored arrays, so a truncated or edited artifact is detectable."""
    hsh = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
        hsh.update(str(a.shape).encode())
        hsh.update(a.tobytes())
    return hsh.hexdigest()


def run_pilot_radial(vspec, fwd, plans, n_replicates):
    """Same pilot, but the FULL 120-component radial output is stored at every stage.

    Nothing is reduced before saving. The three observables the spec pre-registers are derived
    offline from these same stored arrays, so no observable costs its own transport run.
    """
    rnd = vspec["random_number_design"]
    psi = np.array(vspec["operating_point"]["nominal_psi"], float)
    out = {}
    for pl in plans:
        u = np.array(pl["direction"], float)
        hs = np.array(pl["h_absolute"], float)
        raw = []
        for r in range(n_replicates):
            seed = rnd["seed_base"] + rnd["stride"] * r
            keys = fwd["split"](seed)          # one key array, shared by all three estimators
            Rp, Rm = [], []
            t_fwd = time.perf_counter()
            for h in hs:
                Rp.append(fwd["refl"](psi + h * u, keys))
                Rm.append(fwd["refl"](psi - h * u, keys))
            t_fwd = time.perf_counter() - t_fwd
            Rp, Rm = np.stack(Rp), np.stack(Rm)                  # (n_h, 120) each
            D = (Rp - Rm) / (2.0 * hs)[:, None]                  # (n_h, 120)
            a2, a3 = extrapolate(hs, D)                          # (120,) each
            t_ad = time.perf_counter()
            ad = fwd["ad"](psi, keys, u)                         # (120,)
            t_ad = time.perf_counter() - t_ad
            t_pmc = time.perf_counter()
            pm = fwd["pmc"](psi, keys, u)                        # (120,)
            t_pmc = time.perf_counter() - t_pmc
            fpe = t_fwd / (2 * len(hs))
            raw.append({
                "replicate": r, "seed": seed,
                "stored_in_full_120_component_radial_coordinates": True,
                "reduced_before_saving": False,
                "R_plus_h": Rp.tolist(), "R_minus_h": Rm.tolist(),
                "h_absolute": hs.tolist(),
                "difference_quotients_120": D.tolist(),
                "difference_quotient_definition": "(R(psi + h u) - R(psi - h u)) / (2 h), per h, "
                                                  "in the full 120-component radial coordinates",
                "reference_two_term_120": a2.tolist(),
                "reference_three_term_120": a3.tolist(),
                "ad_120": ad.tolist(), "pmc_120": pm.tolist(),
                "seconds": {"forward_pairs_total": round(t_fwd, 2),
                            "forward_per_evaluation": round(fpe, 2),
                            "ad_jvp": round(t_ad, 2), "pmc_replay": round(t_pmc, 2),
                            "ad_in_forward_equivalents": round(t_ad / fpe, 2) if fpe > 0 else None,
                            "pmc_in_forward_equivalents": round(t_pmc / fpe, 2) if fpe > 0
                                                          else None},
                "array_sha256": array_digest(Rp, Rm, a2, a3, ad, pm)})
        out[pl["target"]] = {"plan": pl, "raw_per_replicate": raw}
    return out


OBSERVABLE_ORDER = ("total_escape_per_wavelength", "radial_aggregates", "full_radial_vector")


def component_admissibility(ref, ad, vspec):
    """Which components carry a genuine ensemble reference, and which only carry rounding.

    Where a radial bin is reached by too few photons, the common-random-number central difference
    samples no discontinuity crossing and the h^2 extrapolation collapses onto the realised
    pathwise derivative. AD then equals the reference to machine precision and the paired standard
    error measures rounding. A budget multiplier computed there comes out near zero and would read
    as already resolvable when nothing has been resolved, so such components are excluded from the
    selection and counted rather than quietly dropped.
    """
    rule = vspec["component_admissibility"]
    floor = rule["paired_scatter_floor"]
    n = ref.shape[0]
    nonzero = (ref != 0.0).sum(axis=0)
    se_ref = paired_se(ref)
    se_ad = paired_se(ad - ref)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(se_ref > 0, se_ad / se_ref, 0.0)
    ok_ens = nonzero >= n
    ok_regime = ratio >= floor
    ok = ok_ens & ok_regime
    reason = []
    for j in range(ref.shape[1]):
        if ok[j]:
            reason.append(None)
        elif not ok_ens[j]:
            reason.append(f"reference nonzero in only {int(nonzero[j])} of {n} replicates")
        else:
            reason.append(f"SE(AD - reference) / SE(reference) = {ratio[j]:.3g}, below the floor "
                          f"of {floor}: the central difference has sampled no discontinuity "
                          f"crossing and the extrapolation has returned AD, so the paired "
                          f"standard error is not a measure of reference uncertainty")
    return {"admissible": ok.tolist(), "n_admissible": int(ok.sum()),
            "n_components": int(ref.shape[1]),
            "replicates_with_nonzero_reference": nonzero.tolist(),
            "paired_scatter_ratio": ratio.tolist(),
            "paired_scatter_floor": floor,
            "reason_if_inadmissible": reason,
            "rule": f"every replicate nonzero, and SE(AD - reference) / SE(reference) at least "
                    f"{floor}"}


def derive_observables(stored, vspec):
    """Every observable comes off the SAME stored 120-component arrays. No extra transport."""
    crit = vspec["acceptance_criteria"]
    thr = crit["threshold_sigma"]
    kmax = vspec["observables"]["full_radial_vector"]["declared_max_retained_components"]
    labels = {"total_escape_per_wavelength": [f"{int(w)}nm" for w in
                                              vspec["operating_point"]["wavelengths_nm"]],
              "radial_aggregates": [f"{int(w)}nm_g{g}"
                                    for w in vspec["operating_point"]["wavelengths_nm"]
                                    for g in range(4)],
              "full_radial_vector": None}
    res = {}
    for target, e in stored.items():
        raw = e["raw_per_replicate"]
        n = len(raw)
        per_obs = {}
        for name in OBSERVABLE_ORDER:
            proj = lambda v: observables(np.asarray(v, float), vspec)[name]
            ref = np.stack([proj(r["reference_two_term_120"]) for r in raw])
            ref3 = np.stack([proj(r["reference_three_term_120"]) for r in raw])
            ad = np.stack([proj(r["ad_120"]) for r in raw])
            pm = np.stack([proj(r["pmc_120"]) for r in raw])
            inf = paired_inference(ref, ad, pm, crit)
            inf["truncation_diagnostic_three_minus_two_over_se"] = (
                np.abs(ref3.mean(0) - ref.mean(0)) / np.maximum(paired_se(ref), 1e-300)).tolist()
            inf["component_labels"] = labels[name]
            inf["n_components"] = int(ref.shape[1])

            # the PC-standardized paired-mean discrepancy, for all three paired differences
            disc = {}
            for key, delta in (("ad_minus_reference", ad - ref),
                               ("pmc_minus_reference", pm - ref),
                               ("ad_minus_pmc", ad - pm)):
                disc[key] = pc_standardized_paired_mean_discrepancy(delta, kmax)
            inf["pc_standardized_paired_mean_discrepancy"] = disc
            inf["retained_rank_note"] = (
                f"k = min(n_replicates - 2, {kmax}) = {min(n - 2, kmax)}. With n = {n} replicates "
                f"at most {n - 2} directions can be retained, because the mean and the scatter "
                f"are estimated from the same replicates.")
            inf["not_a_chi_square"] = True
            inf["p_value"] = None

            # admissibility first: a multiplier from a component carrying only rounding is a
            # false signal, not a cheap result
            adm = component_admissibility(ref, ad, vspec)
            inf["component_admissibility"] = adm
            ok = np.asarray(adm["admissible"], bool)

            gap = np.abs(np.asarray(inf["delta_AD_pMC_mean"], float))
            se = np.asarray(inf["delta_AD_se"], float)
            with np.errstate(divide="ignore", invalid="ignore"):
                fac = (se / (gap / thr)) ** 2
            fac = np.where(np.isfinite(fac), fac, np.inf)
            fac_sel = np.where(ok, fac, np.inf)
            best = int(np.argmin(fac_sel)) if ok.any() else None

            # The principal-subspace multiplier must use the REFERENCE-comparison scatter as its
            # denominator. Standardizing by SE(AD - pMC) would answer a question already settled:
            # AD and pMC share random numbers, so their separation is resolved at this budget. The
            # budget question is how much is needed to detect a bias of that size AGAINST the
            # reference, so the effect is mean(AD - pMC) and the scale is the scatter of
            # AD - reference.
            fv_fac, fv_detail = None, None
            if ok.any():
                X = (ad - ref)[:, ok]
                n_r = X.shape[0]
                k = int(min(n_r - 2, kmax))
                if k >= 1:
                    m_eff = (ad - pm)[:, ok].mean(axis=0)
                    Xc = X - X.mean(axis=0)
                    _, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
                    V = Vt[:k].T
                    se_j = sv[:k] / np.sqrt(n_r * (n_r - 1))
                    good = se_j > 0
                    if good.any():
                        zz = np.abs((m_eff @ V)[good] / se_j[good])
                        fv_fac = (float((thr / zz.max()) ** 2) if zz.max() > 0
                                  else float("inf"))
                        fv_detail = {
                            "construction": "project mean(AD - pMC) onto the leading principal "
                                            "directions of the AD-minus-reference replicate "
                                            "scatter, standardize by that scatter's standard "
                                            "error, and ask what budget brings the largest "
                                            "standardized projection to threshold_sigma",
                            "retained_components": int(good.sum()),
                            "per_direction_z": zz.tolist(),
                            "not_a_chi_square": True, "p_value": None}
            disc["ad_minus_pmc_admissible_components_only"] = (
                pc_standardized_paired_mean_discrepancy((ad - pm)[:, ok], kmax)
                if ok.any() else None)

            inf["budget_multiplier_for_reference_resolution_at_ad_pmc_gap_scale"] = {
                "rule": "the smallest real effect worth resolving is the AD-minus-pMC gap; the "
                        "paired standard error must fall to gap / threshold_sigma, and it scales "
                        "as one over the square root of the total photon count",
                "threshold_sigma": thr,
                "computed_on_admissible_components_only": True,
                "n_admissible": adm["n_admissible"], "n_components": adm["n_components"],
                "per_component": [None if not np.isfinite(f) else round(float(f), 1)
                                  for f in fac],
                "per_component_admissible": adm["admissible"],
                "best_component_index": best,
                "best_component_label": (None if best is None else
                                         (labels[name][best] if labels[name]
                                          else f"bin {best}")),
                "best_multiplier": (None if best is None or not np.isfinite(fac_sel[best])
                                    else round(float(fac_sel[best]), 1)),
                "pc_subspace_multiplier": (None if fv_fac is None or not np.isfinite(fv_fac)
                                           else round(fv_fac, 1)),
                "pc_subspace_detail": fv_detail,
                "pc_subspace_rule": "the effect is mean(AD - pMC) and the scale is the "
                                    "AD-minus-reference replicate scatter, over admissible "
                                    "components only. Standardizing by SE(AD - pMC) instead would "
                                    "answer a question already settled, since the two estimators "
                                    "share random numbers and their separation is resolved at this "
                                    "budget."}
            per_obs[name] = inf
        res[target] = {"plan": e["plan"], "by_observable": per_obs}
    return res


def adjacency_consistency(inf, vspec, band_reachable=10.0):
    """Does a cheap radial group have a cheap neighbour at the same wavelength?

    best_multiplier is a minimum over many components and is optimistic by construction, so an
    isolated cheap component is not evidence. For the 16-component aggregates the corroboration
    available is the neighbouring radial group at the same wavelength.
    """
    labels = inf.get("component_labels")
    fac = inf["budget_multiplier_for_reference_resolution_at_ad_pmc_gap_scale"]["per_component"]
    adm = inf["component_admissibility"]["admissible"]
    n_wl = len(vspec["operating_point"]["wavelengths_nm"])
    n_g = len(vspec["observables"]["radial_aggregates"]["bin_group_edges_index"]) - 1
    if labels is None or len(fac) != n_wl * n_g:
        return {"applicable": False,
                "reason": "adjacency is defined for the radial aggregates only"}
    M = np.array([[fac[w * n_g + g] if (adm[w * n_g + g] and fac[w * n_g + g] is not None)
                   else np.inf for g in range(n_g)] for w in range(n_wl)], dtype=float)
    rows = []
    for w in range(n_wl):
        for g in range(n_g):
            if not np.isfinite(M[w, g]) or M[w, g] > band_reachable:
                continue
            nb = [M[w, g2] for g2 in (g - 1, g + 1) if 0 <= g2 < n_g]
            ok = any(np.isfinite(x) and x <= band_reachable for x in nb)
            rows.append({"wavelength_index": w, "group": g,
                         "label": labels[w * n_g + g],
                         "multiplier": round(float(M[w, g]), 1),
                         "neighbour_multipliers": [None if not np.isfinite(x) else round(float(x), 1)
                                                   for x in nb],
                         "corroborated_by_a_neighbour": bool(ok)})
    return {"applicable": True, "band_reachable": band_reachable,
            "cheap_groups": rows,
            "any_corroborated": bool(any(r["corroborated_by_a_neighbour"] for r in rows)),
            "rule": "an isolated cheap group is treated the same way as an isolated bin: not "
                    "evidence of reachability"}


def resource_verdict(derived, vspec):
    """Apply the frozen bands, reading the aggregates and the PC subspace, not the best bin."""
    crit = vspec["resource_decision_criteria"]
    bands = crit["bands"]
    out = {"criteria_frozen_before_results_were_seen":
           crit["frozen_before_the_radial_pilot_results_were_seen"],
           "primary_evidence_in_order": crit["primary_evidence_in_order"],
           "per_target": {}}
    for target, e in derived.items():
        agg = e["by_observable"]["radial_aggregates"]
        fv = e["by_observable"]["full_radial_vector"]
        tot = e["by_observable"]["total_escape_per_wavelength"]
        key = "budget_multiplier_for_reference_resolution_at_ad_pmc_gap_scale"
        adj = adjacency_consistency(agg, vspec)
        cand = [agg[key]["pc_subspace_multiplier"], fv[key]["pc_subspace_multiplier"]]
        cand = [c for c in cand if c is not None]
        pc = min(cand) if cand else None
        agg_best = agg[key]["best_multiplier"]
        # the decision quantity: the aggregates and the PC subspace, with corroboration required
        if pc is None and agg_best is None:
            band, why = "close_the_route", ("no admissible component supports a multiplier at "
                                            "all")
        else:
            decisive = min([x for x in (pc, agg_best) if x is not None])
            corroborated = adj.get("any_corroborated", False)
            if decisive <= 10.0 and corroborated:
                band = "reachable"
                why = "the aggregates and the PC subspace agree and a cheap group is corroborated"
            elif decisive <= 10.0 and not corroborated:
                band, why = "close_the_route", (
                    "a cheap component exists but no neighbouring radial group corroborates it; "
                    "by the frozen single-cheap-bin rule this is recorded as not reachable")
            elif decisive <= 100.0:
                band, why = "targeted_only", "a targeted supplement only"
            else:
                band, why = "close_the_route", "beyond 100 times this budget"
        out["per_target"][target] = {
            "aggregates_best_multiplier": agg_best,
            "aggregates_pc_subspace_multiplier": agg[key]["pc_subspace_multiplier"],
            "full_vector_best_multiplier": fv[key]["best_multiplier"],
            "full_vector_pc_subspace_multiplier": fv[key]["pc_subspace_multiplier"],
            "total_escape_best_multiplier": tot[key]["best_multiplier"],
            "adjacency": adj,
            "band": band, "action": bands[band]["action"], "reason": why,
            "note": "the best multiplier over 120 bins is optimistic by construction and does "
                    "not decide this on its own"}
    return out


def selftest(vspec, bspec):
    psi = np.array(vspec["operating_point"]["nominal_psi"], float)
    labs = vspec["operating_point"]["psi_labels"]
    lo, hi = bounds_arrays(vspec)
    r = vspec["step_size_rule"]
    res, fail = [], []

    def check(name, ok, detail=""):
        res.append({"check": name, "pass": bool(ok), "detail": detail})
        if not ok:
            fail.append(name)

    # 1. the operating point agrees with the frozen production spec
    nom = bspec["parameters"]["nominal_psi"]
    check("operating point matches baseline_spec",
          np.allclose(psi, [nom[l] for l in labs]) and
          vspec["operating_point"]["max_steps"] == bspec["profiles"]["production"]["max_steps"] and
          vspec["operating_point"]["wavelengths_nm"] == bspec["wavelengths_nm"] and
          vspec["operating_point"]["radial_grid"]["n_bins"] ==
          bspec["profiles"]["production"]["radial"]["n_bins"])

    # 2. every planned point is inside its bounds, for all six columns and the near-null direction
    plans = build_plans(vspec, labs + ["near_null"])
    check("all six columns and the near-null direction have feasible step grids",
          len(plans) == 7, f"{len(plans)} plans built")
    worst = min(p["margin_moving_components"] for p in plans)
    tightest = min(plans, key=lambda p: p["margin_moving_components"])["target"]
    check("every perturbed point is strictly inside its bounds", worst > 0,
          f"tightest margin on a moving component {worst:.6g}, at {tightest}")

    # 3. the guard actually bites: a magnitude-only rule leaves the feasible region for StO2_U
    i = labs.index("StO2_U")
    naive_h = max(r["h_relative_grid"]) * r["value_fraction"] * abs(psi[i])
    check("a magnitude-only step rule would violate the StO2_U upper bound",
          psi[i] + naive_h > hi[i],
          f"naive psi+h = {psi[i] + naive_h:.4f} against bound {hi[i]}")
    sp = [p for p in plans if p["target"] == "StO2_U"][0]
    check("the headroom cap binds for StO2_U", sp["binding_cap"]["cap"] == "headroom_fraction",
          f"{sp['binding_cap']}; max point {max(sp['extreme_points']['max']):.6f}")

    # 4. extrapolation algebra against a known limit
    a0, a1, a2 = 1.2345, -0.7, 0.25
    hs = np.array([hr * 0.1 for hr in r["h_relative_grid"]])
    D = (a0 + a1 * hs ** 2 + a2 * hs ** 4)[:, None]
    e2, e3 = extrapolate(hs, D)
    check("three-term fit recovers an exact quartic limit", abs(e3[0] - a0) < 1e-10,
          f"residual {abs(e3[0]-a0):.3e}")
    check("two-term fit is biased by the h^4 term, as expected",
          1e-9 < abs(e2[0] - a0) < 1e-2, f"two-term offset {abs(e2[0]-a0):.3e}")
    e2p, e3p = extrapolate(hs, (a0 + a1 * hs ** 2)[:, None])
    check("both fits are exact when the series truly stops at h^2",
          abs(e2p[0] - a0) < 1e-12 and abs(e3p[0] - a0) < 1e-10)

    # 5. paired inference: each comparison uses its own paired standard error
    crit = vspec["acceptance_criteria"]
    rg = np.random.default_rng(11)
    n_rep = 8
    # shared replicate noise, large; estimator-specific noise, small. AD tracks the reference,
    # pMC carries a genuine offset. A reference-only denominator would hide both facts.
    shared = rg.normal(scale=1.0, size=(n_rep, 1))
    ref = 10.0 + shared + rg.normal(scale=0.002, size=(n_rep, 1))
    ad = 10.0 + shared + rg.normal(scale=0.002, size=(n_rep, 1))
    pmc = 10.3 + shared + rg.normal(scale=0.002, size=(n_rep, 1))
    inf = paired_inference(ref, ad, pmc, crit)
    se_ref = np.asarray(inf["se_reference_descriptive_only"])
    se_pair = np.asarray(inf["delta_AD_pMC_se"])
    check("paired standard error is far smaller than the reference standard error when noise is "
          "shared", float(se_ref[0] / se_pair[0]) > 100.0,
          f"SE(reference) {se_ref[0]:.4g} against SE(paired AD-pMC) {se_pair[0]:.4g}, "
          f"ratio {se_ref[0]/se_pair[0]:.0f}")
    wrong_z = abs(float(np.mean(np.asarray(ad) - np.asarray(pmc)) / se_ref[0]))
    check("a reference-only denominator would have called a real separation unresolved",
          wrong_z < crit["threshold_sigma"] <= abs(inf["separation_z"][0]),
          f"wrong z {wrong_z:.2f} against paired z {abs(inf['separation_z'][0]):.1f}")
    check("paired inference resolves the separation and prefers AD",
          inf["verdict_ad_vs_pmc"][0] == "resolved separation"
          and inf["verdict_ad_vs_reference"][0] == "unresolved"
          and inf["verdict_pmc_vs_reference"][0] == "resolved disagreement"
          and inf["preference"][0] == "favours AD",
          f"{inf['verdict_ad_vs_reference'][0]} / {inf['verdict_pmc_vs_reference'][0]} / "
          f"{inf['verdict_ad_vs_pmc'][0]} -> {inf['preference'][0]}")

    # 5b. perfect shared-noise cancellation drives the paired standard error to zero
    sh = rg.normal(scale=3.0, size=(n_rep, 1))
    r0, a0_ = 5.0 + sh, 5.0 + sh
    d0 = np.asarray(a0_) - np.asarray(r0)
    check("identical shared noise gives a paired difference standard error of zero",
          float(paired_se(d0)[0]) == 0.0 and float(paired_se(np.asarray(r0))[0]) > 0.1,
          f"paired SE {paired_se(d0)[0]:.3e}, reference SE {paired_se(np.asarray(r0))[0]:.4g}")
    inf0 = paired_inference(r0, a0_, None, crit)
    check("a zero paired difference is reported as unresolved, never as validated",
          inf0["verdict_ad_vs_reference"][0] == "unresolved")

    # 5c. with pMC unavailable, no pMC number or verdict is fabricated
    check("pMC unavailable produces no pMC number and no preference",
          inf0["pmc_available"] is False and inf0["z_pMC"] is None
          and inf0["delta_pMC_mean"] is None
          and inf0["verdict_pmc_vs_reference"][0] == "unavailable"
          and inf0["verdict_ad_vs_pmc"][0] == "unavailable"
          and inf0["preference"][0] == "no preference"
          and inf0["label"] == "pMC unavailable; comparison incomplete",
          inf0["label"])

    # 5d. the word validated appears nowhere in a verdict vocabulary
    vocab = set(inf["verdict_ad_vs_reference"] + inf["verdict_pmc_vs_reference"]
                + inf["verdict_ad_vs_pmc"] + inf["preference"]
                + inf0["verdict_ad_vs_reference"] + inf0["verdict_pmc_vs_reference"]
                + inf0["verdict_ad_vs_pmc"] + inf0["preference"])
    check("no verdict claims validation", not any("valid" in v for v in vocab),
          str(sorted(vocab)))

    # 6. observables partition the grid exactly
    rng = np.random.default_rng(7)
    n_wl = len(vspec["operating_point"]["wavelengths_nm"])
    per = vspec["operating_point"]["radial_grid"]["n_bins"]
    Rr = rng.uniform(1e-6, 1e-3, n_wl * per)
    ob = observables(Rr, vspec)
    agg = ob["radial_aggregates"].reshape(n_wl, -1)
    check("radial aggregates sum to total escape",
          np.allclose(agg.sum(axis=1), ob["total_escape_per_wavelength"], rtol=0, atol=1e-15))
    ed = vspec["observables"]["radial_aggregates"]["bin_group_edges_index"]
    _, edges_cm = bin_areas(vspec)
    check("declared group edges in cm match the bin edges they name",
          np.allclose([edges_cm[i] for i in ed],
                      vspec["observables"]["radial_aggregates"]["bin_group_edges_cm"]))
    check("groups cover every bin exactly once", ed[0] == 0 and ed[-1] == per and
          all(ed[i] < ed[i + 1] for i in range(len(ed) - 1)))
    check("component counts match the spec",
          ob["total_escape_per_wavelength"].size ==
          vspec["observables"]["total_escape_per_wavelength"]["n_components"] and
          ob["radial_aggregates"].size ==
          vspec["observables"]["radial_aggregates"]["n_components"] and
          ob["full_radial_vector"].size == n_wl * per)

    # 7. the near-null direction is the pushforward D v_z, not v_z
    u = near_null_direction(vspec)
    v_z = near_null_z_vector(vspec)
    Dv = np.array(bspec_scale(vspec), float) * v_z
    check("the z-space singular vector is pushed forward as D v_z",
          np.allclose(u, Dv, rtol=0, atol=1e-15),
          f"max difference {np.max(np.abs(u - Dv)):.3e}")
    check("the pushforward is materially different from v_z itself",
          not np.allclose(u / np.linalg.norm(u), v_z / np.linalg.norm(v_z), atol=1e-3),
          f"f_D component: v_z {v_z[3]:+.6f} against D v_z {u[3]:+.6f}, a factor of "
          f"{abs(v_z[3] / u[3]):.1f}")
    check("the tangent is used unnormalized, as a unit step in z",
          abs(np.linalg.norm(u) - 1.0) > 1e-6, f"||D v_z|| = {np.linalg.norm(u):.6f}")

    # 7b. the artifact's own checks on that direction
    crb_art = load(CRB)
    nn_art = crb_art["branches"]["multinomial_fixed_launch"]["geometry"]["near_null"]
    check("J_psi @ (D v_z) equals (J_psi D) @ v_z in the artifact",
          nn_art["pushforward_consistency"]["passes"],
          f"{nn_art['pushforward_consistency']['normalized_max_diff']:.3e}")
    check("the whitened response along the direction equals the smallest singular value",
          nn_art["whitened_response_equals_smallest_singular_value"]["passes"],
          f"{nn_art['whitened_response_equals_smallest_singular_value']['relative_difference']:.3e}")

    # 7c. with the coordinates fixed, f_D no longer limits the near-null step
    npl = [p for p in plans if p["target"] == "near_null"][0]
    i_bind = npl["binding_cap"]["binding_component_index"]
    i_sd = labs.index("StO2_D")
    move_sd = abs(max(npl["h_absolute"]) * u[i_sd])
    check("the near-null step is no longer limited by f_D",
          labs[i_bind] != "f_D", f"binding component is {labs[i_bind]}")
    check("the near-null step moves deep StO2 by about 0.078",
          abs(move_sd - 0.078) < 0.002, f"deep StO2 displacement {move_sd:.6f}")

    # 8. random number design
    rnd = vspec["random_number_design"]
    seeds = [rnd["seed_base"] + rnd["stride"] * k for k in range(64)]
    prod = [43000 + 1000 * k for k in range(32)]
    clash = sorted(set(seeds) & set(prod))
    check("replicate seeds are distinct and disjoint from the production pool",
          len(set(seeds)) == len(seeds) and not clash,
          f"64 candidate seeds, {len(clash)} collide with the production pool"
          + (f": {clash}" if clash else ""))

    # 9. the full-vector statistic acts on paired differences and retains at most n-2 components
    fv = vspec["observables"]["full_radial_vector"]
    kmax = fv["declared_max_retained_components"]
    nr = 8
    sh = rng.normal(size=(nr, 1)) * 1.0
    base = rng.normal(size=(1, 40))
    ref_v = base + sh + rng.normal(scale=1e-3, size=(nr, 40))
    ad_v = base + sh + rng.normal(scale=1e-3, size=(nr, 40))
    off_v = base + sh + rng.normal(scale=1e-3, size=(nr, 40)) + 0.05
    st_null = pc_standardized_paired_mean_discrepancy(ad_v - ref_v, kmax)
    st_off = pc_standardized_paired_mean_discrepancy(off_v - ref_v, kmax)
    check("paired-mean discrepancy retains at most n_replicates - 2 components",
          st_null["retained_components"] <= nr - 2
          and st_null["retained_components"] == min(nr - 2, kmax),
          f"{st_null['retained_components']} retained from {nr} replicates, cap {kmax}")
    check("paired-mean discrepancy separates an offset estimator from a matching one",
          st_off["statistic"] > 100.0 * max(st_null["statistic"], 1e-12),
          f"matching {st_null['statistic']:.3g} against offset {st_off['statistic']:.3g}")
    check("paired-mean discrepancy is not labelled a chi-square and carries no p-value",
          st_off["not_a_chi_square"] is True and st_off["p_value"] is None
          and st_off["name"] == fv["statistic_name"])
    check("too few replicates yield no statistic rather than a degenerate one",
          pc_standardized_paired_mean_discrepancy(rng.normal(size=(2, 5)), kmax)
          ["statistic"] is None)

    # 10. the pMC precondition is declared and its status comes from the gate result
    st = replay_kernel_status()
    check("pMC availability is decided by the gate result, not by file presence",
          "precondition" in st and "vendored" in st
          and (st["vendored"] is False or "gate_summary" in st),
          f"kernel present {st.get('kernel_present')}, gate present "
          f"{st.get('gate_artifact_present')}, pMC available {st['vendored']}")

    # 11. pilot wiring, with a synthetic forward model and no transport. The point is that the
    # three estimators see the SAME key array inside a replicate and that every raw array is kept.
    for pmc_on in (True, False):
        seen = []
        n_obs = 6 * len(vspec["operating_point"]["wavelengths_nm"])

        def fake_keys(seed):
            return ("keys", seed)

        def fake_refl(p_, keys):
            seen.append(("refl", keys))
            return np.asarray(p_, float).sum() * np.ones(n_obs) * (1.0 + 1e-6 * hash(keys) % 3)

        def fake_ad(p_, keys, t):
            seen.append(("ad", keys))
            return np.ones(n_obs) * float(np.sum(t))

        def fake_pmc(p_, keys, t):
            seen.append(("pmc", keys))
            return np.ones(n_obs) * float(np.sum(t)) * 1.5

        fwd = {"refl": fake_refl, "ad": fake_ad, "pmc": fake_pmc, "n_wl": 4,
               "split": fake_keys}
        pl1 = [p for p in plans if p["target"] == "f_U"]
        nrep = 5
        got = run_pilot(vspec, fwd, pl1, nrep, pmc_on, lambda v: np.asarray(v, float),
                        "synthetic_wiring_test")
        e = got["f_U"]
        keys_per_rep = {}
        for kind, kk in seen:
            keys_per_rep.setdefault(kk, set()).add(kind)
        want = {"refl", "ad", "pmc"} if pmc_on else {"refl", "ad"}
        check(f"one key array per replicate serves every estimator (pMC {'on' if pmc_on else 'off'})",
              len(keys_per_rep) == nrep and all(v == want for v in keys_per_rep.values()),
              f"{len(keys_per_rep)} distinct key arrays, roles {sorted(want)}")
        check(f"raw per-replicate arrays are stored (pMC {'on' if pmc_on else 'off'})",
              len(e["raw_per_replicate"]) == nrep
              and all(set(r) >= {"replicate", "seed", "difference_quotients_by_h",
                                 "reference_two_term", "reference_three_term", "ad", "pmc"}
                      for r in e["raw_per_replicate"])
              and len(e["raw_per_replicate"][0]["difference_quotients_by_h"])
              == len(vspec["step_size_rule"]["h_relative_grid"]),
              f"{nrep} replicates, {len(e['raw_per_replicate'][0]['difference_quotients_by_h'])} "
              f"step sizes each")
        # the paired differences must be reconstructible from the stored arrays alone
        ref = np.array([r["reference_two_term"] for r in e["raw_per_replicate"]])
        adv = np.array([r["ad"] for r in e["raw_per_replicate"]])
        check(f"paired AD difference is reconstructible from stored arrays "
              f"(pMC {'on' if pmc_on else 'off'})",
              np.allclose(e["inference"]["delta_AD_mean"], (adv - ref).mean(axis=0),
                          rtol=0, atol=1e-15))
        if pmc_on:
            pmv = np.array([r["pmc"] for r in e["raw_per_replicate"]])
            check("paired pMC and AD-minus-pMC differences are reconstructible from stored arrays",
                  np.allclose(e["inference"]["delta_pMC_mean"], (pmv - ref).mean(axis=0),
                              rtol=0, atol=1e-15)
                  and np.allclose(e["inference"]["delta_AD_pMC_mean"],
                                  (adv - pmv).mean(axis=0), rtol=0, atol=1e-15))
        else:
            check("with pMC off the pilot stores None and issues no pMC verdict",
                  all(r["pmc"] is None for r in e["raw_per_replicate"])
                  and e["inference"]["z_pMC"] is None
                  and e["inference"]["verdict_pmc_vs_reference"][0] == "unavailable"
                  and e["inference"]["verdict_ad_vs_pmc"][0] == "unavailable"
                  and e["inference"]["paired_mean_discrepancy"]["pmc"] is None)
        check(f"truncation diagnostic is reported per component (pMC {'on' if pmc_on else 'off'})",
              len(e["inference"]["truncation_diagnostic_three_minus_two_over_se"]) == n_obs)

    # 12. admissibility: a component carrying only rounding must be excluded, not reported as a
    # cheap win. This is the failure the radial smoke run exposed.
    nrep = 6
    rg2 = np.random.default_rng(99)
    healthy_ref = 1e-4 + rg2.normal(scale=1e-6, size=(nrep, 3))
    healthy_ad = healthy_ref + rg2.normal(scale=1e-6, size=(nrep, 3))
    # a sparse component: nonzero in 2 of 6 replicates, and AD identical to the reference there
    sparse = np.zeros((nrep, 1))
    sparse[3, 0], sparse[5, 0] = -3.5e-7, -1.6e-6
    # a pathwise-regime component: every replicate nonzero, but AD tracks the reference so
    # closely that the paired scatter is a tiny fraction of the reference's own
    pw = (1e-4 + rg2.normal(scale=1e-6, size=(nrep, 1)))
    ref_t = np.hstack([healthy_ref, sparse, pw])
    ad_t = np.hstack([healthy_ad, sparse.copy(), pw + rg2.normal(scale=1e-12, size=(nrep, 1))])
    adm = component_admissibility(ref_t, ad_t, vspec)
    check("admissibility accepts healthy components and rejects both degenerate kinds",
          adm["admissible"] == [True, True, True, False, False]
          and adm["n_admissible"] == 3,
          f"{adm['n_admissible']}/5 admissible; reasons "
          f"{[r for r in adm['reason_if_inadmissible'] if r]}")
    check("the sparse component is rejected for lacking ensemble support",
          "nonzero in only 2 of 6" in (adm["reason_if_inadmissible"][3] or ""),
          adm["reason_if_inadmissible"][3])
    check("the pathwise-regime component is rejected on the paired-scatter ratio",
          "below the floor" in (adm["reason_if_inadmissible"][4] or "")
          and adm["paired_scatter_ratio"][4] < vspec["component_admissibility"]
          ["paired_scatter_floor"],
          f"ratio {adm['paired_scatter_ratio'][4]:.3g}")
    check("healthy components sit well above the floor",
          all(adm["paired_scatter_ratio"][j] > 0.3 for j in range(3)),
          f"ratios {[round(adm['paired_scatter_ratio'][j], 3) for j in range(3)]}")
    check("admissibility rules are recorded in the spec, not only in code",
          "component_admissibility" in vspec
          and vspec["component_admissibility"]["rules_fixed_before_the_radial_pilot_was_run"]
          is True)

    return res, fail, plans


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pilot", action="store_true",
                    help="run the pilot transport. Requires the replay equivalence gate to have "
                         "passed if pMC is to be included; otherwise pMC is omitted and the run "
                         "is labelled incomplete.")
    ap.add_argument("--pilot-photons", type=int, default=200000)
    ap.add_argument("--pilot-replicates", type=int, default=6)
    ap.add_argument("--jaxpad-root", default=os.environ.get(
        "JAXPAD_ROOT", os.path.normpath(os.path.join(ROOT, "..", "jax-pad"))))
    ap.add_argument("--pilot-out", default=os.path.join(ROOT, "results",
                                                       "jacobian_validation_pilot.json"))
    ap.add_argument("--pilot-radial", action="store_true",
                    help="store the full 120-component radial output at every stage and derive "
                         "all three pre-registered observables offline from it. Same targets, "
                         "replicates, photon budget, h grid and seeds as the total-escape pilot: "
                         "a deterministic rerun that completes the observable set, not an "
                         "independent repeat.")
    ap.add_argument("--pilot-radial-out", default=os.path.join(
        ROOT, "results", "jacobian_validation_pilot_radial.json"))
    a = ap.parse_args()
    vspec, bspec = load(VSPEC), load(BSPEC)

    if a.selftest:
        res, fail, _ = selftest(vspec, bspec)
        for r in res:
            print(f"  {'PASS' if r['pass'] else 'FAIL'}  {r['check']}"
                  + (f"  [{r['detail']}]" if r["detail"] else ""))
        print(f"  {sum(r['pass'] for r in res)}/{len(res)} checks passed")
        return 1 if fail else 0

    if a.dry_run:
        labs = vspec["operating_point"]["psi_labels"]
        psi = np.array(vspec["operating_point"]["nominal_psi"], float)
        lo, hi = bounds_arrays(vspec)
        res, fails, plans = selftest(vspec, bspec)
        if fails:
            raise SystemExit(f"self-tests must pass before a dry run is meaningful: {fails}")
        print("  step grid, every point checked against the declared bounds before transport\n")
        print(f"  {'target':<11} {'scale':>10} {'h_max':>11} {'largest move':>13} "
              f"{'margin':>10}  binding cap")
        for p in plans:
            i = p["binding_cap"]["binding_component_index"]
            print(f"  {p['target']:<11} {p['scale']:>10.6g} {max(p['h_absolute']):>11.6g} "
                  f"{p['largest_component_displacement']:>13.6g} "
                  f"{p['margin_moving_components']:>10.3g}  {p['binding_cap']['cap']} "
                  f"on {labs[i]}")
        print("  margin is the distance to the nearest bound over the components the step moves")
        print("\n  per-target detail, showing only the components a step moves")
        for p in plans:
            print(f"  {p['target']}")
            for i, rg in p["moving_component_range"].items():
                print(f"    {labs[i]:<11} nominal {psi[i]:<11.6g} bounds [{lo[i]}, {hi[i]}]  "
                      f"swept over [{rg[0]:.6g}, {rg[1]:.6g}]")
            print(f"    h = " + ", ".join(f"{h:.4g}" for h in p["h_absolute"]))

        measured = {
            "reference_photons_per_wavelength": 200000,
            "forward_seconds_at_reference_photons": 372.0 / 7.0,
            "ad_jvp_in_forward_equivalents": 2.0,
            "pmc_replay_in_forward_equivalents": 1.5,
            "basis": "the production pool measured 372 s per seed for forward plus a six-column "
                     "jacfwd at 200000 photons per wavelength over four wavelengths with "
                     "max_steps 4000; forward-mode AD with six tangents costs about seven "
                     "forward evaluations, so one forward evaluation is taken as 372/7 s. The AD "
                     "and pMC factors are conservative placeholders.",
        }
        pmc_ok = replay_kernel_status()["vendored"]
        pilot_now = cost_estimate(vspec, a.pilot_photons, a.pilot_replicates, 2, measured,
                                  include_pmc=False)
        pilot_future = cost_estimate(vspec, a.pilot_photons, a.pilot_replicates, 2, measured,
                                     include_pmc=True)
        full_now = cost_estimate(vspec, a.pilot_photons, a.pilot_replicates, 7, measured,
                                 include_pmc=False)
        full_future = cost_estimate(vspec, a.pilot_photons, a.pilot_replicates, 7, measured,
                                    include_pmc=True)
        print(f"\n  cost. pMC is {'available' if pmc_ok else 'NOT available'}, so the runnable "
              f"plan is {'AD and pMC together against the ensemble reference' if pmc_ok else 'AD against the ensemble reference only'}.")
        print(f"  {'plan':<44} {'FE/rep/target':>14} {'total FE':>10} {'hours':>8}")
        tag_ad = "runnable now" if not pmc_ok else "reduced scope"
        tag_both = "runnable now" if pmc_ok else "blocked, for budgeting"
        for lab, c in ((f"pilot, AD only ({tag_ad})", pilot_now),
                       (f"pilot, AD plus pMC ({tag_both})", pilot_future),
                       (f"all seven targets, AD only ({tag_ad})", full_now),
                       (f"all seven targets, AD plus pMC ({tag_both})", full_future)):
            print(f"  {lab:<44} {c['forward_equivalents_per_replicate_per_target']:>14} "
                  f"{c['total_forward_equivalents']:>10} "
                  f"{c['estimated_wall_clock_hours']:>8.2f}")
        print(f"  {a.pilot_replicates} replicates, {a.pilot_photons} photons per wavelength, "
              f"{pilot_now['estimated_seconds_per_forward_evaluation']:.1f} s per forward "
              f"evaluation (estimated)")
        print(f"  status: {pilot_now['status']}")
        st = replay_kernel_status()
        print(f"\n  perturbation Monte Carlo: available = {st['vendored']}")
        if st["vendored"]:
            g = st["gate_summary"]
            print(f"    equivalence gate passed {g['n_passed']}/{g['n_cases']} cases over "
                  f"{g['total_photon_comparisons']} photon comparisons")
            print(f"    detector bin and escaping weight bitwise identical: "
                  f"{g['det_id_bitwise_identical_in_all_cases']} / "
                  f"{g['weight_bitwise_identical_in_all_cases']}; negative control rejected: "
                  f"{g['negative_control_rejected']}")
            print("    the pilot may include pMC, so no incompleteness label is needed")
        else:
            print(f"    {st['blocking']}")
            print(f"    any pilot run now is labelled: {vspec['pilot']['label']!r}")
        out = {
            "mode": "dry run, design freeze; no transport was run",
            "validation_spec": "spec/validation_spec.json",
            "operating_point": vspec["operating_point"],
            "step_plans": plans,
            "cost_estimate_pilot_ad_only_runnable_now": pilot_now,
            "cost_estimate_pilot_future_ad_plus_pmc": pilot_future,
            "cost_estimate_full_seven_targets_ad_only": full_now,
            "cost_estimate_full_seven_targets_future_ad_plus_pmc": full_future,
            "pilot_label": None if pmc_ok else vspec["pilot"]["label"],
            "pmc_counted_in_runnable_plan": bool(pmc_ok),
            "runnable_plan": ("pilot, AD plus pMC" if pmc_ok else "pilot, AD only"),
            "cost_basis": measured,
            "perturbation_mc_status": st,
            "selftest": {"checks": len(res), "passed": sum(r["pass"] for r in res),
                         "all_passed": not fails},
            "note_on_near_null_step": (
                "with the direction taken as the pushforward D v_z rather than v_z, the step is "
                "capped by StO2_D and moves deep StO2 by 0.078, the same order as the StO2_D "
                "column itself. The earlier f_D cap was an artifact of using the z-space singular "
                "vector as if it were a psi-space direction."),
            "near_null_direction_psi_tangent": near_null_direction(vspec).tolist(),
            "near_null_direction_z": near_null_z_vector(vspec).tolist(),
        }
        dst = os.path.join(ROOT, "results", "jacobian_validation_design.json")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\n  wrote {os.path.relpath(dst, ROOT)}")
        print("  no transport was run")
        return 0

    if a.pilot_radial:
        res, fails, plans = selftest(vspec, bspec)
        if fails:
            raise SystemExit(f"self-tests must pass before the pilot runs: {fails}")
        st = replay_kernel_status()
        if not st["vendored"]:
            raise SystemExit(f"pMC is required for this run and is unavailable: {st['blocking']}")
        targets = list(vspec["pilot"]["columns"])
        pl = build_plans(vspec, targets)
        print(f"  radial-observable pilot: targets {targets}, {a.pilot_replicates} replicates, "
              f"{a.pilot_photons} photons per wavelength")
        print(f"  full 120-component radial output stored at every stage; nothing reduced before "
              f"saving")
        print(f"  gate {st['gate_artifact']} backend {st['gate_backend']}, pMC included")
        fwd = build_forward(vspec, a.jaxpad_root, a.pilot_photons)
        warm = fwd["warm_up"](np.array(vspec["operating_point"]["nominal_psi"], float),
                              fwd["split"](vspec["random_number_design"]["seed_base"]),
                              np.array(pl[0]["direction"], float))
        print(f"  compiled once: forward {warm['refl']}s, AD jvp {warm['ad']}s, "
              f"pMC replay {warm['pmc']}s")
        t0 = time.time()
        stored = run_pilot_radial(vspec, fwd, pl, a.pilot_replicates)
        elapsed = round(time.time() - t0, 1)
        derived = derive_observables(stored, vspec)
        payload = {
            "mode": "radial-observable pilot",
            "relationship_to_the_previous_pilot":
                "a deterministic rerun with the same targets, replicates, photon budget, h grid, "
                "seeds and PRNG key pairing. It completes the pre-registered observable set by "
                "storing the full radial output; it is not an independent repeat experiment and "
                "the total-escape pilot is retained unchanged.",
            "forms_final_conclusions": False,
            "is_a_validation": False,
            "question_it_answers": "whether radial information lowers the photon-budget "
                                   "requirement into a feasible range",
            "photon_budget_unchanged": True,
            "pmc_available": True, "replay_equivalence_status": st,
            "n_replicates": a.pilot_replicates,
            "n_photons_per_wavelength": a.pilot_photons,
            "compile_seconds_paid_once": warm, "elapsed_s": elapsed,
            "observables_derived_offline_from_one_set_of_stored_arrays": list(OBSERVABLE_ORDER),
            "no_observable_cost_its_own_transport_run": True,
            "targets": derived,
            "raw": stored}
        os.makedirs(os.path.dirname(a.pilot_radial_out), exist_ok=True)
        with open(a.pilot_radial_out, "w") as fh:
            json.dump(payload, fh, indent=2)
        for t, e in derived.items():
            print(f"  --- {t} ---")
            for nm in OBSERVABLE_ORDER:
                i = e["by_observable"][nm]
                b = i["budget_multiplier_for_reference_resolution_at_ad_pmc_gap_scale"]
                print(f"    {nm:<30} n={i['n_components']:>3}  best multiplier "
                      f"{b['best_multiplier']} at {b['best_component_label']}  "
                      f"PC-subspace {b['pc_subspace_multiplier']}")
        print(f"  wrote {os.path.relpath(a.pilot_radial_out, ROOT)}")
        return 0

    if a.pilot:
        res, fails, plans = selftest(vspec, bspec)
        if fails:
            raise SystemExit(f"self-tests must pass before the pilot runs: {fails}")
        st = replay_kernel_status()
        pmc_on = bool(st["vendored"])
        targets = list(vspec["pilot"]["columns"])
        pl = [p for p in build_plans(vspec, targets)]
        print(f"  pilot targets {targets}, {a.pilot_replicates} replicates, "
              f"{a.pilot_photons} photons per wavelength")
        print(f"  pMC {'included, equivalence gate passed' if pmc_on else 'OMITTED: ' + str(st['blocking'])}")
        fwd = build_forward(vspec, a.jaxpad_root, a.pilot_photons)
        reduce_to = (lambda v: observables(v, vspec)["total_escape_per_wavelength"])
        warm = fwd["warm_up"](np.array(vspec["operating_point"]["nominal_psi"], float),
                              fwd["split"](vspec["random_number_design"]["seed_base"]),
                              np.array(pl[0]["direction"], float))
        print(f"  compiled once: forward {warm['refl']}s, AD jvp {warm['ad']}s, "
              f"pMC replay {warm['pmc']}s")
        t0 = time.time()
        got = run_pilot(vspec, fwd, pl, a.pilot_replicates, pmc_on, reduce_to,
                        vspec["pilot"]["observable"])
        payload = {"mode": "pilot", "forms_final_conclusions": False,
                   "observable": vspec["pilot"]["observable"],
                   "pmc_available": pmc_on,
                   "label": None if pmc_on else vspec["pilot"]["label"],
                   "replay_equivalence_status": st,
                   "compile_seconds_paid_once": warm,
                   "n_replicates": a.pilot_replicates,
                   "n_photons_per_wavelength": a.pilot_photons,
                   "elapsed_s": round(time.time() - t0, 1),
                   "targets": got}
        os.makedirs(os.path.dirname(a.pilot_out), exist_ok=True)
        with open(a.pilot_out, "w") as fh:
            json.dump(payload, fh, indent=2)
        for name, e in got.items():
            i = e["inference"]
            print(f"  {name}: z_AD {np.array2string(np.array(i['z_AD']), precision=2)} -> "
                  f"{i['verdict_ad_vs_reference']}")
        print(f"  wrote {os.path.relpath(a.pilot_out, ROOT)}")
        return 0

    ap.error("one of --selftest, --dry-run, --pilot or --pilot-radial is required")


if __name__ == "__main__":
    raise SystemExit(main())
