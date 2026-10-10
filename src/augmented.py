"""augmented.py -- nuisance marginalization for MERGE_CHECKS item 2, in both directions.

The production parameter vector is left alone. This module adds an augmented vector used ONLY for
nuisance validation:

    theta = [psi, eta],   psi = [StO2_U, f_U, StO2_D, f_D, mua_DermR, mua_SubQ]

Sweeping nuisance values and recomputing the six-parameter bound at each one treats the nuisance as
KNOWN inside every draw, which is conditional sensitivity. The question is a joint estimation
problem, so both marginalizations are computed from the block Fisher matrix of theta:

    F = [[F_pp, F_pe], [F_ep, F_ee]]

    F_psi_eff = F_pp - F_pe pinv(F_ee) F_ep      information about psi that survives estimating eta
    F_eta_eff = F_ee - F_ep pinv(F_pp) F_pe      information about eta that survives estimating psi

The second one is not symmetry for its own sake. A statement that the data do not constrain the
scattering nuisance must come from F_eta_eff, because bare F_ee assumes psi is KNOWN, which is
exactly the assumption this check exists to remove. A nuisance can look measurable in F_ee and be
unmeasurable once the physiological parameters have to be estimated from the same photons.

Both are computed by projection as the numerical main path, because forming F squares the condition
number and a pseudo-inverse of a near-singular block is the worst place to be careless:

    J_psi_perp = (I - P_eta) J_w_psi,   F_psi_eff = J_psi_perp^T J_psi_perp
    J_eta_perp = (I - P_psi) J_w_eta,   F_eta_eff = J_eta_perp^T J_eta_perp

The Schur forms are kept as cross-checks, not as sources.

Estimability is decided PER PARAMETER. A rank-deficient information matrix does not make every
bound infinite: it makes infinite exactly those parameters whose unit vector has a component in the
null space. Wiping out the bound of a parameter that is still identifiable, because some other
direction is not, would throw away a real result. So for each j the null-space leakage
||N^T e_j|| is computed against a recorded tolerance, and a finite e_j^T F^+ e_j is reported where
it is small.

When the nullity exceeds one there is no unique least-identifiable direction, and reporting one
would be an artifact of whichever basis the eigensolver happened to return. The null space is then
reported as a subspace, and estimator comparisons use principal angles between subspaces rather
than a cosine between vectors. A single near-null direction is reported only when the matrix is
full rank AND its smallest eigenvalue is separated from the next by a declared ratio.
"""
from __future__ import annotations

import numpy as np

PSI_LABELS = ("StO2_U", "f_U", "StO2_D", "f_D", "mua_DermR", "mua_SubQ")


# ---------------------------------------------------------------------------------------------
# eta coordinates. Log for every strictly positive quantity, so positivity is automatic.
# ---------------------------------------------------------------------------------------------

def eta_column_scale(n_eta):
    """Unity. eta is dimensionless because every entry is a log scale or an additive exponent."""
    return np.ones(int(n_eta), dtype=float)


def theta_column_scale(psi_scale, n_eta):
    return np.concatenate([np.asarray(psi_scale, float), eta_column_scale(n_eta)])


def boundaries_from_eta(nominal_cm, log_d_epi, log_d_pap, log_d_ret):
    """Layer boundaries from log thickness scales.

    Only the named layer's own thickness changes; everything below shifts by the same absolute
    amount, so every other layer keeps its thickness and the dimension stays interpretable as a
    depth change rather than a global stretch.
    """
    t = np.diff(np.concatenate([[0.0], np.asarray(nominal_cm, float)]))
    s = np.exp([log_d_epi, log_d_pap, log_d_ret])
    t_new = t.copy()
    t_new[0] *= s[0]
    t_new[1] *= s[0]
    t_new[2] *= s[1]
    t_new[4] *= s[2]
    return np.cumsum(t_new)


def mus_factor_from_eta(log_mus_scale, mus_slope, wavelengths_nm, lam0_nm):
    w = np.asarray(wavelengths_nm, float)
    return np.exp(log_mus_scale) * (w / float(lam0_nm)) ** (-float(mus_slope))


def mel_fraction_from_eta(log_f_mel, nominal):
    return float(nominal) * np.exp(float(log_f_mel))


# ---------------------------------------------------------------------------------------------
# Per-parameter estimability
# ---------------------------------------------------------------------------------------------

def _rank_tol(shape, s):
    return max(shape) * np.finfo(float).eps * (float(s[0]) if len(s) else 0.0)


def estimability(F, scale=None, labels=None, leakage_tol=1e-08):
    """Per-parameter bounds from a possibly rank-deficient information matrix.

    A parameter j is individually estimable when its unit vector has essentially no component in
    the null space, measured as ||N^T e_j||. Where it does, the bound is infinite and the value is
    null; where it does not, a finite e_j^T F^+ e_j is reported. The two cases coexist: a lost
    direction does not erase the bound of a parameter that is still identifiable.

    `scale` maps the scaled coordinate to the reported one, Var(psi_j) = scale_j^2 (F^+)_{jj}.
    """
    F = np.asarray(F, float)
    F = 0.5 * (F + F.T)
    n = F.shape[0]
    lam, V = np.linalg.eigh(F)
    order = np.argsort(lam)[::-1]
    lam, V = lam[order], V[:, order]
    tol = _rank_tol(F.shape, lam[lam > 0] if np.any(lam > 0) else np.array([0.0]))
    keep = lam > tol
    rank = int(keep.sum())
    nullity = n - rank
    N = V[:, ~keep]                                     # null-space basis, n x nullity
    Fp = V[:, keep] @ np.diag(1.0 / lam[keep]) @ V[:, keep].T    # pseudo-inverse

    sc = np.ones(n) if scale is None else np.asarray(scale, float)
    lab = [f"p{j}" for j in range(n)] if labels is None else list(labels)

    per = {}
    for j in range(n):
        leak = float(np.linalg.norm(N[j, :])) if nullity else 0.0
        if leak < leakage_tol:
            var = float(Fp[j, j]) * sc[j] ** 2
            per[lab[j]] = {"individually_estimable": True,
                           "null_space_leakage": leak,
                           "bound": "finite",
                           "sqrt_crlb": float(np.sqrt(max(var, 0.0))),
                           "variance": var}
        else:
            per[lab[j]] = {"individually_estimable": False,
                           "null_space_leakage": leak,
                           "bound": "infinite",
                           "sqrt_crlb": None, "variance": None,
                           "reason": "the unit vector of this parameter has a component in the "
                                     "null space, so no finite bound exists for it"}
    return {
        "n_parameters": n, "rank": rank, "nullity": nullity,
        "rank_tolerance": float(tol), "full_rank": bool(nullity == 0),
        "eigenvalues_descending": lam.tolist(),
        "condition_number": (float(lam[0] / lam[-1]) if lam[-1] > 0 else float("inf")),
        "null_space_basis": N.T.tolist() if nullity else [],
        "null_space_leakage_tolerance": leakage_tol,
        "per_parameter": per,
        "n_individually_estimable": int(sum(v["individually_estimable"] for v in per.values())),
        "note": "estimability is decided per parameter. A rank-deficient matrix makes infinite "
                "exactly those parameters whose unit vector leaks into the null space; the others "
                "keep a finite bound.",
    }


def near_null_report(F, scale=None, labels=None, separation_ratio=10.0, leakage_tol=1e-08):
    """The least-identifiable structure, reported as a direction only when that is well defined.

    A single direction and a cosine are reported only when the matrix is full rank and its
    smallest eigenvalue is separated from the next by `separation_ratio`. Otherwise the smallest
    eigenvalues are degenerate, or there is a null space, and any one eigenvector returned by the
    solver is a basis artifact. In that case a subspace is reported and comparisons must use
    principal angles.
    """
    F = np.asarray(F, float)
    F = 0.5 * (F + F.T)
    n = F.shape[0]
    lam, V = np.linalg.eigh(F)                           # ascending
    tol = _rank_tol(F.shape, np.sort(lam[lam > 0])[::-1] if np.any(lam > 0) else np.array([0.0]))
    nullity = int((lam <= tol).sum())
    sc = np.ones(n) if scale is None else np.asarray(scale, float)
    lab = [f"p{j}" for j in range(n)] if labels is None else list(labels)

    out = {"nullity": nullity, "rank": n - nullity, "rank_tolerance": float(tol),
           "eigenvalues_ascending": lam.tolist()}

    if nullity >= 1:
        B = V[:, :nullity]
        out.update({
            "single_direction_reported": False,
            "reason": f"the information matrix has a {nullity}-dimensional null space, so there "
                      f"is no unique least-identifiable direction and any single eigenvector "
                      f"would be a basis artifact",
            "null_space_dimension": nullity,
            "null_space_basis_z": B.T.tolist(),
            "null_space_basis_psi_tangents": (np.asarray(sc)[:, None] * B).T.tolist(),
            "compare_with": "principal angles or projector distance, not a cosine"})
        return out

    sep = float(lam[1] / lam[0]) if lam[0] > 0 else float("inf")
    if sep < separation_ratio:
        out.update({
            "single_direction_reported": False,
            "reason": f"the two smallest eigenvalues are separated by only {sep:.3g}, below the "
                      f"declared ratio of {separation_ratio}, so the least-identifiable direction "
                      f"is not well determined and a single eigenvector would be a basis artifact",
            "eigenvalue_separation_ratio": sep,
            "declared_separation_ratio": separation_ratio,
            "near_null_subspace_dimension": 2,
            "near_null_subspace_basis_z": V[:, :2].T.tolist(),
            "compare_with": "principal angles or projector distance, not a cosine"})
        return out

    v = V[:, 0]
    if v[int(np.argmax(np.abs(v)))] < 0:
        v = -v
    tan = np.asarray(sc, float) * v
    out.update({
        "single_direction_reported": True,
        "eigenvalue_separation_ratio": sep,
        "declared_separation_ratio": separation_ratio,
        "direction_z": v.tolist(), "tangent_psi": tan.tolist(),
        "dominant_component": lab[int(np.argmax(np.abs(tan)))],
        "smallest_eigenvalue": float(lam[0]),
        "coordinate_note": "the information is in the scaled coordinate z with psi = D z, so the "
                           "psi-space direction is the pushforward D v"})
    return out


def subspace_distance(basis_a, basis_b):
    """Principal angles between two subspaces, plus the projector distance.

    The right comparison when the least-identifiable structure is a subspace rather than a
    direction. A cosine between arbitrarily chosen basis vectors would measure the eigensolver's
    choice, not the geometry.
    """
    A = np.atleast_2d(np.asarray(basis_a, float))
    B = np.atleast_2d(np.asarray(basis_b, float))
    if A.shape[0] <= A.shape[1]:
        A = A.T
    if B.shape[0] <= B.shape[1]:
        B = B.T
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    ang = np.arccos(s)
    Pa, Pb = Qa @ Qa.T, Qb @ Qb.T
    return {"principal_angles_rad": ang.tolist(),
            "principal_angles_deg": np.degrees(ang).tolist(),
            "largest_principal_angle_deg": float(np.degrees(ang).max()),
            "projector_distance": float(np.linalg.norm(Pa - Pb, 2)),
            "dim_a": int(Qa.shape[1]), "dim_b": int(Qb.shape[1])}


# ---------------------------------------------------------------------------------------------
# Both marginalizations
# ---------------------------------------------------------------------------------------------

def _project_off(J_keep, J_drop):
    """(I - P_drop) J_keep, with the projector built from the column space of J_drop."""
    U, s, _ = np.linalg.svd(J_drop, full_matrices=False)
    tol = _rank_tol(J_drop.shape, s)
    keep = s > tol
    Q = U[:, keep]
    return J_keep - Q @ (Q.T @ J_keep), int(keep.sum()), float(tol), s


def block_information(J_w_theta, n_psi, support_mask=None, require_mask=True):
    """Both effective information matrices, by projection, with Schur cross-checks.

    `support_mask` selects the observable components that carry a comparison. It is applied to the
    rows of the whitened Jacobian, so an excluded component contributes no information to either
    block. `require_mask` defaults to TRUE and makes it mandatory. It defaulted to False, which left
    every production caller permissive and contradicted the spec: a Fisher matrix built without the
    mask the equivalence test produced lets components that were never compared contribute
    information, and the gate's mask becomes a record of nothing. A synthetic analysis that
    genuinely wants every row must pass an explicit all-true mask and say so, rather than relying
    on a default.

    F_psi_eff is the information about psi that survives estimating eta. F_eta_eff is the
    information about eta that survives estimating psi, and it is what a claim about the nuisance
    must be read from: bare F_ee assumes psi is known.
    """
    J = np.asarray(J_w_theta, float)
    if require_mask and support_mask is None:
        raise ValueError(
            "support_mask is required here. The equivalence test excluded components that carry "
            "no comparison, and building a Fisher matrix without that mask would let them "
            "contribute information they were never checked for.")
    mask_info = {"support_mask_applied": False, "n_rows_used": int(J.shape[0]),
                 "n_rows_excluded": 0}
    if support_mask is not None:
        mask = np.asarray(support_mask, bool)
        if mask.shape[0] != J.shape[0]:
            raise ValueError(f"support_mask has {mask.shape[0]} entries for {J.shape[0]} rows")
        J = J[mask, :]
        mask_info = {"support_mask_applied": True, "n_rows_used": int(mask.sum()),
                     "n_rows_excluded": int((~mask).sum()),
                     "contract": "excluded components contribute no information to either block"}
    p = int(n_psi)
    Jp, Je = J[:, :p], J[:, p:]

    F = J.T @ J
    F_pp, F_pe, F_ee = F[:p, :p], F[:p, p:], F[p:, p:]

    J_psi_perp, eta_rank, eta_tol, s_eta = _project_off(Jp, Je)
    J_eta_perp, psi_rank, psi_tol, s_psi = _project_off(Je, Jp)
    F_psi_eff = 0.5 * ((J_psi_perp.T @ J_psi_perp) + (J_psi_perp.T @ J_psi_perp).T)
    F_eta_eff = 0.5 * ((J_eta_perp.T @ J_eta_perp) + (J_eta_perp.T @ J_eta_perp).T)

    def agree(A, B):
        d = max(float(np.max(np.abs(B))), np.finfo(float).tiny)
        return float(np.max(np.abs(A - B)) / d)

    schur_psi = F_pp - F_pe @ np.linalg.pinv(F_ee) @ F_pe.T
    schur_eta = F_ee - F_pe.T @ np.linalg.pinv(F_pp) @ F_pe

    s_ee = np.linalg.svd(F_ee, compute_uv=False)
    s_ppb = np.linalg.svd(F_pp, compute_uv=False)
    tol_ee = _rank_tol(F_ee.shape, s_ee)
    tol_pp = _rank_tol(F_pp.shape, s_ppb)

    a_psi, a_eta = agree(schur_psi, F_psi_eff), agree(schur_eta, F_eta_eff)
    return {
        "F_pp": F_pp, "F_ee": F_ee, "F_pe": F_pe,
        "F_psi_eff": F_psi_eff, "F_eta_eff": F_eta_eff,
        "J_psi_perp": J_psi_perp, "J_eta_perp": J_eta_perp,
        "nuisance_column_rank": eta_rank, "nuisance_column_rank_tolerance": eta_tol,
        "nuisance_singular_values": s_eta.tolist(),
        "psi_column_rank": psi_rank, "psi_column_rank_tolerance": psi_tol,
        "psi_singular_values": s_psi.tolist(),
        "F_ee_rank": int((s_ee > tol_ee).sum()), "F_ee_rank_tolerance": float(tol_ee),
        "F_ee_condition_number": (float(s_ee[0] / s_ee[-1]) if s_ee[-1] > 0 else float("inf")),
        "F_ee_is_singular": bool((s_ee > tol_ee).sum() < F_ee.shape[0]),
        "F_pp_rank": int((s_ppb > tol_pp).sum()), "F_pp_rank_tolerance": float(tol_pp),
        "F_pp_condition_number": (float(s_ppb[0] / s_ppb[-1]) if s_ppb[-1] > 0 else float("inf")),
        "schur_vs_projection_psi": a_psi, "schur_vs_projection_eta": a_eta,
        "routes_agree": bool(a_psi < 1e-8 and a_eta < 1e-8),
        "reported_route": "projection",
        "why_projection_is_reported": "algebraically identical and it never forms a pseudo-inverse "
                                      "of a near-singular block; the Schur forms are cross-checks",
        "mask": mask_info,
        "why_F_eta_eff_and_not_F_ee": "bare F_ee assumes psi is known, which is the assumption this "
                                      "check exists to remove. A nuisance can look measurable in "
                                      "F_ee and be unmeasurable once psi must be estimated from the "
                                      "same photons.",
    }


def eta_posterior(F_eta_eff, prior_sd, labels=None):
    """Prior-informed posterior covariance for eta, named separately from the data-only CRLB.

    Cov = inverse(F_eta_eff + diag(1 / prior_sd^2)). The prior precision is positive definite, so
    the sum is invertible even where F_eta_eff is not, and a posterior SD therefore exists for
    every eta component whether or not the data alone identify it. That is precisely why it must
    not be reported under the same name as the data-only bound: the two answer different
    questions, and only the data-only bound says what the measurement itself constrains.
    """
    F = np.asarray(F_eta_eff, float)
    F = 0.5 * (F + F.T)
    sd0 = np.asarray(prior_sd, float)
    n = F.shape[0]
    lab = [f"e{j}" for j in range(n)] if labels is None else list(labels)
    P = F + np.diag(1.0 / sd0 ** 2)
    lam, V = np.linalg.eigh(P)
    Cov = V @ np.diag(1.0 / lam) @ V.T
    Cov = 0.5 * (Cov + Cov.T)
    sd = np.sqrt(np.diag(Cov))
    return {
        "name": "prior_informed_posterior",
        "distinct_from": "data_only_crlb",
        "why_distinct": "the posterior borrows the prior, so it is finite even where the data "
                        "alone identify nothing. Only the data-only bound says what the "
                        "measurement constrains.",
        "prior_sd": sd0.tolist(),
        "posterior_sd": {lab[j]: float(sd[j]) for j in range(n)},
        "posterior_over_prior_sd_ratio": {lab[j]: float(sd[j] / sd0[j]) for j in range(n)},
        "never_wider_than_the_prior": bool(np.all(sd <= sd0 * (1.0 + 1e-12))),
        "data_narrowed_the_prior": {lab[j]: bool(sd[j] < 0.95 * sd0[j]) for j in range(n)},
        "posterior_is_always_proper": "the prior precision is positive definite, so this inverse "
                                      "exists even when F_eta_eff is rank deficient",
    }


def direction_agreement(tan_a, tan_b):
    """|cos| between two psi-space tangents, sign-insensitive."""
    a, b = np.asarray(tan_a, float), np.asarray(tan_b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return abs(float(a @ b / (na * nb)))


# ---------------------------------------------------------------------------------------------
# Downstream certification. A margin on a Jacobian column does not certify a classification.
# ---------------------------------------------------------------------------------------------

def _range_basis(M):
    """An orthonormal basis for the column space, from singular vectors retained by the declared
    rank tolerance.

    QR was wrong here. For a rank-deficient block the QR factor includes arbitrary completion
    vectors that span no part of the actual column space, and principal angles computed against
    them are meaningless. The tolerance is the same max(m, n) * eps * s_max used everywhere else
    in this module, so the basis and the rank reported elsewhere agree.
    """
    M = np.asarray(M, float)
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    tol = _rank_tol(M.shape, s)
    keep = s > tol
    return U[:, keep], int(keep.sum()), float(tol), s


def nuisance_psi_principal_angles(J_w_psi, J_w_eta):
    """Principal angles between the nuisance and physiological column spaces, rank-aware.

    This is the quantity the Schur complement is actually sensitive to. Projecting the nuisance
    column space out of the psi block removes psi information along whatever the two share, so a
    small principal angle means a small perturbation of the nuisance columns moves the psi bound a
    long way. A relative margin on the column norms says nothing about it.

    Each basis comes from retained singular vectors rather than a full QR, because a QR factor of a
    rank-deficient block carries arbitrary completion vectors and the angles against those are an
    artifact of the factorization.
    """
    Qp, rp, tp, sp = _range_basis(J_w_psi)
    Qe, re, te, se_ = _range_basis(J_w_eta)
    if rp == 0 or re == 0:
        return {"principal_angles_deg": [], "smallest_principal_angle_deg": None,
                "largest_cosine": None, "psi_rank": rp, "eta_rank": re,
                "reason": "one of the blocks has no column space at the declared rank tolerance"}
    sv = np.clip(np.linalg.svd(Qp.T @ Qe, compute_uv=False), -1.0, 1.0)
    ang = np.degrees(np.arccos(sv))
    return {"principal_angles_deg": ang.tolist(),
            "smallest_principal_angle_deg": float(np.min(ang)),
            "largest_cosine": float(np.max(sv)),
            "psi_rank": rp, "eta_rank": re,
            "psi_rank_tolerance": tp, "eta_rank_tolerance": te,
            "n_angles": int(ang.size),
            "basis_from": "retained singular vectors at the declared rank tolerance, not a full QR",
            "why_not_qr": "a QR factor of a rank-deficient block includes arbitrary completion "
                          "vectors, and principal angles measured against those are an artifact "
                          "of the factorization rather than a property of the column spaces",
            "why_this_matters": "the Schur complement removes psi information along whatever the "
                                "two column spaces share, so a small angle makes the psi bound "
                                "extremely sensitive to the nuisance columns. A relative margin "
                                "on the column norms does not control it."}


def stress_search_classifications(J_w_psi, J_w_eta, column_rel_uncertainty, psi_scale,
                                  thresholds, labels, support_mask, n_directions=48,
                                  seed=20261005):
    """A HEURISTIC stress search for a classification flip. This is NOT a certification.

    An earlier name implied a certification and the docstring claimed invariance over every
    admissible Jacobian.
    It cannot establish that: the admissible set is a product of per-column balls and this searches
    the leading psi singular directions plus a finite number of random ones, so an untested
    admissible direction can flip a classification the search declared stable. A deterministic
    counterexample against the previous implementation reported a searched maximum of 0.3772730688
    against a threshold of 0.3876343099 while an admissible untested direction reached
    0.3979955510, and the function returned certified true.

    So it never returns a certification. Finding no flip is weak evidence and is reported as
    "no_flip_found"; finding one is conclusive and is reported as "flip_found". Until an exhaustive
    invariance argument exists over the complete admissible set, every classification downstream of
    this must be reported as unresolved.

    `column_rel_uncertainty` is the simultaneous relative bound the equivalence test established
    for the nuisance columns, and `support_mask` is the gate-produced mask that every Fisher matrix
    built here must apply.

    It is still worth running, because a flip it does find is a real finding: a nuisance block
    nearly collinear with a leading psi singular direction has its upper-StO2 bound moved by a
    factor of about 5000 by a five percent column perturbation. What it cannot do is license the
    opposite conclusion.
    """
    Jp = np.asarray(J_w_psi, float)
    Je = np.asarray(J_w_eta, float)
    rng = np.random.default_rng(seed)
    D = np.asarray(psi_scale, float)
    m = float(column_rel_uncertainty)
    cols = np.maximum(np.linalg.norm(Je, axis=0, keepdims=True), np.finfo(float).tiny)

    def classify(Jeta):
        bi = block_information(np.hstack([Jp, Jeta]), Jp.shape[1],
                               support_mask=support_mask, require_mask=True)
        est = estimability(bi["F_psi_eff"], scale=D, labels=labels)
        nn = near_null_report(bi["F_psi_eff"], scale=D, labels=labels)
        vals = {"rank": est["rank"]}
        for name, (param, op, thr) in thresholds.items():
            e = est["per_parameter"][param]
            if not e["individually_estimable"]:
                vals[name] = (op == "ge")
            else:
                vals[name] = (e["sqrt_crlb"] >= thr) if op == "ge" else (e["sqrt_crlb"] < thr)
            vals[f"{name}__value"] = e["sqrt_crlb"]
        vals["single_near_null_direction_reported"] = nn["single_direction_reported"]
        vals["near_null_dominant"] = nn.get("dominant_component")
        vals["near_null_tangent"] = nn.get("tangent_psi")
        return vals

    base = classify(Je)
    # the worst directions are the leading left singular directions of the psi block, where the
    # two column spaces couple hardest; random directions essentially never find them
    Up = np.linalg.svd(Jp, full_matrices=False)[0]
    dirs = [Up[:, i % Up.shape[1]][:, None] * np.ones((1, Je.shape[1]))
            for i in range(Up.shape[1])]
    dirs += [rng.normal(size=Je.shape) for _ in range(n_directions)]

    no_flip = {k: True for k in thresholds}
    no_flip["rank"] = True
    no_flip["single_near_null_direction_reported"] = True
    def _rng_init(v):
        return (float("inf"), float("inf")) if v is None else (v, v)

    worst_val = {k: _rng_init(base[f"{k}__value"]) for k in thresholds}
    worst_cos = 1.0
    for P0 in dirs:
        P = P0 / np.maximum(np.linalg.norm(P0, axis=0, keepdims=True), np.finfo(float).tiny)
        for sgn in (+1.0, -1.0):
            v = classify(Je + sgn * m * P * cols)
            for k in list(thresholds) + ["rank", "single_near_null_direction_reported"]:
                if v[k] != base[k]:
                    no_flip[k] = False
            for k in thresholds:
                a, b = worst_val[k]
                c = v[f"{k}__value"]
                c = float("inf") if c is None else float(c)
                worst_val[k] = (min(a, c), max(b, c))
            if base.get("near_null_tangent") and v.get("near_null_tangent"):
                worst_cos = min(worst_cos, direction_agreement(base["near_null_tangent"],
                                                               v["near_null_tangent"]))

    angles = nuisance_psi_principal_angles(Jp, Je)
    return {
        "column_relative_uncertainty_used": m,
        "n_directions_searched": 2 * len(dirs),
        "search": "the leading left singular directions of the psi block plus random directions, "
                  "both signs. Random directions alone essentially never find the worst case.",
        "baseline": {k: base[k] for k in list(thresholds)
                     + ["rank", "single_near_null_direction_reported", "near_null_dominant"]},
        "baseline_values": {k: base[f"{k}__value"] for k in thresholds},
        "value_range_over_the_admissible_set": {k: list(v) for k, v in worst_val.items()},
        "worst_near_null_cosine": worst_cos,
        "is_a_certification": False,
        "no_flip_found": no_flip,
        "any_flip_found": bool(not all(no_flip.values())),
        "stress_outcome": {k: ("flip_found" if not no_flip[k] else "no_flip_found")
                           for k in list(thresholds)
                           + ["rank", "single_near_null_direction_reported"]},
        "classification_status": {k: ("inequivalent_under_stress" if not no_flip[k]
                                      else "unresolved")
                                  for k in thresholds},
        "why_unresolved_even_when_no_flip_is_found":
            "the search covers the leading psi singular directions plus a finite number of random "
            "ones, which is not the complete admissible set. Finding no flip is weak evidence and "
            "cannot establish invariance, so the classification stays unresolved.",
        "principal_angles": angles,
        "rule": "a flip found is conclusive; no flip found is not. No classification may be "
                "reported as established from this function.",
        "support_mask_applied": True,
        "why_a_column_margin_is_not_enough":
            "a relative error in a nuisance Jacobian column does not control the principal angle "
            "between the nuisance and physiological column spaces, and the Schur complement "
            "depends on that angle. Measured on the frozen production psi block, a five percent "
            "column perturbation moved the upper-StO2 bound by a factor of about 5000 for a block "
            "nearly collinear with a leading psi direction, and holding it to within 40 percent "
            "needed a column margin near 1e-5.",
    }
