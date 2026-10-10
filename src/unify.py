"""Unified whitened-Jacobian / Fisher machinery for the P1+P2 merge.

The merge exists because P1 and P2 do not compute comparable objects. P1 takes the SVD of a raw
Jacobian in log-concentration coordinates with no noise model; P2 eigendecomposes a diagonal-Poisson
Fisher matrix in physiological coordinates with nominal-value scaling. The two coordinate systems are
related by a nonlinear reparameterization, so the connecting matrix is the full Jacobian
T = d(phi)/d(psi), not a diagonal scale.

Everything below is the algebra that puts both into one coordinate system:

    J_psi = J_phi @ T                     reparameterize
    J_w   = C^{-1/2} @ J_psi @ D          whiten and non-dimensionalize
    I     = J_w.T @ J_w                   Fisher information in the whitened, scaled basis

with the consequences that the right singular vectors of J_w are the eigenvectors of I and
lambda_i = s_i**2. Those identities are asserted, not assumed: see verify_spectral_identities.

C is the observation covariance. It is diagonal for the Poisson likelihood and block-diagonal-dense
for the fixed-launch multinomial, where within each wavelength block

    Cov_k = N * (diag(p_k) - outer(p_k, p_k)),   p_kj = A_j * R_kj.

No configuration constant is defined in this module. Everything comes from spec/baseline_spec.json.
"""
from __future__ import annotations

import numpy as np

PSI_LABELS = ("StO2_U", "f_U", "StO2_D", "f_D", "mua_DermR", "mua_SubQ")
PHI_LABELS = ("log_cU_oxy", "log_cU_deoxy", "log_cD_oxy", "log_cD_deoxy", "log_DermR", "log_SubQ")


# --------------------------------------------------------------------------------------- phi(psi)

def phi_of_psi(psi):
    """P1's log-concentration coordinates as a function of P2's physiological coordinates.

    psi = [S_U, f_U, S_D, f_D, mua_DermR, mua_SubQ]
    phi = log([f_U*S_U, f_U*(1-S_U), f_D*S_D, f_D*(1-S_D), mua_DermR, mua_SubQ])
    """
    s_u, f_u, s_d, f_d, dermr, subq = (float(x) for x in psi)
    c = np.array([f_u * s_u, f_u * (1.0 - s_u), f_d * s_d, f_d * (1.0 - s_d), dermr, subq])
    if np.any(c <= 0.0):
        raise ValueError(f"phi_of_psi: non-positive concentration in {c}; log undefined")
    return np.log(c)


def reparam_jacobian(psi):
    """T = d(phi)/d(psi), analytic. Block structured: two 2x2 blocks and two diagonal entries.

    This is emphatically NOT diagonal, which is the whole reason the two spectra cannot be compared
    without it.
    """
    s_u, f_u, s_d, f_d, dermr, subq = (float(x) for x in psi)
    T = np.zeros((6, 6))
    # d log(f*S)/dS = 1/S ;  d log(f*S)/df = 1/f
    T[0, 0], T[0, 1] = 1.0 / s_u, 1.0 / f_u
    # d log(f*(1-S))/dS = -1/(1-S) ;  d log(f*(1-S))/df = 1/f
    T[1, 0], T[1, 1] = -1.0 / (1.0 - s_u), 1.0 / f_u
    T[2, 2], T[2, 3] = 1.0 / s_d, 1.0 / f_d
    T[3, 2], T[3, 3] = -1.0 / (1.0 - s_d), 1.0 / f_d
    T[4, 4] = 1.0 / dermr
    T[5, 5] = 1.0 / subq
    return T


def verify_reparam_jacobian(psi, rel_step=1e-6):
    """Central-difference check of the analytic T. Returns (T_analytic, T_numeric, max_rel_err)."""
    psi = np.asarray(psi, dtype=float)
    T_a = reparam_jacobian(psi)
    T_n = np.zeros((6, 6))
    for j in range(6):
        h = rel_step * abs(psi[j])
        if h == 0.0:
            raise ValueError(f"verify_reparam_jacobian: psi[{j}] is zero; relative step undefined")
        up, dn = psi.copy(), psi.copy()
        up[j] += h
        dn[j] -= h
        T_n[:, j] = (phi_of_psi(up) - phi_of_psi(dn)) / (2.0 * h)
    scale = np.maximum(np.abs(T_a), 1e-300)
    return T_a, T_n, float(np.max(np.abs(T_a - T_n) / scale))


def reparameterize(J_phi, psi):
    """J_psi = J_phi @ T. J_phi is (n_obs, 6) in log-concentration coordinates."""
    J_phi = np.asarray(J_phi, dtype=float)
    if J_phi.ndim != 2 or J_phi.shape[1] != 6:
        raise ValueError(f"reparameterize: expected (n_obs, 6), got {J_phi.shape}")
    return J_phi @ reparam_jacobian(psi)


# ------------------------------------------------------------------------------------- whitening
#
# The observable is the COUNT in a detector bin, not the reflectance density. With p_j = A_j R_j the
# detection probability of bin j, the Jacobian of the probability vector is A @ J, and that factor A
# must appear in the whitening. An earlier version of this module omitted it for the multinomial and
# also inverted the power of N; both are corrected here and both are now checked against an
# independent direct-FIM construction rather than against the same J_w.
#
#   raw          J_w = J D
#   Poisson      J_w = diag(sqrt(N A / R)) J D
#   multinomial  J_w = sqrt(N) M^{-1/2} A J D      per wavelength block,
#                                                   M = diag(p) - outer(p, p),  p = A R
#
# N is the MEASUREMENT budget per wavelength. It is not the transport photon count: the pooled
# transport budget sets the Monte Carlo precision of R and J, and never enters the likelihood.


def _blocks(n_obs, n_wavelengths):
    if n_obs % n_wavelengths:
        raise ValueError(f"{n_obs} observables not divisible by {n_wavelengths} wavelengths")
    per = n_obs // n_wavelengths
    return [slice(k * per, (k + 1) * per) for k in range(n_wavelengths)]


def _check_inputs(J, R, areas, n_measurement_photons):
    J, R, areas = np.asarray(J, float), np.asarray(R, float), np.asarray(areas, float)
    if R.shape != areas.shape:
        raise ValueError(f"R {R.shape} and areas {areas.shape} differ")
    if J.shape[0] != R.size:
        raise ValueError(f"J has {J.shape[0]} rows but R has {R.size} entries")
    if n_measurement_photons is None:
        raise ValueError(
            "n_measurement_photons is None. A profile that defines no measurement budget (p1_legacy) "
            "forms no noise model by design; whitening is undefined without a budget. Supply one "
            "explicitly rather than defaulting to a number nobody chose.")
    bad = ~np.isfinite(R) | (R <= 0.0)
    if np.any(bad):
        raise ValueError(f"{int(bad.sum())} of {R.size} bins have non-positive or non-finite "
                         f"reflectance; whitening is undefined there")
    return J, R, areas, float(n_measurement_photons)


def _inv_sqrt_psd(M, rel_tol=1e-12):
    """Symmetric inverse square root of a positive-definite matrix."""
    w, V = np.linalg.eigh(M)
    keep = w > rel_tol * w.max()
    if not np.all(keep):
        raise np.linalg.LinAlgError(
            f"matrix is rank deficient ({int((~keep).sum())} of {w.size} eigenvalues below "
            f"{rel_tol:g} of the largest); covariance is singular")
    return (V / np.sqrt(w)) @ V.T


def whiten_raw(J, scale):
    """J_w = J D. No noise model: this is the geometric branch."""
    return np.asarray(J, float) * np.asarray(scale, float)[None, :]


def whiten_poisson(J, R, areas, n_measurement_photons, scale):
    """J_w = diag(sqrt(N A / R)) J D."""
    J, R, areas, N = _check_inputs(J, R, areas, n_measurement_photons)
    return (np.sqrt(N * areas / R)[:, None] * J) * np.asarray(scale, float)[None, :]


def whiten_multinomial(J, R, areas, n_measurement_photons, scale, n_wavelengths):
    """J_w = sqrt(N) M^{-1/2} A J D, blockwise per wavelength."""
    J, R, areas, N = _check_inputs(J, R, areas, n_measurement_photons)
    out = np.zeros_like(J)
    for sl in _blocks(R.size, n_wavelengths):
        A = np.diag(areas[sl])
        p = areas[sl] * R[sl]
        tot = p.sum()
        if not (0.0 < tot < 1.0):
            raise ValueError(f"total detection probability {tot:.6g} outside (0, 1); the "
                             f"non-detection cell is degenerate")
        M = np.diag(p) - np.outer(p, p)
        out[sl] = np.sqrt(N) * (_inv_sqrt_psd(M) @ (A @ J[sl]))
    return out * np.asarray(scale, float)[None, :]


def fisher_from_whitened(J_w):
    """I = J_w.T @ J_w."""
    J_w = np.asarray(J_w, float)
    return J_w.T @ J_w


# --- independent direct constructions, for checking the whitened forms ---------------------------
#
# These do NOT reuse J_w. They build the Fisher matrix from its definition, so agreement with
# J_w.T @ J_w is evidence that the whitening is right rather than a tautology.

def fisher_direct_poisson(J, R, areas, n_measurement_photons, scale):
    """I = N D J.T diag(A / R) J D."""
    J, R, areas, N = _check_inputs(J, R, areas, n_measurement_photons)
    D = np.diag(np.asarray(scale, float))
    return N * (D @ J.T @ np.diag(areas / R) @ J @ D)


def fisher_direct_multinomial(J, R, areas, n_measurement_photons, scale, n_wavelengths):
    """I = N D J.T A pinv(M) A J D, summed over wavelength blocks."""
    J, R, areas, N = _check_inputs(J, R, areas, n_measurement_photons)
    D = np.diag(np.asarray(scale, float))
    acc = np.zeros((J.shape[1], J.shape[1]))
    for sl in _blocks(R.size, n_wavelengths):
        A = np.diag(areas[sl])
        p = areas[sl] * R[sl]
        M = np.diag(p) - np.outer(p, p)
        acc += J[sl].T @ A @ np.linalg.pinv(M) @ A @ J[sl]
    return N * (D @ acc @ D)


# ------------------------------------------------------------------------- spectral consistency

def verify_spectral_identities(J_w, rel_tol=1e-9):
    """Assert the identities that make the unified comparison meaningful.

    Returns a dict of residuals. The caller decides whether they pass; nothing is silently accepted.
    """
    J_w = np.asarray(J_w, dtype=float)
    I = fisher_from_whitened(J_w)

    U, s, Vt = np.linalg.svd(J_w, full_matrices=False)
    lam, V = np.linalg.eigh(I)
    order = np.argsort(lam)[::-1]                      # eigh ascends; SVD descends
    lam, V = lam[order], V[:, order]

    # lambda_i == s_i**2.
    #
    # The meaningful criterion is the residual relative to lambda_max, not to each lambda_i.
    # Forming I = J_w.T J_w squares the condition number, so eigh cannot resolve the smallest
    # eigenvalues of an ill-conditioned problem to relative accuracy even when the algebra is exact.
    # That is precisely why the near-null direction must be read from the SVD of J_w and not from an
    # eigendecomposition of I; per-eigenvalue relative error is reported for the record but is not a
    # correctness criterion.
    lam_from_s = s ** 2
    lam_max = float(np.max(np.abs(lam_from_s))) or 1.0
    eig_res_scaled = float(np.max(np.abs(lam - lam_from_s)) / lam_max)
    denom = np.maximum(np.abs(lam_from_s), np.finfo(float).tiny)
    eig_res_each = float(np.max(np.abs(lam - lam_from_s) / denom))
    # the smallest eigenvalue eigh can still resolve, given double precision
    eigh_resolvable_floor = float(np.finfo(float).eps * lam_max)
    lam_min_below_floor = bool(np.min(lam) < eigh_resolvable_floor)

    # eigenvectors == right singular vectors, up to sign
    Vr = Vt.T
    signs = np.sign(np.sum(Vr * V, axis=0))
    signs[signs == 0] = 1.0
    vec_res = float(np.max(np.abs(Vr - V * signs)))

    # J_w @ v_i == s_i * u_i, as a residual relative to the largest singular value. An absolute
    # tolerance is meaningless here: whitening multiplies J by sqrt(N A_j / R_j), so the entries of
    # J_w carry the scale of the photon budget.
    lhs = J_w @ Vr
    rhs = U * s[None, :]
    sv_res = float(np.max(np.abs(lhs - rhs)) / max(s[0], np.finfo(float).tiny))

    # I is symmetric by construction, relative to its largest entry
    I_scale = float(np.max(np.abs(I))) or 1.0
    sym_res = float(np.max(np.abs(I - I.T)) / I_scale)

    cond_J = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
    return {
        "singular_values": s.tolist(),
        "eigenvalues": lam.tolist(),
        "eigenvalue_vs_squared_singular_residual_over_lambda_max": eig_res_scaled,
        "eigenvalue_vs_squared_singular_rel_residual_each": eig_res_each,
        "eigenvector_vs_right_singular_abs_residual": vec_res,
        "Jw_v_minus_s_u_rel_residual": sv_res,
        "fisher_symmetry_rel_residual": sym_res,
        "condition_number_Jw": cond_J,
        "condition_number_I": cond_J ** 2 if np.isfinite(cond_J) else float("inf"),
        "eigh_resolvable_floor": eigh_resolvable_floor,
        "smallest_eigenvalue_below_eigh_floor": lam_min_below_floor,
        "near_null_must_be_read_from_svd": lam_min_below_floor,
        "all_within_tolerance": bool(
            eig_res_scaled < rel_tol and vec_res < 1e-6
            and sv_res < 1e-10 and sym_res < 1e-12),
        "rel_tol": rel_tol,
    }


def direction_alignment(vectors, directions):
    """|<d, v>| for each named direction against each spectral vector, both unit-normalized."""
    out = {}
    for name, d in directions.items():
        d = np.asarray(d, dtype=float)
        d = d / np.linalg.norm(d)
        out[name] = [float(abs(np.dot(d, vectors[:, i] / np.linalg.norm(vectors[:, i]))))
                     for i in range(vectors.shape[1])]
    return out


def sto2_direction_psi(index):
    """Unit StO2 direction in psi coordinates: e_0 for upper, e_2 for deep."""
    e = np.zeros(6)
    e[index] = 1.0
    return e


def sto2_direction_phi(saturation, offset):
    """Unit StO2 direction in phi coordinates, matching P1's own `direction()` helper.

    This equals the pushforward of the psi StO2 basis vector through T, which is the cross-check
    that the reparameterization is the right one.
    """
    e = np.zeros(6)
    e[offset] = 1.0 / saturation
    e[offset + 1] = -1.0 / (1.0 - saturation)
    return e / np.linalg.norm(e)


def near_null_direction(J_w, scale, J_psi=None):
    """The least-identifiable direction, in both coordinate systems, with the pushforward checked.

    J_w = C^{-1/2} J_psi D, so the right singular vectors of J_w live in the SCALED coordinate z,
    where psi = D z. The direction that matters physically is therefore the pushforward D v_z, not
    v_z itself. Confusing the two silently reweights the direction by D: here it would make the
    f_D component about 50 times larger than it should be relative to StO2_D, which in turn makes
    f_D appear to be what limits a step along the direction.

    The psi-space tangent is deliberately left UNNORMALIZED. A scalar step t along D v_z then
    means a displacement of t in the z coordinate, which is the coordinate the singular value
    belongs to. A unit-norm version is returned as well, for display only.
    """
    J_w = np.asarray(J_w, dtype=float)
    D = np.asarray(scale, dtype=float)
    _, s, Vt = np.linalg.svd(J_w, full_matrices=False)
    v_z = Vt[-1]
    if v_z[np.argmax(np.abs(v_z))] < 0:            # sign convention, largest component positive
        v_z = -v_z
    tangent = D * v_z
    n = np.linalg.norm(tangent)

    out = {"near_null_direction_z_from_svd": v_z.tolist(),
           "near_null_tangent_psi_per_unit_z": tangent.tolist(),
           "near_null_tangent_psi_unit_norm_display_only": (tangent / n).tolist(),
           "tangent_norm": float(n),
           "smallest_singular_value": float(s[-1]),
           "coordinate_note": "the SVD of J_w = C^{-1/2} J_psi D gives right singular vectors in "
                              "z, where psi = D z; the psi-space direction is the pushforward "
                              "D v_z, left unnormalized so a unit step means unit displacement "
                              "in z"}

    # the whitened response along this direction is the smallest singular value, by construction
    r = float(np.linalg.norm(J_w @ v_z))
    out["whitened_response_equals_smallest_singular_value"] = {
        "response": r, "smallest_singular_value": float(s[-1]),
        "relative_difference": float(abs(r - s[-1]) / s[-1]) if s[-1] > 0 else None,
        "passes": bool(s[-1] > 0 and abs(r - s[-1]) / s[-1] < 1e-10)}

    # D v_z is the pushforward, so applying J_psi to it must equal applying (J_psi D) to v_z
    if J_psi is not None:
        J_psi = np.asarray(J_psi, dtype=float)
        a, b = J_psi @ tangent, (J_psi * D[None, :]) @ v_z
        d = max(float(np.max(np.abs(b))), np.finfo(float).tiny)
        out["pushforward_consistency"] = {
            "identity": "J_psi @ (D v_z) == (J_psi D) @ v_z",
            "normalized_max_diff": float(np.max(np.abs(a - b)) / d),
            "passes": bool(np.max(np.abs(a - b)) / d < 1e-12)}
    return out
