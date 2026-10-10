#!/usr/bin/env python3
"""unify_spectrum.py -- the one place the three comparable spectra are built.

All three branches consume the SAME pooled Jacobian, the SAME observation grid, the SAME psi
coordinates and the SAME column scale D. Only the likelihood changes, which is what makes the
comparison about whitening and nothing else:

    raw          J_w = J D
    Poisson      J_w = diag(sqrt(N A / R)) J D
    multinomial  J_w = sqrt(N) M^{-1/2} A J D        per wavelength block, M = diag(p) - p p^T

N is the MEASUREMENT budget per wavelength from the spec. It is never the pooled transport photon
count: transport sets the Monte Carlo precision of R and J and does not enter the likelihood.

Coordinate handling is explicit and refuses to guess. A pooled artifact already stores J_psi in psi
coordinates, so reparameterize() must NOT be applied to it; doing so would push an already-pushed
Jacobian through T a second time. A legacy phi-coordinate artifact takes the compatibility branch,
and anything ambiguous is rejected.

Each whitened form is checked against an INDEPENDENT direct construction of the Fisher matrix, not
against the same J_w, so the check is evidence rather than a tautology.

    python3 scripts/unify_spectrum.py --selftest
    python3 scripts/unify_spectrum.py --pooled data/derived/production_pooled.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import unify  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC_PATH = os.path.join(ROOT, "spec", "baseline_spec.json")


def load_spec(path=SPEC_PATH):
    with open(path) as fh:
        return json.load(fh)


def jacobian_coordinates(d):
    """Decide the coordinate system explicitly, or refuse.

    Priority: an explicit `jacobian_coordinates` field, then an unambiguous array name. A file that
    carries both J_psi and J_phi, or neither, is rejected rather than guessed at.
    """
    files = set(d.files)
    if "jacobian_coordinates" in files:
        tag = str(d["jacobian_coordinates"])
        if tag not in ("psi", "phi"):
            raise ValueError(f"jacobian_coordinates is {tag!r}; expected 'psi' or 'phi'")
        name = "J_psi" if tag == "psi" else "J_phi"
        if name not in files:
            raise ValueError(f"jacobian_coordinates says {tag!r} but {name} is absent")
        return tag, name
    has_psi, has_phi = "J_psi" in files, "J_phi" in files
    if has_psi and has_phi:
        raise ValueError("artifact carries both J_psi and J_phi and no jacobian_coordinates field; "
                         "the coordinate system is ambiguous and will not be guessed")
    if has_psi:
        return "psi", "J_psi"
    if has_phi:
        return "phi", "J_phi"
    raise ValueError(f"artifact carries neither J_psi nor J_phi; arrays present: {sorted(files)}")


def column_scale(psi):
    """D: StO2 entries are dimensionless and carry 1; the rest are scaled by their nominal value."""
    return np.array([1.0, psi[1], 1.0, psi[3], psi[4], psi[5]], dtype=float)


def three_branches(J_psi, R, areas, N_meas, D, n_wl, rel_tol=1e-9):
    """Build the three J_w, and check each against an independent direct Fisher construction."""
    Jw = {
        "raw": unify.whiten_raw(J_psi, D),
        "diagonal_poisson": unify.whiten_poisson(J_psi, R, areas, N_meas, D),
        "multinomial_fixed_launch": unify.whiten_multinomial(J_psi, R, areas, N_meas, D, n_wl),
    }
    direct = {
        "diagonal_poisson": unify.fisher_direct_poisson(J_psi, R, areas, N_meas, D),
        "multinomial_fixed_launch": unify.fisher_direct_multinomial(
            J_psi, R, areas, N_meas, D, n_wl),
    }
    out = {}
    for name, W in Jw.items():
        I = unify.fisher_from_whitened(W)
        entry = {"spectral_identities": unify.verify_spectral_identities(W, rel_tol=rel_tol)}
        if name in direct:
            Id = direct[name]
            s = max(float(np.max(np.abs(Id))), np.finfo(float).tiny)
            nd = float(np.max(np.abs(I - Id)) / s)
            entry["direct_fisher_check"] = {
                "max_abs_diff": float(np.max(np.abs(I - Id))),
                "normalized_max_diff": nd,
                "passes": bool(nd < 1e-10),
                "note": "I built from its definition, independently of J_w",
            }
        lam, V = np.linalg.eigh(I)
        order = np.argsort(lam)[::-1]
        lam, V = lam[order], V[:, order]
        entry["eigenvalues"] = lam.tolist()
        entry["condition_number_I"] = float(lam[0] / lam[-1]) if lam[-1] > 0 else float("inf")
        entry["alignment"] = unify.direction_alignment(V, {
            "StO2_upper": unify.sto2_direction_psi(0),
            "StO2_deep": unify.sto2_direction_psi(2)})
        # the near-null direction is read from the SVD of J_w, not from eigh(I): forming I squares
        # the condition number and destroys exactly the direction of interest. The right singular
        # vectors live in the scaled coordinate z, so the psi-space direction is the pushforward
        # D v_z, not v_z.
        _, s_sv, _ = np.linalg.svd(W, full_matrices=False)
        entry["singular_values"] = s_sv.tolist()
        entry["near_null"] = unify.near_null_direction(W, D, J_psi=J_psi)
        out[name] = entry
    return Jw, out


def selftest(spec):
    """Verify the algebra, including each whitening against its independent direct Fisher form."""
    rng = np.random.default_rng(20260930)
    results, failures = [], []
    labels, nom = spec["parameters"]["psi_labels"], spec["parameters"]["nominal_psi"]
    psi = np.array([nom[k] for k in labels], dtype=float)
    D = column_scale(psi)
    n_wl = len(spec["wavelengths_nm"])
    N_meas = spec["profiles"]["production"]["n_measurement_photons_per_wavelength"]

    for per in (30, 38):
        n_obs = per * n_wl
        e = np.linspace(0.0, 1.5, per + 1)
        areas = np.tile(np.pi * (e[1:] ** 2 - e[:-1] ** 2), n_wl)
        base = np.exp(-(np.arange(per) + 0.5) / (0.35 * per)) + 1e-3
        R = np.tile(base, n_wl) * rng.uniform(0.8, 1.2, n_obs)
        p_tot = (areas * R).reshape(n_wl, per).sum(axis=1)
        R = R / (3.0 * np.repeat(p_tot, per))
        J = rng.normal(size=(n_obs, 6))
        J[:, 1] = J[:, 0] * (1.0 + 1e-3)                 # a near-null direction to exercise

        _, br = three_branches(J, R, areas, N_meas, D, n_wl)
        for name, entry in br.items():
            ok = entry["spectral_identities"]["all_within_tolerance"]
            if "direct_fisher_check" in entry:
                ok = ok and entry["direct_fisher_check"]["passes"]
            results.append({"grid_bins_per_wavelength": per, "branch": name, "pass": bool(ok),
                            "identities": entry["spectral_identities"],
                            "direct": entry.get("direct_fisher_check")})
            if not ok:
                failures.append(f"{per}bins:{name}")

    T_a, _, T_err = unify.verify_reparam_jacobian(psi)
    pushed = T_a @ unify.sto2_direction_psi(0)
    cos_u = abs(float(np.dot(pushed / np.linalg.norm(pushed),
                             unify.sto2_direction_phi(psi[0], 0))))
    rep_ok = bool(T_err < 1e-7 and abs(cos_u - 1.0) < 1e-12
                  and not np.allclose(T_a, np.diag(np.diag(T_a))))
    if not rep_ok:
        failures.append("reparameterization")
    return results, failures, {"T_max_rel_err": T_err, "pushforward_cosine": cos_u, "pass": rep_ok}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--pooled", help="pooled artifact carrying J_psi (or legacy J_phi) and R")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "unify_spectrum.json"))
    a = ap.parse_args()
    spec = load_spec()

    if a.selftest:
        results, failures, rep = selftest(spec)
        os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
        with open(os.path.join(ROOT, "results", "unify_selftest.json"), "w") as fh:
            json.dump({"mode": "selftest", "results": results, "failures": failures,
                       "reparameterization": rep}, fh, indent=2)
        for r in results:
            i = r["identities"]
            dmsg = (f"  direct-FIM {r['direct']['normalized_max_diff']:.2e}"
                    if r["direct"] else "  direct-FIM n/a (raw has no likelihood)")
            print(f"  {r['grid_bins_per_wavelength']:>3} bins  {r['branch']:<26} "
                  f"{'PASS' if r['pass'] else 'FAIL'}  lam-vs-s2 "
                  f"{i['eigenvalue_vs_squared_singular_residual_over_lambda_max']:.2e}{dmsg}")
        print(f"  reparameterization: T err {rep['T_max_rel_err']:.2e}, "
              f"pushforward cosine {rep['pushforward_cosine']:.12f} -> "
              f"{'PASS' if rep['pass'] else 'FAIL'}")
        if failures:
            print(f"  FAILED: {failures}")
            return 1
        print(f"  {len(results)}/{len(results)} branch checks verified")
        return 0

    if not a.pooled:
        ap.error("either --selftest or --pooled is required")

    d = np.load(a.pooled)
    coords, jname = jacobian_coordinates(d)
    psi = np.asarray(d["psi_nominal"], dtype=float)
    R, areas = np.asarray(d["R"], float), np.asarray(d["areas"], float)
    n_wl = int(d["n_wavelengths"])
    N_meas = spec["profiles"]["production"]["n_measurement_photons_per_wavelength"]
    D = column_scale(psi)

    if coords == "psi":
        J_psi = np.asarray(d[jname], float)
        reparam_applied = False
    else:
        # compatibility branch: a legacy phi artifact is pushed through T exactly once
        J_psi = unify.reparameterize(np.asarray(d[jname], float), psi)
        reparam_applied = True

    Jw, branches = three_branches(J_psi, R, areas, N_meas, D, n_wl)

    payload = {
        "pooled_artifact": os.path.relpath(a.pooled, ROOT),
        "jacobian_coordinates": coords,
        "jacobian_array": jname,
        "reparameterize_applied": reparam_applied,
        "n_observables": int(R.size), "n_wavelengths": n_wl,
        "psi_nominal": psi.tolist(), "column_scale_D": D.tolist(),
        "n_measurement_photons_per_wavelength": N_meas,
        "pooled_transport_photons_per_wavelength": (
            int(d["n_transport_photons"]) * int(d["n_seeds"]) if "n_seeds" in d.files else None),
        "note": "all three branches share one J_psi, one grid, one coordinate system and one D; "
                "only the likelihood differs",
        "branches": branches,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"  {os.path.basename(a.pooled)}: coordinates={coords} ({jname}), "
          f"reparameterize_applied={reparam_applied}")
    print(f"  N_measurement={N_meas}/wavelength, pooled transport="
          f"{payload['pooled_transport_photons_per_wavelength']}/wavelength")
    for name, e in branches.items():
        dc = e.get("direct_fisher_check")
        print(f"  {name:<26} cond(I) {e['condition_number_I']:.3e}"
              + (f"  direct-FIM {dc['normalized_max_diff']:.2e} "
                 f"{'PASS' if dc['passes'] else 'FAIL'}" if dc else "  (no likelihood)"))
    print(f"  wrote {os.path.relpath(a.out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
