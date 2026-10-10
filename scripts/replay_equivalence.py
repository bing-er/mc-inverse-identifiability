#!/usr/bin/env python3
"""replay_equivalence.py -- the gate that must pass before any perturbation Monte Carlo number.

The production kernel is the only physical truth. This script asks one question and refuses to
accept a weaker answer: does the instrumented replay kernel walk the SAME trajectory, photon by
photon, under the same PRNG keys, the same random-number consumption order, the same branch order,
the same max_steps, the same radial grid, the same layer stack and the same detector definition?

Comparing pooled reflectance alone would not answer it. Two kernels can agree on an aggregate while
disagreeing on individual photons, and a pathlength moment is a per-photon quantity. So the checks
are per photon:

    escaped or not            det_id >= 0 must agree for every photon
    detector bin              det_id must agree exactly for every photon
    escaping weight           w_refl must agree to a stated tolerance
    aggregate reflectance     R must agree, as a consequence rather than as the test
    pathlength identity       w_exit == w0 exp(-sum_j mua_j L_j) for every escaped photon

The last one is not a transport check at all: it verifies that the recorded pathlengths are the
ones that produced the weight, which is what makes the pMC derivative meaningful. It is an
independent check because production never stores a pathlength.

A failure stops here. There is no override flag.

    python3 scripts/replay_equivalence.py --n 2000 --seeds 910000,920007,930014
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOMINAL_BOUNDARIES = [0.0020, 0.0100, 0.0250, 0.0330, 0.1830, 0.1910, 0.8000]

# Criteria. The detector bin and the escaped flag are integer facts and must match exactly. The
# weight is floating point produced by the same operations in the same order, so it is expected to
# match bitwise; the tolerance exists so that a non-bitwise but numerically irrelevant difference
# is reported rather than silently accepted as a hard failure.
TOL = {"det_id_mismatches_allowed": 0,
       "escaped_flag_mismatches_allowed": 0,
       "weight_max_abs": 1e-15,
       "weight_max_rel": 1e-12,
       "reflectance_max_rel": 1e-12,
       "pathlength_identity_max_rel": 1e-10}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(a, b):
    d = np.maximum(np.abs(b), np.finfo(float).tiny)
    return float(np.max(np.abs(a - b) / d)) if a.size else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=2000, help="photons per wavelength per seed")
    ap.add_argument("--seeds", default="910000,920007,930014,940021,950028")
    ap.add_argument("--jaxpad-root", default=os.environ.get(
        "JAXPAD_ROOT", os.path.normpath(os.path.join(ROOT, "..", "jax-pad"))))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "replay_equivalence.json"))
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    t0 = time.time()

    spec = json.load(open(os.path.join(ROOT, "spec", "baseline_spec.json")))
    prof = spec["profiles"]["production"]
    kappa = spec["blood_coordinate"]["legacy_to_physical_divisor"]
    nom, labels = spec["parameters"]["nominal_psi"], spec["parameters"]["psi_labels"]
    radial, max_steps = prof["radial"], prof["max_steps"]

    # jax-pad owns the name `src`, and the replay kernel must import the production physics from
    # it, so the replay module is loaded by explicit path rather than by package name.
    sys.path.insert(0, os.path.abspath(a.jaxpad_root))
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from scripts.inverse.phase2_v12_runner import (                   # noqa: E402
        Phase2RunConfig, scattering_for_transport, _build_mu)
    import scripts.archive.inverse.multi_wl_v12_colab as v12          # noqa: E402
    from src.pad.physics_engine_v4 import (                           # noqa: E402
        simulate_one_photon_rigorous as prod_photon)
    from src.pad.mc_forward import forward_y_multi_layer as prod_forward   # noqa: E402
    import importlib.util                                             # noqa: E402
    _rk_path = os.path.join(ROOT, "src", "replay_kernel.py")
    _spec = importlib.util.spec_from_file_location("mcii_replay_kernel", _rk_path)
    _rk = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_rk)
    replay_moments, replay_per_photon = _rk.replay_moments, _rk.replay_per_photon
    replay_per_photon_DEFECTIVE = _rk.replay_per_photon_DEFECTIVE

    n_wl = int(v12.N_WL)
    edges = jnp.linspace(radial["r_min_cm"], radial["r_max_cm"], radial["n_bins"] + 1,
                         dtype=jnp.float64)
    psi0 = np.array([nom[k] for k in labels], dtype=float)
    cfg = Phase2RunConfig(n_photons_final=a.n, seed_keys=seeds[0], seed_noise=seeds[0] + 10000,
                          sto2_ubnp_true=psi0[0], sto2_dbnp_true=psi0[2],
                          b_ubnp_true=psi0[1] * kappa, b_dbnp_true=psi0[3] * kappa,
                          max_steps=max_steps)
    bounds = jnp.array(NOMINAL_BOUNDARIES, dtype=jnp.float64)
    mus_list = [scattering_for_transport(v12.MUSS_WL[k], v12.GS_WL[k], cfg.scattering_convention)
                for k in range(n_wl)]

    def psi_to_theta(psi):
        s_u, f_u, s_d, f_d, dermr, subq = psi
        c = jnp.stack([kappa * f_u * s_u, kappa * f_u * (1.0 - s_u),
                       kappa * f_d * s_d, kappa * f_d * (1.0 - s_d), dermr, subq])
        return jnp.log(c + v12.EPS)

    mu_list, _, _ = _build_mu(psi_to_theta(jnp.asarray(psi0)))

    prod_per_photon = jax.jit(jax.vmap(
        prod_photon, in_axes=(0, None, None, None, None, None, None, None)),
        static_argnums=7)

    e = np.asarray(edges)
    areas = np.pi * (e[1:] ** 2 - e[:-1] ** 2)

    print(f"  backend {jax.default_backend()} {jax.devices()}  x64 "
          f"{jax.config.read('jax_enable_x64')}")
    print(f"  {a.n} photons per wavelength per seed, {len(seeds)} seeds, {n_wl} wavelengths, "
          f"max_steps {max_steps}, {radial['n_bins']} bins to {radial['r_max_cm']} cm")
    print(f"  {'seed':>8} {'lam':>5} {'photons':>8} {'det_id':>8} {'escaped':>8} "
          f"{'|dw|':>10} {'rel dw':>10} {'rel dR':>10} {'pathlen id':>11}")

    cases, all_pass = [], True
    for s in seeds:
        keys = jax.random.split(jax.random.PRNGKey(s), a.n)
        for k in range(n_wl):
            opt = dict(muss=mus_list[k], gs=v12.GS_WL[k], ns=v12.NS_WL[k],
                       layer_boundaries=bounds, detector_r_edges=edges, max_steps=max_steps)
            d_p, w_p = prod_per_photon(keys, mu_list[k], opt["muss"], opt["gs"], opt["ns"],
                                       opt["layer_boundaries"], opt["detector_r_edges"],
                                       max_steps)
            d_r, w_r, pl_r = replay_per_photon(muas=mu_list[k], keys=keys, **opt)
            d_p, w_p = np.asarray(d_p), np.asarray(w_p)
            d_r, w_r, pl_r = np.asarray(d_r), np.asarray(w_r), np.asarray(pl_r)

            R_p = np.asarray(prod_forward(muas=mu_list[k], keys=keys, **opt))
            R_r, S_r = replay_moments(muas=mu_list[k], keys=keys, **opt)
            R_r, S_r = np.asarray(R_r), np.asarray(S_r)

            n_det = int((d_p != d_r).sum())
            n_esc = int(((d_p >= 0) != (d_r >= 0)).sum())
            dw_abs = float(np.max(np.abs(w_p - w_r))) if w_p.size else 0.0
            esc = d_p >= 0
            dw_rel = rel(w_r[esc], w_p[esc])
            dR_rel = rel(R_r, R_p)

            # pathlength identity, on escaped photons only: w_exit == w0 exp(-sum_j mua_j L_j)
            ns0 = float(np.asarray(v12.NS_WL[k])[0])
            w0 = 1.0 - ((1.0 - ns0) / (1.0 + ns0)) ** 2
            mua = np.asarray(mu_list[k], dtype=float)
            w_from_pl = w0 * np.exp(-(pl_r[esc] @ mua))
            pl_rel = rel(w_from_pl, w_r[esc])

            ok = (n_det <= TOL["det_id_mismatches_allowed"]
                  and n_esc <= TOL["escaped_flag_mismatches_allowed"]
                  and dw_abs <= TOL["weight_max_abs"] and dw_rel <= TOL["weight_max_rel"]
                  and dR_rel <= TOL["reflectance_max_rel"]
                  and pl_rel <= TOL["pathlength_identity_max_rel"])
            all_pass = all_pass and ok

            bitwise_w = bool(np.array_equal(w_p, w_r))
            bitwise_d = bool(np.array_equal(d_p, d_r))
            cases.append({
                "seed": s, "wavelength_nm": spec["wavelengths_nm"][k], "n_photons": int(a.n),
                "n_escaped": int(esc.sum()),
                "det_id_mismatches": n_det, "escaped_flag_mismatches": n_esc,
                "det_id_bitwise_identical": bitwise_d,
                "weight_max_abs_diff": dw_abs, "weight_max_rel_diff": dw_rel,
                "weight_bitwise_identical": bitwise_w,
                "reflectance_max_rel_diff": dR_rel,
                "pathlength_identity_max_rel_diff": pl_rel,
                "pathlength_mean_total_cm": float(pl_r[esc].sum(axis=1).mean()) if esc.any()
                                            else None,
                "S_finite": bool(np.all(np.isfinite(S_r))),
                "passes": bool(ok)})
            print(f"  {s:>8} {spec['wavelengths_nm'][k]:>5.0f} {int(esc.sum()):>8} "
                  f"{n_det:>8} {n_esc:>8} {dw_abs:>10.2e} {dw_rel:>10.2e} {dR_rel:>10.2e} "
                  f"{pl_rel:>11.2e}  {'PASS' if ok else 'FAIL'}")

    # Negative control. A gate that cannot fail proves nothing, so a replay with a deliberately
    # altered random-number stream is run and asserted to be REJECTED by the same criteria.
    nc_keys = jax.random.split(jax.random.PRNGKey(seeds[0]), min(a.n, 400))
    opt0 = dict(muss=mus_list[0], gs=v12.GS_WL[0], ns=v12.NS_WL[0], layer_boundaries=bounds,
                detector_r_edges=edges, max_steps=max_steps)
    d_p0, w_p0 = prod_per_photon(nc_keys, mu_list[0], opt0["muss"], opt0["gs"], opt0["ns"],
                                 opt0["layer_boundaries"], opt0["detector_r_edges"], max_steps)
    d_b, w_b, _ = replay_per_photon_DEFECTIVE(muas=mu_list[0], keys=nc_keys, **opt0)
    d_p0, w_p0, d_b, w_b = (np.asarray(x) for x in (d_p0, w_p0, d_b, w_b))
    nc_det = int((d_p0 != d_b).sum())
    nc_esc = int(((d_p0 >= 0) != (d_b >= 0)).sum())
    nc_dw = float(np.max(np.abs(w_p0 - w_b)))
    # verify the prefix property in-run, so the claim above is evidence rather than assertion
    _k = jax.random.PRNGKey(7)
    prefix_property = bool(np.array_equal(np.asarray(jax.random.split(_k, 2)),
                                          np.asarray(jax.random.split(_k, 3))[:2]))
    nc_rejected = (nc_det > TOL["det_id_mismatches_allowed"]
                   or nc_esc > TOL["escaped_flag_mismatches_allowed"]
                   or nc_dw > TOL["weight_max_abs"])
    negative_control = {
        "purpose": "confirm the gate can fail; a replay whose random-number stream differs must "
                   "be rejected by the same criteria that accept the real one",
        "defect_injected": "the outer loop swaps the two halves of the free-path key split, so "
                           "the key carried forward and the key used for tau are exchanged. The "
                           "physics is untouched; only the random-number consumption order "
                           "changes.",
        "a_rejected_candidate_control": {
            "candidate": "split the key three ways and discard one output",
            "why_it_was_rejected": "JAX's threefry split is a prefix, so split(k, 3)[:2] is "
                                   "bitwise equal to split(k, 2) and the stream is unchanged. The "
                                   "gate correctly reported zero mismatches for it, which is why "
                                   "it is not a usable control.",
            "threefry_split_is_a_prefix": True},
        "n_photons": int(nc_keys.shape[0]),
        "det_id_mismatches": nc_det, "escaped_flag_mismatches": nc_esc,
        "weight_max_abs_diff": nc_dw,
        "rejected_by_the_gate": bool(nc_rejected),
        "threefry_split_prefix_property_verified_in_run": prefix_property}
    print(f"\n  negative control: defective replay over {nc_keys.shape[0]} photons -> "
          f"{nc_det} detector-bin mismatches, {nc_esc} escaped-flag mismatches, "
          f"max |dw| {nc_dw:.3e}")
    print(f"    rejected by the gate: {nc_rejected}"
          + ("" if nc_rejected else "   [the gate cannot fail and is therefore worthless]"))
    all_pass = all_pass and nc_rejected

    payload = {
        "gate": "photon-level equivalence of the replay kernel against the production kernel",
        "production_kernel_is_the_only_physical_truth": True,
        "production_physics_unmodified": True,
        "pooled_reflectance_alone_is_not_accepted_as_evidence":
            "two kernels can agree on an aggregate while disagreeing per photon, and a pathlength "
            "moment is a per-photon quantity",
        "what_is_shared_with_production":
            "the Fresnel reflectance, the Henyey-Greenstein sampler, the direction update and the "
            "termination constants are imported from the production module, not copied. The only "
            "new code is one extra state element carrying per-layer pathlength, threaded through "
            "the identical loop structure.",
        "trajectory_invariants_held_fixed": [
            "same PRNG keys", "same random-number consumption order",
            "same branch order", "same max_steps", "same radial grid",
            "same layer boundaries and optical properties", "same detector definition"],
        "criteria": TOL,
        "design": {"n_photons_per_wavelength_per_seed": int(a.n), "seeds": seeds,
                   "n_wavelengths": n_wl, "max_steps": max_steps,
                   "radial_grid": radial, "layer_boundaries": NOMINAL_BOUNDARIES,
                   "psi_nominal": psi0.tolist(), "blood_coordinate": "physical_bvf"},
        "cases": cases,
        "negative_control": negative_control,
        "summary": {
            "n_cases": len(cases), "n_passed": sum(c["passes"] for c in cases),
            "all_passed": bool(all_pass),
            "negative_control_rejected": bool(nc_rejected),
            "total_photon_comparisons": int(a.n) * len(seeds) * n_wl,
            "det_id_mismatches_total": sum(c["det_id_mismatches"] for c in cases),
            "escaped_flag_mismatches_total": sum(c["escaped_flag_mismatches"] for c in cases),
            "weight_bitwise_identical_in_all_cases":
                bool(all(c["weight_bitwise_identical"] for c in cases)),
            "det_id_bitwise_identical_in_all_cases":
                bool(all(c["det_id_bitwise_identical"] for c in cases)),
            "worst_weight_max_abs_diff": max(c["weight_max_abs_diff"] for c in cases),
            "worst_reflectance_max_rel_diff": max(c["reflectance_max_rel_diff"] for c in cases),
            "worst_pathlength_identity_max_rel_diff":
                max(c["pathlength_identity_max_rel_diff"] for c in cases)},
        "environment": {
            "jax_version": jax.__version__, "jax_backend": jax.default_backend(),
            "jax_devices": [str(d) for d in jax.devices()],
            "x64_enabled": bool(jax.config.read("jax_enable_x64")),
            "numpy_version": np.__version__, "python": platform.python_version(),
            "platform": platform.platform()},
        "elapsed_s": round(time.time() - t0, 1),
        "verdict": "pass, the gate can fail and the real replay is accepted; the pMC route is "
                   "unblocked" if all_pass else
                   "FAIL, the pMC route stays blocked and no pilot may run",
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(payload, fh, indent=2)

    sm = payload["summary"]
    print(f"\n  {sm['n_passed']}/{sm['n_cases']} cases passed over "
          f"{sm['total_photon_comparisons']} photon comparisons")
    print(f"  detector id bitwise identical in all cases: "
          f"{sm['det_id_bitwise_identical_in_all_cases']}")
    print(f"  escaping weight bitwise identical in all cases: "
          f"{sm['weight_bitwise_identical_in_all_cases']}")
    print(f"  worst pathlength identity residual {sm['worst_pathlength_identity_max_rel_diff']:.3e}")
    print(f"  wrote {os.path.relpath(a.out, ROOT)}")
    print(f"  {payload['verdict']}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
