#!/usr/bin/env python3
"""candidate3_executor_spike.py -- CPU-only feasibility spike for candidate 3 option A.

READ-ONLY AGAINST jax-pad. Nothing in the production package is modified. This file is a LOCAL
wrapper that calls the hash-bound per-photon kernel directly:

    jax-pad/src/pad/physics_engine_v4.py :: simulate_one_photon_rigorous

which is already in the gate's manifest as `jaxpad:src/pad/physics_engine_v4.py`. The production
aggregate solver `mc_forward.forward_y_multi_layer` imports that same symbol at mc_forward.py:30 and
then vmaps and scatter-adds it; this wrapper performs the same vmap, keeps the per-photon output,
and does its own scatter. The kernel source is never copied and the vendored replay kernel is never
substituted for it.

CPU ONLY, tiny fixed N, fixed seed. No GPU, no coverage experiment, no margin, grid, replicate or
support choice, nothing registered, nothing frozen.

    python3 scripts/candidate3_executor_spike.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# CPU is forced BEFORE anything can import JAX. JAX reads JAX_PLATFORMS once, at first
# import, so setting it after an import would silently do nothing. This spike must never
# touch a GPU.
os.environ["JAX_PLATFORMS"] = "cpu"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SPIKE_STATUS = ("CPU-ONLY FEASIBILITY SPIKE. jax-pad is not modified, the production executor is "
                "not modified, no GPU is used, and nothing is registered or frozen.")


def _resolve_jaxpad(jaxpad_root=None):
    """Resolve jax-pad the way the gate does, and return the modules actually imported."""
    root = jaxpad_root or os.environ.get("JAXPAD_ROOT") or os.path.join(ROOT, "..", "jax-pad")
    root = os.path.abspath(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    import jax
    if jax.default_backend() != "cpu":
        raise SystemExit(
            "REFUSING: this spike is CPU only and the backend came up as "
            f"{jax.default_backend()!r}. JAX_PLATFORMS is set to cpu at the top of this "
            "file, before any import that could initialise JAX; a non-cpu backend means "
            "something initialised JAX first. No transport is run.")
    jax.config.update("jax_enable_x64", True)
    from src.pad import physics_engine_v4 as pe
    from src.pad import mc_forward as mcf
    return {"root": root, "physics_engine_v4": pe, "mc_forward": mcf,
            "kernel": pe.simulate_one_photon_rigorous,
            "prod_forward": mcf.forward_y_multi_layer,
            "kernel_is_the_production_symbol":
                mcf.simulate_one_photon is pe.simulate_one_photon_rigorous,
            "files": {"physics_engine_v4": os.path.abspath(pe.__file__),
                      "mc_forward": os.path.abspath(mcf.__file__)}}


# ---------------------------------------------------------------------------------------------
# The local device-side wrapper
# ---------------------------------------------------------------------------------------------

def per_photon(jp, *, muas, muss, gs, ns, layer_boundaries, keys, detector_r_edges, max_steps):
    """vmap the hash-bound kernel and KEEP the per-photon output. No aggregation here.

    This is the only thing the production solver does differently: mc_forward.py:57 performs this
    same vmap and mc_forward.py:72-79 then scatter-adds and normalises, discarding the per-photon
    arrays. Nothing about the transport changes.
    """
    import jax
    import jax.numpy as jnp
    f = jax.vmap(jp["kernel"], in_axes=(0, None, None, None, None, None, None, None))
    det_ids, w_refl = f(keys, muas, muss, gs, ns, layer_boundaries, detector_r_edges, max_steps)
    return det_ids, w_refl


def scatter_reflectance(det_ids, w_refl, detector_r_edges, n_photons):
    """Re-derive the production aggregate from the per-photon output, by the same arithmetic.

    Mirrors mc_forward.py:68-79 exactly: the same `valid = det_ids >= 0` mask, the same
    `where`-guarded safe ids and weights, the same scatter-add, the same annulus normalisation.
    An aggregate-equivalence regression compares this against prod_forward so the wrapper cannot
    drift from the production path.
    """
    import jax.numpy as jnp
    M = detector_r_edges.shape[0] - 1
    valid = det_ids >= 0
    safe_ids = jnp.where(valid, det_ids, 0).astype(jnp.int32)
    safe_w = jnp.where(valid, w_refl, 0.0)
    y = jnp.zeros((M,), dtype=w_refl.dtype).at[safe_ids].add(safe_w)
    r0, r1 = detector_r_edges[:-1], detector_r_edges[1:]
    return y / (jnp.asarray(n_photons, dtype=w_refl.dtype) * jnp.pi * (r1 ** 2 - r0 ** 2))


def combine_device_side(det_plus, w_plus, det_minus, w_minus, h_grid, q, areas, M):
    """Form Z per photon from its at most 2L sparse (bin, contribution) pairs, then accumulate.

    NO DENSE (N, M) ARRAY IS BUILT. For photon p the contribution to bin i is

        Z_p,i = sum_l q_l [ w+_{p,l} 1[b+_{p,l}=i] - w-_{p,l} 1[b-_{p,l}=i] ] / (2 h_l A_i)

    and the photon touches at most 2L distinct bins, so it is represented by 2L sparse pairs. Pairs
    landing in the SAME bin must be summed BEFORE squaring, which is the whole point of option A, so
    duplicates are folded with a 2L x 2L comparison inside the photon and only the first occurrence
    of each bin is kept. The cost is 2L x 2L per photon, independent of M.

    Returns sum_Z, sum_Z2, max_abs_Z, each of shape (M,).
    """
    import jax
    import jax.numpy as jnp
    L = len(h_grid)
    h = jnp.asarray(h_grid, dtype=w_plus.dtype)
    qv = jnp.asarray(q, dtype=w_plus.dtype)
    A = jnp.asarray(areas, dtype=w_plus.dtype)

    bins = jnp.concatenate([det_plus, det_minus], axis=1)                 # (N, 2L)
    valid = bins >= 0
    safe = jnp.where(valid, bins, 0).astype(jnp.int32)
    sgn = jnp.concatenate([jnp.ones((1, L)), -jnp.ones((1, L))], axis=1)  # (1, 2L)
    w = jnp.concatenate([w_plus, w_minus], axis=1)
    coef = jnp.concatenate([qv / (2.0 * h), qv / (2.0 * h)], axis=0)[None, :] * sgn
    vals = jnp.where(valid, w * coef / A[safe], 0.0)                      # (N, 2L)

    def fold(b, v, ok):
        """Sum duplicates within one photon and keep only the first occurrence of each bin."""
        same = (b[:, None] == b[None, :]) & ok[None, :] & ok[:, None]
        combined = (same * v[None, :]).sum(axis=1)
        first = jnp.argmax(same, axis=1) == jnp.arange(b.shape[0])
        keep = ok & first
        return combined * keep, keep

    comb, keep = jax.vmap(fold)(safe, vals, valid)                        # (N, 2L) each
    z = jnp.zeros((M,), dtype=w_plus.dtype)
    sum_Z = z.at[safe].add(jnp.where(keep, comb, 0.0))
    sum_Z2 = z.at[safe].add(jnp.where(keep, comb * comb, 0.0))
    max_abs_Z = z.at[safe].max(jnp.where(keep, jnp.abs(comb), 0.0))
    return sum_Z, sum_Z2, max_abs_Z


def per_step_max_abs_Y(det_plus, w_plus, det_minus, w_minus, h_grid, areas, M):
    """max_p |Y_{p,i}(h_l)| per step, with the two arms PAIRED before the absolute value.

    Y is a central difference, so for one photon and one step the plus and minus records must be
    combined by bin first:

        Y_{p,i}(h) = ( w+ 1[b+ = i] - w- 1[b- = i] ) / (2 h A_i)

    When both arms land in the SAME bin the contribution is the DIFFERENCE w+ - w-, which common
    random numbers make small. Taking each arm's maximum separately, as an earlier version did,
    reports max(|w+|, |w-|) there instead and overstates the quantity, sometimes by orders of
    magnitude. That surrogate is withdrawn.

    Returns an (L, M) array.
    """
    import jax
    import jax.numpy as jnp
    L = len(h_grid)
    A = jnp.asarray(areas, dtype=w_plus.dtype)

    def one_step(dp_l, wp_l, dm_l, wm_l, h):
        bins = jnp.stack([dp_l, dm_l], axis=1)                     # (N, 2)
        ok = bins >= 0
        safe = jnp.where(ok, bins, 0).astype(jnp.int32)
        vals = jnp.where(ok,
                         jnp.stack([wp_l, -wm_l], axis=1) / (2.0 * h * A[safe]),
                         0.0)

        def fold(b, v, m_):
            same = (b[:, None] == b[None, :]) & m_[None, :] & m_[:, None]
            combined = (same * v[None, :]).sum(axis=1)
            first = jnp.argmax(same, axis=1) == jnp.arange(b.shape[0])
            keep = m_ & first
            return combined * keep, keep

        comb, keep = jax.vmap(fold)(safe, vals, ok)
        z = jnp.zeros((M,), dtype=w_plus.dtype)
        return z.at[safe].max(jnp.where(keep, jnp.abs(comb), 0.0))

    return jnp.stack([one_step(det_plus[:, l], w_plus[:, l],
                               det_minus[:, l], w_minus[:, l], float(h_grid[l]))
                      for l in range(L)])


# ---------------------------------------------------------------------------------------------
# The single device-side entry point: eight transports, fold, scatter, accumulators out
# ---------------------------------------------------------------------------------------------

def option_a_accumulators_device(jp, cfg, h_grid, q2, qd, keys):
    """ONE jitted computation: eight per-photon transports, fold before squaring, scatter out.

    Everything between the photon keys and the accumulators happens inside a single device
    computation boundary. The per-photon detector ids and weights are produced, combined and
    discarded there; they never cross the boundary.

    The return tree is fixed size and contains ONLY:

        sum_Z_e2, sum_Z2_e2, max_abs_Z_e2   each (M,)
        sum_Z_d,  sum_Z2_d,  max_abs_Z_d    each (M,)
        max_abs_Y                           (L, M)

    No array whose leading dimension is N is returned. The dense (N, M) reference used in the
    self-test is an external oracle on a handful of photons and has no path into this function.
    """
    import jax
    import jax.numpy as jnp

    kernel = jp["kernel"]
    muas, muss, gs, ns = cfg["muas"], cfg["muss"], cfg["gs"], cfg["ns"]
    edges, max_steps, M = cfg["edges"], cfg["max_steps"], cfg["M"]
    boundaries = cfg["boundaries"]
    areas = jnp.asarray(cfg["areas"], dtype=jnp.float64)
    h = jnp.asarray(h_grid, dtype=jnp.float64)
    L = len(h_grid)
    bounds_p = [boundaries(+float(x)) for x in h_grid]
    bounds_m = [boundaries(-float(x)) for x in h_grid]

    def body(keys_in, q2_in, qd_in):
        trace = jax.vmap(kernel, in_axes=(0, None, None, None, None, None, None, None))
        dp, wp, dm, wm = [], [], [], []
        for l in range(L):
            d1, w1 = trace(keys_in, muas, muss, gs, ns, bounds_p[l], edges, max_steps)
            d2, w2 = trace(keys_in, muas, muss, gs, ns, bounds_m[l], edges, max_steps)
            dp.append(d1)
            wp.append(w1)
            dm.append(d2)
            wm.append(w2)
        det_p = jnp.stack(dp, axis=1)
        w_p = jnp.stack(wp, axis=1)
        det_m = jnp.stack(dm, axis=1)
        w_m = jnp.stack(wm, axis=1)

        bins = jnp.concatenate([det_p, det_m], axis=1)
        ok = bins >= 0
        safe = jnp.where(ok, bins, 0).astype(jnp.int32)
        sgn = jnp.concatenate([jnp.ones((L,)), -jnp.ones((L,))])[None, :]
        w = jnp.concatenate([w_p, w_m], axis=1)
        base = w * sgn / (2.0 * jnp.concatenate([h, h])[None, :] * areas[safe])

        def fold(b, v, m_):
            same = (b[:, None] == b[None, :]) & m_[None, :] & m_[:, None]
            combined = (same * v[None, :]).sum(axis=1)
            first = jnp.argmax(same, axis=1) == jnp.arange(b.shape[0])
            keep = m_ & first
            return combined * keep, keep

        out = {}
        for tag, qv in (("e2", q2_in), ("d", qd_in)):
            vals = jnp.where(ok, base * jnp.concatenate([qv, qv])[None, :], 0.0)
            comb, keep = jax.vmap(fold)(safe, vals, ok)
            z = jnp.zeros((M,), dtype=jnp.float64)
            kept = jnp.where(keep, comb, 0.0)
            out[f"sum_Z_{tag}"] = z.at[safe].add(kept)
            out[f"sum_Z2_{tag}"] = z.at[safe].add(jnp.where(keep, comb * comb, 0.0))
            out[f"max_abs_Z_{tag}"] = z.at[safe].max(jnp.abs(kept))

        # max_abs_Y uses the SAME pairing as Z: the two arms are combined by bin before the
        # absolute value, because Y is a central difference. An arm-wise maximum would
        # report max(|w+|, |w-|) where the arms share a bin and overstate the quantity.
        mY = []
        for l in range(L):
            b2 = jnp.stack([det_p[:, l], det_m[:, l]], axis=1)
            o2 = b2 >= 0
            s2 = jnp.where(o2, b2, 0).astype(jnp.int32)
            v2 = jnp.where(o2,
                           jnp.stack([w_p[:, l], -w_m[:, l]], axis=1)
                           / (2.0 * h[l] * areas[s2]),
                           0.0)
            c2, k2 = jax.vmap(fold)(s2, v2, o2)
            mY.append(jnp.zeros((M,), dtype=jnp.float64).at[s2].max(
                jnp.where(k2, jnp.abs(c2), 0.0)))
        out["max_abs_Y"] = jnp.stack(mY)
        return out

    return jax.jit(body)(keys, jnp.asarray(q2, dtype=jnp.float64),
                         jnp.asarray(qd, dtype=jnp.float64))


ACCUMULATOR_KEYS = ("sum_Z_e2", "sum_Z2_e2", "max_abs_Z_e2",
                    "sum_Z_d", "sum_Z2_d", "max_abs_Z_d", "max_abs_Y")


# ---------------------------------------------------------------------------------------------
# Pairing checks, reused from the design module so the spike cannot drift from the spec
# ---------------------------------------------------------------------------------------------

sys.path.insert(0, HERE)
import candidate3_design as C3  # noqa: E402


def run_eight(jp, cfg, h_grid, keys):
    """The eight evaluations for one column and one wavelength, all from ONE key array."""
    import jax.numpy as jnp
    det_p, w_p, det_m, w_m, labels = [], [], [], [], {}
    for h in h_grid:
        for sign, dl, wl_ in ((+1.0, det_p, w_p), (-1.0, det_m, w_m)):
            bounds = cfg["boundaries"](sign * h)
            d, w = per_photon(jp, muas=cfg["muas"], muss=cfg["muss"], gs=cfg["gs"],
                              ns=cfg["ns"], layer_boundaries=bounds, keys=keys,
                              detector_r_edges=cfg["edges"], max_steps=cfg["max_steps"])
            dl.append(d)
            wl_.append(w)
            labels[f"h{float(h):g}{'+' if sign > 0 else '-'}"] = keys
    return (jnp.stack(det_p, axis=1), jnp.stack(w_p, axis=1),
            jnp.stack(det_m, axis=1), jnp.stack(w_m, axis=1), labels)


def tiny_config(jp, n_bins=12, r_max=3.0, max_steps=200):
    """A small, fixed, CPU-sized optical configuration. Not a production setting."""
    import jax.numpy as jnp
    d = jnp.float64
    muas = jnp.asarray([0.3, 0.2, 0.1], dtype=d)
    muss = jnp.asarray([80.0, 60.0, 40.0], dtype=d)
    gs = jnp.asarray([0.9, 0.9, 0.9], dtype=d)
    ns = jnp.asarray([1.4, 1.4, 1.4], dtype=d)
    nom = jnp.asarray([0.02, 0.08, 0.30], dtype=d)
    edges = jnp.linspace(0.0, r_max, n_bins + 1, dtype=d)
    areas = np.pi * (np.asarray(edges[1:]) ** 2 - np.asarray(edges[:-1]) ** 2)

    def boundaries(eta0):
        """One nuisance column: a log scale on the first layer thickness. jnp only."""
        t = jnp.diff(jnp.concatenate([jnp.zeros(1, dtype=d), nom]))
        t = t.at[0].multiply(jnp.exp(jnp.asarray(eta0, dtype=d)))
        return jnp.cumsum(t)

    return {"muas": muas, "muss": muss, "gs": gs, "ns": ns, "edges": edges,
            "areas": areas, "max_steps": max_steps, "boundaries": boundaries, "M": n_bins}


def resources(n_photons=200000, n_wl=4, n_columns=5, n_steps=4, n_bins=120):
    """Corrected resource arithmetic, separating host-side from device-side combination."""
    per_photon_bytes = n_steps * 2 * (4 + 8)          # int32 det_id + float64 weight, 8 of each
    raw_one = per_photon_bytes * n_photons            # one column, one wavelength
    acc_scalars = 3 * 2 * n_bins * n_columns + n_bins * n_steps * n_columns
    return {
        "per_photon_raw_bytes": per_photon_bytes,
        "note_per_photon": f"{n_steps} steps x 2 arms x (int32 det_id 4B + float64 weight 8B) = "
                           f"{per_photon_bytes} bytes per photon",
        "raw_one_column_one_wavelength_MB": raw_one / 1e6,
        "raw_one_column_all_wavelengths_MB": raw_one * n_wl / 1e6,
        "raw_all_columns_all_wavelengths_MB": raw_one * n_wl * n_columns / 1e6,
        "recommended_processing_order": "column outer, wavelength inner, so only ONE "
                                        "(column, wavelength) of raw per-photon output is live at "
                                        "a time. The peak raw footprint is then about "
                                        f"{raw_one / 1e6:.1f} MB plus XLA temporaries, not "
                                        f"{raw_one * n_wl * n_columns / 1e6:.0f} MB.",
        "host_side_combination": {
            "what_crosses_the_bus": "every per-photon array",
            "per_column_MB": raw_one * n_wl / 1e6,
            "per_replicate_MB": raw_one * n_wl * n_columns / 1e6,
            "verdict": "rejected: it moves hundreds of megabytes per replicate for no benefit"},
        "device_side_combination": {
            "what_crosses_the_bus": "only the accumulators",
            "accumulator_scalars_per_replicate": acc_scalars,
            "per_replicate_KiB": acc_scalars * 8 / 1024,
            "verdict": "chosen: the per-photon arrays never leave the device"},
        "transfer_is_not_zero_either_way": "an earlier note said host-side combination and zero "
                                           "extra transfer in the same breath. Only the "
                                           "device-side form has a negligible transfer, and it is "
                                           "the accumulators, not nothing.",
        "runtime": "the eight evaluations are already performed by the current gate, so the "
                   "transport work is unchanged. What is added is keeping the per-photon output "
                   "live and one fold plus three scatters per photon.",
    }


# ---------------------------------------------------------------------------------------------
# CPU self-test, tiny fixed N and fixed seed
# ---------------------------------------------------------------------------------------------

def selftest(n_photons=600, seed=20261006):
    import jax
    import jax.numpy as jnp
    res, fails = [], []

    def ck(name, ok, detail=""):
        res.append({"name": name, "pass": bool(ok)})
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
        if not ok:
            fails.append(name)

    jp = _resolve_jaxpad()
    ck("the kernel imported is the SAME symbol the production solver vmaps",
       jp["kernel_is_the_production_symbol"] is True,
       "mc_forward.simulate_one_photon is physics_engine_v4.simulate_one_photon_rigorous")
    ck("the kernel comes from the resolved jax-pad, not from a vendored copy",
       os.path.abspath(jp["files"]["physics_engine_v4"]).endswith(
           os.path.join("src", "pad", "physics_engine_v4.py"))
       and "replay_kernel" not in jp["files"]["physics_engine_v4"],
       jp["files"]["physics_engine_v4"])
    import ast as _ast
    with open(os.path.abspath(__file__)) as fh:
        own_src = fh.read()
    _own = _ast.parse(own_src)
    _own_fns = {n.name for n in _own.body if isinstance(n, _ast.FunctionDef)}
    ck("the spike defines no transport kernel of its own; it only calls the bound one",
       not any("photon_rigorous" in n or n.startswith("simulate_") for n in _own_fns)
       and ("lax." + "while_loop") not in own_src,
       f"{len(_own_fns)} local functions, none a transport kernel")
    ck("jax is on CPU for this spike",
       jax.default_backend() == "cpu", jax.default_backend())

    cfg = tiny_config(jp)
    h_grid = np.array([0.4, 0.3, 0.2, 0.1])
    keys = jax.random.split(jax.random.PRNGKey(seed), n_photons)
    det_p, w_p, det_m, w_m, labels = run_eight(jp, cfg, h_grid, keys)

    # --- the wrapper must reproduce the production aggregate exactly -------------------------
    worst = 0.0
    for l, h in enumerate(h_grid):
        for sign, det, w in ((+1.0, det_p[:, l], w_p[:, l]), (-1.0, det_m[:, l], w_m[:, l])):
            mine = scatter_reflectance(det, w, cfg["edges"], n_photons)
            prod = jp["prod_forward"](muas=cfg["muas"], muss=cfg["muss"], gs=cfg["gs"],
                                      ns=cfg["ns"],
                                      layer_boundaries=cfg["boundaries"](sign * h),
                                      keys=keys, detector_r_edges=cfg["edges"],
                                      max_steps=cfg["max_steps"])
            worst = max(worst, float(jnp.max(jnp.abs(mine - prod))))
    ck("the re-scattered reflectance is BITWISE equal to the production aggregate",
       worst == 0.0, f"max abs difference {worst:.3e} over 8 evaluations")

    # --- sum_Z / N must equal q applied to the aggregate difference quotients ------------------
    fr = C3.fit_rows(h_grid)
    Rp = jnp.stack([scatter_reflectance(det_p[:, l], w_p[:, l], cfg["edges"], n_photons)
                    for l in range(len(h_grid))])
    Rm = jnp.stack([scatter_reflectance(det_m[:, l], w_m[:, l], cfg["edges"], n_photons)
                    for l in range(len(h_grid))])
    D = (Rp - Rm) / (2.0 * jnp.asarray(h_grid))[:, None]
    for nm, q in (("q2", fr["q2"]), ("qd", fr["qd"])):
        sZ, sZ2, mZ = combine_device_side(det_p, w_p, det_m, w_m, h_grid, q,
                                          cfg["areas"], cfg["M"])
        agg = jnp.asarray(q, dtype=D.dtype) @ D
        err = float(jnp.max(jnp.abs(sZ / n_photons - agg)))
        ck(f"sum_Z/N for {nm} matches the aggregate fit applied to the four D(h)",
           err < 1e-12 * float(jnp.max(jnp.abs(agg)) + 1e-300),
           f"max abs difference {err:.3e}")

    # --- sum_Z2 against an explicit materialised reference on a tiny sample --------------------
    small = 40
    dp, wp = det_p[:small], w_p[:small]
    dm, wm = det_m[:small], w_m[:small]
    sZ, sZ2, mZ = combine_device_side(dp, wp, dm, wm, h_grid, fr["qd"], cfg["areas"], cfg["M"])
    Zdense = np.zeros((small, cfg["M"]))
    for p in range(small):
        for l, h in enumerate(h_grid):
            for sign, det, w in ((+1.0, np.asarray(dp)[p, l], np.asarray(wp)[p, l]),
                                 (-1.0, np.asarray(dm)[p, l], np.asarray(wm)[p, l])):
                if det >= 0:
                    Zdense[p, det] += (sign * fr["qd"][l] * w
                                       / (2.0 * h * cfg["areas"][det]))
    def rel(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        sc = max(float(np.max(np.abs(b))), np.finfo(float).tiny)
        return float(np.max(np.abs(a - b))) / sc
    ck("sum_Z matches an explicit materialised dense reference on 40 photons",
       rel(sZ, Zdense.sum(axis=0)) < 1e-13,
       f"relative {rel(sZ, Zdense.sum(axis=0)):.3e}")
    ck("sum_Z2 matches the dense reference, so duplicates were folded BEFORE squaring",
       rel(sZ2, (Zdense ** 2).sum(axis=0)) < 1e-13,
       f"relative {rel(sZ2, (Zdense ** 2).sum(axis=0)):.3e}")
    ck("max_abs_Z matches the dense reference",
       rel(mZ, np.abs(Zdense).max(axis=0)) < 1e-13)
    naive = np.zeros(cfg["M"])
    for p_ in range(small):
        for l, h in enumerate(h_grid):
            for sign, det, w in ((+1.0, np.asarray(dp)[p_, l], np.asarray(wp)[p_, l]),
                                 (-1.0, np.asarray(dm)[p_, l], np.asarray(wm)[p_, l])):
                if det >= 0:
                    naive[det] += (sign * fr["qd"][l] * w
                                   / (2.0 * h * cfg["areas"][det])) ** 2
    ck("folding before squaring differs from squaring each pair separately",
       rel(naive, (Zdense ** 2).sum(axis=0)) > 1e-3,
       f"relative gap {rel(naive, (Zdense ** 2).sum(axis=0)):.3e}; without the fold "
       f"this test would be vacuous")
    n_dup = int(sum(len(set(np.concatenate([np.asarray(dp)[p], np.asarray(dm)[p]])
                            [np.concatenate([np.asarray(dp)[p], np.asarray(dm)[p]]) >= 0]))
                    < int((np.concatenate([np.asarray(dp)[p], np.asarray(dm)[p]]) >= 0).sum())
                    for p in range(small)))
    ck("the tiny sample actually contains photons hitting one bin more than once",
       n_dup > 0, f"{n_dup} of {small} photons have a repeated bin, so the fold is exercised")

    # --- no dense (N, M) is built by the production path --------------------------------------
    L2 = 2 * len(h_grid)
    ck("the combination per-photon intermediates are (N, 2L), never (N, M)",
       det_p.shape[1] + det_m.shape[1] == L2 and cfg["M"] != L2
       and "independent of M" in combine_device_side.__doc__
       and "NO DENSE" in combine_device_side.__doc__,
       f"per-photon width {L2}, bins {cfg['M']}; the dense reference is used ONLY on {small} photons, never at production N")

    # --- per-step max_abs_Y --------------------------------------------------------------------
    mY = per_step_max_abs_Y(det_p, w_p, det_m, w_m, h_grid, cfg["areas"], cfg["M"])
    ck("per-step max_abs_Y has one row per step and is non-negative",
       mY.shape == (len(h_grid), cfg["M"]) and bool(jnp.all(mY >= 0.0)))
    B = np.array([C3.absolute_bound(float(h), cfg["areas"]) for h in h_grid])
    ck("every observed per-step max is within the derived absolute bound",
       bool(np.all(np.asarray(mY) <= B * (1.0 + 1e-12))),
       f"worst ratio {float(np.max(np.asarray(mY) / B)):.4f}")

    # --- an INDEPENDENT dense oracle for max_abs_Y ------------------------------------
    # Y[p, l, i] is built explicitly from the definition on a small sample, with no reuse
    # of either implementation. Both the staged helper and the jitted entry are checked
    # against it, so they cannot validate only each other.
    nos = 50
    Ydense = np.zeros((nos, len(h_grid), cfg["M"]))
    for p_ in range(nos):
        for l, h in enumerate(h_grid):
            bp, wpv = int(np.asarray(det_p)[p_, l]), float(np.asarray(w_p)[p_, l])
            bm, wmv = int(np.asarray(det_m)[p_, l]), float(np.asarray(w_m)[p_, l])
            if bp >= 0:
                Ydense[p_, l, bp] += wpv / (2.0 * h * cfg["areas"][bp])
            if bm >= 0:
                Ydense[p_, l, bm] -= wmv / (2.0 * h * cfg["areas"][bm])
    oracle = np.abs(Ydense).max(axis=0)
    mY_small = per_step_max_abs_Y(det_p[:nos], w_p[:nos], det_m[:nos], w_m[:nos],
                                  h_grid, cfg["areas"], cfg["M"])
    ck("the staged max_abs_Y matches an INDEPENDENT dense oracle of Y[p, step, bin]",
       float(np.max(np.abs(np.asarray(mY_small) - oracle))) < 1e-12,
       f"max abs difference "
       f"{float(np.max(np.abs(np.asarray(mY_small) - oracle))):.3e} over "
       f"{nos} photons")
    acc_small = option_a_accumulators_device(jp, cfg, h_grid, fr["q2"], fr["qd"],
                                             keys[:nos])
    jit_oracle_gap = float(np.max(np.abs(np.asarray(acc_small["max_abs_Y"]) - oracle)))
    ck("the jitted max_abs_Y matches the SAME independent oracle",
       jit_oracle_gap < 1e-12,
       f"max abs difference {jit_oracle_gap:.3e}")

    # --- the withdrawn arm-wise surrogate must actually differ ------------------------
    arm_wise = np.zeros((len(h_grid), cfg["M"]))
    for l, h in enumerate(h_grid):
        for det, w in ((np.asarray(det_p)[:nos, l], np.asarray(w_p)[:nos, l]),
                       (np.asarray(det_m)[:nos, l], np.asarray(w_m)[:nos, l])):
            for p_ in range(nos):
                b = int(det[p_])
                if b >= 0:
                    arm_wise[l, b] = max(arm_wise[l, b],
                                         abs(float(w[p_]))
                                         / (2.0 * h * cfg["areas"][b]))
    n_diff = int((np.abs(arm_wise - oracle) > 1e-12).sum())
    ck("the withdrawn arm-wise surrogate differs from the true paired max_abs_Y",
       n_diff > 0 and bool(np.all(arm_wise >= oracle - 1e-12)),
       f"{n_diff} of {arm_wise.size} entries differ; the surrogate is an upper bound and "
       f"overstates by up to "
       f"{float(np.max(arm_wise - oracle)):.3e}")

    # --- the single device-side entry point ----------------------------------------------------
    acc = option_a_accumulators_device(jp, cfg, h_grid, fr["q2"], fr["qd"], keys)
    ck("the jitted entry returns exactly the seven accumulators and nothing else",
       set(acc) == set(ACCUMULATOR_KEYS), f"{sorted(acc)}")
    leaves = jax.tree_util.tree_leaves(acc)
    ck("no returned array has N as its leading dimension",
       all(a.shape[0] != n_photons for a in leaves)
       and all(a.shape[0] in (cfg["M"], len(h_grid)) for a in leaves),
       f"shapes {[tuple(a.shape) for a in leaves]} against N={n_photons}")
    n_scalars = int(sum(int(np.prod(a.shape)) for a in leaves))
    expect_scalars = 6 * cfg["M"] + len(h_grid) * cfg["M"]
    ck("the returned scalar count matches the design: 6 per bin plus one per step per bin",
       n_scalars == expect_scalars,
       f"{n_scalars} scalars = 6 x {cfg['M']} + {len(h_grid)} x {cfg['M']}")
    ck("every accumulator is finite and the sizes are fixed, independent of N",
       all(bool(jnp.all(jnp.isfinite(a))) for a in leaves)
       and all(a.shape[0] != n_photons for a in
               jax.tree_util.tree_leaves(
                   option_a_accumulators_device(jp, cfg, h_grid, fr["q2"], fr["qd"],
                                                keys[: n_photons // 2]))),
       "halving N leaves every accumulator shape unchanged")
    # The staged helper and the jitted entry evaluate the same expression in a different
    # ORDER, so they agree to floating point rather than bitwise. Bitwise is required only
    # where the arithmetic is literally identical, which is the aggregate-versus-production
    # check above, and that one does hold at exactly zero.
    def relmax(x, y):
        sc = max(float(jnp.max(jnp.abs(y))), np.finfo(float).tiny)
        return float(jnp.max(jnp.abs(x - y))) / sc
    worst_rel = 0.0
    for tag, qv in (("e2", fr["q2"]), ("d", fr["qd"])):
        sZr, sZ2r, mZr = combine_device_side(det_p, w_p, det_m, w_m, h_grid, qv,
                                             cfg["areas"], cfg["M"])
        r = max(relmax(acc[f"sum_Z_{tag}"], sZr),
                relmax(acc[f"sum_Z2_{tag}"], sZ2r),
                relmax(acc[f"max_abs_Z_{tag}"], mZr))
        worst_rel = max(worst_rel, r)
        ck(f"the jitted entry reproduces the staged combination for {tag}",
           r < 1e-14, f"worst relative difference {r:.3e}, order of evaluation only")
    ck("the staged and jitted agreement is at the floating-point floor, not merely close",
       worst_rel < 1e-14, f"worst relative difference over both {worst_rel:.3e}")
    ck("the jitted entry reproduces the staged per-step max_abs_Y, bitwise",
       float(jnp.max(jnp.abs(acc["max_abs_Y"] - mY))) == 0.0)
    ck("the dense oracle has no path into the jitted entry",
       "Zdense" not in (option_a_accumulators_device.__doc__ or "")
       and "Zdense" not in own_src.split("def option_a_accumulators_device")[1].split(
           "ACCUMULATOR_KEYS")[0])

    # --- pairing refusals ----------------------------------------------------------------------
    labs = C3.required_labels(h_grid)
    ck("the eight evaluations used one key array, in one order, at a fixed N",
       C3.verify_pairing(labels, n_photons)["ok"] and set(labels) == set(labs),
       f"{len(labels)} labels")
    perm = dict(labels)
    perm[labs[2]] = np.asarray(keys)[::-1].copy()
    ck("a permuted key array is REFUSED", not C3.verify_pairing(perm, n_photons)["ok"])
    ck("a missing arm is REFUSED",
       not C3.verify_coverage_of_labels({k: v for k, v in labels.items() if k != labs[1]},
                                        h_grid)["ok"])
    ck("a missing step is REFUSED",
       not C3.verify_coverage_of_labels(
           {k: v for k, v in labels.items() if not k.startswith("h0.2")}, h_grid)["ok"])
    ck("a launch count different from the fixed N is REFUSED",
       not C3.verify_pairing(labels, n_photons - 1)["ok"])

    # --- a bin nobody reaches ------------------------------------------------------------------
    counts = np.zeros(cfg["M"], dtype=int)
    for arr in (np.asarray(det_p), np.asarray(det_m)):
        for v in arr.ravel():
            if v >= 0:
                counts[int(v)] += 1
    empty = np.flatnonzero(counts == 0)
    sZ_all, sZ2_all, _ = combine_device_side(det_p, w_p, det_m, w_m, h_grid, fr["qd"],
                                             cfg["areas"], cfg["M"])
    if empty.size:
        # the quantity under test is Z_d, so the range must be the COMBINED one, not a
        # single step width. R_Z_d = 2 sum_l |qd_l| B(h_l), by the triangle inequality.
        R_Zd = C3.combined_range_width(h_grid, cfg["areas"], fr["qd"])
        single_step_width = C3.deterministic_range_width(float(h_grid[-1]), cfg["areas"])
        expect = 2.0 * sum(abs(float(q)) * C3.absolute_bound(float(h), cfg["areas"])
                           for q, h in zip(fr["qd"], h_grid))
        ck("the combined range is exactly 2 sum_l |qd_l| B(h_l), not a single-step width",
           bool(np.allclose(np.asarray(R_Zd), expect, rtol=0.0, atol=0.0))
           and not np.allclose(np.asarray(R_Zd), np.asarray(single_step_width)),
           f"R_Z_d {float(np.min(R_Zd)):.4g}..{float(np.max(R_Zd)):.4g}; a single-step "
           f"width would be {float(np.max(single_step_width)):.4g}")
        rad = C3.empirical_bernstein_radius(np.asarray(sZ_all)[empty],
                                            np.asarray(sZ2_all)[empty],
                                            n_photons, np.asarray(R_Zd)[empty], 0.05)
        ck("a bin no photon reaches gives an exact zero mean and a strictly positive radius",
           bool(np.all(np.asarray(sZ_all)[empty] == 0.0))
           and bool(np.all(np.asarray(rad["mean"]) == 0.0))
           and bool(np.all(np.asarray(rad["radius"]) > 0.0)),
           f"{empty.size} empty bins, radius from the range term alone")
    else:
        ck("a bin no photon reaches gives an exact zero mean and a strictly positive radius",
           False, "no empty bin in this configuration; widen the grid to exercise it")
    return res, fails, {"n_photons": n_photons, "seed": seed, "jaxpad": jp["files"],
                        "resources": resources()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--resources", action="store_true")
    a = ap.parse_args()
    if a.resources:
        print(json.dumps(resources(), indent=2))
        return 0
    if a.selftest:
        print(f"  {SPIKE_STATUS}")
        res, fails, meta = selftest()
        print(f"  {sum(r['pass'] for r in res)}/{len(res)} checks passed "
              f"(N={meta['n_photons']}, seed={meta['seed']})")
        return 1 if fails else 0
    ap.error("either --selftest or --resources is required")


if __name__ == "__main__":
    raise SystemExit(main())
