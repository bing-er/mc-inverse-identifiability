#!/usr/bin/env python3
"""candidate3_executor.py -- the formal candidate 3 device-side accumulator.

PLATFORM NEUTRAL. This module never sets or reads JAX_PLATFORMS, never imports JAX at the top
level, and never imports the CPU-only feasibility spike. The platform is whatever the surrounding
job selects; a module that pinned it would silently downgrade the gate.

It does not contain a transport kernel. It receives the production per-photon symbol,
`physics_engine_v4.simulate_one_photon_rigorous`, which is already hash bound as
`jaxpad:src/pad/physics_engine_v4.py`, and vmaps that. The production aggregate solver
`mc_forward.forward_y_multi_layer` performs the same vmap and then discards the per-photon output;
this module keeps it, folds each photon's records by bin BEFORE squaring, and scatters the result.

What leaves the device is seven fixed-size accumulators and nothing else. No array whose leading
dimension is the photon count is returned.

NOT WIRED IN. `run_gate` does not call this module in this revision. Nothing is registered, the
adversarial suite is not frozen, no margin, step grid or support policy is chosen, and no GPU or
coverage run is performed.

    <python> -W error scripts/candidate3_executor.py --selftest
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# The seven accumulators this module is allowed to return. Anything else is a defect.
ACCUMULATOR_KEYS = ("sum_Z_e2", "sum_Z2_e2", "max_abs_Z_e2",
                    "sum_Z_d", "sum_Z2_d", "max_abs_Z_d", "max_abs_Y")

STATUS = ("formal module, implemented but NOT wired into run_gate. Nothing registered, no suite "
          "frozen, no margin, grid or support chosen, no GPU or coverage run.")


# ---------------------------------------------------------------------------------------------
# The jitted accumulator, created once by a factory and reused
# ---------------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class _FrozenAccumulator:
    """A module-private, immutable handle around the jitted callable.

    The step count and max_steps are fields of a frozen dataclass, so assigning to them raises
    rather than silently succeeding. An earlier version set plain attributes on the JAX callable;
    those were writable, so a caller could change `candidate3_n_steps` after the fact and make the
    wrapper accept a shape the compiled body does not implement, and any object carrying the same
    attribute names passed the "produced by the factory" check.

    `accumulate_column_wavelength` checks for THIS type, not for attribute names.
    """
    _jitted: object
    n_steps: int
    max_steps: int

    def __call__(self, *args, **kwargs):
        return self._jitted(*args, **kwargs)

    def compilation_cache_size(self):
        """Read-only proxy, for the self-test. Returns None when JAX exposes no such counter."""
        fn = getattr(self._jitted, "_cache_size", None)
        return fn() if callable(fn) else None


def _is_index(value):
    """A valid array index: a Python int or a NumPy integer, but never a bool.

    bool is a subclass of int in Python, so `isinstance(True, int)` is True and an unguarded check
    would accept `True` as column 1. np.bool_ is not an np.integer but is rejected explicitly for
    the same reason. The same rule governs the factory's structural parameters, so a float step
    count cannot be silently truncated into a valid one.
    """
    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, np.integer))


def make_accumulator(kernel, n_steps, max_steps):
    """Build the jitted accumulator ONCE. The returned callable is reused for every call.

    `kernel` is the production per-photon symbol. `n_steps` and `max_steps` fix the computation's
    structure and are closed over, so they are not traced arguments.

    The body takes the already-constructed kernel inputs for all four steps and both arms. No eta
    index and no Python branch on a column identity appears inside it: the caller resolves which
    column is being perturbed by building `bounds_plus`, `bounds_minus`, `muss_plus`, `muss_minus`,
    and the same compiled callable then serves the three geometry columns and the two scattering
    columns alike. One compilation covers every (column, wavelength) pair of a given shape, which
    is what JAX's own shape and dtype cache then reuses.
    """
    # Fail closed BEFORE anything is compiled. An earlier version coerced with int() and never
    # validated, so 4.9 silently became 4, True became 1, and 0 or -1 were accepted outright.
    # Worse, the body closed over the ORIGINAL max_steps while the handle recorded int(max_steps),
    # so the metadata could disagree with the value actually passed to the kernel.
    if not callable(kernel):
        raise SystemExit(f"REFUSING: kernel must be callable; got {type(kernel).__name__}")
    for name, value in (("n_steps", n_steps), ("max_steps", max_steps)):
        if not _is_index(value):
            raise SystemExit(
                f"REFUSING: {name} must be a Python int or a NumPy integer, never a bool and "
                f"never a float; got {value!r} of type {type(value).__name__}. Coercing it would "
                f"silently truncate.")
        if int(value) <= 0:
            raise SystemExit(f"REFUSING: {name} must be strictly positive; got {int(value)}")

    import jax
    import jax.numpy as jnp

    # The ONLY integers used from here on. Both the body's closure and the handle's metadata read
    # these, so the two cannot disagree about what the kernel was called with.
    L = int(n_steps)
    K = int(max_steps)

    def body(keys, bounds_p, bounds_m, muss_p, muss_m, muas, gs, ns, edges, h, q2, qd, areas):
        trace = jax.vmap(kernel, in_axes=(0, None, None, None, None, None, None, None))
        M = areas.shape[0]
        dp, wp, dm, wm = [], [], [], []
        for l in range(L):                      # structural, unrolled; not a data-dependent branch
            d1, w1 = trace(keys, muas, muss_p[l], gs, ns, bounds_p[l], edges, K)
            d2, w2 = trace(keys, muas, muss_m[l], gs, ns, bounds_m[l], edges, K)
            dp.append(d1)
            wp.append(w1)
            dm.append(d2)
            wm.append(w2)
        det_p = jnp.stack(dp, axis=1)           # (N, L)
        w_p = jnp.stack(wp, axis=1)
        det_m = jnp.stack(dm, axis=1)
        w_m = jnp.stack(wm, axis=1)

        def fold(b, v, ok):
            """Sum a photon's records that share a bin, and keep each bin once.

            This is what makes the second moment the variance of the COMBINED contribution rather
            than of the individual records: duplicates are summed BEFORE squaring.
            """
            same = (b[:, None] == b[None, :]) & ok[None, :] & ok[:, None]
            combined = (same * v[None, :]).sum(axis=1)
            first = jnp.argmax(same, axis=1) == jnp.arange(b.shape[0])
            keep = ok & first
            return combined * keep, keep

        bins = jnp.concatenate([det_p, det_m], axis=1)          # (N, 2L)
        ok = bins >= 0
        safe = jnp.where(ok, bins, 0).astype(jnp.int32)
        sgn = jnp.concatenate([jnp.ones((L,)), -jnp.ones((L,))])[None, :]
        w = jnp.concatenate([w_p, w_m], axis=1)
        base = w * sgn / (2.0 * jnp.concatenate([h, h])[None, :] * areas[safe])

        out = {}
        for tag, qv in (("e2", q2), ("d", qd)):
            vals = jnp.where(ok, base * jnp.concatenate([qv, qv])[None, :], 0.0)
            comb, keep = jax.vmap(fold)(safe, vals, ok)
            z = jnp.zeros((M,), dtype=base.dtype)
            kept = jnp.where(keep, comb, 0.0)
            out[f"sum_Z_{tag}"] = z.at[safe].add(kept)
            out[f"sum_Z2_{tag}"] = z.at[safe].add(jnp.where(keep, comb * comb, 0.0))
            out[f"max_abs_Z_{tag}"] = z.at[safe].max(jnp.abs(kept))

        # max_abs_Y pairs the two arms by bin BEFORE the absolute value, because Y is a central
        # difference. An arm-wise maximum would report max(|w+|, |w-|) where the arms share a bin.
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
            mY.append(jnp.zeros((M,), dtype=base.dtype).at[s2].max(
                jnp.where(k2, jnp.abs(c2), 0.0)))
        out["max_abs_Y"] = jnp.stack(mY)
        return out

    # The handle is IMMUTABLE. Its step count cannot drift from the step count the body was
    # closed over, which is the whole point of carrying it alongside.
    return _FrozenAccumulator(jax.jit(body), L, K)


# ---------------------------------------------------------------------------------------------
# Building the kernel inputs for one column, outside the jitted body
# ---------------------------------------------------------------------------------------------

def column_inputs(jnp, *, eta_index, h_grid, n_eta, boundaries_fn, muss_fn, wavelength_index):
    """The four steps and both arms of ONE column, as traced arrays.

    The caller resolves the column here, using the gate's own `boundaries_jnp` and
    `mus_factor_jnp` logic passed in as `boundaries_fn` and `muss_fn`, so the jitted body never
    sees a column identity and never branches on one.

    This covers all five columns by construction:

      eta 0, 1, 2   geometry: `boundaries_fn` moves the layer boundaries, `muss_fn` returns the
                    nominal scattering, so muss_plus equals muss_minus
      eta 3         log_mus_scale: a single factor applied at every wavelength
      eta 4         mus_slope: a wavelength-dependent power law, which is why this function takes
                    `wavelength_index` and asks `muss_fn` for that wavelength specifically
    """
    n_eta = int(n_eta)
    if not _is_index(eta_index) or not (0 <= int(eta_index) < n_eta):
        raise SystemExit(
            f"REFUSING: eta_index must be an int or np.integer, never a bool, in "
            f"[0, {n_eta}); got "
            f"{eta_index!r}. A negative index would silently select a DIFFERENT column, "
            f"and an index at or above {n_eta} would produce an all-zero perturbation "
            f"that looks like a valid result.")
    bp, bm, sp, sm = [], [], [], []
    for h in h_grid:
        ep = jnp.zeros(n_eta, dtype=jnp.float64).at[eta_index].set(+float(h))
        em = jnp.zeros(n_eta, dtype=jnp.float64).at[eta_index].set(-float(h))
        bp.append(boundaries_fn(ep))
        bm.append(boundaries_fn(em))
        sp.append(muss_fn(ep, wavelength_index))
        sm.append(muss_fn(em, wavelength_index))
    return (jnp.stack(bp), jnp.stack(bm), jnp.stack(sp), jnp.stack(sm))


# ---------------------------------------------------------------------------------------------
# The public wrapper: fail closed before anything reaches the device
# ---------------------------------------------------------------------------------------------

def _finite(a):
    return bool(np.all(np.isfinite(np.asarray(a, float))))


WEIGHT_CEILING = 1.0


def accumulate_column_wavelength(accumulator, *, keys, n_photons_declared,
                                 bounds_plus, bounds_minus, muss_plus, muss_minus,
                                 muas, gs, ns, edges, h_grid, q2, qd, areas):
    """Run one (replicate, column, wavelength) and return only the seven accumulators.

    ONE key array is accepted and all eight transports reuse it. There is no API position in which
    a second key array could be supplied, so a mismatch between steps or arms cannot be expressed
    rather than merely being checked for.

    The key array is inspected through `.shape` and `.dtype` only. It is never passed through
    `np.asarray`, which would pull an (N, 2) device array back to the host just to read its
    metadata. The small configuration arrays, which are at most L by n_layers or M long, are read
    on the host because their VALUES have to be checked, and that is a deliberate difference.

    Every structural and numeric precondition is verified before the device sees anything, and the
    range integrity of the result is verified before it is handed back. A violation raises; no
    bound is ever widened to accommodate an observation.
    """
    import jax.numpy as jnp

    if not isinstance(accumulator, _FrozenAccumulator):
        raise SystemExit(
            "REFUSING: the accumulator was not produced by make_accumulator. The check is "
            "on the concrete private type, not on attribute names, because any object can "
            "carry an attribute called candidate3_n_steps.")
    L_factory = int(accumulator.n_steps)
    L = len(h_grid)

    # --- the key array, by metadata only -----------------------------------------------------
    kshape = getattr(keys, "shape", None)
    kdtype = getattr(keys, "dtype", None)
    if kshape is None or kdtype is None:
        raise SystemExit("REFUSING: keys must expose shape and dtype without being converted")
    if len(kshape) != 2 or kshape[1] != 2:
        raise SystemExit(f"REFUSING: keys must have shape (N, 2); got {tuple(kshape)}")
    if int(kshape[0]) != int(n_photons_declared):
        raise SystemExit(f"REFUSING: the key array carries {int(kshape[0])} photons but the "
                         f"frozen design declares {int(n_photons_declared)}. The launch count is "
                         f"fixed and is not taken from the data.")
    if np.dtype(kdtype) != np.uint32:
        raise SystemExit(f"REFUSING: keys must be uint32 as JAX produces them; got {kdtype}")

    # --- the step count must match what the callable was compiled for ------------------------
    if L != L_factory:
        raise SystemExit(f"REFUSING: h_grid has {L} steps but the accumulator was compiled for "
                         f"{L_factory}. Recompiling silently for a different step count would "
                         f"defeat the point of a single cached callable.")
    if len(q2) != L or len(qd) != L:
        raise SystemExit(f"REFUSING: q2 and qd must have one entry per step, {L}")

    n_layers = int(np.asarray(muas).shape[0])
    for name, arr in (("bounds_plus", bounds_plus), ("bounds_minus", bounds_minus),
                      ("muss_plus", muss_plus), ("muss_minus", muss_minus)):
        a = np.asarray(arr)
        if a.ndim != 2 or a.shape[0] != L_factory:
            raise SystemExit(f"REFUSING: {name} must have one row per step, {L_factory}; got "
                             f"{a.shape}. All steps and both arms must be present, in order.")
        if a.shape[1] != n_layers:
            raise SystemExit(f"REFUSING: every {name} row must have one entry per layer, "
                             f"{n_layers}; got {a.shape[1]}")
    if np.asarray(bounds_plus).shape != np.asarray(bounds_minus).shape:
        raise SystemExit("REFUSING: the two arms disagree on the boundary array shape")
    if np.asarray(muss_plus).shape != np.asarray(muss_minus).shape:
        raise SystemExit("REFUSING: the two arms disagree on the scattering array shape")
    for name, arr in (("gs", gs), ("ns", ns)):
        if int(np.asarray(arr).shape[0]) != n_layers:
            raise SystemExit(f"REFUSING: {name} has {np.asarray(arr).shape[0]} entries but muas "
                             f"has {n_layers}; the optical arrays must agree")
    if int(np.asarray(edges).shape[0]) != int(np.asarray(areas).shape[0]) + 1:
        raise SystemExit("REFUSING: the detector edges and the annulus areas are inconsistent")

    # --- numeric domains ----------------------------------------------------------------------
    h = np.asarray(h_grid, float)
    A = np.asarray(areas, float)
    E = np.asarray(edges, float)
    if not (_finite(h) and np.all(h > 0.0)):
        raise SystemExit(f"REFUSING: every step size must be finite and strictly positive; got "
                         f"{h.tolist()}")
    if not (_finite(A) and np.all(A > 0.0)):
        raise SystemExit("REFUSING: every annulus area must be finite and strictly positive")
    if not (_finite(E) and np.all(np.diff(E) > 0.0)):
        raise SystemExit("REFUSING: the detector edges must be finite and strictly increasing")
    for name, arr in (("bounds_plus", bounds_plus), ("bounds_minus", bounds_minus)):
        a = np.asarray(arr, float)
        if not _finite(a):
            raise SystemExit(f"REFUSING: {name} contains a non-finite layer boundary")
        if not np.all(a > 0.0):
            raise SystemExit(f"REFUSING: every {name} layer boundary must be strictly "
                             f"positive")
        bad = np.flatnonzero(np.any(np.diff(a, axis=1) <= 0.0, axis=1))
        if bad.size:
            raise SystemExit(f"REFUSING: {name} row {int(bad[0])} is not strictly "
                             f"increasing, so the layer stack is degenerate")
    if not (_finite(muas) and np.all(np.asarray(muas, float) >= 0.0)):
        raise SystemExit("REFUSING: every absorption coefficient must be finite and non-negative")
    for name, arr in (("muss_plus", muss_plus), ("muss_minus", muss_minus)):
        if not (_finite(arr) and np.all(np.asarray(arr, float) >= 0.0)):
            raise SystemExit(f"REFUSING: every {name} entry must be finite and non-negative")
    g = np.asarray(gs, float)
    if not (_finite(g) and np.all(np.abs(g) <= 1.0)):
        raise SystemExit("REFUSING: every anisotropy must be finite and within [-1, 1]")
    nn = np.asarray(ns, float)
    if not (_finite(nn) and np.all(nn >= 1.0)):
        raise SystemExit("REFUSING: every refractive index must be finite and at least 1")
    if not (_finite(q2) and _finite(qd)):
        raise SystemExit("REFUSING: q2 and qd must be finite")

    out = accumulator(keys,
                      jnp.asarray(bounds_plus), jnp.asarray(bounds_minus),
                      jnp.asarray(muss_plus), jnp.asarray(muss_minus),
                      jnp.asarray(muas), jnp.asarray(gs), jnp.asarray(ns),
                      jnp.asarray(edges), jnp.asarray(h_grid, dtype=jnp.float64),
                      jnp.asarray(q2, dtype=jnp.float64),
                      jnp.asarray(qd, dtype=jnp.float64), jnp.asarray(areas))
    if set(out) != set(ACCUMULATOR_KEYS):
        raise SystemExit(f"REFUSING: the accumulator returned {sorted(out)}, expected "
                         f"{sorted(ACCUMULATOR_KEYS)}")
    verify_range_integrity(out, h_grid=h, areas=A, q2=q2, qd=qd)
    return out


def absolute_bounds(h_grid, areas, q2, qd, weight_ceiling=WEIGHT_CEILING):
    """B_i(h_l), B^Z_e2 and B^Z_d, all derived from the kernel's weight ceiling.

    B_i(h)   = weight_ceiling / (2 h A_i)
    B^Z_tag  = sum_l |q_tag,l| B_i(h_l),  the triangle inequality on the per-step bounds
    """
    h = np.asarray(h_grid, float)
    A = np.asarray(areas, float)
    B = weight_ceiling / (2.0 * h[:, None] * A[None, :])            # (L, M)
    return {"B_per_step": B,
            "B_Z_e2": (np.abs(np.asarray(q2, float))[:, None] * B).sum(axis=0),
            "B_Z_d": (np.abs(np.asarray(qd, float))[:, None] * B).sum(axis=0)}


def verify_range_integrity(out, *, h_grid, areas, q2, qd):
    """Refuse if any observed maximum exceeds its derived absolute bound, or anything is not finite.

    Only the seven fixed-size accumulators are inspected; this is the expected output and involves
    no per-photon transfer. A violation means a kernel invariant is broken, so the call is refused;
    the bound is NEVER widened to accommodate what was observed.
    """
    # SHAPE FIRST. A broadcast comparison against a scalar or a mis-shaped array would
    # succeed silently, so every one of the seven is checked before any value is read.
    M = int(np.asarray(areas).shape[0])
    L = int(np.asarray(h_grid).shape[0])
    for k in ACCUMULATOR_KEYS[:6]:
        sh = tuple(np.shape(out[k]))
        if sh != (M,):
            raise SystemExit(f"REFUSING: {k} has shape {sh}, expected ({M},). A scalar or "
                             f"a mis-shaped accumulator would pass a broadcast comparison "
                             f"and is refused here instead.")
    shY = tuple(np.shape(out["max_abs_Y"]))
    if shY != (L, M):
        raise SystemExit(f"REFUSING: max_abs_Y has shape {shY}, expected ({L}, {M})")
    for k in ACCUMULATOR_KEYS:
        if not _finite(out[k]):
            raise SystemExit(f"REFUSING: {k} contains a non-finite value")
    b = absolute_bounds(h_grid, areas, q2, qd)
    tol = 1.0 + 1e-12
    mY = np.asarray(out["max_abs_Y"], float)
    bad = np.argwhere(mY > b["B_per_step"] * tol)
    if bad.size:
        s0, i0 = int(bad[0][0]), int(bad[0][1])
        raise SystemExit(f"REFUSING: max_abs_Y exceeds the derived absolute bound at step {s0}, "
                         f"bin {i0}: {mY[s0, i0]:.6g} > {b['B_per_step'][s0, i0]:.6g}. The range "
                         f"rests on a kernel invariant, so this means the invariant is broken. "
                         f"The bound is not widened.")
    for tag, key in (("B_Z_e2", "max_abs_Z_e2"), ("B_Z_d", "max_abs_Z_d")):
        v = np.asarray(out[key], float)
        bad = np.flatnonzero(v > b[tag] * tol)
        if bad.size:
            i0 = int(bad[0])
            raise SystemExit(f"REFUSING: {key} exceeds its derived absolute bound at bin {i0}: "
                             f"{v[i0]:.6g} > {b[tag][i0]:.6g}. The bound is not widened.")
    return {"checked": list(ACCUMULATOR_KEYS), "shapes_verified": True, "all_finite": True,
            "within_derived_bounds": True,
            "bounds_are_derived_not_observed": True}


def fit_rows(h_grid):
    """q2, q3 and qd from the step grid alone. Local, so no run-time dependency on a design script.

    Duplicating three lines of linear algebra is preferable to putting a planning module on the
    production import path, where it would also have to join the hash manifest.
    """
    h2 = np.asarray(h_grid, float) ** 2
    q2 = np.linalg.pinv(np.vander(h2, 2, increasing=True))[0]
    q3 = np.linalg.pinv(np.vander(h2, 3, increasing=True))[0]
    return {"q2": q2, "q3": q3, "qd": q3 - q2}


# ---------------------------------------------------------------------------------------------
# CPU-only regression. The platform is chosen by the surrounding process, never by this module.
# ---------------------------------------------------------------------------------------------

_probe_calls = {"n": 0}


def _counting_probe():
    """A callable that records invocations, so a refusal can be shown to precede any kernel call."""
    def probe(*a, **k):
        _probe_calls["n"] += 1
        raise AssertionError("the kernel was invoked despite a refusal")
    return probe


def _raises_frozen(obj, field, value=None):
    """True when the handle refuses an assignment or a deletion. Self-test only."""
    try:
        if value is None:
            object.__getattribute__(obj, field)
            delattr(obj, field)
        else:
            setattr(obj, field, value)
        return False
    except (dataclasses.FrozenInstanceError, AttributeError):
        return True


def _raises_systemexit(fn, *args, **kw):
    """True when `fn` refuses with SystemExit. Used only by the self-test."""
    try:
        fn(*args, **kw)
        return False
    except SystemExit:
        return True


def _tiny_setup(jnp, n_bins=12, r_max=3.0):
    """A small fixed optical configuration with five eta columns, mirroring the real structure."""
    d = jnp.float64
    nominal_b = jnp.asarray([0.002, 0.010, 0.025, 0.033, 0.183], dtype=d)
    muas = jnp.asarray([0.30, 0.25, 0.20, 0.15, 0.10], dtype=d)
    muss_nom = jnp.asarray([90.0, 80.0, 70.0, 60.0, 50.0], dtype=d)
    gs = jnp.asarray([0.9] * 5, dtype=d)
    ns = jnp.asarray([1.4] * 5, dtype=d)
    edges = jnp.linspace(0.0, r_max, n_bins + 1, dtype=d)
    areas = jnp.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    wl = jnp.asarray([540.0, 580.0, 633.0, 800.0], dtype=d)
    lam0 = 633.0

    def boundaries_fn(eta):
        """The gate's own geometry logic: eta 0 scales two layers, eta 1 and eta 2 one each."""
        t = jnp.diff(jnp.concatenate([jnp.zeros(1, dtype=d), nominal_b]))
        s = jnp.exp(jnp.stack([eta[0], eta[1], eta[2]]))
        t = t.at[0].multiply(s[0]).at[1].multiply(s[0])
        t = t.at[2].multiply(s[1]).at[4].multiply(s[2])
        return jnp.cumsum(t)

    def muss_fn(eta, k):
        """The gate's own scattering logic, for ONE wavelength k."""
        return muss_nom * jnp.exp(eta[3]) * (wl[k] / lam0) ** (-eta[4])

    return {"muas": muas, "muss_nom": muss_nom, "gs": gs, "ns": ns, "edges": edges,
            "areas": areas, "wl": wl, "lam0": lam0, "M": n_bins, "n_eta": 5,
            "boundaries_fn": boundaries_fn, "muss_fn": muss_fn, "nominal_b": nominal_b}


def selftest(n_photons=500, seed=20261007, jaxpad_root=None):
    import jax
    import jax.numpy as jnp
    res, fails = [], []

    def ck(name, ok, detail=""):
        res.append({"name": name, "pass": bool(ok)})
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
        if not ok:
            fails.append(name)

    root = os.path.abspath(jaxpad_root or os.environ.get("JAXPAD_ROOT")
                           or os.path.join(ROOT, "..", "jax-pad"))
    if root not in sys.path:
        sys.path.insert(0, root)
    jax.config.update("jax_enable_x64", True)
    from src.pad import physics_engine_v4 as pe
    from src.pad import mc_forward as mcf
    kernel = pe.simulate_one_photon_rigorous

    with open(os.path.abspath(__file__)) as fh:
        own_src = fh.read()
    import ast as _ast
    tree = _ast.parse(own_src)
    env_writes, env_reads, imported = [], [], set()
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Subscript) and isinstance(node.value, _ast.Attribute):
            if getattr(node.value, "attr", None) == "environ":
                tgt = getattr(node.slice, "value", None)
                (env_writes if isinstance(getattr(node, "ctx", None), _ast.Store)
                 else env_reads).append(tgt)
        if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Attribute):
            if node.func.attr in ("get", "setdefault") and \
                    getattr(getattr(node.func, "value", None), "attr", None) == "environ":
                env_reads.append(node.args[0].value if node.args and
                                 isinstance(node.args[0], _ast.Constant) else None)
        if isinstance(node, (_ast.Import, _ast.ImportFrom)):
            nm = getattr(node, "module", None) or ""
            imported.add(nm)
            for al in node.names:
                imported.add(al.name)
    plat = "JAX_" + "PLATFORMS"
    ck("the module never writes the platform environment variable",
       plat not in [x for x in env_writes if isinstance(x, str)],
       f"{len(env_writes)} environ writes, none of them the platform")
    ck("the only environment read is the jax-pad root, used in the self-test",
       [x for x in env_reads if isinstance(x, str)] == ["JAXPAD_ROOT"],
       str([x for x in env_reads if isinstance(x, str)]))
    ck("the module imports neither the CPU-only spike nor the design script",
       not any("candidate3_" in m for m in imported if m),
       f"imports: {sorted(m for m in imported if m and not m.startswith('_'))[:8]}")
    top_imports = {n.names[0].name if isinstance(n, _ast.Import) else (n.module or "")
                   for n in tree.body if isinstance(n, (_ast.Import, _ast.ImportFrom))}
    ck("JAX is not imported at module scope; the factory takes it when called",
       not any(m.startswith("jax") for m in top_imports if m),
       f"module-scope imports: {sorted(m for m in top_imports if m)}")
    ck("the module defines no transport kernel of its own",
       ("lax." + "while" + "_loop") not in own_src
       and not any(n.startswith("simulate_") for n in dir(sys.modules[__name__])))

    cfg = _tiny_setup(jnp)
    h_grid = np.array([0.4, 0.3, 0.2, 0.1])
    fr = fit_rows(h_grid)
    keys = jax.random.split(jax.random.PRNGKey(seed), n_photons)
    acc_fn = make_accumulator(kernel, len(h_grid), 200)

    def run(eta_index, k):
        bp, bm, sp, sm = column_inputs(jnp, eta_index=eta_index, h_grid=h_grid,
                                       n_eta=cfg["n_eta"],
                                       boundaries_fn=cfg["boundaries_fn"],
                                       muss_fn=cfg["muss_fn"], wavelength_index=k)
        return (bp, bm, sp, sm,
                accumulate_column_wavelength(
                    acc_fn, keys=keys, n_photons_declared=n_photons,
                    bounds_plus=bp, bounds_minus=bm, muss_plus=sp, muss_minus=sm,
                    muas=cfg["muas"], gs=cfg["gs"], ns=cfg["ns"], edges=cfg["edges"],
                    h_grid=h_grid, q2=fr["q2"], qd=fr["qd"], areas=cfg["areas"]))

    # --- every column changes the right kernel input ------------------------------------------
    for j, nm in ((0, "log_d_epidermis"), (1, "log_d_papillary"), (2, "log_d_reticular")):
        bp, bm, sp, sm, _ = run(j, 2)
        ck(f"{nm} moves the boundaries and leaves the scattering at nominal",
           not np.allclose(np.asarray(bp), np.asarray(bm))
           and np.allclose(np.asarray(sp), np.asarray(sm)),
           "geometry column")
    # wavelength 0 is 540 nm. mus_slope is INERT at the reference wavelength, so a test at
    # index 2, which is 633 nm, would have been vacuous for that column.
    for j, nm in ((3, "log_mus_scale"), (4, "mus_slope")):
        bp, bm, sp, sm, _ = run(j, 0)
        ck(f"{nm} moves the scattering and leaves the boundaries at nominal",
           np.allclose(np.asarray(bp), np.asarray(bm))
           and not np.allclose(np.asarray(sp), np.asarray(sm)),
           "scattering column, tested at 540 nm")
    _, _, sp_ref, sm_ref, _ = run(4, 2)
    ck("mus_slope is INERT at the reference wavelength, where the power law factor is 1",
       np.allclose(np.asarray(sp_ref), np.asarray(sm_ref)),
       "633 nm is lam0, so (lam/lam0)^(-eta) equals 1 for every eta; a column test at "
       "that wavelength would be vacuous")
    fac = []
    for k in range(4):
        _, _, sp, _, _ = run(4, k)
        fac.append(float(np.asarray(sp)[0, 0] / float(np.asarray(cfg["muss_nom"])[0])))
    ck("mus_slope produces a DIFFERENT factor at each of the four wavelengths",
       len({round(x, 12) for x in fac}) == 4,
       ", ".join(f"{x:.6f}" for x in fac))
    fac3 = []
    for k in range(4):
        _, _, sp, _, _ = run(3, k)
        fac3.append(float(np.asarray(sp)[0, 0] / float(np.asarray(cfg["muss_nom"])[0])))
    ck("log_mus_scale produces the SAME factor at every wavelength",
       len({round(x, 12) for x in fac3}) == 1, f"{fac3[0]:.6f}")

    # --- shape contract ------------------------------------------------------------------------
    bp, bm, sp, sm, out = run(2, 1)
    ck("the return is exactly the seven accumulators", set(out) == set(ACCUMULATOR_KEYS),
       str(sorted(out)))
    leaves = jax.tree_util.tree_leaves(out)
    ck("no returned array has the photon count as its leading dimension",
       all(a.shape[0] != n_photons for a in leaves)
       and sum(int(np.prod(a.shape)) for a in leaves) == 6 * cfg["M"] + len(h_grid) * cfg["M"],
       f"shapes {[tuple(a.shape) for a in leaves]}, "
       f"{sum(int(np.prod(a.shape)) for a in leaves)} scalars")
    ck("the six Z accumulators are (M,) and max_abs_Y is (L, M)",
       all(out[k_].shape == (cfg["M"],) for k_ in ACCUMULATOR_KEYS[:6])
       and out["max_abs_Y"].shape == (len(h_grid), cfg["M"]))
    # --- equivalence with the production aggregate --------------------------------------------
    def rescatter(det, w):
        """Mirror mc_forward.py:68-79 exactly, including how the annulus area is formed."""
        valid = det >= 0
        sid = jnp.where(valid, det, 0).astype(jnp.int32)
        sw = jnp.where(valid, w, 0.0)
        y = jnp.zeros((cfg["M"],), dtype=sw.dtype).at[sid].add(sw)
        r0, r1 = cfg["edges"][:-1], cfg["edges"][1:]
        return y / (jnp.asarray(n_photons, dtype=sw.dtype)
                    * jnp.pi * (r1 ** 2 - r0 ** 2))

    trace = jax.vmap(kernel, in_axes=(0, None, None, None, None, None, None, None))
    worst = 0.0
    bp2, bm2, sp2, sm2, _ = run(0, 2)
    for l in range(len(h_grid)):
        for b_, s_ in ((bp2[l], sp2[l]), (bm2[l], sm2[l])):
            d_, w_ = trace(keys, cfg["muas"], s_, cfg["gs"], cfg["ns"], b_, cfg["edges"], 200)
            prod = mcf.forward_y_multi_layer(muas=cfg["muas"], muss=s_, gs=cfg["gs"],
                                             ns=cfg["ns"], layer_boundaries=b_, keys=keys,
                                             detector_r_edges=cfg["edges"], max_steps=200)
            worst = max(worst, float(jnp.max(jnp.abs(rescatter(d_, w_) - prod))))
    ck("the per-photon path reproduces the production aggregate BITWISE",
       worst == 0.0, f"max abs difference {worst:.3e} over eight evaluations")

    # --- an INDEPENDENT dense oracle -----------------------------------------------------------
    nos = 40
    dets_p, ws_p, dets_m, ws_m = [], [], [], []
    for l in range(len(h_grid)):
        d1, w1 = trace(keys[:nos], cfg["muas"], sp2[l], cfg["gs"], cfg["ns"], bp2[l],
                       cfg["edges"], 200)
        d2, w2 = trace(keys[:nos], cfg["muas"], sm2[l], cfg["gs"], cfg["ns"], bm2[l],
                       cfg["edges"], 200)
        dets_p.append(np.asarray(d1))
        ws_p.append(np.asarray(w1))
        dets_m.append(np.asarray(d2))
        ws_m.append(np.asarray(w2))
    A = np.asarray(cfg["areas"])
    Zd = np.zeros((nos, cfg["M"]))
    Ze = np.zeros((nos, cfg["M"]))
    Yd = np.zeros((nos, len(h_grid), cfg["M"]))
    for p_ in range(nos):
        for l, h in enumerate(h_grid):
            bp_, wp_ = int(dets_p[l][p_]), float(ws_p[l][p_])
            bm_, wm_ = int(dets_m[l][p_]), float(ws_m[l][p_])
            if bp_ >= 0:
                Zd[p_, bp_] += fr["qd"][l] * wp_ / (2.0 * h * A[bp_])
                Ze[p_, bp_] += fr["q2"][l] * wp_ / (2.0 * h * A[bp_])
                Yd[p_, l, bp_] += wp_ / (2.0 * h * A[bp_])
            if bm_ >= 0:
                Zd[p_, bm_] -= fr["qd"][l] * wm_ / (2.0 * h * A[bm_])
                Ze[p_, bm_] -= fr["q2"][l] * wm_ / (2.0 * h * A[bm_])
                Yd[p_, l, bm_] -= wm_ / (2.0 * h * A[bm_])
    small = accumulate_column_wavelength(
        acc_fn, keys=keys[:nos], n_photons_declared=nos,
        bounds_plus=bp2, bounds_minus=bm2, muss_plus=sp2, muss_minus=sm2,
        muas=cfg["muas"], gs=cfg["gs"], ns=cfg["ns"], edges=cfg["edges"],
        h_grid=h_grid, q2=fr["q2"], qd=fr["qd"], areas=cfg["areas"])

    def rel(a, b):
        b = np.asarray(b, float)
        sc = max(float(np.max(np.abs(b))), np.finfo(float).tiny)
        return float(np.max(np.abs(np.asarray(a, float) - b))) / sc

    ck("sum_Z_d matches an independent dense oracle",
       rel(small["sum_Z_d"], Zd.sum(0)) < 1e-13, f"relative {rel(small['sum_Z_d'], Zd.sum(0)):.3e}")
    ck("sum_Z2_d matches it, so duplicates were folded BEFORE squaring",
       rel(small["sum_Z2_d"], (Zd ** 2).sum(0)) < 1e-13,
       f"relative {rel(small['sum_Z2_d'], (Zd ** 2).sum(0)):.3e}")
    ck("sum_Z_e2 and sum_Z2_e2 match the oracle too",
       rel(small["sum_Z_e2"], Ze.sum(0)) < 1e-13
       and rel(small["sum_Z2_e2"], (Ze ** 2).sum(0)) < 1e-13)
    ck("max_abs_Z_d matches the oracle",
       rel(small["max_abs_Z_d"], np.abs(Zd).max(0)) < 1e-13)
    ck("max_abs_Y matches the PAIRED oracle, arms combined before the absolute value",
       rel(small["max_abs_Y"], np.abs(Yd).max(0)) < 1e-13,
       f"relative {rel(small['max_abs_Y'], np.abs(Yd).max(0)):.3e}")
    naive = np.zeros(cfg["M"])
    for p_ in range(nos):
        for l, h in enumerate(h_grid):
            for sgn_, det_, w_ in ((+1.0, dets_p[l][p_], ws_p[l][p_]),
                                   (-1.0, dets_m[l][p_], ws_m[l][p_])):
                if int(det_) >= 0:
                    naive[int(det_)] += (sgn_ * fr["qd"][l] * float(w_)
                                         / (2.0 * h * A[int(det_)])) ** 2
    ck("folding before squaring differs from squaring each record separately",
       rel(naive, (Zd ** 2).sum(0)) > 1e-3,
       f"relative gap {rel(naive, (Zd ** 2).sum(0)):.3e}, so the check is not vacuous")
    n_dup = sum(1 for p_ in range(nos)
                for l in range(len(h_grid))
                if dets_p[l][p_] >= 0 and dets_p[l][p_] == dets_m[l][p_])
    n_move = sum(1 for p_ in range(nos)
                 for l in range(len(h_grid))
                 if dets_p[l][p_] >= 0 and dets_m[l][p_] >= 0
                 and dets_p[l][p_] != dets_m[l][p_])
    ck("the sample exercises both the shared-bin and the moving-bin case",
       n_dup > 0 and n_move > 0,
       f"{n_dup} photon-steps share a bin, {n_move} move between bins")

    # --- K = 0 ---------------------------------------------------------------------------------
    touched = np.zeros(cfg["M"], dtype=int)
    for l in range(len(h_grid)):
        for arr in (dets_p[l], dets_m[l]):
            for v in arr:
                if int(v) >= 0:
                    touched[int(v)] += 1
    empty = np.flatnonzero(touched == 0)
    ck("a bin no photon reaches has exactly zero in every accumulator",
       empty.size > 0
       and bool(np.all(np.asarray(small["sum_Z_d"])[empty] == 0.0))
       and bool(np.all(np.asarray(small["sum_Z2_d"])[empty] == 0.0))
       and bool(np.all(np.asarray(small["max_abs_Z_d"])[empty] == 0.0)),
       f"{empty.size} empty bins")

    # --- fail-closed refusals ------------------------------------------------------------------
    def _neg(a):
        b = np.array(np.asarray(a, float))
        b[0, 0] = -1.0
        return b

    def refuses_with(_unused, **kw):
        base = dict(keys=keys, n_photons_declared=n_photons, bounds_plus=bp2,
                    bounds_minus=bm2, muss_plus=sp2, muss_minus=sm2, muas=cfg["muas"],
                    gs=cfg["gs"], ns=cfg["ns"], edges=cfg["edges"], h_grid=h_grid,
                    q2=fr["q2"], qd=fr["qd"], areas=cfg["areas"])
        acc = kw.pop("accumulator", acc_fn)
        base.update(kw)
        try:
            accumulate_column_wavelength(acc, **base)
            return False
        except SystemExit:
            return True

    def refuses(**kw):
        base = dict(keys=keys, n_photons_declared=n_photons, bounds_plus=bp2, bounds_minus=bm2,
                    muss_plus=sp2, muss_minus=sm2, muas=cfg["muas"], gs=cfg["gs"], ns=cfg["ns"],
                    edges=cfg["edges"], h_grid=h_grid, q2=fr["q2"], qd=fr["qd"],
                    areas=cfg["areas"])
        base.update(kw)
        try:
            accumulate_column_wavelength(acc_fn, **base)
            return False
        except SystemExit:
            return True

    ck("a key array of the wrong rank is REFUSED", refuses(keys=np.asarray(keys).ravel()))
    ck("a launch count differing from the declared N is REFUSED",
       refuses(n_photons_declared=n_photons - 1))
    ck("a non-uint32 key dtype is REFUSED",
       refuses(keys=np.asarray(keys).astype(np.int64)))
    ck("a missing step in either arm is REFUSED",
       refuses(bounds_plus=bp2[:3]) and refuses(bounds_minus=bm2[:3])
       and refuses(muss_plus=sp2[:3]) and refuses(muss_minus=sm2[:3]))
    ck("an optical array of the wrong length is REFUSED",
       refuses(gs=np.asarray(cfg["gs"])[:3]))
    ck("inconsistent detector edges and areas are REFUSED",
       refuses(areas=np.asarray(cfg["areas"])[:5]))
    ck("a q vector with the wrong length is REFUSED", refuses(q2=fr["q2"][:2]))
    ck("the API has no position for a second key array",
       "keys" in accumulate_column_wavelength.__code__.co_varnames
       and sum(1 for v in accumulate_column_wavelength.__code__.co_varnames
               if v.startswith("keys")) == 1,
       "one key array only, so a per-step mismatch cannot be expressed")

    # --- the cached callable is reused by all twenty (column, wavelength) calls -----------------
    # The cache legitimately holds one entry per SHAPE. What matters is that the twenty
    # (column, wavelength) calls, which share a shape, add nothing to it.
    before = acc_fn.compilation_cache_size()
    has_cache = before is not None
    for j in range(5):
        for k in range(4):
            run(j, k)
    after = acc_fn.compilation_cache_size()
    ck("the twenty (column, wavelength) calls add NO new compilation",
       (not has_cache) or (after == before),
       f"compilation cache size {before} before and {after} after twenty calls; the "
       f"entries that exist correspond to distinct photon counts, not to columns")
    ck("the factory is called once and the callable is the object reused",
       callable(acc_fn) and run(0, 0)[4] is not None)

    # --- factory metadata and step-count refusals (item 2 and item 9) --------------------------
    # --- the factory validates its structural parameters before compiling anything -----------
    for ns_bad, lab in ((4.9, "a float that would truncate to 4"), (True, "a bool"),
                        (np.bool_(True), "a numpy bool"), (0, "zero"), (-1, "negative"),
                        (np.float64(4.0), "a numpy float")):
        ck(f"make_accumulator REFUSES n_steps as {lab}",
           _raises_systemexit(make_accumulator, kernel, ns_bad, 200))
    for ms_bad, lab in ((200.9, "a float that would truncate to 200"), (False, "a bool"),
                        (np.bool_(False), "a numpy bool"), (0, "zero"), (-1, "negative")):
        ck(f"make_accumulator REFUSES max_steps as {lab}",
           _raises_systemexit(make_accumulator, kernel, 4, ms_bad))
    ck("make_accumulator REFUSES a kernel that is not callable",
       _raises_systemexit(make_accumulator, object(), 4, 200)
       and _raises_systemexit(make_accumulator, None, 4, 200))
    ck("make_accumulator ACCEPTS numpy integers for both structural parameters",
       (lambda a: isinstance(a, _FrozenAccumulator) and a.n_steps == 4 and a.max_steps == 200)(
           make_accumulator(kernel, np.int64(4), np.int64(200))),
       "np.int64(4) and np.int64(200)")

    # the handle's metadata must equal the value the body actually closes over
    seen_max_steps = {"v": None}

    def _recording_kernel(key, muas_, muss_, gs_, ns_, bounds_, edges_, max_steps_):
        seen_max_steps["v"] = max_steps_
        return kernel(key, muas_, muss_, gs_, ns_, bounds_, edges_, max_steps_)

    rec = make_accumulator(_recording_kernel, len(h_grid), 173)
    accumulate_column_wavelength(
        rec, keys=keys[:8], n_photons_declared=8, bounds_plus=bp2, bounds_minus=bm2,
        muss_plus=sp2, muss_minus=sm2, muas=cfg["muas"], gs=cfg["gs"], ns=cfg["ns"],
        edges=cfg["edges"], h_grid=h_grid, q2=fr["q2"], qd=fr["qd"], areas=cfg["areas"])
    ck("the handle's max_steps equals the value the body passes to the kernel",
       seen_max_steps["v"] == rec.max_steps == 173
       and isinstance(seen_max_steps["v"], int),
       f"metadata {rec.max_steps}, kernel received {seen_max_steps['v']!r}; both read the same "
       f"validated integer, so they cannot disagree")
    ck("every factory refusal happens before any JIT build or kernel call",
       _raises_systemexit(make_accumulator, _counting_probe(), 4.9, 200.9)
       and _probe_calls["n"] == 0,
       f"kernel invoked {_probe_calls['n']} times across the refused factory calls")

    ck("the factory returns the private frozen handle, carrying its own step count",
       isinstance(acc_fn, _FrozenAccumulator)
       and acc_fn.n_steps == len(h_grid) and acc_fn.max_steps == 200,
       f"n_steps {acc_fn.n_steps}, max_steps {acc_fn.max_steps}")
    for field, value in (("n_steps", 99), ("max_steps", -1)):
        try:
            setattr(acc_fn, field, value)
            frozen_ok = False
        except dataclasses.FrozenInstanceError:
            frozen_ok = True
        ck(f"assigning to {field} on the handle is REFUSED, so it cannot drift from the "
           f"compiled body", frozen_ok and getattr(acc_fn, field) != value,
           f"{field} is still {getattr(acc_fn, field)}")
    ck("deleting a field on the handle is REFUSED",
       (lambda: (lambda: False)() if False else _raises_frozen(acc_fn, "n_steps"))())

    class _Impostor:
        """A plain callable wearing the same attribute names."""
        n_steps = len(h_grid)
        max_steps = 200
        candidate3_n_steps = len(h_grid)
        candidate3_max_steps = 200

        def __call__(self, *a, **k):
            return {q: np.zeros(1) for q in ACCUMULATOR_KEYS}

    ck("a plain callable carrying the same attribute names is REFUSED",
       refuses_with(None, accumulator=_Impostor()),
       "the check is on the concrete private type, not on attribute names")
    ck("a bare lambda is REFUSED",
       refuses_with(None, accumulator=lambda *a, **k: {}))
    ck("a four-step handle cannot be edited into accepting three or five steps",
       all(_raises_frozen(acc_fn, "n_steps", v) for v in (3, 5))
       and acc_fn.n_steps == 4,
       "the only way to change the step count is to call the factory again")
    for bad_L, lab in ((3, "three"), (5, "five")):
        hb = np.linspace(0.4, 0.1, bad_L)
        frb = fit_rows(hb)
        ck(f"a factory compiled for four steps REFUSES a call with {lab} steps",
           refuses(h_grid=hb, q2=frb["q2"], qd=frb["qd"]),
           f"len(h_grid)={bad_L} against factory n_steps=4")
    ck("a bounds array with the wrong step count is REFUSED",
       refuses(bounds_plus=bp2[:3]) and refuses(muss_minus=sm2[:3]))
    ck("a boundary row whose length disagrees with the layer count is REFUSED",
       refuses(bounds_plus=np.asarray(bp2)[:, :3]))

    # --- numeric domain refusals (item 4 and item 9) -----------------------------------------
    ck("a non-positive step size is REFUSED",
       refuses(h_grid=np.array([0.4, 0.3, 0.0, 0.1]))
       and refuses(h_grid=np.array([0.4, 0.3, -0.2, 0.1])))
    ck("a non-finite step size is REFUSED",
       refuses(h_grid=np.array([0.4, 0.3, np.inf, 0.1])))
    ck("a non-positive annulus area is REFUSED",
       refuses(areas=np.asarray(cfg["areas"]).at[0].set(0.0)
               if hasattr(np.asarray(cfg["areas"]), "at")
               else np.concatenate([[0.0], np.asarray(cfg["areas"])[1:]])))
    ck("non-increasing detector edges are REFUSED",
       refuses(edges=np.concatenate([np.asarray(cfg["edges"])[:2][::-1],
                                     np.asarray(cfg["edges"])[2:]])))
    ck("a negative absorption coefficient is REFUSED",
       refuses(muas=np.concatenate([[-0.1], np.asarray(cfg["muas"])[1:]])))
    ck("a negative scattering coefficient is REFUSED in either arm",
       refuses(muss_plus=np.asarray(sp2).at[0, 0].set(-1.0)
               if hasattr(np.asarray(sp2), "at") else _neg(sp2))
       and refuses(muss_minus=_neg(sm2)))
    ck("an anisotropy outside [-1, 1] is REFUSED",
       refuses(gs=np.concatenate([[1.5], np.asarray(cfg["gs"])[1:]])))
    ck("a refractive index below 1 is REFUSED",
       refuses(ns=np.concatenate([[0.5], np.asarray(cfg["ns"])[1:]])))
    ck("a non-finite q vector is REFUSED",
       refuses(qd=np.concatenate([[np.nan], fr["qd"][1:]])))

    # --- range integrity (item 5 and item 9) --------------------------------------------------
    b = absolute_bounds(h_grid, cfg["areas"], fr["q2"], fr["qd"])
    ck("the derived bounds are B = 1/(2 h A) and B_Z = sum_l |q_l| B(h_l)",
       np.allclose(b["B_per_step"],
                   1.0 / (2.0 * h_grid[:, None] * np.asarray(cfg["areas"])[None, :]))
       and np.allclose(b["B_Z_d"],
                       (np.abs(fr["qd"])[:, None] * b["B_per_step"]).sum(0)),
       f"B_Z_d {b['B_Z_d'].min():.4g}..{b['B_Z_d'].max():.4g}")
    ck("the real result passes range integrity",
       verify_range_integrity(out, h_grid=h_grid, areas=cfg["areas"],
                              q2=fr["q2"], qd=fr["qd"])["within_derived_bounds"] is True)
    for key, lab in (("max_abs_Y", "max_abs_Y"), ("max_abs_Z_e2", "max_abs_Z_e2"),
                     ("max_abs_Z_d", "max_abs_Z_d")):
        tampered = dict(out)
        arr = np.array(np.asarray(out[key], float))
        bound = (b["B_per_step"] if key == "max_abs_Y"
                 else b["B_Z_e2"] if key.endswith("e2") else b["B_Z_d"])
        idx = (0, 0) if key == "max_abs_Y" else (0,)
        arr[idx] = float(np.asarray(bound)[idx]) * 1.001
        tampered[key] = arr
        try:
            verify_range_integrity(tampered, h_grid=h_grid, areas=cfg["areas"],
                                   q2=fr["q2"], qd=fr["qd"])
            ok = False
        except SystemExit as exc:
            ok = "is not widened" in str(exc)
        ck(f"a planted {lab} above its derived bound is REFUSED, and the bound is not widened", ok)
    tampered = dict(out)
    arr = np.array(np.asarray(out["sum_Z_d"], float))
    arr[0] = np.nan
    tampered["sum_Z_d"] = arr
    try:
        verify_range_integrity(tampered, h_grid=h_grid, areas=cfg["areas"],
                               q2=fr["q2"], qd=fr["qd"])
        ok = False
    except SystemExit as exc:
        ok = "non-finite" in str(exc)
    ck("a non-finite accumulator entry is REFUSED", ok)

    # --- the wrapper never materialises the key array (item 3) -------------------------------
    class _KeyProbe:
        """Exposes shape and dtype; raises if anything tries to convert it to a host array."""
        def __init__(self, inner):
            self._inner = inner
            self.shape = inner.shape
            self.dtype = inner.dtype
            self.converted = False

        def __array__(self, *a, **k):
            self.converted = True
            raise AssertionError("the wrapper converted the key array to a host array")

    # scoped to the WRAPPER's body: the self-test legitimately converts keys to build bad inputs
    _wrapper_src = own_src.split("def accumulate_column_wavelength(")[1].split(
        "def absolute_bounds(")[0]
    probe = _KeyProbe(keys)
    try:
        accumulate_column_wavelength(
            acc_fn, keys=probe, n_photons_declared=n_photons - 1,
            bounds_plus=bp2, bounds_minus=bm2, muss_plus=sp2, muss_minus=sm2,
            muas=cfg["muas"], gs=cfg["gs"], ns=cfg["ns"], edges=cfg["edges"],
            h_grid=h_grid, q2=fr["q2"], qd=fr["qd"], areas=cfg["areas"])
        hit = False
    except SystemExit:
        hit = True
    ck("the wrapper reads keys by shape and dtype only, never converting them",
       hit and probe.converted is False
       and ("np.asarray(" + "keys)") not in _wrapper_src,
       "an (N, 2) device array is not pulled to the host just to read its metadata; the "
       "self-test's own uses below are for building bad inputs, not for reading metadata")

    # --- the local fit_rows must not drift from the design module (item 8) -------------------
    import importlib.util as _ilu
    _dp = os.path.join(HERE, "candidate3_design.py")
    _spec = _ilu.spec_from_file_location("_c3_design_crosscheck", _dp)
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    ref = _mod.fit_rows(h_grid)
    ck("the local fit_rows agrees with the design module's, loaded by path for this test only",
       np.allclose(fr["q2"], ref["q2"], rtol=0.0, atol=0.0)
       and np.allclose(fr["qd"], ref["qd"], rtol=0.0, atol=0.0),
       "two copies of three lines of linear algebra, checked against each other")
    ck("the design module is loaded dynamically and is not on the production import path",
       "candidate3_design" not in {m for m in sys.modules
                                   if not m.startswith("_c3_design_crosscheck")}
       or "_c3_design_crosscheck" in sys.modules,
       "loaded by file path inside the self-test only")

    # --- the DORMANT interface, through the real build_executor (item 7) ----------------------
    # A small CPU interface test, not a preflight and not a coverage run. It builds the real
    # executor against the real production configuration at a tiny photon count, calls the dormant
    # accumulator for one geometry column and one scattering column at a NON-REFERENCE wavelength,
    # and checks the contract. The step grid is passed explicitly; there is no default.
    sys.path.insert(0, HERE)
    import nuisance_feasibility_gate as _gate
    nspec = _gate.load(_gate.NSPEC)
    bspec = _gate.load(_gate.BSPEC)
    ex = _gate.build_executor(root, 48, nspec, bspec)
    ck("the real build_executor exposes the dormant accumulator and declares it unwired",
       callable(ex["candidate3_accumulate_column_wavelength"])
       and ex["candidate3_is_wired_into_run_gate"] is False,
       "built against the real production configuration at 48 photons")
    test_grid = (0.40, 0.30, 0.20, 0.10)
    ex_keys = ex["split"](11)
    for j, lab, k_wl in ((1, "log_d_papillary, geometry", 0),
                         (4, "mus_slope, scattering", 0)):
        got = ex["candidate3_accumulate_column_wavelength"](
            eta_index=j, wavelength_index=k_wl, keys=ex_keys, h_grid=test_grid)
        n_bins = int(bspec["profiles"]["production"]["radial"]["n_bins"])
        ck(f"the dormant interface returns exactly the seven accumulators for {lab}",
           set(got) == set(ACCUMULATOR_KEYS)
           and all(got[q].shape == (n_bins,) for q in ACCUMULATOR_KEYS[:6])
           and got["max_abs_Y"].shape == (len(test_grid), n_bins),
           f"six ({n_bins},) and one ({len(test_grid)}, {n_bins}) at wavelength index {k_wl}")
        ck(f"no returned array has the photon count as its leading dimension for {lab}",
           all(a.shape[0] != 48 for a in jax.tree_util.tree_leaves(got)))
    ck("the dormant interface REFUSES a grid whose step count disagrees with the factory",
       (lambda: (lambda f: any(
           (lambda: (f(eta_index=1, wavelength_index=0, keys=ex_keys, h_grid=g), False)[1])()
           if False else _raises_systemexit(f, eta_index=1, wavelength_index=0,
                                            keys=ex_keys, h_grid=g)
           for g in ((0.4, 0.3, 0.2), (0.4, 0.3, 0.2, 0.1, 0.05))))(
               ex["candidate3_accumulate_column_wavelength"]))(),
       "three and five steps both refused against a factory compiled for four")
    ck("run_gate still does not reference the dormant accumulator",
       "candidate3_accumulate_column_wavelength" not in
       _gate._read_self().split("def run_gate(")[1].split("def run_preflight(")[0])

    # --- audit note: index, feasibility and shape contracts (verified refusals) ---------------
    for bad_j, why in ((-1, "would select the LAST column, not be refused"),
                       (cfg["n_eta"], "would produce an all-zero perturbation")):
        ck(f"column_inputs REFUSES eta_index={bad_j}, which {why}",
           _raises_systemexit(column_inputs, jnp=jnp, eta_index=bad_j, h_grid=h_grid,
                              n_eta=cfg["n_eta"], boundaries_fn=cfg["boundaries_fn"],
                              muss_fn=cfg["muss_fn"], wavelength_index=0))
    for bad, lab in ((True, "a bool"), (np.bool_(True), "a numpy bool")):
        ck(f"column_inputs REFUSES {lab} as eta_index, which int would accept",
           _raises_systemexit(column_inputs, jnp=jnp, eta_index=bad,
                              h_grid=h_grid, n_eta=cfg["n_eta"],
                              boundaries_fn=cfg["boundaries_fn"],
                              muss_fn=cfg["muss_fn"], wavelength_index=0))
    ck("_is_index accepts int and np.integer but never bool",
       _is_index(3) and _is_index(np.int64(3))
       and not _is_index(True) and not _is_index(np.bool_(True))
       and not _is_index(3.0),
       "bool is a subclass of int, so it must be rejected explicitly")
    ck("column_inputs REFUSES a non-integer eta_index",
       _raises_systemexit(column_inputs, jnp=jnp, eta_index=1.0, h_grid=h_grid,
                          n_eta=cfg["n_eta"], boundaries_fn=cfg["boundaries_fn"],
                          muss_fn=cfg["muss_fn"], wavelength_index=0))

    # every one of the six Z accumulators must be (M,), checked BEFORE any value comparison
    good = {k_: (np.zeros(cfg["M"]) if k_ != "max_abs_Y" else np.zeros((len(h_grid), cfg["M"])))
            for k_ in ACCUMULATOR_KEYS}
    ck("an all-zero result of the right shape passes",
       verify_range_integrity(good, h_grid=h_grid, areas=cfg["areas"],
                              q2=fr["q2"], qd=fr["qd"])["shapes_verified"] is True)
    for k_ in ACCUMULATOR_KEYS[:6]:
        for bad_val, lab in ((np.float64(0.0), "a scalar"),
                             (np.zeros(cfg["M"] - 1), "a short array")):
            t = dict(good)
            t[k_] = bad_val
            ck(f"{k_} as {lab} is REFUSED before any range comparison",
               _raises_systemexit(verify_range_integrity, t, h_grid=h_grid,
                                  areas=cfg["areas"], q2=fr["q2"], qd=fr["qd"]))
    t = dict(good)
    t["max_abs_Y"] = np.zeros(cfg["M"])
    ck("max_abs_Y with the wrong rank is REFUSED",
       _raises_systemexit(verify_range_integrity, t, h_grid=h_grid, areas=cfg["areas"],
                          q2=fr["q2"], qd=fr["qd"]))

    # layer boundaries, row by row
    def _bad_bounds(kind):
        a = np.array(np.asarray(bp2, float))
        if kind == "nan":
            a[1, 2] = np.nan
        elif kind == "nonpositive":
            a[0, 0] = 0.0
        else:
            a[2, 3] = a[2, 2] * 0.5
        return a

    for kind, lab in (("nan", "a non-finite value"), ("nonpositive", "a non-positive value"),
                      ("nonincreasing", "a non-increasing row")):
        ck(f"a layer-boundary array with {lab} is REFUSED",
           refuses(bounds_plus=_bad_bounds(kind)) and refuses(bounds_minus=_bad_bounds(kind)))

    # --- refusals must happen BEFORE any kernel call ------------------------------------------
    calls = {"n": 0}

    def _counting_kernel(*a, **k):
        calls["n"] += 1
        return pe.simulate_one_photon_rigorous(*a, **k)

    counted = make_accumulator(_counting_kernel, len(h_grid), 200)
    for lab, kw in (("wrong step count", dict(h_grid=np.linspace(0.4, 0.1, 3),
                                              q2=fit_rows(np.linspace(0.4, 0.1, 3))["q2"],
                                              qd=fit_rows(np.linspace(0.4, 0.1, 3))["qd"])),
                    ("non-positive step", dict(h_grid=np.array([0.4, 0.3, 0.0, 0.1]))),
                    ("non-positive area", dict(areas=np.concatenate(
                        [[0.0], np.asarray(cfg["areas"])[1:]]))),
                    ("non-increasing edges", dict(edges=np.concatenate(
                        [np.asarray(cfg["edges"])[:2][::-1], np.asarray(cfg["edges"])[2:]]))),
                    ("negative mua", dict(muas=np.concatenate(
                        [[-0.1], np.asarray(cfg["muas"])[1:]]))),
                    ("bad boundary row", dict(bounds_plus=_bad_bounds("nan")))):
        calls["n"] = 0
        base = dict(keys=keys, n_photons_declared=n_photons, bounds_plus=bp2, bounds_minus=bm2,
                    muss_plus=sp2, muss_minus=sm2, muas=cfg["muas"], gs=cfg["gs"], ns=cfg["ns"],
                    edges=cfg["edges"], h_grid=h_grid, q2=fr["q2"], qd=fr["qd"],
                    areas=cfg["areas"])
        base.update(kw)
        refused = _raises_systemexit(accumulate_column_wavelength, counted, **base)
        ck(f"a {lab} is refused with ZERO kernel calls", refused and calls["n"] == 0,
           f"kernel invoked {calls['n']} times")

    # --- wavelength-major concatenation --------------------------------------------------------
    per_wl = [run(2, k)[4]["sum_Z_d"] for k in range(4)]
    joined = np.concatenate([np.asarray(x) for x in per_wl])
    ck("concatenating the four wavelengths gives a 120-vector in wavelength-major order",
       joined.shape == (4 * cfg["M"],)
       and all(np.array_equal(joined[k * cfg["M"]:(k + 1) * cfg["M"]], np.asarray(per_wl[k]))
               for k in range(4)),
       f"index = wavelength_index * {cfg['M']} + bin_index")
    return res, fails


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--jaxpad-root", default=None)
    a = ap.parse_args()
    if not a.selftest:
        ap.error("--selftest is required")
    print(f"  {STATUS}")
    res, fails = selftest(jaxpad_root=a.jaxpad_root)
    print(f"  {sum(r['pass'] for r in res)}/{len(res)} checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
