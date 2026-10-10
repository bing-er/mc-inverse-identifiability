"""replay_kernel.py -- the production trajectory, with a per-layer pathlength accumulator.

The production kernel is the only physical truth here. This module does not modify it, does not
re-derive it, and does not treat any other project's replay kernel as correct for this one: P5's
kernel was instrumented against a different production kernel and cannot be assumed equivalent.

Everything physical is IMPORTED from the production module rather than copied, so the Fresnel
reflectance, the Henyey-Greenstein sampler, the direction update and the termination constants are
literally the production implementations:

    get_fresnel_reflectance, sample_henyey_greenstein, update_direction, ALIVE, DEAD, W_MIN

The only new code is the outer and marching loop with one extra state element, a length-L vector of
per-layer pathlengths. It mirrors the production loop statement for statement so that the random
number consumption is identical:

    outer step: split(k, 2) -> tau0 from uniform(k_tau, 1e-12, 1.0)
    march step: on an interface hit only, split(k1, 2) -> uniform(k_ref) for the Fresnel draw
    scatter:    sample_henyey_greenstein, which itself splits into three and draws twice

and so that the branch order, the fori_loop bound of 10 marching iterations, the eps nudges, the
layer-jump on transmit and the weight update are the same. Pathlength is accumulated exactly where
the production weight update happens, with the same d_move, so the recorded lengths are the ones
that produced the weight. Nothing is accumulated for the geometric eps nudge, because production
applies no attenuation there either.

Why pathlength is what is needed. Production attenuates continuously, w <- w exp(-mua d), so an
escaping weight is w0 exp(-sum_j mua_j L_j) and

    d w_exit / d mua_j = -L_j w_exit

which gives the perturbation Monte Carlo derivative of the binned reflectance,

    d R_i / d mua_j = -(1 / (N A_i)) sum_{p in bin i} W_p L_{p,j}

That this identity holds on the recorded numbers is itself a check on the accumulator, and
scripts/replay_equivalence.py asserts it photon by photon.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, random, vmap

from src.pad.physics_engine_v4 import (          # noqa: E402  the production implementations
    ALIVE, DEAD, W_MIN, get_fresnel_reflectance, sample_henyey_greenstein, update_direction)


@partial(jax.jit, static_argnames=("max_steps",))
def simulate_one_photon_replay(key, muas, muss, gs, ns, layer_boundaries, detector_r_edges,
                               max_steps):
    """Production transport, plus per-layer pathlength. Returns (det_id, w_refl, pathlengths)."""
    dtype = muas.dtype
    n_air = jnp.asarray(1.0, dtype=dtype)
    eps_z = jnp.asarray(1e-5, dtype=dtype)
    eps_dir = jnp.asarray(1e-7, dtype=dtype)
    L = muas.shape[0]

    r_sp = ((n_air - ns[0]) / (n_air + ns[0])) ** 2
    w0 = jnp.asarray(1.0, dtype=dtype) - r_sp

    init = (
        jnp.asarray(0.0, dtype=dtype), jnp.asarray(0.0, dtype=dtype), jnp.asarray(0.0, dtype=dtype),
        jnp.asarray(0.0, dtype=dtype), jnp.asarray(0.0, dtype=dtype), jnp.asarray(1.0, dtype=dtype),
        w0, ALIVE, key, jnp.int32(0), jnp.int32(-1), jnp.asarray(0.0, dtype=dtype),
        jnp.zeros((L,), dtype=dtype),                       # the only added state
    )

    def outer_cond(c):
        status, w, step, det_id = c[7], c[6], c[9], c[10]
        return (status == ALIVE) & (w > W_MIN) & (step < max_steps) & (det_id < 0)

    def outer_body(c):
        x, y, z, ux, uy, uz, w, status, k, step, det_id, w_refl, pl = c

        uz_safe = jnp.where(jnp.abs(uz) < eps_dir, jnp.sign(uz + eps_dir) * eps_dir, uz)

        l_idx0 = jnp.searchsorted(layer_boundaries, z, side="right")
        l_idx0 = jnp.clip(l_idx0, 0, L - 1)

        k, k_tau = random.split(k, 2)
        tau0 = -jnp.log(random.uniform(k_tau, minval=1e-12, maxval=1.0)).astype(dtype)

        m_init = (tau0, x, y, z, ux, uy, uz_safe, w, l_idx0, status, det_id, w_refl, k, pl)

        def march_cond(m):
            tau_rem, _, _, _, _, _, _, _, _, st, d_id, _, _, _ = m
            return (st == ALIVE) & (d_id < 0) & (tau_rem > 0.0)

        def march_body(m):
            tau_rem, x1, y1, z1, ux1, uy1, uz1, w1, l1, st1, d_id1, wr1, k1, pl1 = m

            mua, mus = muas[l1], muss[l1]
            mut = mua + mus
            going_down = uz1 > 0

            z_up = jnp.where(l1 == 0, 0.0, layer_boundaries[l1 - 1])
            z_dn = layer_boundaries[l1]
            z_int = jnp.where(going_down, z_dn, z_up)

            d_int = (z_int - z1) / uz1
            d_tau = tau_rem / mut
            hit_interface = d_int < d_tau
            d_move = jnp.where(hit_interface, jnp.maximum(d_int, eps_z), d_tau)

            x2, y2, z2 = x1 + d_move * ux1, y1 + d_move * uy1, z1 + d_move * uz1
            w2 = w1 * jnp.exp(-mua * d_move)
            tau2 = tau_rem - mut * d_move
            pl2 = pl1.at[l1].add(d_move)       # exactly where production updates the weight

            def hit():
                top_hit, bot_hit = (~going_down) & (l1 == 0), (going_down) & (l1 == (L - 1))
                n1 = ns[l1]
                l_next = jnp.clip(l1 + jnp.where(going_down, 1, -1), 0, L - 1)
                n2 = jnp.where(top_hit | bot_hit, n_air, ns[l_next])

                R = get_fresnel_reflectance(n1, n2, uz1)
                k2, k_ref = random.split(k1, 2)
                reflect = (random.uniform(k_ref) < R)
                transmit = ~reflect

                eta = n1 / n2
                cos_i = jnp.abs(uz1)
                sin2_t = eta ** 2 * (1.0 - cos_i ** 2)
                cos_t = jnp.sqrt(jnp.maximum(0.0, 1.0 - sin2_t))

                uz3 = jnp.where(reflect, -uz1, jnp.sign(uz1) * cos_t)

                exit_top = top_hit & transmit
                r = jnp.sqrt(x2 ** 2 + y2 ** 2)
                idx = jnp.searchsorted(detector_r_edges, r, side="right") - 1
                valid_det = (idx >= 0) & (idx < detector_r_edges.shape[0] - 1)
                d_id2 = jnp.where(exit_top & valid_det, idx.astype(jnp.int32), d_id1)
                wr2 = jnp.where(exit_top & valid_det, w2, 0.0)
                st2 = jnp.where(exit_top | (bot_hit & transmit), DEAD, st1)

                l2 = jnp.where((~(top_hit | bot_hit)) & transmit, l_next, l1)

                z3 = jnp.where(transmit, z2 + jnp.sign(uz3) * eps_z * 2.0,
                               z2 - jnp.sign(uz1) * eps_z * 2.0)

                return (tau2, x2, y2, z3, ux1, uy1, uz3, w2, l2, st2, d_id2, wr2, k2, pl2)

            def no_hit():
                return (tau2, x2, y2, z2, ux1, uy1, uz1, w2, l1, st1, d_id1, wr1, k1, pl2)

            return lax.cond(hit_interface, hit, no_hit)

        m_final = lax.fori_loop(0, 10,
                                lambda i, m: lax.cond(march_cond(m), march_body, lambda x: x, m),
                                m_init)
        _, xm, ym, zm, uxm, uym, uzm, wm, lm, stm, didm, wrm, k_after, plm = m_final

        def do_scatter(k_in):
            ct, ph, k_out = sample_henyey_greenstein(k_in, gs[lm])
            ux2, uy2, uz2 = update_direction(uxm, uym, uzm, ct, ph)
            return ux2, uy2, uz2, k_out

        ux_new, uy_new, uz_new, k_new = lax.cond((stm == ALIVE) & (didm < 0),
                                                 lambda: do_scatter(k_after),
                                                 lambda: (uxm, uym, uzm, k_after))

        return (xm, ym, zm, ux_new, uy_new, uz_new, wm,
                jnp.where(wm <= W_MIN, DEAD, stm), k_new, step + 1, didm, wrm, plm)

    final = lax.while_loop(outer_cond, outer_body, init)
    return final[10], final[11], final[12]


@partial(jax.jit, static_argnames=("max_steps",))
def replay_moments(*, muas, muss, gs, ns, layer_boundaries, keys, detector_r_edges, max_steps):
    """Reflectance and its per-layer pathlength moments, from one pass over the photons.

    Returns (R, S) with R the usual binned reflectance density and
    S[i, j] = (1 / (N A_i)) sum_{p in bin i} W_p L_{p,j}, so that dR_i/dmua_j = -S[i, j].

    The accumulation mirrors the production forward solver: the same `valid = det_ids >= 0` mask,
    the same scatter-add, the same annulus-area normalization.
    """
    M = detector_r_edges.shape[0] - 1
    n_photons = jnp.asarray(keys.shape[0], dtype=muas.dtype)

    det_ids, w_refl, pls = vmap(
        simulate_one_photon_replay,
        in_axes=(0, None, None, None, None, None, None, None),
    )(keys, muas, muss, gs, ns, layer_boundaries, detector_r_edges, max_steps)

    valid = det_ids >= 0
    safe_ids = jnp.where(valid, det_ids, 0).astype(jnp.int32)
    safe_w = jnp.where(valid, w_refl, 0.0)

    r0, r1 = detector_r_edges[:-1], detector_r_edges[1:]
    annulus_area = jnp.pi * (r1 ** 2 - r0 ** 2)
    denom = n_photons * annulus_area

    y = jnp.zeros((M,), dtype=w_refl.dtype).at[safe_ids].add(safe_w)
    s = jnp.zeros((M, muas.shape[0]), dtype=w_refl.dtype).at[safe_ids].add(
        safe_w[:, None] * pls)

    return y / denom, s / denom[:, None]


@partial(jax.jit, static_argnames=("max_steps",))
def replay_per_photon(*, muas, muss, gs, ns, layer_boundaries, keys, detector_r_edges, max_steps):
    """Per-photon detector id, escaping weight and pathlengths, for the equivalence gate."""
    return vmap(
        simulate_one_photon_replay,
        in_axes=(0, None, None, None, None, None, None, None),
    )(keys, muas, muss, gs, ns, layer_boundaries, detector_r_edges, max_steps)


@partial(jax.jit, static_argnames=("max_steps",))
def simulate_one_photon_replay_DEFECTIVE(key, muas, muss, gs, ns, layer_boundaries,
                                         detector_r_edges, max_steps):
    """A deliberately broken replay, used only as a negative control for the equivalence gate.

    It is identical to simulate_one_photon_replay except that the outer loop swaps the two halves
    of the free-path key split: the key carried forward and the key used for tau are exchanged.
    That is exactly the class of mistake the gate exists to catch, a wrong random-number
    consumption order with the physics untouched. A gate that cannot fail proves nothing, so
    scripts/replay_equivalence.py runs this and asserts that it IS rejected.

    A three-way split discarding one output would NOT be a valid negative control. JAX's threefry
    split is a prefix: split(k, 3)[:2] is bitwise equal to split(k, 2), so that change leaves the
    stream untouched. The first attempt at this control used it, the gate reported zero mismatches,
    and the control was replaced. The property is verified in the gate's own output.
    """
    dtype = muas.dtype
    n_air = jnp.asarray(1.0, dtype=dtype)
    eps_z = jnp.asarray(1e-5, dtype=dtype)
    eps_dir = jnp.asarray(1e-7, dtype=dtype)
    L = muas.shape[0]
    w0 = jnp.asarray(1.0, dtype=dtype) - ((n_air - ns[0]) / (n_air + ns[0])) ** 2

    init = (jnp.asarray(0.0, dtype=dtype), jnp.asarray(0.0, dtype=dtype),
            jnp.asarray(0.0, dtype=dtype), jnp.asarray(0.0, dtype=dtype),
            jnp.asarray(0.0, dtype=dtype), jnp.asarray(1.0, dtype=dtype),
            w0, ALIVE, key, jnp.int32(0), jnp.int32(-1), jnp.asarray(0.0, dtype=dtype),
            jnp.zeros((L,), dtype=dtype))

    def outer_cond(c):
        return (c[7] == ALIVE) & (c[6] > W_MIN) & (c[9] < max_steps) & (c[10] < 0)

    def outer_body(c):
        x, y, z, ux, uy, uz, w, status, k, step, det_id, w_refl, pl = c
        uz_safe = jnp.where(jnp.abs(uz) < eps_dir, jnp.sign(uz + eps_dir) * eps_dir, uz)
        l_idx0 = jnp.clip(jnp.searchsorted(layer_boundaries, z, side="right"), 0, L - 1)
        k_tau, k = random.split(k, 2)                   # THE DEFECT: halves swapped
        tau0 = -jnp.log(random.uniform(k_tau, minval=1e-12, maxval=1.0)).astype(dtype)
        m_init = (tau0, x, y, z, ux, uy, uz_safe, w, l_idx0, status, det_id, w_refl, k, pl)

        def march_cond(m):
            return (m[9] == ALIVE) & (m[10] < 0) & (m[0] > 0.0)

        def march_body(m):
            tau_rem, x1, y1, z1, ux1, uy1, uz1, w1, l1, st1, d_id1, wr1, k1, pl1 = m
            mua, mus = muas[l1], muss[l1]
            mut = mua + mus
            going_down = uz1 > 0
            z_up = jnp.where(l1 == 0, 0.0, layer_boundaries[l1 - 1])
            z_int = jnp.where(going_down, layer_boundaries[l1], z_up)
            d_int = (z_int - z1) / uz1
            d_tau = tau_rem / mut
            hit_interface = d_int < d_tau
            d_move = jnp.where(hit_interface, jnp.maximum(d_int, eps_z), d_tau)
            x2, y2, z2 = x1 + d_move * ux1, y1 + d_move * uy1, z1 + d_move * uz1
            w2 = w1 * jnp.exp(-mua * d_move)
            tau2 = tau_rem - mut * d_move
            pl2 = pl1.at[l1].add(d_move)

            def hit():
                top_hit, bot_hit = (~going_down) & (l1 == 0), going_down & (l1 == (L - 1))
                n1 = ns[l1]
                l_next = jnp.clip(l1 + jnp.where(going_down, 1, -1), 0, L - 1)
                n2 = jnp.where(top_hit | bot_hit, n_air, ns[l_next])
                R = get_fresnel_reflectance(n1, n2, uz1)
                k2, k_ref = random.split(k1, 2)
                reflect = random.uniform(k_ref) < R
                transmit = ~reflect
                eta, cos_i = n1 / n2, jnp.abs(uz1)
                cos_t = jnp.sqrt(jnp.maximum(0.0, 1.0 - eta ** 2 * (1.0 - cos_i ** 2)))
                uz3 = jnp.where(reflect, -uz1, jnp.sign(uz1) * cos_t)
                exit_top = top_hit & transmit
                r = jnp.sqrt(x2 ** 2 + y2 ** 2)
                idx = jnp.searchsorted(detector_r_edges, r, side="right") - 1
                valid = (idx >= 0) & (idx < detector_r_edges.shape[0] - 1)
                d_id2 = jnp.where(exit_top & valid, idx.astype(jnp.int32), d_id1)
                wr2 = jnp.where(exit_top & valid, w2, 0.0)
                st2 = jnp.where(exit_top | (bot_hit & transmit), DEAD, st1)
                l2 = jnp.where((~(top_hit | bot_hit)) & transmit, l_next, l1)
                z3 = jnp.where(transmit, z2 + jnp.sign(uz3) * eps_z * 2.0,
                               z2 - jnp.sign(uz1) * eps_z * 2.0)
                return (tau2, x2, y2, z3, ux1, uy1, uz3, w2, l2, st2, d_id2, wr2, k2, pl2)

            def no_hit():
                return (tau2, x2, y2, z2, ux1, uy1, uz1, w2, l1, st1, d_id1, wr1, k1, pl2)

            return lax.cond(hit_interface, hit, no_hit)

        m_final = lax.fori_loop(0, 10,
                                lambda i, m: lax.cond(march_cond(m), march_body, lambda x: x, m),
                                m_init)
        _, xm, ym, zm, uxm, uym, uzm, wm, lm, stm, didm, wrm, k_after, plm = m_final

        def do_scatter(k_in):
            ct, ph, k_out = sample_henyey_greenstein(k_in, gs[lm])
            return (*update_direction(uxm, uym, uzm, ct, ph), k_out)

        ux_new, uy_new, uz_new, k_new = lax.cond((stm == ALIVE) & (didm < 0),
                                                 lambda: do_scatter(k_after),
                                                 lambda: (uxm, uym, uzm, k_after))
        return (xm, ym, zm, ux_new, uy_new, uz_new, wm,
                jnp.where(wm <= W_MIN, DEAD, stm), k_new, step + 1, didm, wrm, plm)

    final = lax.while_loop(outer_cond, outer_body, init)
    return final[10], final[11], final[12]


@partial(jax.jit, static_argnames=("max_steps",))
def replay_per_photon_DEFECTIVE(*, muas, muss, gs, ns, layer_boundaries, keys, detector_r_edges,
                                max_steps):
    """Negative control only. Never used for any reported number."""
    return vmap(simulate_one_photon_replay_DEFECTIVE,
                in_axes=(0, None, None, None, None, None, None, None))(
        keys, muas, muss, gs, ns, layer_boundaries, detector_r_edges, max_steps)
