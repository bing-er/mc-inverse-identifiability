"""
src/pad/physics_engine_v4.py

JAX-PAD: Differentiable Monte Carlo Framework [v4.1.0-stable]
Component: Rigorous Multi-Layer Physics Engine

Description:
    Core transport kernel implementing the Radiative Transfer Equation (RTE)
    for multi-layered cutaneous tissue. Features include:
    - Fresnel/Snell interface logic with Total Internal Reflection (TIR).
    - Robust "Layer Jump" logic to prevent boundary sticking/deadlocks.
    - Pathwise Automatic Differentiation (PAD) via continuous absorption.
    - Vectorized scattering using the Henyey-Greenstein phase function.

Author: Binger Yu
Date: January 15, 2026
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from jax import lax, random
from functools import partial
from src.pad.config_v4_633nm import MAX_STEPS


# Simulation Constants
ALIVE = jnp.int32(1)
DEAD = jnp.int32(0)
W_MIN = 1e-4  # Weight threshold for termination


@jax.jit
def get_fresnel_reflectance(n1: jax.Array, n2: jax.Array, cos_theta_i: jax.Array) -> jax.Array:
    """Calculates internal reflectance using Fresnel equations (PAD-friendly)."""
    cos_theta_i = jnp.abs(cos_theta_i)
    # Snell's Law check
    sin2_theta_t = (n1 / n2) ** 2 * (1.0 - cos_theta_i ** 2)

    # Total Internal Reflection (TIR) condition
    tir = sin2_theta_t >= 1.0
    cos_theta_t = jnp.sqrt(jnp.maximum(0.0, 1.0 - sin2_theta_t))

    #
    rs = (n1 * cos_theta_i - n2 * cos_theta_t) / (n1 * cos_theta_i + n2 * cos_theta_t)
    rp = (n2 * cos_theta_i - n1 * cos_theta_t) / (n2 * cos_theta_i + n1 * cos_theta_t)
    r_f = 0.5 * (rs ** 2 + rp ** 2)

    return jnp.where(tir, 1.0, r_f)


@partial(jax.jit, static_argnames=("max_steps",))
def simulate_one_photon_rigorous(
        key: jax.Array,
        muas: jax.Array,
        muss: jax.Array,
        gs: jax.Array,
        ns: jax.Array,
        layer_boundaries: jax.Array,
        detector_r_edges: jax.Array,
        max_steps: int = MAX_STEPS,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Simulates a single photon trajectory through a layered medium.
    Utilizes a marching algorithm to resolve layer crossings accurately.
    """

    dtype = muas.dtype
    n_air = jnp.asarray(1.0, dtype=dtype)

    # Numerical Safety Buffers
    eps_z = jnp.asarray(1e-5, dtype=dtype)  # Push off interface to avoid re-hits
    eps_dir = jnp.asarray(1e-7, dtype=dtype)  # Avoid division by zero for horizontal paths

    L = muas.shape[0]

    # --- Launch: Specular reflection (air -> stratum corneum) ---
    r_sp = ((n_air - ns[0]) / (n_air + ns[0])) ** 2
    w0 = jnp.asarray(1.0, dtype=dtype) - r_sp

    # Initial State: (x, y, z, ux, uy, uz, w, status, key, step, det_id, w_refl)
    init = (
        jnp.asarray(0.0, dtype=dtype), jnp.asarray(0.0, dtype=dtype), jnp.asarray(0.0, dtype=dtype),
        jnp.asarray(0.0, dtype=dtype), jnp.asarray(0.0, dtype=dtype), jnp.asarray(1.0, dtype=dtype),
        w0, ALIVE, key, jnp.int32(0), jnp.int32(-1), jnp.asarray(0.0, dtype=dtype),
    )

    def outer_cond(c):
        status, w, step, det_id = c[7], c[6], c[9], c[10]
        return (status == ALIVE) & (w > W_MIN) & (step < max_steps) & (det_id < 0)

    def outer_body(c):
        x, y, z, ux, uy, uz, w, status, k, step, det_id, w_refl = c

        # Direction normalization for vertical stability
        uz_safe = jnp.where(jnp.abs(uz) < eps_dir, jnp.sign(uz + eps_dir) * eps_dir, uz)

        # Map current Z to anatomical layer index
        l_idx0 = jnp.searchsorted(layer_boundaries, z, side="right")
        l_idx0 = jnp.clip(l_idx0, 0, L - 1)

        # Sample free path (Beer-Lambert distribution)
        k, k_tau = random.split(k, 2)
        tau0 = -jnp.log(random.uniform(k_tau, minval=1e-12, maxval=1.0)).astype(dtype)

        m_init = (tau0, x, y, z, ux, uy, uz_safe, w, l_idx0, status, det_id, w_refl, k)

        def march_cond(m):
            tau_rem, _, _, _, _, _, _, _, _, st, d_id, _, _ = m
            return (st == ALIVE) & (d_id < 0) & (tau_rem > 0.0)

        def march_body(m):
            """Internal layer marching logic: handles interface interactions."""
            tau_rem, x1, y1, z1, ux1, uy1, uz1, w1, l1, st1, d_id1, wr1, k1 = m

            mua, mus = muas[l1], muss[l1]
            mut = mua + mus
            going_down = uz1 > 0

            # Determine distance to next interface vs distance to next collision
            z_up = jnp.where(l1 == 0, 0.0, layer_boundaries[l1 - 1])
            z_dn = layer_boundaries[l1]
            z_int = jnp.where(going_down, z_dn, z_up)

            d_int = (z_int - z1) / uz1
            d_tau = tau_rem / mut
            hit_interface = d_int < d_tau
            d_move = jnp.where(hit_interface, jnp.maximum(d_int, eps_z), d_tau)

            # Move and update weight (Pathwise Absorption)
            x2, y2, z2 = x1 + d_move * ux1, y1 + d_move * uy1, z1 + d_move * uz1
            w2 = w1 * jnp.exp(-mua * d_move)
            tau2 = tau_rem - mut * d_move

            def hit():
                """Refraction/Reflection logic upon reaching a boundary."""
                top_hit, bot_hit = (~going_down) & (l1 == 0), (going_down) & (l1 == (L - 1))
                n1 = ns[l1]
                l_next = jnp.clip(l1 + jnp.where(going_down, 1, -1), 0, L - 1)
                n2 = jnp.where(top_hit | bot_hit, n_air, ns[l_next])

                # Fresnel calculations
                R = get_fresnel_reflectance(n1, n2, uz1)
                k2, k_ref = random.split(k1, 2)
                reflect = (random.uniform(k_ref) < R)
                transmit = ~reflect

                # Update directions based on Snell's Law
                eta = n1 / n2
                cos_i = jnp.abs(uz1)
                sin2_t = eta ** 2 * (1.0 - cos_i ** 2)
                cos_t = jnp.sqrt(jnp.maximum(0.0, 1.0 - sin2_t))

                uz3 = jnp.where(reflect, -uz1, jnp.sign(uz1) * cos_t)

                # Scoring exits at Z=0 (Top)
                exit_top = top_hit & transmit
                r = jnp.sqrt(x2 ** 2 + y2 ** 2)
                idx = jnp.searchsorted(detector_r_edges, r, side="right") - 1
                valid_det = (idx >= 0) & (idx < detector_r_edges.shape[0] - 1)
                d_id2 = jnp.where(exit_top & valid_det, idx.astype(jnp.int32), d_id1)
                wr2 = jnp.where(exit_top & valid_det, w2, 0.0)
                st2 = jnp.where(exit_top | (bot_hit & transmit), DEAD, st1)

                l2 = jnp.where((~(top_hit | bot_hit)) & transmit, l_next, l1)

                # ADVANCED: Layer Jump to prevent "Interface Trapping" deadlocks
                z3 = jnp.where(transmit, z2 + jnp.sign(uz3) * eps_z * 2.0, z2 - jnp.sign(uz1) * eps_z * 2.0)

                return (tau2, x2, y2, z3, ux1, uy1, uz3, w2, l2, st2, d_id2, wr2, k2)

            def no_hit():
                return (tau2, x2, y2, z2, ux1, uy1, uz1, w2, l1, st1, d_id1, wr1, k1)

            return lax.cond(hit_interface, hit, no_hit)

        # SAFETY: fori_loop replaces while_loop to prevent infinite hangs on boundaries
        m_final = lax.fori_loop(0, 10, lambda i, m: lax.cond(march_cond(m), march_body, lambda x: x, m), m_init)
        _, xm, ym, zm, uxm, uym, uzm, wm, lm, stm, didm, wrm, k_after = m_final

        # --- Scattering event (Henyey-Greenstein) ---
        def do_scatter(k_in):
            ct, ph, k_out = sample_henyey_greenstein(k_in, gs[lm])
            ux2, uy2, uz2 = update_direction(uxm, uym, uzm, ct, ph)
            return ux2, uy2, uz2, k_out

        ux_new, uy_new, uz_new, k_new = lax.cond((stm == ALIVE) & (didm < 0),
                                                 lambda: do_scatter(k_after),
                                                 lambda: (uxm, uym, uzm, k_after))

        return (xm, ym, zm, ux_new, uy_new, uz_new, wm, jnp.where(wm <= W_MIN, DEAD, stm), k_new, step + 1, didm, wrm)

    final = lax.while_loop(outer_cond, outer_body, init)
    return final[10], final[11]  # Returns (Detector ID, Weight)


@partial(jax.jit, static_argnames=("max_steps",))
def simulate_one_photon_rigorous_energy(
        key: jax.Array,
        muas: jax.Array,
        muss: jax.Array,
        gs: jax.Array,
        ns: jax.Array,
        layer_boundaries: jax.Array,
        detector_r_edges: jax.Array,
        max_steps: int = MAX_STEPS,
):
    """
    Same transport as simulate_one_photon_rigorous, but also returns
    energy accounting terms for conservation checks.

    Returns:
        det_id (int32): detector bin id, -1 if not binned
        w_refl_bin (float): diffuse reflected weight that falls into a detector bin
        w_refl_total (float): diffuse reflected weight exiting top (all radii)
        w_tran_total (float): transmitted weight exiting bottom
        w_abs_total (float): absorbed weight accumulated along path
        r_sp (float): specular reflectance at launch
    """

    dtype = muas.dtype
    n_air = jnp.asarray(1.0, dtype=dtype)

    eps_z = jnp.asarray(1e-5, dtype=dtype)
    eps_dir = jnp.asarray(1e-7, dtype=dtype)

    L = muas.shape[0]

    # --- Launch: Specular reflection (air -> layer 0) ---
    r_sp = ((n_air - ns[0]) / (n_air + ns[0])) ** 2
    w0 = jnp.asarray(1.0, dtype=dtype) - r_sp

    # State adds: w_abs_total, w_tran_total, w_refl_total
    # (x,y,z, ux,uy,uz, w, status, key, step, det_id, w_refl_bin, w_refl_total, w_tran_total, w_abs_total)
    init = (
        jnp.asarray(0.0, dtype=dtype), jnp.asarray(0.0, dtype=dtype), jnp.asarray(0.0, dtype=dtype),
        jnp.asarray(0.0, dtype=dtype), jnp.asarray(0.0, dtype=dtype), jnp.asarray(1.0, dtype=dtype),
        w0, ALIVE, key, jnp.int32(0),
        jnp.int32(-1),                 # det_id
        jnp.asarray(0.0, dtype=dtype), # w_refl_bin
        jnp.asarray(0.0, dtype=dtype), # w_refl_total
        jnp.asarray(0.0, dtype=dtype), # w_tran_total
        jnp.asarray(0.0, dtype=dtype), # w_abs_total
    )

    def outer_cond(c):
        status, w, step, det_id = c[7], c[6], c[9], c[10]
        return (status == ALIVE) & (w > W_MIN) & (step < max_steps) & (det_id < 0)

    def outer_body(c):
        x, y, z, ux, uy, uz, w, status, k, step, det_id, w_refl_bin, w_refl_total, w_tran_total, w_abs_total = c

        uz_safe = jnp.where(jnp.abs(uz) < eps_dir, jnp.sign(uz + eps_dir) * eps_dir, uz)

        l_idx0 = jnp.searchsorted(layer_boundaries, z, side="right")
        l_idx0 = jnp.clip(l_idx0, 0, L - 1)

        k, k_tau = random.split(k, 2)
        tau0 = -jnp.log(random.uniform(k_tau, minval=1e-12, maxval=1.0)).astype(dtype)

        m_init = (tau0, x, y, z, ux, uy, uz_safe, w, l_idx0, status, det_id,
                  w_refl_bin, w_refl_total, w_tran_total, w_abs_total, k)

        def march_cond(m):
            tau_rem, _, _, _, _, _, _, _, _, st, d_id, _, _, _, _, _ = m
            return (st == ALIVE) & (d_id < 0) & (tau_rem > 0.0)

        def march_body(m):
            (tau_rem, x1, y1, z1, ux1, uy1, uz1, w1, l1, st1, d_id1,
             wr_bin1, wr_tot1, wt_tot1, wa_tot1, k1) = m

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

            # Move and continuous absorption
            x2, y2, z2 = x1 + d_move * ux1, y1 + d_move * uy1, z1 + d_move * uz1
            w2 = w1 * jnp.exp(-mua * d_move)
            tau2 = tau_rem - mut * d_move

            # absorbed increment for this segment
            wa2 = wa_tot1 + (w1 - w2)

            def hit():
                top_hit = (~going_down) & (l1 == 0)
                bot_hit = (going_down) & (l1 == (L - 1))

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

                # --- exits ---
                exit_top = top_hit & transmit
                exit_bot = bot_hit & transmit

                # total diffuse reflectance (all radii)
                wr_tot2 = wr_tot1 + jnp.where(exit_top, w2, 0.0)

                # transmitted total
                wt_tot2 = wt_tot1 + jnp.where(exit_bot, w2, 0.0)

                # binned reflectance (only if within detector range)
                r = jnp.sqrt(x2 ** 2 + y2 ** 2)
                idx = jnp.searchsorted(detector_r_edges, r, side="right") - 1
                valid_det = (idx >= 0) & (idx < detector_r_edges.shape[0] - 1)

                d_id2 = jnp.where(exit_top & valid_det, idx.astype(jnp.int32), d_id1)
                wr_bin2 = jnp.where(exit_top & valid_det, w2, 0.0)

                st2 = jnp.where(exit_top | exit_bot, DEAD, st1)
                l2 = jnp.where((~(top_hit | bot_hit)) & transmit, l_next, l1)

                z3 = jnp.where(transmit,
                               z2 + jnp.sign(uz3) * eps_z * 2.0,
                               z2 - jnp.sign(uz1) * eps_z * 2.0)

                return (tau2, x2, y2, z3, ux1, uy1, uz3, w2, l2, st2, d_id2,
                        wr_bin2, wr_tot2, wt_tot2, wa2, k2)

            def no_hit():
                return (tau2, x2, y2, z2, ux1, uy1, uz1, w2, l1, st1, d_id1,
                        wr_bin1, wr_tot1, wt_tot1, wa2, k1)

            return lax.cond(hit_interface, hit, no_hit)

        m_final = lax.fori_loop(
            0, 10,
            lambda i, m: lax.cond(march_cond(m), march_body, lambda x: x, m),
            m_init
        )

        (_, xm, ym, zm, uxm, uym, uzm, wm, lm, stm, didm,
         wr_bin_m, wr_tot_m, wt_tot_m, wa_tot_m, k_after) = m_final

        def do_scatter(k_in):
            ct, ph, k_out = sample_henyey_greenstein(k_in, gs[lm])
            ux2, uy2, uz2 = update_direction(uxm, uym, uzm, ct, ph)
            return ux2, uy2, uz2, k_out

        ux_new, uy_new, uz_new, k_new = lax.cond(
            (stm == ALIVE) & (didm < 0),
            lambda: do_scatter(k_after),
            lambda: (uxm, uym, uzm, k_after)
        )

        return (xm, ym, zm, ux_new, uy_new, uz_new,
                wm, jnp.where(wm <= W_MIN, DEAD, stm), k_new,
                step + 1, didm, wr_bin_m, wr_tot_m, wt_tot_m, wa_tot_m)

    final = lax.while_loop(outer_cond, outer_body, init)

    det_id = final[10]
    w_refl_bin = final[11]
    w_refl_total = final[12]
    w_tran_total = final[13]
    w_abs_total = final[14]

    return det_id, w_refl_bin, w_refl_total, w_tran_total, w_abs_total, r_sp


@jax.jit
def sample_henyey_greenstein(key: jax.Array, g: jax.Array):
    """Samples cosine of the scattering angle."""
    key, k1, k2 = random.split(key, 3)
    rnd = random.uniform(k1, minval=1e-9, maxval=1.0)

    def g_nonzero(g_val):
        temp = (1.0 - g_val ** 2) / (1.0 - g_val + 2.0 * g_val * rnd)
        return (1.0 + g_val ** 2 - temp ** 2) / (2.0 * g_val)

    cos_theta = lax.cond(jnp.abs(g) < 1e-6, lambda g_v: 2.0 * rnd - 1.0, g_nonzero, g)
    return jnp.clip(cos_theta, -1.0, 1.0), 2.0 * jnp.pi * random.uniform(k2), key


@jax.jit
def update_direction(ux, uy, uz, cos_theta, phi):
    """Updates direction vectors using spherical coordinate transformation."""
    sin_theta = jnp.sqrt(jnp.maximum(0.0, 1.0 - cos_theta ** 2))
    sin_phi, cos_phi = jnp.sin(phi), jnp.cos(phi)

    def general():
        temp = jnp.sqrt(jnp.maximum(1e-12, 1.0 - uz ** 2))
        return (sin_theta * (ux * uz * cos_phi - uy * sin_phi) / temp + ux * cos_theta,
                sin_theta * (uy * uz * cos_phi + ux * sin_phi) / temp + uy * cos_theta,
                -sin_theta * cos_phi * temp + uz * cos_theta)

    return lax.cond(jnp.abs(uz) > 0.99999,
                    lambda _: (sin_theta * cos_phi, sin_theta * sin_phi, jnp.sign(uz) * cos_theta),
                    lambda _: general(), operand=None)