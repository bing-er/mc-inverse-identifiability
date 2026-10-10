"""
JAX-PAD: Differentiable Monte Carlo Framework [v4.1.0-stable]
Component: Multi-Layer Forward Solver (Rigorous Kernel)

Description:
    This module implements a vectorized, JAX-native forward solver for light
    transport in 7-layered cutaneous tissue. Optimized for the Meglinski
    anatomical model, it leverages Pathwise Automatic Differentiation (PAD)
    to enable high-fidelity physiological sensing.

Key Features:
    - Stable Photon Marching: Resolves boundary deadlocks via eps-nudging.
    - XLA Optimization: Support for N=100,000+ photons with linear scaling.
    - Spectral-Spatial Sync: Validated for 'W-Curve' and radial decay profiles.

Author: Binger Yu
Date: January 21, 2026
"""

from __future__ import annotations
from functools import partial

import jax
import jax.numpy as jnp
from jax import vmap

# Keep the public API stable while routing all forward calls through the
# validated v4 transport kernel. This lets older scripts continue to import
# forward_y_multi_layer without silently using the older physics path.
from src.pad.physics_engine_v4 import simulate_one_photon_rigorous as simulate_one_photon
from src.pad import config as CFG

VERSION = getattr(CFG, "VERSION", "v4.x (no VERSION in config)")
MAX_STEPS = CFG.MAX_STEPS


@partial(jax.jit, static_argnames=("max_steps",))
def forward_y_multi_layer(
    *,
    muas: jax.Array,
    muss: jax.Array,
    gs: jax.Array,
    ns: jax.Array,
    layer_boundaries: jax.Array,
    keys: jax.Array,
    detector_r_edges: jax.Array,
    max_steps: int = MAX_STEPS,
) -> jax.Array:
    """
    Computes diffuse reflectance Rd(r) for a multi-layered geometry.
    """
    M = detector_r_edges.shape[0] - 1
    # Use float32/64 to match absorption precision for normalization
    n_photons = jnp.asarray(keys.shape[0], dtype=muas.dtype)

    # Parallelize the 2,000-step rigorous trajectories across available JAX devices
    det_ids, w_refl = vmap(
        simulate_one_photon,
        in_axes=(0, None, None, None, None, None, None, None),
    )(keys, muas, muss, gs, ns, layer_boundaries, detector_r_edges, max_steps)

    # Initialize the spatial detection accumulator
    y = jnp.zeros((M,), dtype=w_refl.dtype)

    # Filtering logic to handle 'dead' conditions:
    # Photons that are absorbed or transmitted past depth are assigned ID -1
    valid = det_ids >= 0
    safe_ids = jnp.where(valid, det_ids, 0).astype(jnp.int32)
    safe_w = jnp.where(valid, w_refl, 0.0)

    # Scatter-add accumulated weights into radial bins
    y = y.at[safe_ids].add(safe_w)

    # Geometric normalization for Diffuse Reflectance (1/cm^2)
    r0, r1 = detector_r_edges[:-1], detector_r_edges[1:]
    annulus_area = jnp.pi * (r1**2 - r0**2)

    # Return averaged reflectance per unit area
    return y / (n_photons * annulus_area)
