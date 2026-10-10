"""

src/pad/config_v4_633nm.py

JAX-PAD Configuration File - Version 4.1.0-stable
Project: Differentiable Monte Carlo Framework
Optimized for 7-Layer Skin Inversion (Phase 1b)
Wavelength Target: 800 nm (Deep Tissue Probing)

Author: Binger Yu
Date: January 25, 2026
"""

import jax.numpy as jnp

# Core Versioning
VERSION = "v4.1.0-stable-633nm"

# --- Simulation Parameters ---
# Higher photon count for N_PHOTONS_FAST to overcome lower signal contrast at 800nm
N_PHOTONS_FAST = 2048
N_PHOTONS_FINAL = 100000
MAX_STEPS = 500

# --- Random Number Generation ---
SEED_KEYS = 42
SEED_NOISE = 1

# --- Physical Constants ---
DEFAULT_EPS = 1e-12

# --- 800nm Baseline mu_a (Target for your test_jacobians.py) ---
# Derived from isosbestic blood point and reduced melanin index
# TARGET_MUA_800NM = jnp.array([0.05, 0.10, 0.25, 1.20, 0.20, 0.50, 0.15], dtype=jnp.float32)

TARGET_MUA_633NM = jnp.array([0.10, 0.15, 0.51, 2.62, 0.49, 0.95, 0.45], dtype=jnp.float32)

# --- Inversion Mask / Loss Knobs ---
LOG_FRAC_THRESHOLD = 1e-3
LOG_USE_WEIGHTS = True
DETECTOR_R_MAX = 2.0
# Increased cutoff to 1.5cm; 800nm photons travel further, providing deep data
MAX_R_CUTOFF = 1.5

RUN_FINAL = False
FINAL_STEPS = 200
FINAL_LR = 0.002
FINAL_REPEATS = 1