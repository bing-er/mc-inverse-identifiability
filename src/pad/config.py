"""
JAX-PAD Configuration File - Version 4.1.0-stable
Project: Differentiable Monte Carlo Framework
Optimized for 7-Layer Skin Inversion (Forward Validation)

src/pad/config.py

Author: Binger Yu
Date: January 21, 2026
"""

# Core Versioning
VERSION = "v4.1.0-stable"

# --- GLOBAL SIMULATION SETTINGS ---
# Rigorous validation requires high N and deep trajectories (MAX_STEPS)
# Photon packet count
N_PHOTONS = 100000

# Max scattering/interface events per photon
MAX_STEPS = 2000

# Updated seeds for the January 21th run to ensure statistical independence
SEED_KEYS = 20260121
SEED_NOISE = 20260122
