#!/usr/bin/env python3
"""probe_determinism.py -- how large is the repeat non-determinism, and can it be turned off?

The preflight found that calling the forward model twice with the SAME photon keys does not give
the same answer. A boolean is not enough to act on: a 1e-16 jitter from scatter-add atomics is
irrelevant to this design, while a 1e-3 one would invalidate it. This measures the magnitude, finds
out whether it is the binned aggregation or the transport, and checks whether XLA's determinism
flags remove it.

It also measures the quantity that actually matters, which is not the reflectance but the central
difference: the method relies on common random numbers cancelling between plus h and minus h, so
the question is how much of the difference quotient is non-determinism rather than signal.

    python3 scripts/probe_determinism.py
"""
import json, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JP = os.environ.get("JAXPAD_ROOT", os.path.normpath(os.path.join(ROOT, "..", "jax-pad")))
N = int(os.environ.get("PROBE_PHOTONS", 2000))
REPEATS = int(os.environ.get("PROBE_REPEATS", 5))

sys.path.insert(0, os.path.abspath(JP))
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from scripts.inverse.phase2_v12_runner import (Phase2RunConfig, scattering_for_transport,
                                               _build_mu)
import scripts.archive.inverse.multi_wl_v12_colab as v12
from src.pad.mc_forward import forward_y_multi_layer as prod_forward
from src.pad.physics_engine_v4 import simulate_one_photon_rigorous as prod_photon

bspec = json.load(open(os.path.join(ROOT, "spec", "baseline_spec.json")))
prof = bspec["profiles"]["production"]
radial, max_steps = prof["radial"], prof["max_steps"]
kappa = bspec["blood_coordinate"]["legacy_to_physical_divisor"]
labels = bspec["parameters"]["psi_labels"]
psi0 = np.array([bspec["parameters"]["nominal_psi"][k] for k in labels], float)
n_wl = int(v12.N_WL)
edges = jnp.linspace(radial["r_min_cm"], radial["r_max_cm"], radial["n_bins"] + 1,
                     dtype=jnp.float64)
bounds = jnp.asarray([0.0020, 0.0100, 0.0250, 0.0330, 0.1830, 0.1910, 0.8000],
                     dtype=jnp.float64)
cfg = Phase2RunConfig(n_photons_final=N, seed_keys=0, seed_noise=1,
                      sto2_ubnp_true=psi0[0], sto2_dbnp_true=psi0[2],
                      b_ubnp_true=psi0[1] * kappa, b_dbnp_true=psi0[3] * kappa,
                      max_steps=max_steps)
c = jnp.stack([kappa * psi0[1] * psi0[0], kappa * psi0[1] * (1.0 - psi0[0]),
               kappa * psi0[3] * psi0[2], kappa * psi0[3] * (1.0 - psi0[2]), psi0[4], psi0[5]])
mu_list, _, _ = _build_mu(jnp.log(c + v12.EPS))
mus = [scattering_for_transport(v12.MUSS_WL[k], v12.GS_WL[k], cfg.scattering_convention)
       for k in range(n_wl)]
keys = jax.random.split(jax.random.PRNGKey(2600000), N)


def R_of(scale):
    """Binned reflectance, with the reduced scattering multiplied by `scale`."""
    return np.asarray(jnp.concatenate([
        prod_forward(muas=mu_list[k], keys=keys, muss=mus[k] * scale, gs=v12.GS_WL[k],
                     ns=v12.NS_WL[k], layer_boundaries=bounds, detector_r_edges=edges,
                     max_steps=max_steps) for k in range(n_wl)]))


def per_photon():
    """Detector id and escaping weight per photon, with no aggregation in the path."""
    f = jax.jit(jax.vmap(prod_photon, in_axes=(0, None, None, None, None, None, None, None)),
                static_argnums=7)
    d, w = f(keys, mu_list[0], mus[0], v12.GS_WL[0], v12.NS_WL[0], bounds, edges, max_steps)
    return np.asarray(d), np.asarray(w)


def rel(a, b):
    d = max(float(np.max(np.abs(b))), np.finfo(float).tiny)
    return float(np.max(np.abs(a - b)) / d)


out = {"backend": jax.default_backend(), "jax_version": jax.__version__,
       "n_photons_per_wavelength": N, "repeats": REPEATS,
       "xla_flags": os.environ.get("XLA_FLAGS", ""),
       "ptxas_version": os.environ.get("MCII_PTXAS_VERSION"),
       "deterministic_env": {k: os.environ.get(k) for k in
                             ("XLA_FLAGS", "TF_DETERMINISTIC_OPS", "JAX_DETERMINISTIC_OPS")}}

# 1. the binned reflectance, repeated
Rs = [R_of(1.0) for _ in range(REPEATS)]
spread = max(rel(Rs[i], Rs[0]) for i in range(1, REPEATS))
out["binned_reflectance_repeat"] = {
    "max_relative_spread": spread, "bitwise_identical": bool(
        all(np.array_equal(Rs[i], Rs[0]) for i in range(1, REPEATS)))}

# 2. the per-photon results, repeated. If these are stable and the binned sums are not, the
#    non-determinism is in the scatter-add, not in the transport.
P = [per_photon() for _ in range(REPEATS)]
out["per_photon_repeat"] = {
    "detector_id_bitwise_identical": bool(
        all(np.array_equal(P[i][0], P[0][0]) for i in range(1, REPEATS))),
    "weight_bitwise_identical": bool(
        all(np.array_equal(P[i][1], P[0][1]) for i in range(1, REPEATS))),
    "weight_max_relative_spread": max(rel(P[i][1], P[0][1]) for i in range(1, REPEATS))}

# 3. the quantity that matters: the central difference at a moderate step
h = 0.20 * 0.20          # h_relative x prior_sd for log_mus_scale
Dq = []
for _ in range(REPEATS):
    Dq.append((R_of(float(np.exp(+h))) - R_of(float(np.exp(-h)))) / (2.0 * h))
Dq = np.stack(Dq)
sig = float(np.max(np.abs(Dq.mean(0))))
jit_ = float(np.max(np.abs(Dq - Dq.mean(0))))
out["difference_quotient_repeat"] = {
    "h": h, "max_abs_signal": sig, "max_abs_repeat_jitter": jit_,
    "jitter_over_signal": (jit_ / sig if sig > 0 else None),
    "note": "the method relies on common random numbers cancelling between plus h and minus h, so "
            "this ratio is what decides whether repeat non-determinism matters"}

# key the filename off the determinism flag itself. Keying it off XLA_FLAGS being non-empty was
# wrong: the module load sets --xla_gpu_cuda_data_dir, so it is never empty and the default run's
# artifact was overwritten by the flagged one.
det_on = "xla_gpu_deterministic_ops=true" in os.environ.get("XLA_FLAGS", "")
out["deterministic_ops_flag_set"] = det_on
dst = os.path.join(ROOT, "results",
                   f"determinism_probe{'_deterministic' if det_on else '_default'}.json")
with open(dst, "w") as fh:
    json.dump(out, fh, indent=2)
print(f"  backend {out['backend']}  XLA_FLAGS {out['xla_flags']!r}")
print(f"  binned reflectance repeat: max rel spread {spread:.3e}  "
      f"bitwise {out['binned_reflectance_repeat']['bitwise_identical']}")
pp = out["per_photon_repeat"]
print(f"  per-photon repeat: det_id bitwise {pp['detector_id_bitwise_identical']}  "
      f"weight bitwise {pp['weight_bitwise_identical']}  "
      f"weight spread {pp['weight_max_relative_spread']:.3e}")
dq = out["difference_quotient_repeat"]
print(f"  difference quotient at h={h:.4f}: signal {dq['max_abs_signal']:.4e}  "
      f"jitter {dq['max_abs_repeat_jitter']:.4e}  ratio {dq['jitter_over_signal']:.3e}")
print(f"  wrote {os.path.relpath(dst, ROOT)}")
