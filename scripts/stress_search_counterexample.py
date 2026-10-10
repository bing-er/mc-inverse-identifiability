#!/usr/bin/env python3
"""stress_search_counterexample.py -- the stress search is not a certification, deterministically.

stress_search_classifications once carried a name implying certification and claimed invariance over every
admissible Jacobian. It searches the leading psi left-singular directions, each applied identically
to all eta columns, plus a finite number of random directions. The admissible set is a product of
per-column balls, so a perturbation that assigns a DIFFERENT singular direction to each column is
admissible and is not in the search set.

This constructs one, on the frozen production psi block, and records that the search reports no
flip while that untested direction crosses a threshold placed between the two. It runs no transport.

    python3 scripts/stress_search_counterexample.py
"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import augmented as aug           # noqa: E402
import unify                      # noqa: E402
from unify_spectrum import column_scale   # noqa: E402

MARGIN, SEED = 0.05, 3

d = np.load(os.path.join(ROOT, "data", "derived", "production_pooled.npz"))
spec = json.load(open(os.path.join(ROOT, "spec", "baseline_spec.json")))
J = np.asarray(d["J_psi"], float); R = np.asarray(d["R"], float)
areas = np.asarray(d["areas"], float); psi = np.asarray(d["psi_nominal"], float)
n_wl = int(d["n_wavelengths"])
N = spec["profiles"]["production"]["n_measurement_photons_per_wavelength"]
labels = list(unify.PSI_LABELS)
Dp = column_scale(psi)
Jw = unify.whiten_multinomial(J, R, areas, N, Dp, n_wl)
s = np.linalg.svd(Jw, compute_uv=False)
U = np.linalg.svd(Jw, full_matrices=False)[0]
MASK = np.ones(Jw.shape[0], dtype=bool)
rng = np.random.default_rng(SEED)
Je = (0.3 * U @ rng.normal(size=(6, 5))
      + 0.7 * np.linalg.qr(rng.normal(size=(Jw.shape[0], 5)))[0]) * float(s[0]) * 0.2
cols = np.linalg.norm(Je, axis=0, keepdims=True)


def value(Jeta):
    bi = aug.block_information(np.hstack([Jw, Jeta]), 6, support_mask=MASK, require_mask=True)
    return aug.estimability(bi["F_psi_eff"], scale=Dp,
                            labels=labels)["per_parameter"]["StO2_U"]["sqrt_crlb"]


probe = {"upper": ("StO2_U", "lt", 1e9)}
r = aug.stress_search_classifications(Jw, Je, MARGIN, Dp, probe, labels, MASK,
                                      n_directions=800, seed=9)
baseline = r["baseline_values"]["upper"]
searched = r["value_range_over_the_admissible_set"]["upper"][1]

best, which = searched, None
for shift in range(U.shape[1]):
    idx = [(shift + k) % U.shape[1] for k in range(5)]
    P = np.stack([U[:, i] for i in idx], axis=1)
    P = P / np.linalg.norm(P, axis=0, keepdims=True)
    for sgn in (1.0, -1.0):
        v = value(Je + sgn * MARGIN * P * cols)
        if v and v > best:
            best, which = v, {"column_direction_indices": idx, "sign": sgn}

assert best > searched, "the construction did not exceed the search; the counterexample failed"
threshold = 0.5 * (searched + best)
check = aug.stress_search_classifications(Jw, Je, MARGIN, Dp,
                                          {"upper": ("StO2_U", "lt", threshold)},
                                          labels, MASK, n_directions=800, seed=9)
out = {
    "claim_tested": "that the stress search establishes invariance over the admissible set",
    "verdict": "it does not",
    "runs_transport": False,
    "construction": "the search applies one direction to all eta columns; this assigns a "
                    "DIFFERENT psi left-singular direction to each column, which is admissible "
                    "and is not in the search set",
    "margin": MARGIN, "seed": SEED,
    "baseline": baseline, "searched_max": searched, "admissible_untested": best,
    "threshold": threshold, "untested_direction": which,
    "search_any_flip_found": check["any_flip_found"],
    "search_status": check["classification_status"]["upper"],
    "is_a_certification": check["is_a_certification"],
    "what_the_old_implementation_would_have_returned": "certified true",
    "user_reported_counterexample": {
        "baseline": 0.3393281036, "searched_max": 0.3772730688,
        "admissible_untested": 0.3979955510, "threshold": 0.3876343099,
        "note": "independently constructed during review; reproduced here in structure on the "
                "frozen production psi block"},
}
dst = os.path.join(ROOT, "results", "stress_search_counterexample.json")
with open(dst, "w") as fh:
    json.dump(out, fh, indent=2)
print(f"  baseline            {baseline:.10f}")
print(f"  searched maximum    {searched:.10f}")
print(f"  threshold           {threshold:.10f}")
print(f"  admissible untested {best:.10f}   {which}")
print(f"  search reports any_flip_found={check['any_flip_found']}, "
      f"status={check['classification_status']['upper']}, "
      f"is_a_certification={check['is_a_certification']}")
print(f"  wrote {os.path.relpath(dst, ROOT)}")
