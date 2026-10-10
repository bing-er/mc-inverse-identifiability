#!/usr/bin/env python3
"""Record what the replay equivalence gate was run against, so the result is traceable."""
import hashlib, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JP = os.environ.get("JAXPAD_ROOT", os.path.normpath(os.path.join(ROOT, "..", "jax-pad")))


def h(p):
    d = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            d.update(c)
    return d.hexdigest()


def main():
    name = (sys.argv[1] if len(sys.argv) > 1 else "replay_equivalence.json")
    gate = os.path.join(ROOT, "results", name)
    if not os.path.exists(gate):
        sys.exit("the gate has not been run")
    g = json.load(open(gate))
    prod = {
        "src/pad/physics_engine_v4.py": os.path.join(JP, "src/pad/physics_engine_v4.py"),
        "src/pad/mc_forward.py": os.path.join(JP, "src/pad/mc_forward.py"),
        "scripts/inverse/phase2_v12_runner.py": os.path.join(
            JP, "scripts/inverse/phase2_v12_runner.py"),
        "scripts/archive/inverse/multi_wl_v12_colab.py": os.path.join(
            JP, "scripts/archive/inverse/multi_wl_v12_colab.py"),
    }
    pv = {
        "analysis": "photon-level equivalence of the replay kernel against the production kernel",
        "date": "2026-10-01",
        "verdict": g["verdict"],
        "all_passed": g["summary"]["all_passed"],
        "negative_control_rejected": g["summary"]["negative_control_rejected"],
        "production_kernel_is_the_only_physical_truth": True,
        "production_physics_unmodified":
            "nothing under the production tree was edited; the replay module imports the Fresnel "
            "reflectance, the Henyey-Greenstein sampler, the direction update and the termination "
            "constants from it rather than copying them",
        "other_projects_replay_kernels_not_used_as_truth":
            "P5's replay kernel was instrumented against a different production kernel. The method "
            "was borrowed; the implementation was written against this kernel and verified here.",
        "production_source_hashes": {k: h(v) for k, v in prod.items() if os.path.exists(v)},
        "replay_source_hash": {"src/replay_kernel.py": h(os.path.join(ROOT, "src/replay_kernel.py"))},
        "gate_script_hash": {"scripts/replay_equivalence.py":
                             h(os.path.join(ROOT, "scripts/replay_equivalence.py"))},
        "spec_hashes": {
            "spec/baseline_spec.json": h(os.path.join(ROOT, "spec/baseline_spec.json")),
            "spec/validation_spec.json": h(os.path.join(ROOT, "spec/validation_spec.json"))},
        "output_hashes": {f"results/{name}": h(gate)},
        "gate_backend": g["environment"]["jax_backend"],
        "backend_note": "bitwise agreement under one XLA backend does not carry over to another, "
                        "so a pilot must run against a gate taken on its own backend",
        "criteria": g["criteria"],
        "design": g["design"],
        "summary": g["summary"],
        "negative_control": g["negative_control"],
        "environment": g["environment"],
        "trajectory_invariants_held_fixed": g["trajectory_invariants_held_fixed"],
        "consequence_for_the_pilot":
            ("the pMC route is unblocked and the pilot may include it"
             if g["summary"]["all_passed"] else
             "the pMC route stays blocked; no pilot may report a pMC number"),
    }
    dst = os.path.join(ROOT, "results",
                       name.replace(".json", "_provenance.json"))
    with open(dst, "w") as fh:
        json.dump(pv, fh, indent=2)
    print(f"  wrote {os.path.relpath(dst, ROOT)}  sha256 {h(dst)}")
    print(f"  gate output sha256 {pv['output_hashes'][f'results/{name}']}  "
          f"backend {pv['gate_backend']}")
    for k, v in pv["production_source_hashes"].items():
        print(f"    production {k}  {v[:16]}...")
    print(f"    replay     src/replay_kernel.py  "
          f"{pv['replay_source_hash']['src/replay_kernel.py'][:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
