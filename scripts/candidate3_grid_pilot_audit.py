#!/usr/bin/env python3
"""Independent implementation audit for the Candidate 3 grid pilot freeze.

The audit deliberately reimplements the canonical payload and manifest instead of importing the
runner's freeze functions. It runs no transport, writes no artifact and never changes the spec.

    <python> -W error scripts/candidate3_grid_pilot_audit.py --expect draft
    <python> -W error scripts/candidate3_grid_pilot_audit.py --expect frozen
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
JAXPAD_DEFAULT = os.path.normpath(os.path.join(ROOT, "..", "jax-pad"))
SPEC = os.path.join(ROOT, "spec", "candidate3_grid_pilot.json")
RUNNER = os.path.join(ROOT, "scripts", "candidate3_grid_pilot.py")
ANALYSIS = os.path.join(ROOT, "scripts", "candidate3_grid_analysis.py")
PREFLIGHT = os.path.join(ROOT, "scripts", "candidate3_grid_preflight.py")
PREFLIGHT_SBATCH = os.path.join(ROOT, "scripts", "candidate3_grid_preflight_fir.sbatch")
PILOT_SBATCH = os.path.join(ROOT, "scripts", "candidate3_grid_pilot_fir.sbatch")
DETERMINISTIC_FLAG = "xla_gpu_deterministic_ops=true"


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def expected_paths(jaxpad_root):
    """Exact manifest, independently restated for audit."""
    return {
        "docs/planning/CANDIDATE3_GRID_PILOT_PLAN.md": os.path.join(
            ROOT, "docs", "planning", "CANDIDATE3_GRID_PILOT_PLAN.md"),
        "jaxpad:scripts/archive/inverse/multi_wl_v12_colab.py": os.path.join(
            jaxpad_root, "scripts", "archive", "inverse", "multi_wl_v12_colab.py"),
        "jaxpad:scripts/inverse/phase2_v12_runner.py": os.path.join(
            jaxpad_root, "scripts", "inverse", "phase2_v12_runner.py"),
        "jaxpad:src/pad/config_v4_633nm.py": os.path.join(
            jaxpad_root, "src", "pad", "config_v4_633nm.py"),
        "jaxpad:src/pad/mc_forward.py": os.path.join(
            jaxpad_root, "src", "pad", "mc_forward.py"),
        "jaxpad:src/pad/physics_engine_v4.py": os.path.join(
            jaxpad_root, "src", "pad", "physics_engine_v4.py"),
        "scripts/candidate3_executor.py": os.path.join(
            ROOT, "scripts", "candidate3_executor.py"),
        "scripts/candidate3_grid_analysis.py": ANALYSIS,
        "scripts/candidate3_grid_pilot.py": RUNNER,
        "scripts/candidate3_grid_pilot_audit.py": os.path.abspath(__file__),
        "scripts/candidate3_grid_preflight.py": PREFLIGHT,
        "scripts/candidate3_grid_preflight_fir.sbatch": PREFLIGHT_SBATCH,
        "scripts/candidate3_grid_pilot_fir.sbatch": PILOT_SBATCH,
        "scripts/nuisance_feasibility_gate.py": os.path.join(
            ROOT, "scripts", "nuisance_feasibility_gate.py"),
        "spec/baseline_spec.json": os.path.join(ROOT, "spec", "baseline_spec.json"),
        "spec/nuisance_spec.json": os.path.join(ROOT, "spec", "nuisance_spec.json"),
        "src/augmented.py": os.path.join(ROOT, "src", "augmented.py"),
    }


def independent_freeze_material(spec, jaxpad_root):
    paths = expected_paths(jaxpad_root)
    missing = [name for name, path in paths.items() if not os.path.isfile(path)]
    hashes = {name: sha256(path) for name, path in paths.items() if os.path.isfile(path)}
    payload = json.loads(json.dumps(spec))
    payload["frozen"] = True
    payload["status"] = "FROZEN NON-CONFIRMATORY GRID-IDENTIFICATION PILOT"
    payload["freeze_hash"] = None
    payload["normative_input_hashes"] = hashes
    digest = hashlib.sha256(b"candidate3-grid-pilot-v1\0" + canonical_bytes(payload)).hexdigest()
    return hashes, digest, missing


def module_scope_imports(path):
    with open(path) as fh:
        tree = ast.parse(fh.read(), filename=path)
    names = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    return names


def parser_options_without_importing_jax(path, module_name):
    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    imported = set(sys.modules) - before
    options = {o for action in module._build_parser()._actions for o in action.option_strings}
    return options, imported


def run_selftest(path, python):
    env = dict(os.environ)
    # Self-tests validate semantics, not persistent-cache deserialization.  FIR's JAX emits
    # PjRt-IFRT compatibility warnings whenever a cached executable is read; keep the strict
    # stderr-zero audit meaningful by compiling these tiny synthetic tests without that cache.
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    proc = subprocess.run([python, "-W", "error", path, "--selftest"], cwd=ROOT,
                          env=env, text=True, capture_output=True, check=False)
    return {"returncode": proc.returncode, "stdout": proc.stdout,
            "stderr": proc.stderr, "stderr_bytes": len(proc.stderr.encode("utf-8"))}


def audit(expect, jaxpad_root, python):
    design = load_json(SPEC)
    baseline = load_json(os.path.join(ROOT, "spec", "baseline_spec.json"))
    nuisance = load_json(os.path.join(ROOT, "spec", "nuisance_spec.json"))
    checks = []

    def ck(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    rd, sr = design.get("run_design", {}), design.get("selection_rule", {})
    expected_columns = [x["eta_name"] for x in nuisance["nuisance_dimensions"]]
    ck("schema and non-confirmatory purpose are explicit",
       design.get("schema_version") == "candidate3-grid-pilot-v1"
       and "non-confirmatory" in design.get("status", "").lower()
       and "without producing a candidate 3 verdict" in design.get("purpose", "").lower())
    ck("photon, replicate, wavelength and column design is exact",
       rd.get("n_photons_per_wavelength") == 200000
       and rd.get("n_replicates") == 6
       and rd.get("wavelengths_nm") == baseline["wavelengths_nm"]
       and rd.get("columns") == expected_columns)
    ep = design.get("execution_plan") or {}
    ck("execution partitions exactly one frozen replicate per array task",
       ep.get("array_indices") == list(range(6))
       and ep.get("replicates_per_task") == 1
       and ep.get("partial_shards_are_not_analyzable") is True)
    ck("candidate h set is exact and data independent",
       rd.get("h_candidates") == [0.5, 0.4, 0.3, 0.2, 0.1, 0.07])
    seeds = rd.get("replicate_seeds") or []
    ck("six uint32 seeds follow the frozen prime stride",
       len(seeds) == 6 and rd.get("seed_stride") == 104729
       and seeds == [3200000000 + 104729 * i for i in range(6)]
       and all(0 <= x < 2 ** 32 for x in seeds))
    ck("selection rule fixes four steps, both scales and common-grid budget",
       sr.get("subset_size") == 4 and sr.get("must_include_h_at_least") == 0.3
       and sr.get("must_include_h_at_most") == 0.1
       and sr.get("common_grid_ratio_budget") == 1.25
       and sr.get("tie_absolute_tolerance") == 1e-12)
    mm = design.get("measurement_model", {})
    bp = baseline["profiles"]["production"]
    ck("measurement model matches the committed production headline",
       mm.get("likelihood") == bp.get("likelihood") == "multinomial_fixed_launch"
       and mm.get("support") == "literal_all_120"
       and bp["min_count_rule"]["enabled"] is False
       and bp["n_observables"] == 120)
    rc = design.get("resource_cap", {})
    ck("resource and stopping limits are fixed",
       rc.get("host_memory_gb_per_task") == 24 and rc.get("shard_wall_hours") == 4
       and rc.get("maximum_concurrent_tasks") == 6
       and rc.get("aggregate_gpu_hours") == 24
       and "10 GB" in rc.get("gpu_per_task", "")
       and "no interim" in rd.get("stopping_rule", ""))
    pf = design.get("preflight", {})
    ck("target-GPU preflight probes, seed and MIG limits are fixed",
       pf.get("seed") == 3100000000
       and pf.get("probes") == [
           {"column": "log_d_papillary", "wavelength_index": 0},
           {"column": "mus_slope", "wavelength_index": 0}]
       and pf.get("required_device_kind_contains") == "MIG 1g.10gb"
       and pf.get("visible_gpu_count") == 1
       and "informational" in pf.get("nvidia_smi_memory_scope", "")
       and pf.get("projection_overhead_seconds_per_shard") == 300
       and pf.get("trials_are_not_pilot_trials") is True)
    forbidden = " ".join(design.get("forbidden") or []).lower()
    ck("verdict, coverage, wiring and trial reuse are forbidden",
       all(term in forbidden for term in ("verdict", "coverage", "run_gate", "reuse")))

    runner_imports = module_scope_imports(RUNNER)
    analysis_imports = module_scope_imports(ANALYSIS)
    preflight_imports = module_scope_imports(PREFLIGHT)
    ck("runner, analysis and preflight import no JAX at module scope",
       not any(x == "jax" or x.startswith("jax.")
               for x in runner_imports + analysis_imports + preflight_imports),
       f"runner={runner_imports}, analysis={analysis_imports}, preflight={preflight_imports}")
    runner_opts, runner_loaded = parser_options_without_importing_jax(
        RUNNER, "candidate3_grid_pilot_audit_runner")
    analysis_opts, analysis_loaded = parser_options_without_importing_jax(
        ANALYSIS, "candidate3_grid_pilot_audit_analysis")
    preflight_opts, preflight_loaded = parser_options_without_importing_jax(
        PREFLIGHT, "candidate3_grid_pilot_audit_preflight")
    ck("importing all three modules initializes no JAX module",
       not any(x == "jax" or x.startswith("jax.")
               for x in runner_loaded | analysis_loaded | preflight_loaded))
    ck("runner has no design override flags",
       not ({"--photons", "--replicates", "--seed", "--h-grid", "--spec"} & runner_opts))
    ck("analysis has no transport/run flag", "--run" not in analysis_opts
       and "--jaxpad-root" not in analysis_opts)
    ck("preflight has no design or output override flags",
       not ({"--photons", "--probe", "--seed", "--out", "--spec"} & preflight_opts))

    rt = run_selftest(RUNNER, python)
    at = run_selftest(ANALYSIS, python)
    pt = run_selftest(PREFLIGHT, python)
    ck("runner selftest passes with stderr zero",
       rt["returncode"] == 0 and rt["stderr_bytes"] == 0 and "15/15 checks passed" in rt["stdout"],
       f"rc={rt['returncode']} stderr={rt['stderr_bytes']}")
    ck("analysis selftest passes with stderr zero",
       at["returncode"] == 0 and at["stderr_bytes"] == 0 and "12/12 checks passed" in at["stdout"],
       f"rc={at['returncode']} stderr={at['stderr_bytes']}")
    ck("preflight selftest passes with stderr zero",
       pt["returncode"] == 0 and pt["stderr_bytes"] == 0 and "12/12 checks passed" in pt["stdout"],
       f"rc={pt['returncode']} stderr={pt['stderr_bytes']}")

    with open(PREFLIGHT_SBATCH) as fh:
        preflight_job = fh.read()
    with open(PILOT_SBATCH) as fh:
        pilot_job = fh.read()
    exact_resource = ("#SBATCH --mem=24G", "#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1")
    ck("both launchers request the exact frozen host and GPU resources",
       all(x in preflight_job and x in pilot_job for x in exact_resource)
       and "#SBATCH --time=04:00:00" in pilot_job
       and "#SBATCH --array=0-5" in pilot_job
       and "--run-shard" in pilot_job)
    ck("launchers set determinism and expose no design override",
       DETERMINISTIC_FLAG in preflight_job and DETERMINISTIC_FLAG in pilot_job
       and not any(x in preflight_job + pilot_job
                   for x in ("--photons", "--replicates", "--seed", "--h-grid", "--spec")))

    hashes, digest, missing = independent_freeze_material(design, jaxpad_root)
    ck("every exact-manifest input exists", not missing, str(missing))
    expected_manifest = set(expected_paths(jaxpad_root))
    ck("manifest key set is exact", set(hashes) == expected_manifest,
       f"n={len(hashes)}")
    if expect == "draft":
        ck("draft is deliberately unable to run",
           design.get("frozen") is False and design.get("freeze_hash") is None
           and design.get("normative_input_hashes") == {})
    else:
        ck("frozen flag, exact manifest and freeze hash all match independently",
           design.get("frozen") is True
           and design.get("normative_input_hashes") == hashes
           and design.get("freeze_hash") == digest,
           f"recorded={design.get('freeze_hash')} computed={digest}")

    failed = [x for x in checks if not x["pass"]]
    return {"schema_version": "candidate3-grid-pilot-freeze-audit-v1",
            "expect": expect, "passes": not failed, "checks": checks,
            "n_checks": len(checks), "n_failed": len(failed),
            "computed_freeze_hash": digest, "computed_input_hashes": hashes,
            "transport_run": False, "candidate3_verdict": None,
            "coverage_claimed": False}


def _build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expect", choices=("draft", "frozen"), required=True)
    ap.add_argument("--jaxpad-root", default=os.environ.get("JAXPAD_ROOT", JAXPAD_DEFAULT))
    ap.add_argument("--python", default=sys.executable)
    return ap


def main():
    a = _build_parser().parse_args()
    result = audit(a.expect, os.path.abspath(a.jaxpad_root), os.path.abspath(a.python))
    for item in result["checks"]:
        print(f"  {'PASS' if item['pass'] else 'FAIL'}  {item['check']}"
              + (f"  [{item['detail']}]" if item["detail"] else ""))
    print(f"  {result['n_checks'] - result['n_failed']}/{result['n_checks']} checks passed")
    print(f"  computed freeze hash: {result['computed_freeze_hash']}")
    print("  transport_run=false; candidate3_verdict=null; coverage_claimed=false")
    return 0 if result["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
