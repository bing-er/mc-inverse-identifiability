#!/usr/bin/env python3
"""Target-GPU preflight for the frozen Candidate 3 grid-identification pilot.

This is not the pilot. It runs fixed, non-reusable probes to verify the production accumulator,
repeat determinism, target H100 MIG allocation and the four-hour shard projection. It cannot select a
grid, claim coverage or emit a Candidate 3 verdict.

    <python> -W error scripts/candidate3_grid_preflight.py --selftest
    <python> -W error scripts/candidate3_grid_preflight.py --run --jaxpad-root PATH
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SPEC = os.path.join(ROOT, "spec", "candidate3_grid_pilot.json")
PILOT = os.path.join(ROOT, "scripts", "candidate3_grid_pilot.py")
ARTIFACT = os.path.join(ROOT, "results", "candidate3_grid_pilot_preflight.json")
DETERMINISTIC_FLAG = "xla_gpu_deterministic_ops=true"
ENVIRONMENT_KEYS = ("backend", "jax_version", "device_kind", "ptxas_version", "cuda_module",
                    "gpu_name", "gpu_memory_total_mib", "visible_gpu_count")
OUTPUT_KEYS = ("sum_plus", "sum_minus", "sum_Y", "sum_Y2", "max_abs_Y")


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def array_digest(value):
    a = np.ascontiguousarray(np.asarray(value))
    h = hashlib.sha256()
    h.update(str(a.dtype).encode("ascii"))
    h.update(str(a.shape).encode("ascii"))
    h.update(a.tobytes())
    return h.hexdigest()


def _nvidia_smi():
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
             "--format=csv,noheader,nounits"], text=True, capture_output=True,
            check=False, timeout=20)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc), "gpus": []}
    if proc.returncode != 0:
        return {"error": proc.stderr.strip() or f"exit {proc.returncode}", "gpus": []}
    rows = []
    for line in proc.stdout.splitlines():
        fields = [x.strip() for x in line.split(",")]
        if len(fields) != 3:
            return {"error": f"unexpected nvidia-smi row: {line!r}", "gpus": []}
        try:
            rows.append({"name": fields[0], "memory_total_mib": int(fields[1]),
                         "memory_used_mib": int(fields[2])})
        except ValueError:
            return {"error": f"non-integer nvidia-smi memory: {line!r}", "gpus": []}
    return {"error": None, "gpus": rows}


def current_environment():
    import jax

    devices = jax.devices()
    smi = _nvidia_smi()
    first = smi["gpus"][0] if len(smi["gpus"]) == 1 else {}
    return {
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "device_kind": getattr(devices[0], "device_kind", None) if devices else None,
        "ptxas_version": os.environ.get("MCII_PTXAS_VERSION"),
        "cuda_module": os.environ.get("MCII_CUDA_MODULE"),
        "gpu_name": first.get("name"),
        "gpu_memory_total_mib": first.get("memory_total_mib"),
        "gpu_memory_used_mib": first.get("memory_used_mib"),
        "visible_gpu_count": len(smi["gpus"]),
        "nvidia_smi_error": smi["error"],
        "deterministic_flag_set": DETERMINISTIC_FLAG in os.environ.get("XLA_FLAGS", ""),
        "xla_python_client_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
    }


def environment_passes(env, design):
    pf = design["preflight"]
    name = str(env.get("gpu_name") or env.get("device_kind") or "")
    device_kind = str(env.get("device_kind") or "")
    return (env.get("backend") == "gpu"
            and pf["required_gpu_name_contains"].lower() in name.lower()
            and pf["required_device_kind_contains"].lower() in device_kind.lower()
            and env.get("visible_gpu_count") == pf["visible_gpu_count"]
            and env.get("nvidia_smi_error") is None
            and env.get("deterministic_flag_set") is True
            and env.get("xla_python_client_preallocate") == "false"
            and bool(env.get("ptxas_version")) and bool(env.get("cuda_module")))


def preflight_status(design, jaxpad_root, freeze_status_fn, *, artifact_path=ARTIFACT,
                     environment=None):
    freeze = freeze_status_fn(design, jaxpad_root)
    problems = []
    if not freeze["frozen"]:
        problems.append("the pilot design is not currently frozen and hash-bound")
    if not os.path.isfile(artifact_path):
        return {"passes": False, "problems": problems + ["no target-GPU preflight artifact"],
                "artifact": None, "freeze": freeze}
    try:
        artifact = load_json(artifact_path)
    except (OSError, ValueError) as exc:
        return {"passes": False, "problems": problems + [f"unreadable preflight: {exc}"],
                "artifact": None, "freeze": freeze}
    if artifact.get("schema_version") != "candidate3-grid-pilot-preflight-v1":
        problems.append("preflight schema version differs")
    if artifact.get("passes") is not True:
        problems.append("preflight did not pass")
    if artifact.get("freeze_hash") != design.get("freeze_hash"):
        problems.append("preflight freeze hash differs")
    if artifact.get("normative_input_hashes") != design.get("normative_input_hashes"):
        problems.append("preflight normative input hashes differ")
    if artifact.get("pilot_transport_run") is not False:
        problems.append("preflight incorrectly claims a pilot transport run")
    if artifact.get("candidate3_verdict") is not None or artifact.get("coverage_claimed") is not False:
        problems.append("preflight contains a forbidden verdict or coverage claim")
    if artifact.get("projected_shard_wall_seconds", float("inf")) > design["resource_cap"][
            "shard_wall_hours"] * 3600:
        problems.append("preflight projection exceeds the frozen per-shard wall-time cap")
    if artifact.get("repeat_bitwise_identical") is not True:
        problems.append("preflight repeat-determinism check failed")
    if artifact.get("preflight_seed") != design["preflight"]["seed"]:
        problems.append("preflight seed differs from the frozen seed")
    if artifact.get("preflight_seed_reused_by_pilot") is not False:
        problems.append("preflight does not prove its seed is excluded from the pilot")
    if artifact.get("preflight_transport_run") is not True:
        problems.append("preflight did not execute its declared production-shape probes")
    expected_calls = (design["run_design"]["n_replicates"]
                      * len(design["run_design"]["columns"])
                      * len(design["run_design"]["wavelengths_nm"]))
    if artifact.get("production_call_count") != expected_calls:
        problems.append("preflight projection uses the wrong production call count")
    if artifact.get("production_call_count_per_shard") != expected_calls // design[
            "run_design"]["n_replicates"]:
        problems.append("preflight projection uses the wrong per-shard production call count")
    if artifact.get("environment_stable") is not True:
        problems.append("preflight environment changed during its probes")
    records = artifact.get("probe_records") or []
    declared = design["preflight"]["probes"]
    sequence = [(declared[0]["column"], declared[0]["wavelength_index"]),
                (declared[0]["column"], declared[0]["wavelength_index"]),
                (declared[1]["column"], declared[1]["wavelength_index"])]
    actual = [(x.get("column"), x.get("wavelength_index")) for x in records]
    if actual != sequence:
        problems.append("preflight probe sequence differs from the frozen sequence")
    if (len(records) != 3
            or any(set(x.get("output_digests") or {}) != set(OUTPUT_KEYS) for x in records)
            or records[0].get("output_digests") != records[1].get("output_digests")):
        problems.append("preflight output digests do not prove a bitwise repeated probe")
    recorded = artifact.get("environment") or {}
    if not environment_passes(recorded, design):
        problems.append("recorded target-GPU environment violates the frozen resource contract")
    now = current_environment() if environment is None else environment
    for key in ENVIRONMENT_KEYS:
        if recorded.get(key) != now.get(key):
            problems.append(f"current environment differs from preflight for {key}")
    if not environment_passes(now, design):
        problems.append("current environment violates the frozen resource contract")
    return {"passes": not problems, "problems": problems, "artifact": artifact,
            "freeze": freeze, "environment_now": now}


def require_preflight_passed(design, jaxpad_root, freeze_status_fn):
    status = preflight_status(design, jaxpad_root, freeze_status_fn)
    if not status["passes"]:
        raise SystemExit("REFUSING to run pilot: target-GPU preflight is absent or stale: "
                         + "; ".join(status["problems"]))
    return status["artifact"]


def run_preflight(design, jaxpad_root, pilot):
    freeze = pilot.freeze_status(design, jaxpad_root)
    if not freeze["frozen"]:
        raise SystemExit("REFUSING preflight: pilot design is not frozen and hash-bound: "
                         + "; ".join(freeze["problems"]))
    if DETERMINISTIC_FLAG not in os.environ.get("XLA_FLAGS", ""):
        raise SystemExit(f"REFUSING preflight: XLA_FLAGS lacks {DETERMINISTIC_FLAG!r}")
    nspec = load_json(pilot.NUISANCE_SPEC)
    bspec = load_json(pilot.BASELINE_SPEC)
    executor = pilot.build_pilot_executor(jaxpad_root, design, nspec, bspec)
    before = current_environment()
    if not environment_passes(before, design):
        raise SystemExit("REFUSING preflight: target environment violates the frozen H100 MIG "
                         f"contract: {before}")

    pf = design["preflight"]
    keys = executor["split"](int(pf["seed"]))
    probes = pf["probes"]
    records, outputs = [], []
    sequence = [probes[0], probes[0], probes[1]]
    for index, probe in enumerate(sequence):
        column = design["run_design"]["columns"].index(probe["column"])
        result, seconds, parameter_digest = executor["accumulate"](
            column, int(probe["wavelength_index"]), keys)
        outputs.append(result)
        records.append({"sequence_index": index, "column": probe["column"],
                        "wavelength_index": int(probe["wavelength_index"]),
                        "wall_seconds": float(seconds),
                        "parameter_digest": parameter_digest,
                        "output_digests": {k: array_digest(v) for k, v in result.items()}})

    repeated = all(np.array_equal(outputs[0][k], outputs[1][k]) for k in outputs[0])
    n_calls = (design["run_design"]["n_replicates"]
               * len(design["run_design"]["columns"])
               * len(design["run_design"]["wavelengths_nm"]))
    compile_and_first = records[0]["wall_seconds"]
    cached_upper = max(records[1]["wall_seconds"], records[2]["wall_seconds"])
    calls_per_shard = n_calls // design["run_design"]["n_replicates"]
    projected = (float(pf["projection_overhead_seconds_per_shard"]) + compile_and_first
                 + (calls_per_shard - 1) * cached_upper)
    after = current_environment()
    environment_stable = all(before.get(k) == after.get(k) for k in ENVIRONMENT_KEYS)
    passes = (repeated and environment_stable and environment_passes(after, design)
              and projected <= design["resource_cap"]["shard_wall_hours"] * 3600)
    return {
        "schema_version": "candidate3-grid-pilot-preflight-v1",
        "status": "TARGET-GPU PREFLIGHT; NOT THE GRID PILOT",
        "passes": bool(passes),
        "freeze_hash": design["freeze_hash"],
        "normative_input_hashes": design["normative_input_hashes"],
        "environment": after,
        "preflight_seed": int(pf["seed"]),
        "preflight_seed_reused_by_pilot": False,
        "probe_records": records,
        "repeat_bitwise_identical": bool(repeated),
        "environment_stable": bool(environment_stable),
        "production_call_count": int(n_calls),
        "production_call_count_per_shard": int(calls_per_shard),
        "projected_shard_wall_seconds": float(projected),
        "projection_formula": "300 s overhead + first compile+call + 19 * max(two cached calls)",
        "pilot_transport_run": False,
        "preflight_transport_run": True,
        "candidate3_verdict": None,
        "coverage_claimed": False,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _load_pilot():
    spec = importlib.util.spec_from_file_location("candidate3_grid_pilot_for_preflight", PILOT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selftest():
    design = load_json(SPEC)
    checks = []

    def ck(name, ok):
        checks.append((name, bool(ok)))

    env = {"backend": "gpu", "jax_version": "test",
           "device_kind": "NVIDIA H100 80GB HBM3 MIG 1g.10gb",
           "ptxas_version": "12.9.1", "cuda_module": "cuda/12.9",
           "gpu_name": "NVIDIA H100 80GB HBM3 MIG 1g.10gb",
           "gpu_memory_total_mib": 81559, "gpu_memory_used_mib": 9620,
           "visible_gpu_count": 1, "nvidia_smi_error": None,
           "deterministic_flag_set": True, "xla_python_client_preallocate": "false"}
    artifact = {"schema_version": "candidate3-grid-pilot-preflight-v1", "passes": True,
                "freeze_hash": design["freeze_hash"],
                "normative_input_hashes": design["normative_input_hashes"],
                "environment": env, "projected_shard_wall_seconds": 3600,
                "repeat_bitwise_identical": True, "pilot_transport_run": False,
                "preflight_transport_run": True, "preflight_seed": 3100000000,
                "preflight_seed_reused_by_pilot": False, "environment_stable": True,
                "production_call_count": 120,
                "production_call_count_per_shard": 20,
                "probe_records": [
                    {"column": "log_d_papillary", "wavelength_index": 0,
                     "output_digests": {k: "a" for k in OUTPUT_KEYS}},
                    {"column": "log_d_papillary", "wavelength_index": 0,
                     "output_digests": {k: "a" for k in OUTPUT_KEYS}},
                    {"column": "mus_slope", "wavelength_index": 0,
                     "output_digests": {k: "b" for k in OUTPUT_KEYS}}],
                "candidate3_verdict": None, "coverage_claimed": False}

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "preflight.json")

        def status_for(value):
            with open(path, "w") as fh:
                json.dump(value, fh)
            return preflight_status(
                design, "/unused", lambda d, r: {"frozen": True, "problems": []},
                artifact_path=path, environment=env)

        ck("a complete matching preflight passes", status_for(artifact)["passes"])
        for field, value, label in (
                ("freeze_hash", "0" * 64, "stale freeze hash"),
                ("normative_input_hashes", {}, "stale input hashes"),
                ("passes", False, "failed preflight"),
                ("repeat_bitwise_identical", False, "non-deterministic repeat"),
                ("pilot_transport_run", True, "pilot-run claim"),
                ("candidate3_verdict", "equivalent", "verdict"),
                ("coverage_claimed", True, "coverage claim"),
                ("projected_shard_wall_seconds", 14401, "over-cap projection")):
            bad = json.loads(json.dumps(artifact)); bad[field] = value
            ck(f"a {label} is refused", not status_for(bad)["passes"])
        bad = json.loads(json.dumps(artifact))
        bad["environment"]["device_kind"] = "NVIDIA H100 80GB HBM3"
        ck("a full H100 without the MIG 1g.10gb identity is refused",
           not status_for(bad)["passes"])
        status_for(artifact)
        changed = dict(env); changed["jax_version"] = "other"
        ck("a current environment different from the preflight is refused",
           not preflight_status(
               design, "/unused", lambda d, r: {"frozen": True, "problems": []},
               artifact_path=path, environment=changed)["passes"])
    options = {o for a in _build_parser()._actions for o in a.option_strings}
    ck("the command line cannot override photons, probes, seed or output",
       not ({"--photons", "--probe", "--seed", "--out", "--spec"} & options))
    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"  {len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


def _build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--jaxpad-root", default=os.environ.get(
        "JAXPAD_ROOT", os.path.normpath(os.path.join(ROOT, "..", "jax-pad"))))
    return ap


def main():
    args = _build_parser().parse_args()
    if sum((args.selftest, args.run)) != 1:
        raise SystemExit("choose exactly one of --selftest or --run")
    if args.selftest:
        return selftest()
    if os.path.exists(ARTIFACT):
        raise SystemExit(f"REFUSING to overwrite the existing preflight artifact: {ARTIFACT}")
    design, pilot = load_json(SPEC), _load_pilot()
    result = run_preflight(design, os.path.abspath(args.jaxpad_root), pilot)
    os.makedirs(os.path.dirname(ARTIFACT), exist_ok=True)
    with open(ARTIFACT, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"  wrote {os.path.relpath(ARTIFACT, ROOT)}")
    print(f"  projected shard wall time: {result['projected_shard_wall_seconds'] / 3600:.3f} h")
    print("  PREFLIGHT ONLY; pilot_transport_run=false; no verdict or coverage claim")
    return 0 if result["passes"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
