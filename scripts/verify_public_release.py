#!/usr/bin/env python3
"""Validate the clean public package and selected reproduced headline values."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]

REQUIRED = (
    "README.md",
    "CITATION.cff",
    "LICENSE",
    "LICENSE-DATA",
    "PROVENANCE.md",
    "requirements-lock.txt",
    "reproduce.sh",
    "spec/baseline_spec.json",
    "data/derived/production_pooled.npz",
    "results/production_crb.json",
    "results/production_stream_loo.json",
    "results/production_pmc_jacobian.json",
)

FORBIDDEN_PARTS = {
    "logs",
    "tmp",
    "_to_delete",
    "iop_submission",
    "ioplatextemplate",
}

FORBIDDEN_NAMES = {
    "cover_letter.pdf",
    "cover_letter.tex",
}

LOCAL_EXCLUDED_PARTS = {
    ".git",
    ".idea",
    ".venv",
    "venv",
    "docs",
    "__pycache__",
    "reproduced",
    "tmp",
    "logs",
    "_to_delete",
}

TRANSPORT_HASHES = {
    "src/pad/mc_forward.py":
        "9e1236fa90da223d56788ed2bccdd1e32eccb787788f03a4f0e5ebca47bf4b0f",
    "src/pad/physics_engine_v4.py":
        "4616e22226a015645f15c0dfa8582fab73a3c18d4bb6b1ff26bb7d54bae77f68",
}

TEXT_SUFFIXES = {
    ".cff", ".json", ".md", ".py", ".sh", ".sbatch", ".tex", ".tsv", ".txt", ".yml", ".yaml"
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def is_local_excluded(path: Path) -> bool:
    """Return whether a worktree path is local-only and absent from the release manifest."""
    return any(part in LOCAL_EXCLUDED_PARTS for part in path.relative_to(ROOT).parts)


def check_manifest() -> list[str]:
    manifest = ROOT / "MANIFEST.sha256"
    if not manifest.is_file():
        return ["MANIFEST.sha256 is missing"]
    errors: list[str] = []
    for number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            expected, relative = raw.split("  ", 1)
        except ValueError:
            errors.append(f"malformed manifest line {number}")
            continue
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"manifest file missing: {relative}")
        elif sha256(path) != expected:
            errors.append(f"manifest hash mismatch: {relative}")
    return errors


def write_manifest() -> None:
    entries: list[tuple[str, str]] = []
    for path in ROOT.rglob("*"):
        if (not path.is_file() or is_local_excluded(path) or path.name == "MANIFEST.sha256"):
            continue
        entries.append((path.relative_to(ROOT).as_posix(), sha256(path)))
    entries.sort()
    text = "".join(f"{digest}  {relative}\n" for relative, digest in entries)
    (ROOT / "MANIFEST.sha256").write_text(text, encoding="utf-8")


def public_boundary_errors() -> list[str]:
    errors: list[str] = []
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"required file missing: {relative}")

    for path in ROOT.rglob("*"):
        if is_local_excluded(path):
            continue
        relative = path.relative_to(ROOT)
        if any(part in FORBIDDEN_PARTS for part in relative.parts):
            errors.append(f"forbidden internal path present: {relative}")
        if path.name.lower() in FORBIDDEN_NAMES:
            errors.append(f"submission correspondence present: {relative}")
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if relative != Path("scripts/verify_public_release.py") and (
            "PRIVATE KEY-----" in text or re.search(
                r"(?i)(api[_-]?key|access[_-]?token|password)\s*[:=]\s*['\"][^'\"]+",
                text,
            )
        ):
            errors.append(f"possible credential material: {relative}")

    for relative, expected in TRANSPORT_HASHES.items():
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"transport source missing: {relative}")
        elif sha256(path) != expected:
            errors.append(f"transport source hash mismatch: {relative}")

    for path in ROOT.rglob("*.json"):
        if is_local_excluded(path):
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # fail closed on any malformed released JSON
            errors.append(f"invalid JSON {path.relative_to(ROOT)}: {exc}")

    for subdir in ("production_cuda129", "production_pmc"):
        count = len(list((ROOT / "data" / "derived" / subdir).glob("seed_*.npz")))
        if count != 32:
            errors.append(f"expected 32 streams in {subdir}, found {count}")
    return errors


def pointer(document: object, keys: tuple[str, ...]) -> object:
    value = document
    for key in keys:
        value = value[key]  # type: ignore[index]
    return value


def reproduced_errors(root: Path) -> list[str]:
    comparisons = (
        (
            "headline deep-StO2 sqrt(CRLB)",
            ROOT / "results/production_crb.json",
            root / "results/production_crb.json",
            ("branches", "multinomial_fixed_launch", "crlb", "sqrt_crlb_psi", "StO2_D"),
        ),
        (
            "LOO minimum headline bound",
            ROOT / "results/production_stream_loo.json",
            root / "results/production_stream_loo.json",
            ("leave_one_stream_out_summary", "multinomial_fixed_launch", "deep_StO2_sqrt_crlb", "minimum"),
        ),
        (
            "LOO maximum headline bound",
            ROOT / "results/production_stream_loo.json",
            root / "results/production_stream_loo.json",
            ("leave_one_stream_out_summary", "multinomial_fixed_launch", "deep_StO2_sqrt_crlb", "maximum"),
        ),
    )
    errors: list[str] = []
    for label, frozen_path, new_path, keys in comparisons:
        frozen = pointer(json.loads(frozen_path.read_text()), keys)
        reproduced = pointer(json.loads(new_path.read_text()), keys)
        scale = max(1.0, abs(float(frozen)))
        if abs(float(frozen) - float(reproduced)) > 1e-12 * scale:
            errors.append(f"{label} differs: frozen={frozen!r}, reproduced={reproduced!r}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--reproduced", type=Path)
    args = parser.parse_args()

    if args.write_manifest:
        write_manifest()
        print("wrote MANIFEST.sha256")
        return 0

    errors = check_manifest()
    if not args.manifest_only:
        errors.extend(public_boundary_errors())
        if args.reproduced is not None:
            errors.extend(reproduced_errors(args.reproduced.resolve()))

    if errors:
        for error in errors:
            print(f"FAIL  {error}")
        return 1

    print("PASS  release manifest")
    if not args.manifest_only:
        print("PASS  public-package boundary and credential scan")
        print("PASS  frozen transport source hashes")
        print("PASS  all released JSON artifacts parse")
        print("PASS  32 AD streams and 32 replay-pMC streams")
        if args.reproduced is not None:
            print("PASS  reproduced primary and leave-one-stream-out headlines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
