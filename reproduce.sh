#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
MODE="${1:-verify}"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export JAXPAD_ROOT="$ROOT"

case "$MODE" in
  verify)
    "$PYTHON" -W error "$ROOT/scripts/verify_public_release.py"
    ;;

  manifest)
    "$PYTHON" -W error "$ROOT/scripts/verify_public_release.py" --manifest-only
    ;;

  analysis)
    OUT="$ROOT/reproduced"
    mkdir -p "$OUT/data" "$OUT/results"
    "$PYTHON" -W error "$ROOT/scripts/check_pool.py" \
      --pool "$ROOT/data/derived/production_cuda129" \
      --out "$OUT/data/production_pooled.npz" \
      --report "$OUT/results/pool_check.json"
    "$PYTHON" -W error "$ROOT/scripts/production_crb.py" \
      --pooled "$OUT/data/production_pooled.npz" \
      --spectrum "$ROOT/results/unify_spectrum.json" \
      --out "$OUT/results/production_crb.json"
    "$PYTHON" -W error "$ROOT/scripts/production_stream_loo.py" \
      --streams "$ROOT/data/derived/production_cuda129" \
      --pooled "$OUT/data/production_pooled.npz" \
      --crlb "$OUT/results/production_crb.json" \
      --out "$OUT/results/production_stream_loo.json"
    "$PYTHON" -W error "$ROOT/scripts/verify_public_release.py" \
      --reproduced "$OUT"
    ;;

  figures)
    "$PYTHON" -W error "$ROOT/scripts/manuscript_figures.py"
    ;;

  *)
    echo "usage: $0 {verify|manifest|analysis|figures}" >&2
    exit 2
    ;;
esac
