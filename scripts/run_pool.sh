#!/usr/bin/env bash
# run_pool.sh -- build the production transport pool, one seed per process, resumable.
#
# Deliberately scheduler agnostic. Each seed is an independent process writing its own file, so this
# same script drives a laptop, a workstation with -j, or an array job where the scheduler sets the
# task index. Nothing is shared between seeds, so a partial pool is always a valid partial pool and
# a rerun costs only the seeds that are missing.
#
#   bash scripts/run_pool.sh                      # every missing seed, sequentially
#   bash scripts/run_pool.sh -j 4                 # four at a time
#   bash scripts/run_pool.sh --seeds 44000,45000  # a named subset
#   bash scripts/run_pool.sh --list               # what is present and what is missing
#
# For an array job, give the scheduler's task index and let it pick the seed:
#   bash scripts/run_pool.sh --index "$TASK_INDEX"
#
# PYTHON should point at the pinned interpreter; the system one will not do.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
PY="${PYTHON:-python3}"
OUTDIR="$ROOT/data/derived/production"

SEED_BASE=43000
SEED_STRIDE=1000
N_SEEDS=32

all_seeds() { for i in $(seq 0 $((N_SEEDS - 1))); do echo $((SEED_BASE + i * SEED_STRIDE)); done; }

JOBS=1; SEEDS=""; INDEX=""; LIST=0
while [ $# -gt 0 ]; do
  case "$1" in
    -j|--jobs)  JOBS="$2"; shift 2 ;;
    --seeds)    SEEDS="$(echo "$2" | tr ',' ' ')"; shift 2 ;;
    --index)    INDEX="$2"; shift 2 ;;
    --list)     LIST=1; shift ;;
    -h|--help)  sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ "$LIST" -eq 1 ]; then
  present=0; missing=0
  for s in $(all_seeds); do
    if [ -f "$OUTDIR/seed_$s.npz" ]; then present=$((present+1)); else
      missing=$((missing+1)); echo "  missing $s"; fi
  done
  echo "  $present of $N_SEEDS present, $missing missing"
  exit 0
fi

if [ -n "$INDEX" ]; then
  if [ "$INDEX" -lt 0 ] || [ "$INDEX" -ge "$N_SEEDS" ]; then
    echo "index $INDEX outside 0..$((N_SEEDS - 1))" >&2; exit 2
  fi
  SEEDS=$((SEED_BASE + INDEX * SEED_STRIDE))
fi
[ -z "$SEEDS" ] && SEEDS="$(all_seeds)"

todo=""
for s in $SEEDS; do
  [ -f "$OUTDIR/seed_$s.npz" ] && echo "  seed $s already present, skipping" || todo="$todo $s"
done
[ -z "$todo" ] && { echo "  nothing to do"; exit 0; }

echo "  running:$todo  (jobs=$JOBS, PYTHON=$PY)"
fail=0
if [ "$JOBS" -le 1 ]; then
  for s in $todo; do
    $PY scripts/production_collect.py --seed "$s" || { echo "  seed $s FAILED" >&2; fail=1; }
  done
else
  running=0
  for s in $todo; do
    $PY scripts/production_collect.py --seed "$s" &
    running=$((running+1))
    if [ "$running" -ge "$JOBS" ]; then wait -n 2>/dev/null || wait; running=$((running-1)); fi
  done
  wait || fail=1
fi

present=0
for s in $(all_seeds); do [ -f "$OUTDIR/seed_$s.npz" ] && present=$((present+1)); done
echo "  pool: $present of $N_SEEDS seeds present"
exit "$fail"
