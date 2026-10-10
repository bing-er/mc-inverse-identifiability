#!/bin/bash
#SBATCH --job-name=mcii_ngate
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --output=logs/mcii_ngate_%j.out
#SBATCH --error=logs/mcii_ngate_%j.err
#
# MERGE_CHECKS item 2: the NOMINAL feasibility gate. Zero draws. This is not the nuisance study.
#
# It will refuse to start unless all of the following hold, and none of them can be overridden:
#   - the nuisance citation gate is open, which needs every dimension to satisfy its prior_kind
#     policy AND an explicit release decision in the spec
#   - the deterministic XLA flag is set in THIS process, checked here rather than trusted from a
#     preflight record
#   - the real-executor preflight artifact passes AND every bound hash still matches the files on
#     disk, covering the nuisance runner, the specs, src/augmented.py and the jax-pad runner,
#     forward solver, physics engine and config resolved from the modules actually imported
#
# The release covers the feasibility gate ONLY: per-column AD-versus-ensemble equivalence,
# finite-difference convergence, and estimator-route selection with cost measurement. It does not
# authorize or establish any downstream scientific classification, and it does not authorize the
# 48-draw nuisance study, which stays a separate decision and is NOT launched from here.
#
# Twelve hours requested against a 9.5 hour projection, because the per-evaluation cost is an
# estimate from the transport pool and the five-column eta jacfwd has never been timed here.

set -euo pipefail
cd "$HOME/scratch/mc-inverse-identifiability"
mkdir -p logs results

module load python/3.11 scipy-stack/2025a cuda/12.9
if   [ -d "$HOME/scratch/venvs/jaxpad" ]; then source "$HOME/scratch/venvs/jaxpad/bin/activate"
elif [ -d "$HOME/.venvs/jaxpad-fir"    ]; then source "$HOME/.venvs/jaxpad-fir/bin/activate"
else echo "ERROR: no jaxpad venv found"; exit 1; fi

export PYTHONPATH="$HOME/scratch/jax-pad:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="$HOME/scratch/jax-pad/jax_cache_cuda129"
export JAXPAD_ROOT="$HOME/scratch/jax-pad"
export MCII_CUDA_MODULE="cuda/12.9"
# Required, not a preference, and the gate checks for it in its own process. Measured on this
# hardware the GPU scatter-add is non-deterministic at a relative 1.3e-15 without it, so repeated
# calls with the same photon keys do not agree bitwise and the common-random-number cancellation
# this method depends on is not exact.
export XLA_FLAGS="--xla_gpu_deterministic_ops=true ${XLA_FLAGS:-}"

echo "=== nominal feasibility gate on $(hostname) ==="
which ptxas; ptxas --version
python -c "import jax, scipy; print('JAX', jax.__version__, '| backend', jax.default_backend(), jax.devices(), '| scipy', scipy.__version__)"

PTXAS_VER="$(ptxas --version | grep -oE 'V[0-9]+\.[0-9]+\.[0-9]+' | tr -d V)"
case "$PTXAS_VER" in
  12.[0-5].*|12.6.[0-2]|11.*) echo "REFUSING: ptxas $PTXAS_VER predates the sm90 clamping fix"; exit 1 ;;
esac
echo "ptxas $PTXAS_VER accepted"
export MCII_PTXAS_VERSION="$PTXAS_VER"

python -u scripts/nuisance_feasibility_gate.py --selftest
python -u scripts/nuisance_feasibility_gate.py --dry-run
# No --replicates or --photons: those flags were removed on purpose so the run design can only
# come from the frozen spec. Passing them would be an unrecognized argument and the job would stop
# here, which is the intended behaviour.
python -u scripts/nuisance_feasibility_gate.py --run
