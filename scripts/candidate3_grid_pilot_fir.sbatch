#!/bin/bash
#SBATCH --job-name=mcii_c3grid
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --array=0-5
#SBATCH --output=logs/mcii_c3grid_%A_%a.out
#SBATCH --error=logs/mcii_c3grid_%A_%a.err

set -euo pipefail
cd "$HOME/scratch/mc-inverse-identifiability"
mkdir -p logs results

module load python/3.11 scipy-stack/2025a cuda/12.9
if   [ -d "$HOME/scratch/venvs/jaxpad" ]; then source "$HOME/scratch/venvs/jaxpad/bin/activate"
elif [ -d "$HOME/.venvs/jaxpad-fir" ]; then source "$HOME/.venvs/jaxpad-fir/bin/activate"
else echo "ERROR: no jaxpad venv found"; exit 1; fi

export PYTHONPATH="$HOME/scratch/jax-pad:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_deterministic_ops=true ${XLA_FLAGS:-}"
export JAX_COMPILATION_CACHE_DIR="$HOME/scratch/jax-pad/jax_cache_cuda129"
export JAXPAD_ROOT="$HOME/scratch/jax-pad"
export MCII_CUDA_MODULE="cuda/12.9"

which ptxas
ptxas --version
PTXAS_VER="$(ptxas --version | grep -oE 'V[0-9]+\.[0-9]+\.[0-9]+' | tr -d V)"
case "$PTXAS_VER" in
  12.[0-5].*|12.6.[0-2]|11.*) echo "REFUSING: ptxas $PTXAS_VER predates the sm90 clamping fix"; exit 1 ;;
esac
export MCII_PTXAS_VERSION="$PTXAS_VER"

python -W error -u scripts/candidate3_grid_pilot.py --freeze-status
python -W error -u scripts/candidate3_grid_pilot_audit.py --expect frozen
python -W error -u scripts/candidate3_grid_preflight.py --selftest
python -W error -u scripts/candidate3_grid_pilot.py --selftest
python -W error -u scripts/candidate3_grid_pilot.py --run-shard
