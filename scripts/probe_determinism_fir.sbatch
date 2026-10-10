#!/bin/bash
#SBATCH --job-name=mcii_detprobe
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --output=logs/mcii_detprobe_%j.out
#SBATCH --error=logs/mcii_detprobe_%j.err
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
PTXAS_VER="$(ptxas --version | grep -oE 'V[0-9]+\.[0-9]+\.[0-9]+' | tr -d V)"
export MCII_PTXAS_VERSION="$PTXAS_VER"
echo "=== probe on $(hostname), ptxas $PTXAS_VER ==="
echo "--- default ---"
python -u scripts/probe_determinism.py
echo
echo "--- with XLA determinism flags ---"
XLA_FLAGS="--xla_gpu_deterministic_ops=true" TF_DETERMINISTIC_OPS=1 \
  python -u scripts/probe_determinism.py || echo "(the flag may not be supported by this XLA build)"
