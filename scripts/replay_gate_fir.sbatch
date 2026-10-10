#!/bin/bash
#SBATCH --job-name=mcii_gate
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --output=logs/mcii_gate_%j.out
#SBATCH --error=logs/mcii_gate_%j.err
#
# Photon-level equivalence of the replay kernel against the production kernel, ON THE GPU.
#
# The gate already passed locally on CPU, bitwise, and that is not sufficient. Bitwise agreement
# under one XLA backend does not carry over to another: the GPU backend fuses and reorders
# differently, and the pilot runs on the GPU. So the gate is re-run here, on the hardware and the
# compiler the pilot will actually use, before any pMC number is produced.
#
# The same CUDA 12.9 requirement applies as for the transport pool. ptxas up to and including
# 12.6.2 can wrongly optimize sm90 kernels with abs() and min/max clamping, which this transport
# kernel does throughout. H100 is sm90.

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

echo "=== replay equivalence gate on $(hostname) ==="
which ptxas; ptxas --version
python -c "import jax, scipy; print('JAX', jax.__version__, '| backend', jax.default_backend(), jax.devices(), '| scipy', scipy.__version__)"

PTXAS_VER="$(ptxas --version | grep -oE 'V[0-9]+\.[0-9]+\.[0-9]+' | tr -d V)"
case "$PTXAS_VER" in
  12.[0-5].*|12.6.[0-2]|11.*) echo "REFUSING: ptxas $PTXAS_VER predates the sm90 clamping fix"; exit 1 ;;
esac
echo "ptxas $PTXAS_VER accepted"
export MCII_PTXAS_VERSION="$PTXAS_VER"

python -u scripts/replay_equivalence.py \
  --n 4000 --seeds 910000,920007,930014,940021,950028 \
  --out results/replay_equivalence_gpu.json "$@"

python -u scripts/write_replay_provenance.py replay_equivalence_gpu.json
