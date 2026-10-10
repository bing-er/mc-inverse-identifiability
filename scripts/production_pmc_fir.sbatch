#!/bin/bash
#SBATCH --job-name=mcii_pmcjac
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --output=logs/mcii_pmcjac_%j.out
#SBATCH --error=logs/mcii_pmcjac_%j.err
#
# Full-grid replay-pMC production Jacobian, on the production backend.
#
# The fail-closed guard requires the frozen GPU photon-level equivalence gate, matching source and
# production-stream provenance, and aggregate reflectance agreement at the gate's recorded
# tolerance. It therefore remains tied to cuda/12.9 on H100 (sm90), without demanding bitwise
# equality from two differently fused XLA reductions. The run is resumable at seed granularity.

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

echo "=== replay-pMC production Jacobian on $(hostname) ==="
which ptxas; ptxas --version
python -c "import jax; print('JAX', jax.__version__, '| backend', jax.default_backend(), jax.devices())"

PTXAS_VER="$(ptxas --version | grep -oE 'V[0-9]+\.[0-9]+\.[0-9]+' | tr -d V)"
case "$PTXAS_VER" in
  12.[0-5].*|12.6.[0-2]|11.*) echo "REFUSING: ptxas $PTXAS_VER predates the sm90 clamping fix"; exit 1 ;;
esac
echo "ptxas $PTXAS_VER accepted"
export MCII_PTXAS_VERSION="$PTXAS_VER"

python -u scripts/production_pmc_jacobian.py --run "$@"
python -u scripts/production_pmc_jacobian.py --analyze
