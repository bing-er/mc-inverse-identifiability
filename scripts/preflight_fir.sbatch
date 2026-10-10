#!/bin/bash
#SBATCH --job-name=mcii_preflt
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --output=logs/mcii_preflt_%j.out
#SBATCH --error=logs/mcii_preflt_%j.err
#
# Real-executor preflight for the nuisance feasibility gate. Two thousand photons per wavelength,
# so it is minutes rather than hours.
#
# It validates CODE, not priors: it compiles the traced eta path and checks that at eta = 0 the
# reflectance reproduces the production forward solver called directly with the same keys. The
# mocked self-tests cannot answer that, because their response is an analytic function rather than
# transport. For the same reason the preflight does not require the nuisance citation gate and
# cannot write the gate's artifact.
#
# CUDA 12.9 for the same reason as every other job here: ptxas up to 12.6.2 can wrongly optimize
# sm90 kernels with abs and min/max clamping, which this transport kernel does throughout.

set -euo pipefail
cd "$HOME/scratch/mc-inverse-identifiability"
mkdir -p logs results

module load python/3.11 scipy-stack/2025a cuda/12.9
if   [ -d "$HOME/scratch/venvs/jaxpad" ]; then source "$HOME/scratch/venvs/jaxpad/bin/activate"
elif [ -d "$HOME/.venvs/jaxpad-fir"    ]; then source "$HOME/.venvs/jaxpad-fir/bin/activate"
else echo "ERROR: no jaxpad venv found"; exit 1; fi

export PYTHONPATH="$HOME/scratch/jax-pad:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Required, not a preference. Measured on this hardware the GPU scatter-add is non-deterministic at
# a relative 1.3e-15, so repeated calls with the same photon keys do not agree bitwise and the
# common-random-number cancellation the ensemble method depends on is not exact. This flag removes
# it entirely at no measured cost in correctness, and the preflight refuses to pass without it.
export XLA_FLAGS="--xla_gpu_deterministic_ops=true ${XLA_FLAGS:-}"
export JAX_COMPILATION_CACHE_DIR="$HOME/scratch/jax-pad/jax_cache_cuda129"
export JAXPAD_ROOT="$HOME/scratch/jax-pad"
export MCII_CUDA_MODULE="cuda/12.9"

echo "=== nuisance executor preflight on $(hostname) ==="
which ptxas; ptxas --version
python -c "import jax, scipy; print('JAX', jax.__version__, '| backend', jax.default_backend(), jax.devices(), '| scipy', scipy.__version__)"

PTXAS_VER="$(ptxas --version | grep -oE 'V[0-9]+\.[0-9]+\.[0-9]+' | tr -d V)"
case "$PTXAS_VER" in
  12.[0-5].*|12.6.[0-2]|11.*) echo "REFUSING: ptxas $PTXAS_VER predates the sm90 clamping fix"; exit 1 ;;
esac
echo "ptxas $PTXAS_VER accepted"
export MCII_PTXAS_VERSION="$PTXAS_VER"

python -u scripts/nuisance_feasibility_gate.py --selftest
python -u scripts/nuisance_feasibility_gate.py --preflight --preflight-photons 2000
