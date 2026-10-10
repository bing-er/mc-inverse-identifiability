#!/bin/bash
#SBATCH --job-name=mcii_pradial
#SBATCH --time=07:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --output=logs/mcii_pradial_%j.out
#SBATCH --error=logs/mcii_pradial_%j.err
#
# MERGE_CHECKS item 1: the radial-observable pilot.
#
# Same two targets, same 6 replicates, same 200000 photons per wavelength, same h grid, same seeds
# and the same PRNG key pairing as the total-escape pilot. The photon budget is NOT raised. The one
# change is that the full 120-component radial output is stored at every stage, so the three
# pre-registered observables are derived offline from one set of arrays rather than each costing
# its own transport run.
#
# It answers one question: whether radial information lowers the photon-budget requirement into a
# feasible range, in particular for StO2_D, which on total escaped weight alone needs between 1500
# and 9000 times this budget. It is a deterministic rerun completing the observable set, not an
# independent repeat, and it is not a validation of either estimator.
#
# The previous total-escape pilot is retained and must not be overwritten.
#
# CUDA 12.9 is required for the same reason as every other job here: ptxas up to 12.6.2 can
# wrongly optimize sm90 kernels with abs and min/max clamping, which this kernel does throughout.

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

echo "=== radial-observable pilot on $(hostname) ==="
which ptxas; ptxas --version
python -c "import jax, scipy; print('JAX', jax.__version__, '| backend', jax.default_backend(), jax.devices(), '| scipy', scipy.__version__)"

PTXAS_VER="$(ptxas --version | grep -oE 'V[0-9]+\.[0-9]+\.[0-9]+' | tr -d V)"
case "$PTXAS_VER" in
  12.[0-5].*|12.6.[0-2]|11.*) echo "REFUSING: ptxas $PTXAS_VER predates the sm90 clamping fix"; exit 1 ;;
esac
echo "ptxas $PTXAS_VER accepted"
export MCII_PTXAS_VERSION="$PTXAS_VER"

# refuse to clobber the total-escape pilot
if [ ! -f results/jacobian_validation_pilot.json ]; then
  echo "REFUSING: the total-escape pilot artifact is missing; it must be present and retained"; exit 1
fi

python -u scripts/jacobian_validation.py --selftest
python -u scripts/jacobian_validation.py --pilot-radial \
  --pilot-photons 200000 --pilot-replicates 6 "$@"
