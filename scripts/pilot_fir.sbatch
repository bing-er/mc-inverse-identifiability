#!/bin/bash
#SBATCH --job-name=mcii_pilot
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --output=logs/mcii_pilot_%j.out
#SBATCH --error=logs/mcii_pilot_%j.err
#
# MERGE_CHECKS item 1, pilot. Two columns only: f_U to exercise the pipeline end to end, and
# StO2_D to size the budget for the near-null direction. It forms no conclusion about any column.
#
# The reference, AD and pMC are produced in ONE run on the SAME replicates, with a single PRNG key
# array per replicate shared by the plus-h and minus-h forward evaluations, the AD jvp and the pMC
# replay. Every raw per-replicate array is stored, so the three paired differences are built from
# the stored numbers rather than reassembled from separate runs.
#
# This job is submitted with an afterok dependency on the GPU equivalence gate. pMC availability is
# read from the gate artifact at run time, so if the gate had not passed the pilot would omit pMC
# and label itself incomplete rather than silently produce a pMC number.
#
# Six hours requested against a two hour estimate: the per-evaluation cost is an estimate derived
# from the transport pool timing, and the pMC pass has never been timed on this hardware.

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

echo "=== pilot on $(hostname) ==="
which ptxas; ptxas --version
python -c "import jax, scipy; print('JAX', jax.__version__, '| backend', jax.default_backend(), jax.devices(), '| scipy', scipy.__version__)"

PTXAS_VER="$(ptxas --version | grep -oE 'V[0-9]+\.[0-9]+\.[0-9]+' | tr -d V)"
case "$PTXAS_VER" in
  12.[0-5].*|12.6.[0-2]|11.*) echo "REFUSING: ptxas $PTXAS_VER predates the sm90 clamping fix"; exit 1 ;;
esac
echo "ptxas $PTXAS_VER accepted"
export MCII_PTXAS_VERSION="$PTXAS_VER"

python -u scripts/jacobian_validation.py --selftest
python -u scripts/jacobian_validation.py --pilot \
  --pilot-photons 200000 --pilot-replicates 6 "$@"
