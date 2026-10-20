#!/bin/bash
#SBATCH --job-name=exp06_prep
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
# Download (unless PREP_ARGS=--no-download) and convert CIFAR-100 + Tiny-ImageNet-200 to 32x32 uint8 .npy + manifest.
# FarmShare: submit_exp06.sh prep (sbatch on normal). Sherlock: submit_exp06.sh download on the login node, then
# submit_exp06.sh prep runs this in a foreground `srun -p dev` with --no-download.
source "${EXP06_CODE:-${SLURM_SUBMIT_DIR:-$(dirname "$0")}}/env.sh"
set -euo pipefail
python "$EXP06_CODE/prep_data.py" --out "$EXP06_ROOT/data" --raw "$EXP06_ROOT/raw" ${PREP_ARGS:-}
