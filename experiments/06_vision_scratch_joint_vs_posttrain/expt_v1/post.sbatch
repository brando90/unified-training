#!/bin/bash
#SBATCH --job-name=exp06_post
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:30:00
# After the array (afterany): analysis over the full 30-cell denominator, then one results tarball without checkpoints.
source "${EXP06_CODE:-${SLURM_SUBMIT_DIR:-$(dirname "$0")}}/env.sh"
set -uo pipefail
python "$EXP06_CODE/analyze.py" --runs "$EXP06_ROOT/runs" --cells "$EXP06_CODE/cells.tsv" --out "$EXP06_ROOT/analysis"
cd "$EXP06_ROOT"
tar czf exp06_results.tgz --exclude='ckpt.pt' --exclude='ckpt.pt.tmp' analysis runs logs jobs.txt data/manifest.json \
  requirements.lock.txt smoke/*.log 2>/dev/null
sha256sum exp06_results.tgz | tee exp06_results.tgz.sha256
ls -la exp06_results.tgz
