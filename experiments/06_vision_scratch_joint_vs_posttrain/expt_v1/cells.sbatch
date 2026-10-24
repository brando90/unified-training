#!/bin/bash
#SBATCH --job-name=exp06_cells
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 2-00:00:00
#SBATCH --signal=B:USR1@600
# One measured cell per array task (row SLURM_ARRAY_TASK_ID of cells.tsv). Rerunning a task resumes from its checkpoint;
# a complete cell exits at once. submit_exp06.sh cells adds the partition, array (%4) and mail flags.
source "${EXP06_CODE:-${SLURM_SUBMIT_DIR:-$(dirname "$0")}}/env.sh"
set -uo pipefail
read -r task model arm seed < <(awk -F'\t' -v t="$SLURM_ARRAY_TASK_ID" 'NR>1 && $1==t' "$EXP06_CODE/cells.tsv")
[ -n "${arm:-}" ] || { echo "no row $SLURM_ARRAY_TASK_ID in cells.tsv"; exit 1; }
out="$EXP06_ROOT/runs/$model/$arm/seed$seed"
echo "task=$task model=$model arm=$arm seed=$seed host=$(hostname) job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} amp=$EXP06_AMP"
nvidia-smi --query-gpu=name --format=csv,noheader
python "$EXP06_CODE/train.py" --arm "$arm" --model "$model" --seed "$seed" --data "$EXP06_ROOT/data" --out "$out" \
  --amp "$EXP06_AMP" &
pid=$!
trap 'kill -USR1 $pid' USR1   # near the time limit: checkpoint and exit 3; resubmit the task to resume
wait $pid; rc=$?
[ $rc -gt 128 ] && { wait $pid; rc=$?; }
echo "exit $rc"
exit $rc
