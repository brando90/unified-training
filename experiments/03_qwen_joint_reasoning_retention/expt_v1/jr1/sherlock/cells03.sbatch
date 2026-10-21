#!/usr/bin/env bash
#SBATCH -J jr1-c03
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 2-00:00:00
#SBATCH -o logs/%x_%A_%a.log
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@900
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brandojazz@gmail.com,brando.science@gmail.com
# GPU type: submit.sh adds -C "GPU_SKU:$GPU_SKU" (one SKU for all measured cells; env.sh default L40S); the job
# refuses to run on any other GPU. Submit from $JR1_STATE_ROOT with submit.sh, never bare sbatch.
set -euo pipefail
: "${JR1_STATE_ROOT:=/scratch/users/brando9/ut_jr1}"
JR1_SH=${JR1_SH:-$JR1_STATE_ROOT/code/experiments/03_qwen_joint_reasoning_retention/expt_v1/jr1/sherlock}
source "$JR1_SH/env.sh"
source "$JR1_SH/common.sh"
cd "$JR1_STATE_ROOT"
# Exp 03 measured cells: array 0-36. Index i -> seed 1101 + i//6, method i%6 of (sequential replay fixed_joint
# smooth_joint aioli jr1), so each finished block of six is a complete paired seed; index 36 = untouched base
# (full + lc evaluation, no training). Each cell: train with the frozen per-method config (--resume: a requeue
# continues from the last atomic state), lc eval at ckpt 000/025/050/075/100, full eval at ckpt_100, CELL_DONE.
jr1_require PREP_OK "run prep first"
I=${SLURM_ARRAY_TASK_ID:?cells03.sbatch is an array job}
eval "$("$JR1_PY" -m jr1.sherlock.tools cell --exp 03 --index "$I")"
D=runs/03/$cell
mkdir -p "$D"
jr1_slurm_note "$D"
jr1_env_receipt "$D/env.json"
if [ "$method" = base ]; then
  jr1_eval models/base full "$D/eval/full_000"
  jr1_eval models/base lc "$D/eval/lc_000"
  date -Iseconds > "$D/CELL_DONE"; jr1_log "base evaluated"; exit 0
fi
CFG=$JR1_PROTOCOL_DIR/$method.json
jr1_require "$CFG" "the frozen per-method config (AT-572)"
jr1_py -m jr1.sherlock.tools check-frozen --config "$CFG"
cp -n "$CFG" "$D/protocol_config.json" 2>/dev/null || true
jr1_log "train $method seed $seed"
jr1_py -m jr1.train --method "$method" --seed "$seed" --config "$CFG" --out "$D" --resume
jr1_status_is "$D/receipt.json" complete || { jr1_log "train did not complete"; exit 7; }
jr1_eval_cell "$D"
date -Iseconds > "$D/CELL_DONE"
jr1_log "cell $cell done"
