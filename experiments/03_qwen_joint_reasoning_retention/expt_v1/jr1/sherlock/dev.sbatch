#!/usr/bin/env bash
#SBATCH -J jr1-dev
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
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
# Development selection array (AT-572): indexes 0-23 = 6 methods x 4 candidates of the frozen grid (dev_grid.json),
# index 24 = the untouched base model on the dev suite. Dev seed 9001, training pools only, selection signals from
# the tune split only (evaluate.py --suite dev). Each candidate trains to the same charged work target under the same
# wall-clock allowance (JR1_DEV_CAND_SECONDS); over-allowance candidates are recorded and never selected.
# Submit: submit.sh dev   (array 0-24). Then on the login node: tools select (see SUBMIT.md).
jr1_require PREP_OK "run prep first"
I=${SLURM_ARRAY_TASK_ID:?dev.sbatch is an array job}
eval "$("$JR1_PY" -m jr1.sherlock.tools cell --exp dev --index "$I")"
D=dev/$candidate
mkdir -p "$D"
jr1_slurm_note "$D"
jr1_env_receipt "$D/env.json"
if [ "$method" = base ]; then
  jr1_eval models/base dev "$D/eval_dev"
  jr1_log "dev base done"; exit 0
fi
jr1_py -m jr1.sherlock.tools dev-config --index "$I" --out "$D/config.json"
ALLOW=${JR1_DEV_CAND_SECONDS:-900}
set +e
jr1_run timeout --signal=TERM "$ALLOW" "$JR1_PY" -m jr1.train --method "$method" --seed 9001 --config "$D/config.json" \
  --out "$D" --resume
rc=$?
set -e
jr1_py -m jr1.sherlock.tools dev-mark --index "$I" --dir "$D" --rc "$rc" --allowance-s "$ALLOW"
if [ "$rc" = 0 ]; then
  jr1_eval "$D/ckpt_100" dev "$D/eval_dev"
else
  jr1_log "candidate $candidate stopped (rc=$rc): over allowance or failed; recorded, not selectable"
fi
jr1_log "dev $candidate done"
