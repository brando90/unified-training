#!/usr/bin/env bash
#SBATCH -J jr1-c04
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
# Exp 04 measured cells: array 0-5. Index i -> seed 1101 + i//2, order (interleaved, stage_sorted)[i%2]: replay of
# the frozen J-R1 dev-seed-9001 schedule (schedule04.sbatch), matched objective microbatches, same evaluation as 03.
jr1_require PREP_OK "run prep first"
I=${SLURM_ARRAY_TASK_ID:?cells04.sbatch is an array job}
eval "$("$JR1_PY" -m jr1.sherlock.tools cell --exp 04 --index "$I")"
S=runs/04/schedule_jr1_9001.jsonl
jr1_require "$S.sha256" "schedule04.sbatch must finish first"
[ "$(sha256sum "$S" | cut -d' ' -f1)" = "$(cat "$S.sha256")" ] || { jr1_log "REFUSED: schedule hash changed"; exit 8; }
[ -z "${JR1_SCHEDULE04_SHA256:-}" ] || [ "$JR1_SCHEDULE04_SHA256" = "$(cat "$S.sha256")" ] || {
  jr1_log "REFUSED: schedule differs from PROTOCOL's JR1_SCHEDULE04_SHA256"; exit 8; }
CFG=$JR1_PROTOCOL_DIR/jr1.json
jr1_py -m jr1.sherlock.tools check-frozen --config "$CFG"
D=runs/04/$cell
mkdir -p "$D"
jr1_slurm_note "$D"
jr1_env_receipt "$D/env.json"
cp -n "$CFG" "$D/protocol_config.json" 2>/dev/null || true
jr1_log "replay $order seed $seed"
jr1_py -m jr1.train --replay-schedule "$S" --order "$order" --seed "$seed" --config "$CFG" --out "$D" --resume
jr1_status_is "$D/receipt.json" complete || { jr1_log "replay did not complete"; exit 7; }
jr1_eval_cell "$D"
date -Iseconds > "$D/CELL_DONE"
jr1_log "cell $cell done"
