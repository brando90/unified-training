#!/usr/bin/env bash
#SBATCH -J jr1-sched04
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 2-00:00:00
#SBATCH -o logs/%x_%j.log
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
# Exp 04 source schedule: the frozen J-R1 method (PROTOCOL v1 config) on dev seed 9001, recording every realized
# objective microbatch. cells04 replays exactly this file. Writes runs/04/schedule_jr1_9001.jsonl(+.meta.json, .sha256).
jr1_require PREP_OK "run prep first"
CFG=$JR1_PROTOCOL_DIR/jr1.json
jr1_require "$CFG" "the frozen per-method configs from AT-572"
jr1_py -m jr1.sherlock.tools check-frozen --config "$CFG"
D=runs/04/schedule_source_jr1_s9001
S=runs/04/schedule_jr1_9001.jsonl
mkdir -p "$D"
jr1_slurm_note "$D"
jr1_env_receipt "$D/env.json"
jr1_py -m jr1.train --method jr1 --seed 9001 --config "$CFG" --out "$D" --record-schedule "$S" --resume
sha256sum "$S" | cut -d' ' -f1 > "$S.sha256"
jr1_log "schedule recorded: $S sha256 $(cat "$S.sha256")"
