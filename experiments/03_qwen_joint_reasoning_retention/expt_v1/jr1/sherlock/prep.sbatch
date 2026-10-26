#!/usr/bin/env bash
#SBATCH -J jr1-prep
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
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
# Prep: venv (idempotent), environment receipt on the GPU SKU, data prepare + verify + hash check against the
# committed jr1/data_manifest.json, Qwen2.5-1.5B at the pinned revision into $HF_HOME with every file's sha256
# checked against the Hub's own hashes. Writes receipts/{env_prep,data_check}.json, models/model_manifest.json, PREP_OK.
jr1_log "prep start (GPU_SKU=$GPU_SKU, code $JR1_CODE_COMMIT)"
rm -f PREP_OK
[ "${JR1_LOCAL:-0}" = 1 ] || bash "$JR1_SH/setup_env.sh"   # local CPU runs bring their own venv
jr1_env_receipt receipts/env_prep.json
if [ "${JR1_TINY:-0}" = 1 ]; then
  jr1_require "${JR1_TINY_EVAL_FROM:-}" "JR1_TINY_EVAL_FROM: a prepared data dir to take eval items from"
  rm -rf data
  jr1_py -m jr1.sherlock.tools tiny-data --out data --eval-from "$JR1_TINY_EVAL_FROM" --n "${JR1_TINY_N:-3}"
  if [ -n "${JR1_PREP_VERIFY_DIR:-}" ]; then   # exercise the hash check on a real prepared copy
    jr1_py -m jr1.data verify --root "$JR1_PREP_VERIFY_DIR"
    jr1_py -m jr1.sherlock.tools check-data --data "$JR1_PREP_VERIFY_DIR" \
      --manifest "$JR1_EXPT/jr1/data_manifest.json" --out receipts/data_check.json
  fi
else
  if ! { [ -f data/manifest.json ] && jr1_py -m jr1.data verify --root data; }; then
    jr1_log "data prepare (WikiText-103, GSM8K, UltraFeedback, MATH-500, MMLU, ARC-Easy at pinned revisions)"
    HF_HUB_OFFLINE=0 jr1_py -m jr1.data prepare --root data
    jr1_py -m jr1.data verify --root data
  fi
  jr1_py -m jr1.sherlock.tools check-data --data data --manifest "$JR1_EXPT/jr1/data_manifest.json" \
    --out receipts/data_check.json
fi
HF_HUB_OFFLINE=0 jr1_py -m jr1.sherlock.tools prep-model --out models
date -Iseconds > PREP_OK
jr1_log "prep done: receipts/env_prep.json receipts/data_check.json models/model_manifest.json"
