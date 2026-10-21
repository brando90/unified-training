# JR1 campaign environment. Source it: on Sherlock (default) or locally with JR1_LOCAL=1 (CPU e2e).
# Every path the jobs use comes from here; override any variable before sourcing.
export JR1_STATE_ROOT=${JR1_STATE_ROOT:-/scratch/users/brando9/ut_jr1}
export JR1_CODE=${JR1_CODE:-$JR1_STATE_ROOT/code}                 # the unpacked bundle (or a repo checkout)
export JR1_EXPT=$JR1_CODE/experiments/03_qwen_joint_reasoning_retention/expt_v1
export JR1_SH=$JR1_EXPT/jr1/sherlock
export JR1_VENV=${JR1_VENV:-$JR1_STATE_ROOT/venv}
export JR1_PROTOCOL_DIR=${JR1_PROTOCOL_DIR:-$JR1_EXPT/jr1/configs/protocol_v1}   # per-method frozen configs (AT-572)
export JR1_EVAL_BACKEND=${JR1_EVAL_BACKEND:-hf}                  # protocol choice: HF generate (see README)
export GPU_SKU=${GPU_SKU:-L40S}                                   # one SKU for every measured cell (calibration decides)
export HF_HOME=${HF_HOME:-$JR1_STATE_ROOT/hf}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}                         # only prep downloads
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONPATH=$JR1_EXPT${PYTHONPATH:+:$PYTHONPATH}
if [ -f "$JR1_CODE/CODE_COMMIT" ]; then
  export JR1_CODE_COMMIT=$(cat "$JR1_CODE/CODE_COMMIT")
else
  export JR1_CODE_COMMIT=$(git -C "$JR1_CODE" rev-parse HEAD 2>/dev/null || echo unknown)
fi
if [ "${JR1_LOCAL:-0}" != 1 ] && command -v module >/dev/null 2>&1; then
  module load python/3.12.1 >/dev/null 2>&1 || true
fi
export JR1_PY=$JR1_VENV/bin/python
mkdir -p "$JR1_STATE_ROOT"/{logs,receipts}
