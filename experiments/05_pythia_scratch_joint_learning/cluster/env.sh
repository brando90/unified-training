# Exp 05 environment. Source it on Sherlock (V100, fp16) or FarmShare (L40S, bf16), or locally with P05_LOCAL=1.
# Set P05_CLUSTER=sherlock|farmshare explicitly (SUBMIT.md does); the guess is $SHERLOCK / the hostname.
# Every path the jobs use comes from here; override any variable before sourcing.
if [ -z "${P05_CLUSTER:-}" ]; then
  if [ "${P05_LOCAL:-0}" = 1 ]; then P05_CLUSTER=local
  elif [ -n "${SHERLOCK:-}" ] || [[ $(hostname -f 2>/dev/null) == *sherlock* ]]; then P05_CLUSTER=sherlock
  else P05_CLUSTER=farmshare; fi
fi
export P05_CLUSTER
case $P05_CLUSTER in
  sherlock)  : "${P05_STATE_ROOT:=/scratch/users/brando9/ut_p05}"; : "${P05_PRECISION:=fp16}"; : "${P05_REQUIRE:=v100_32gb}"
             : "${P05_JR1_DATA:=/scratch/users/brando9/ut_jr1/data}";;   # reuse JR1 prep when it exists
  farmshare) : "${P05_STATE_ROOT:=/home/users/brando9/ut_p05}"; : "${P05_PRECISION:=bf16}"; : "${P05_REQUIRE:=sm80}"
             : "${JR1_BASE_PYTHON:=/usr/bin/python3.12}";;                  # not the Jupyter python3 (3.14)
  local)     : "${P05_STATE_ROOT:?P05_STATE_ROOT}"; : "${P05_PRECISION:=bf16}"; : "${P05_REQUIRE:=none}";;
  *) echo "unknown P05_CLUSTER=$P05_CLUSTER" >&2; return 2 2>/dev/null || exit 2;;
esac
export P05_STATE_ROOT P05_PRECISION P05_REQUIRE
export P05_CODE=${P05_CODE:-$P05_STATE_ROOT/code}                       # the unpacked bundle (or a repo checkout)
export P05_HOME=$P05_CODE/experiments/05_pythia_scratch_joint_learning
export P05_SH=$P05_HOME/cluster
export JR1_EXPT=$P05_CODE/experiments/03_qwen_joint_reasoning_retention/expt_v1
export JR1_SH=$JR1_EXPT/jr1/sherlock
export P05_JR1_DATA=${P05_JR1_DATA:-$P05_STATE_ROOT/jr1_data}          # jr1/data.py root (prepared if missing)
[ -f "$P05_JR1_DATA/manifest.json" ] || [ "$P05_CLUSTER" != sherlock ] || P05_JR1_DATA=$P05_STATE_ROOT/jr1_data
export P05_DATA_ROOT=${P05_DATA_ROOT:-$P05_STATE_ROOT/data}            # p05/data.py output
# jr1's setup_env.sh builds the venv from jr1's hashed lock (torch 2.6.0+cu124 wheels: V100 sm70 and L40S sm89,
# glibc 2.17+), so both clusters run the same torch/transformers. It reads JR1_STATE_ROOT/JR1_CODE/JR1_VENV.
export JR1_STATE_ROOT=$P05_STATE_ROOT/jr1_env JR1_CODE=$P05_CODE
export JR1_VENV=${P05_VENV:-$P05_STATE_ROOT/venv}
[ -n "${JR1_BASE_PYTHON:-}" ] && export JR1_BASE_PYTHON
export HF_HOME=${HF_HOME:-$P05_STATE_ROOT/hf}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}                              # only prep downloads
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONPATH=$P05_HOME:$JR1_EXPT${PYTHONPATH:+:$PYTHONPATH}
if [ -f "$P05_CODE/CODE_COMMIT" ]; then
  export P05_CODE_COMMIT=$(cat "$P05_CODE/CODE_COMMIT")
else
  export P05_CODE_COMMIT=$(git -C "$P05_CODE" rev-parse HEAD 2>/dev/null || echo unknown)
fi
export JR1_CODE_COMMIT=$P05_CODE_COMMIT GPU_SKU=${GPU_SKU:-$P05_REQUIRE}
if [ "$P05_CLUSTER" = sherlock ] && command -v module >/dev/null 2>&1; then
  module load python/3.12.1 >/dev/null 2>&1 || true
fi
export P05_PY=${P05_PY:-$JR1_VENV/bin/python}
export JR1_PY=$P05_PY
mkdir -p "$P05_STATE_ROOT"/{logs,receipts,runs} "$JR1_STATE_ROOT"/{logs,receipts}
