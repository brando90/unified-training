# Shared functions for the JR1 sbatch scripts (sourced after env.sh).

jr1_log() { echo "[$(date '+%F %T')] ${SLURM_JOB_ID:+job $SLURM_JOB_ID }$*"; }

# Requeue near the time limit (#SBATCH --signal=B:USR1@900): Slurm restarts the task, which resumes from its last
# atomic state (the trainer saves state every 30 min and at every checkpoint; evaluation resumes per benchmark).
_jr1_on_usr1() {
  local n=${SLURM_RESTART_COUNT:-0}
  if [ -n "${SLURM_JOB_ID:-}" ] && [ "$n" -lt "${JR1_MAX_REQUEUE:-3}" ]; then
    jr1_log "time limit near: scontrol requeue $SLURM_JOB_ID (restart $((n + 1)) of max ${JR1_MAX_REQUEUE:-3})"
    scontrol requeue "$SLURM_JOB_ID"
  else
    jr1_log "time limit near and requeue budget used ($n); resubmit this index by hand, it resumes"
  fi
}
trap _jr1_on_usr1 USR1

# Run a command in the background and wait, so the USR1 trap fires while it runs.
jr1_run() {
  "$@" &
  local pid=$! rc
  while true; do
    rc=0; wait "$pid" || rc=$?
    kill -0 "$pid" 2>/dev/null || return "$rc"
  done
}

jr1_py() { jr1_run "$JR1_PY" "$@"; }

# Environment receipt; on Sherlock the GPU must match GPU_SKU.
jr1_env_receipt() {
  local out=$1
  if [ "${JR1_LOCAL:-0}" = 1 ]; then
    jr1_py -m jr1.sherlock.tools env-receipt --out "$out"
  else
    jr1_py -m jr1.sherlock.tools env-receipt --out "$out" --require-gpu
  fi
}

# Per-cell Slurm record (appended on every start, so requeues stay visible).
jr1_slurm_note() {
  local dir=$1; mkdir -p "$dir"
  "$JR1_PY" - "$dir" <<'PY'
import json, os, sys, datetime
from pathlib import Path
p = Path(sys.argv[1]) / "slurm.json"
d = json.loads(p.read_text()) if p.exists() else {"starts": []}
keys = ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_NODELIST", "SLURM_RESTART_COUNT",
        "GPU_SKU", "JR1_CODE_COMMIT", "JR1_EVAL_BACKEND")
d["starts"].append({"time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                    **{k.lower(): os.environ.get(k) for k in keys}})
p.write_text(json.dumps(d, indent=1))
PY
}

jr1_eval() {   # jr1_eval <ckpt> <suite> <out>
  jr1_py -m jr1.evaluate --ckpt "$1" --suite "$2" --out "$3" --backend "$JR1_EVAL_BACKEND" --resume \
    ${JR1_EVAL_ARGS:-}
}

# lc suite at every retained checkpoint (0/25/50/75/100% of charged work) and the full suite at the terminal one.
jr1_eval_cell() {
  local d=$1 ck
  for ck in 000 025 050 075 100; do
    jr1_log "eval lc ckpt_$ck"
    jr1_eval "$d/ckpt_$ck" lc "$d/eval/lc_$ck"
  done
  jr1_log "eval full ckpt_100"
  jr1_eval "$d/ckpt_100" full "$d/eval/full_100"
}

jr1_status_is() {   # jr1_status_is <receipt.json> <status>
  "$JR1_PY" -c 'import json, sys; sys.exit(json.load(open(sys.argv[1]))["status"] != sys.argv[2])' "$1" "$2"
}

jr1_require() { [ -e "$1" ] || { jr1_log "REFUSED: missing $1 ($2)"; exit 2; }; }
