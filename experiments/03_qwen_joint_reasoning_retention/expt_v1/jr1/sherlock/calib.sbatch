#!/usr/bin/env bash
#SBATCH -J jr1-calib
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
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
# Calibration on one GPU SKU (run once per SKU: GPU_SKU=L40S and GPU_SKU=H100_SXM5). Measures, into calib/$GPU_SKU:
#  - one short real run per method (dev seed 9001, development config): per-objective update time, per-method
#    work/s incl. controller probes, peak memory (train/<method>/{log.jsonl,receipt.json})
#  - restart: an interrupted run resumed from its atomic state (restart.json)
#  - checkpoint save/load (ckpt_timing.json)
#  - full and lc evaluation wall time and memory on one checkpoint (eval_full, eval_lc). Scores in those
#    directories are never read: calibration.json carries times only, no test feedback reaches any choice.
# Then calibration.json with per-cell and 03+04 GPU-hour estimates at the config's work target.
jr1_require PREP_OK "run prep first"
C=calib/$GPU_SKU
mkdir -p "$C/train"
jr1_env_receipt "$C/env.json"
CFG=${JR1_CALIB_CONFIG:-$JR1_EXPT/jr1/configs/qwen15b_v0.json}
[ "${JR1_TINY:-0}" = 1 ] && CFG=${JR1_CALIB_CONFIG:-$JR1_EXPT/jr1/configs/tiny_cpu.json}
for m in sequential replay fixed_joint smooth_joint aioli jr1; do
  n=${JR1_CALIB_UPDATES:-8}
  case $m in aioli|jr1) n=${JR1_CALIB_UPDATES_ADAPTIVE:-60};; esac
  if [ -f "$C/train/$m/receipt.json" ] && jr1_status_is "$C/train/$m/receipt.json" debug_stopped; then
    jr1_log "calib $m already measured"; continue
  fi
  rm -rf "$C/train/$m"
  jr1_log "calib train $m ($n updates)"
  jr1_py -m jr1.train --method "$m" --seed 9001 --config "$CFG" --out "$C/train/$m" --max-updates "$n"
done
# restart cost: interrupt after 2 actions, then resume to 4 updates in a new process
R=$C/restart_fixed_joint
if [ ! -f "$C/restart.json" ]; then
  rm -rf "$R"
  jr1_py -m jr1.train --method fixed_joint --seed 9001 --config "$CFG" --out "$R" --max-updates 4 --stop-after-actions 2
  t0=$(date +%s.%N)
  jr1_py -m jr1.train --method fixed_joint --seed 9001 --config "$CFG" --out "$R" --max-updates 4 --resume
  t1=$(date +%s.%N)
  "$JR1_PY" - "$R" "$t0" "$t1" "$C/restart.json" <<'PY'
import json, sys
from pathlib import Path
r, t0, t1, out = Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
rec = json.loads((r / "receipt.json").read_text())
log = [json.loads(l) for l in (r / "log.jsonl").read_text().splitlines()]
json.dump({"resume_process_s": round(t1 - t0, 3), "status": rec["status"], "resumes": rec["resumes"],
           "actions": rec["actions"], "log_rows": len(log),
           "note": "process wall of the resumed run: interpreter + model load + state load + remaining actions"},
          open(out, "w"), indent=1)
PY
fi
jr1_py -m jr1.sherlock.tools ckpt-timing --ckpt "$C/train/fixed_joint/ckpt_000" --scratch "$C/tmp_save" \
  --out "$C/ckpt_timing.json"
jr1_log "calib eval lc + full (times only)"
jr1_eval "$C/train/fixed_joint/ckpt_100" lc "$C/eval_lc"
jr1_eval "$C/train/fixed_joint/ckpt_100" full "$C/eval_full"
TARGET=$("$JR1_PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['work']['target'])" "$CFG")
jr1_py -m jr1.sherlock.tools calib-report --calib "$C" --work-target "${JR1_WORK_TARGET:-$TARGET}" \
  --out "$C/calibration.json"
jr1_log "calib done: $C/calibration.json"
