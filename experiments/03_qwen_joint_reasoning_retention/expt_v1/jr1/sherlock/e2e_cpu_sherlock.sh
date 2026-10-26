#!/usr/bin/env bash
# CPU end-to-end of the Sherlock scripts themselves (the same .sbatch files, run with bash) over the tiny model:
#   prep -> calib -> dev array (25 tasks) -> select -> freeze per-method configs -> schedule04 -> cells03 (one cell per
#   method, seed 1101, + the untouched base) -> cells04 (all 6) -> collect, then checks every result.
# Usage (from anywhere): JR1_VENV=<venv with the lock's pins> JR1_TINY_EVAL_FROM=<a jr1.data prepared dir> \
#        [JR1_PREP_VERIFY_DIR=<the same dir: also run the real data hash check>] bash jr1/sherlock/e2e_cpu_sherlock.sh
set -euo pipefail
SH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export JR1_CODE=$(cd "$SH/../../../../.." && pwd)
export JR1_LOCAL=1 JR1_TINY=1 GPU_SKU=cpu JR1_EVAL_BACKEND=hf
export JR1_STATE_ROOT=${JR1_STATE_ROOT:-$(mktemp -d)}
export JR1_VENV=${JR1_VENV:?set JR1_VENV to a venv with requirements-sherlock pins}
export JR1_TINY_EVAL_FROM=${JR1_TINY_EVAL_FROM:?set JR1_TINY_EVAL_FROM to a prepared data dir}
export JR1_PROTOCOL_DIR=$JR1_STATE_ROOT/protocol_v1_tiny
export JR1_DEV_CAND_SECONDS=${JR1_DEV_CAND_SECONDS:-900}
export JR1_CALIB_UPDATES=${JR1_CALIB_UPDATES:-4} JR1_CALIB_UPDATES_ADAPTIVE=${JR1_CALIB_UPDATES_ADAPTIVE:-30}
export JR1_SH=$SH
export PYTHONPATH=$(cd "$SH/../.." && pwd)   # expt_v1, as env.sh sets it for the sbatch files
P=${JR1_E2E_PARALLEL:-4}
mkdir -p "$JR1_STATE_ROOT/logs"
cd "$JR1_STATE_ROOT"
echo "e2e state root: $JR1_STATE_ROOT"
step() { echo "[$(date '+%T')] $*"; }
array() {   # array <stage> <indexes...>: run tasks P at a time, like Slurm array tasks
  local stage=$1; shift
  printf '%s\n' "$@" | xargs -P "$P" -I{} bash -c \
    "SLURM_ARRAY_TASK_ID={} bash '$SH/$stage.sbatch' > 'logs/${stage}_{}.log' 2>&1 || { echo '$stage {} FAILED'; tail -20 'logs/${stage}_{}.log'; exit 255; }"
}
step prep;        bash "$SH/prep.sbatch" > logs/prep.log 2>&1
step calib;       bash "$SH/calib.sbatch" > logs/calib.log 2>&1
step "dev array (25)"; array dev $(seq 0 24)
step select
"$JR1_VENV/bin/python" -m jr1.sherlock.tools select --dev dev --out dev/selection.json
"$JR1_VENV/bin/python" -m jr1.sherlock.tools freeze-configs --selection dev/selection.json --work-target 30000 \
  --out "$JR1_PROTOCOL_DIR"
step schedule04;  bash "$SH/schedule04.sbatch" > logs/schedule04.log 2>&1
step "cells03 (${JR1_E2E_CELLS03:-0 1 2 3 4 5 36})"; array cells03 ${JR1_E2E_CELLS03:-0 1 2 3 4 5 36}
step "cells04 (${JR1_E2E_CELLS04:-0 1 2 3 4 5})";   array cells04 ${JR1_E2E_CELLS04:-0 1 2 3 4 5}
step "rerun one finished cell (requeue path: must resume, not retrain)"
SLURM_ARRAY_TASK_ID=0 bash "$SH/cells03.sbatch" > logs/cells03_0_rerun.log 2>&1
step collect;     bash "$SH/collect.sh" > logs/collect.log 2>&1
step check
"$JR1_VENV/bin/python" - "$JR1_STATE_ROOT" <<'PY'
import json, sys, tarfile
from pathlib import Path
root = Path(sys.argv[1])
ok = lambda c, m: print(("PASS " if c else "FAIL ") + m) or c
res = []
res.append(ok((root / "PREP_OK").exists() and (root / "models/model_manifest.json").exists(), "prep receipts"))
cal = json.loads((root / "calib/cpu/calibration.json").read_text())
res.append(ok(all(cal["methods"][m].get("status") == "debug_stopped" for m in cal["methods"]) and
              set(cal["objectives"]) == {"pt", "sft", "dpo", "rl"} and set(cal["eval"]) == {"lc", "full"},
              f"calibration: 6 methods, 4 objectives timed, lc+full eval timed; totals {cal.get('totals_gpu_hours', {}).get('all')}"))
sel = json.loads((root / "dev/selection.json").read_text())
res.append(ok(len(sel["selected"]) == 6, f"dev selection: {sel['selected']['jr1']['candidate']} for jr1 ({sel['selected']['jr1']['how']})"))
idx = json.loads(sorted((root / "collect").glob("index.json"))[0].read_text())
done = [c for c in idx["cells"] if c["status"] == "complete"]
res.append(ok(idx["counts"]["03"]["planned"] == 37 and idx["counts"]["04"]["planned"] == 6,
              f"index denominators 37 + 6; counts {idx['counts']}"))
for c in done:
    d = root / "runs" / c["exp"] / c["cell"]
    if c["method"] == "base":
        evs = ["full_000", "lc_000"]
    else:
        evs = [f"lc_{k}" for k in ("000", "025", "050", "075", "100")] + ["full_100"]
        r = json.loads((d / "receipt.json").read_text())
        res.append(ok(r["status"] == "complete" and r["milestones_done"] == [0.0, 0.25, 0.5, 0.75, 1.0],
                      f"{c['exp']}/{c['cell']}: trained, work {r['work_total']}, 5 checkpoints"))
    st = {e: json.loads((d / "eval" / e / "summary.json").read_text())["status"] for e in evs}
    res.append(ok(set(st.values()) == {"complete"}, f"{c['exp']}/{c['cell']}: evals {st}"))
res.append(ok(len(done) == 13, f"{len(done)} cells complete (7 of 03 + 6 of 04 expected)"))
rerun = (root / "logs/cells03_0_rerun.log").read_text()
r0 = json.loads((root / "runs/03/sequential_s1101/receipt.json").read_text())
res.append(ok(r0["resumes"] >= 1 and "cell sequential_s1101 done" in rerun, f"rerun resumed (resumes={r0['resumes']})"))
tgz = sorted((root / "collect").glob("jr1_results_*.tgz"))[-1]
names = tarfile.open(tgz).getnames()
res.append(ok(any(n.endswith("full_100/summary.json") for n in names) and not any("/state/" in n or n.endswith(".safetensors") for n in names),
              f"collect tarball {tgz.name}: {len(names)} files, no weights or optimizer state"))
print("E2E", "PASSED" if all(res) else "FAILED", f"({sum(res)}/{len(res)} checks)")
sys.exit(0 if all(res) else 1)
PY
