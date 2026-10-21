#!/usr/bin/env bash
# CPU end-to-end of the trainer CLI on the tiny offline Qwen2: every method for 2 updates, a recorded
# schedule, and both Exp 04 replay orders. Usage: bash jr1/e2e_cpu.sh [python]   (from expt_v1/)
set -euo pipefail
PY=${1:-python}
export JR1_STATE_ROOT=${JR1_STATE_ROOT:-$(mktemp -d)}
$PY -c "import os; from jr1.tiny import write_synthetic_data; write_synthetic_data(os.environ['JR1_STATE_ROOT'] + '/data')"
CFG=jr1/configs/tiny_cpu.json
OUT=$JR1_STATE_ROOT/e2e
for m in sequential replay fixed_joint smooth_joint aioli jr1; do
  extra=()
  [ "$m" = jr1 ] && extra=(--record-schedule "$OUT/schedule_jr1_9001.jsonl")
  seed=1101; [ "$m" = jr1 ] && seed=9001
  $PY -m jr1.train --method $m --seed $seed --config $CFG --out "$OUT/$m" --max-updates 2 "${extra[@]}"
done
for order in interleaved stage_sorted; do
  $PY -m jr1.train --replay-schedule "$OUT/schedule_jr1_9001.jsonl" --order $order --seed 1101 --config $CFG \
      --out "$OUT/replay_$order"
done
$PY - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
print("| cell | status | actions | updates (objective/purpose, optimizer step) | work total | checkpoints |")
print("|---|---|---|---|---|---|")
for cell in sorted(p for p in out.iterdir() if p.is_dir()):
    r = json.loads((cell / "receipt.json").read_text())
    log = [json.loads(l) for l in (cell / "log.jsonl").read_text().splitlines()]
    ups = ", ".join(f"{x['objective']}/{x['purpose']}{'' if x['optimizer_step'] else ' (zero-grad)'}" for x in log if x["action"] == "update")
    ck = " ".join(sorted(p.name[5:] for p in cell.glob("ckpt_*")))
    print(f"| {cell.name} | {r['status']} | {r['actions']} | {ups} | {r['work_total']} | {ck} |")
PY
