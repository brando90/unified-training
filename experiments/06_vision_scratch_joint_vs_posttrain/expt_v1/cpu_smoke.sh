#!/bin/bash
# Local CPU smoke (no GPU, no real data): every arm x 3 seeds on synthetic data with the 'smoke' ViT, a stop/resume
# check that must reproduce the uninterrupted run exactly, and analyze.py over the 15 cells.
#   bash cpu_smoke.sh <python with torch, numpy, pillow, matplotlib> <work dir>
set -euo pipefail
PY=${1:?python}; W=${2:?workdir}; C="$(cd "$(dirname "$0")" && pwd)"
rm -rf "$W"; mkdir -p "$W/runs/smoke"
$PY "$C/prep_data.py" --out "$W/data" --synthetic | tail -1
COMMON=(--model smoke --data "$W/data" --steps 100 --batch 24 --eval-every 25 --amp none --device cpu)
printf 'task\tmodel\tarm\tseed\n' > "$W/cells.tsv"; t=0
for arm in target_only sequential sequential_replay fixed_joint smooth_joint; do
  for seed in 1 2 3; do
    printf '%s\tsmoke\t%s\t%s\n' $t $arm $seed >> "$W/cells.tsv"; t=$((t+1))
    $PY "$C/train.py" --arm $arm --seed $seed --out "$W/runs/smoke/$arm/seed$seed" "${COMMON[@]}" > "$W/$arm.$seed.log" 2>&1
    $PY - "$W/runs/smoke/$arm/seed$seed" <<'PYEOF'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1]); fin = json.loads((d / 'final.json').read_text())
cur = [json.loads(l) for l in (d / 'curve.jsonl').read_text().splitlines()]
def first_last(k):
    v = [r[k] for r in cur if r[k] is not None]
    return (round(v[0], 3), round(v[-1], 3)) if v else None
print(f"{fin['arm']:18s} seed{fin['seed']} {fin['status']} src={fin['src_images']} tgt={fin['tgt_images']} "
      f"loss_src first->last={first_last('train_loss_src')} loss_tgt={first_last('train_loss_tgt')} "
      f"cifar={fin['final_cifar100_test_top1']} tin={fin['final_tin200_val_top1']}")
PYEOF
  done
done
# Resume: stop exactly at sequential_replay's phase boundary (step 40 of 100), again mid phase 2 between evals
# (step 57), then finish; the curve and final metrics must equal seed 1's uninterrupted run.
R="$W/resume"
set +e; $PY "$C/train.py" --arm sequential_replay --seed 1 --out "$R" "${COMMON[@]}" --stop-after 40 > "$W/resume.1.log" 2>&1; rc1=$?
$PY "$C/train.py" --arm sequential_replay --seed 1 --out "$R" "${COMMON[@]}" --stop-after 17 > "$W/resume.1b.log" 2>&1; rc2=$?; set -e
echo "stop-after rc=$rc1 $rc2 (expect 3 3)"; test $rc1 -eq 3 && test $rc2 -eq 3
grep -h resumed "$W/resume.1b.log"
$PY "$C/train.py" --arm sequential_replay --seed 1 --out "$R" "${COMMON[@]}" > "$W/resume.2.log" 2>&1
grep -h resumed "$W/resume.2.log"
cmp <(cut -d, -f1-6 "$R/curve.jsonl"; $PY -c "import json;d=json.load(open('$R/final.json'));print(d['final_cifar100_test_top1'],d['final_tin200_val_top1'])") \
    <(cut -d, -f1-6 "$W/runs/smoke/sequential_replay/seed1/curve.jsonl"; $PY -c "import json;d=json.load(open('$W/runs/smoke/sequential_replay/seed1/final.json'));print(d['final_cifar100_test_top1'],d['final_tin200_val_top1'])") \
  && echo "RESUME EXACT: curve and final metrics identical to the uninterrupted run"
$PY "$C/train.py" --arm sequential_replay --seed 1 --out "$R" "${COMMON[@]}" | tail -1   # complete cell: no-op
$PY "$C/analyze.py" --runs "$W/runs" --cells "$W/cells.tsv" --out "$W/analysis" > "$W/analyze.log"
cat "$W/analysis/tables.md"; ls "$W/analysis"
echo CPU SMOKE PASS
