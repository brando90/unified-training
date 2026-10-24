#!/usr/bin/env bash
# Exp 05 CPU smoke test (no GPU, offline after the Pythia config/tokenizer files are cached):
#  1. synthetic jr1-layout data -> p05.data prepare (64-token packed blocks) + verify
#  2. all 3 methods x 3 sizes (14m, 70m, 160m; real Pythia architectures, random init) through p05.train/jr1.train
#     with tiny batches and a tiny work target, from the DRAFT configs with only batch sizes, lengths and the work
#     target shrunk (fp32 compute on CPU; the bf16 and fp16 autocast paths once each on 14m); each cell must finish 'complete' with every objective updated at least once
#  3. one cell through the cluster job body (cluster/jobs.sh cells v2, P05_LOCAL=1): train, scratch_lc at 5
#     checkpoints, scratch_full at 000 and 100, CELL_DONE
#  4. p05.calibrate on all 3 sizes (2 timed updates per objective) and p05.collect over the runs
# Usage: bash smoke_cpu.sh [python]   (from experiments/05_pythia_scratch_joint_learning)
# Needs ~25 GB free under $P05_SMOKE (fp32 checkpoints and resume state of 12 cells, 160m ones ~5 GB each).
set -euo pipefail
PY=${1:-python}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH=$HERE:$HERE/../03_qwen_joint_reasoning_retention/expt_v1${PYTHONPATH:+:$PYTHONPATH}
export P05_SMOKE=${P05_SMOKE:-$(mktemp -d)}
export TOKENIZERS_PARALLELISM=false
echo "smoke root $P05_SMOKE"
$PY -m p05.synthetic --out "$P05_SMOKE/jr1_data" --n 24
$PY -m p05.data prepare --jr1-data "$P05_SMOKE/jr1_data" --out "$P05_SMOKE/data" --block-len 64 \
  --lambada-parquet "$P05_SMOKE/jr1_data/lambada.parquet" --no-manifest-check
$PY -m p05.data verify --out "$P05_SMOKE/data"

# Smoke configs: the committed DRAFT configs with tiny objectives (and nothing else changed).
mkdir -p "$P05_SMOKE/configs"
$PY - "$P05_SMOKE/configs" <<'PY'
import json, sys
from pathlib import Path
from p05 import spec
from p05.freeze import HOME
out = Path(sys.argv[1])
tiny = {"pt": {"batch_size": 2, "max_len": 64}, "sft": {"batch_size": 2, "max_len": 64},
        "dpo": {"batch_size": 1, "max_len": 64}, "rl": {"prompts": 1, "group_size": 2, "max_new_tokens": 8, "max_prompt_len": 32}}
for v, sizes in spec.VERSIONS.items():
    for size in sizes:
        for prec in spec.PRECISIONS:
            name = spec.config_name(size, prec)
            cfg = json.loads((HOME / f"expt_{v}" / "configs" / name).read_text())
            for k, o in tiny.items():
                cfg["objectives"][k].update(o)
            cfg["work"]["target"] = 2500
            cfg["protocol"] += "-CPU-SMOKE"
            (out / name).write_text(json.dumps(cfg, indent=2, sort_keys=True))
            if prec == "bf16":   # CPUs without bf16 units run autocast ~10x slower: the 3x3 grid trains in fp32
                cfg["precision"]["compute"] = "fp32"
                cfg["protocol"] = cfg["protocol"].replace("bf16", "fp32")
                (out / name.replace("bf16", "fp32")).write_text(json.dumps(cfg, indent=2, sort_keys=True))
PY

for size in 14m 70m 160m; do
  v=v2; [ $size = 160m ] && v=v1
  for method in sequential fixed_joint smooth_joint; do
    for prec in fp32 bf16 fp16; do
      # the bf16 and fp16 autocast paths (GPU paths; slow on CPU): once each, on 14m fixed_joint
      [ $prec != fp32 ] && { [ $size != 14m ] || [ $method != fixed_joint ]; } && continue
      $PY -m p05.train --method $method --seed 1 --config "$P05_SMOKE/configs/pythia${size}_${prec}.json" \
        --out "$P05_SMOKE/runs/$v/$size/${method}_s1_$prec" --data-root "$P05_SMOKE/data" --allow-draft
    done
  done
done
$PY - "$P05_SMOKE/runs" <<'PY'
import json, sys
from pathlib import Path
rows, bad = [], []
for rec in sorted(Path(sys.argv[1]).glob("*/*/*/receipt.json")):
    r = json.loads(rec.read_text())
    ups = {o: e["updates"] for o, e in r["exposure"].items()}
    ok = r["status"] == "complete" and all(ups.values()) and r["overshoot_bounded"] and \
        sorted(p.name for p in rec.parent.glob("ckpt_*")) == ["ckpt_000", "ckpt_025", "ckpt_050", "ckpt_075", "ckpt_100"]
    rows.append(f"| {rec.parent.relative_to(Path(sys.argv[1]))} | {r['status']} | {r['precision']['compute']} | "
                f"{ups['pt']}/{ups['sft']}/{ups['dpo']}/{ups['rl']} | {r['optimizer_steps']} | {r['work_total']} | "
                f"{r['work_overshoot']} | {r['p05']['observed_pt_tokens']} | {'PASS' if ok else 'FAIL'} |")
    bad += [] if ok else [str(rec)]
print("| cell | status | compute | updates pt/sft/dpo/rl | optimizer steps | work total | overshoot | PT tokens | check |")
print("|---|---|---|---|---|---|---|---|---|")
print("\n".join(rows))
print(f"{len(rows) - len(bad)}/{len(rows)} smoke cells pass")
sys.exit(1 if bad or len(rows) != 11 else 0)
PY

# One cell through the cluster job body (v2 index 0 = 14m sequential seed 1), with evaluation.
mkdir -p "$P05_SMOKE/state" && date -Iseconds > "$P05_SMOKE/state/PREP_OK"
P05_LOCAL=1 P05_CLUSTER=local P05_STATE_ROOT=$P05_SMOKE/state P05_CODE=$(cd "$HERE/../.." && pwd) \
  P05_DATA_ROOT=$P05_SMOKE/data P05_PY=$(command -v $PY) P05_PRECISION=fp32 P05_CONFIG_DIR=$P05_SMOKE/configs \
  P05_TRAIN_ARGS=--allow-draft P05_INDEX=0 HF_HUB_OFFLINE=1 \
  bash "$HERE/cluster/jobs.sh" cells v2
d=$P05_SMOKE/state/runs/v2/14m/sequential_s1
test -f "$d/CELL_DONE"
for e in lc_000 lc_025 lc_050 lc_075 lc_100 full_000 full_100; do
  $PY -c 'import json,sys; s=json.load(open(sys.argv[1])); assert "lambada" in s and "wikitext" in s, sys.argv[1]; print(sys.argv[1].split("/")[-2], {b: s[b].get("acc", s[b].get("nll_per_token")) for b in s["benches"]})' "$d/eval/$e/summary.json"
done
$PY -m p05.collect --runs "$P05_SMOKE/state/runs" --out "$P05_SMOKE/collect"
head -3 "$P05_SMOKE/collect/collect.md"

$PY -m p05.calibrate --precision bf16 --compute fp32 --data-root "$P05_SMOKE/data" --out "$P05_SMOKE/calib" --warmup 1 --updates 2 \
  --objective-overrides '{"pt": {"batch_size": 2, "max_len": 64}, "sft": {"batch_size": 2, "max_len": 64}, "dpo": {"batch_size": 1, "max_len": 64}, "rl": {"prompts": 1, "group_size": 2, "max_new_tokens": 8, "max_prompt_len": 32}}'
cat "$P05_SMOKE/calib/calibration.md"
echo "SMOKE OK ($P05_SMOKE)"
