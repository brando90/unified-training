#!/bin/bash
#SBATCH --job-name=exp06_smoke
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:45:00
# GPU smoke: 200 steps per arm for ViT-Tiny and ViT-Small on the real data, then a stop/resume check.
# Prints img/s, peak memory and the projected hours of a full S-step cell; writes $EXP06_ROOT/smoke/PASS on success.
source "${EXP06_CODE:-${SLURM_SUBMIT_DIR:-$(dirname "$0")}}/env.sh"
set -uo pipefail
S_FULL=60000; B=192
SM="$EXP06_ROOT/smoke"; rm -rf "$SM"; mkdir -p "$SM"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
fail=0
for model in tiny small; do
  mkdir -p "$SM/$model"
  for arm in target_only sequential sequential_replay fixed_joint smooth_joint; do
    out="$SM/$model/$arm"
    python "$EXP06_CODE/train.py" --arm $arm --model $model --seed 1 --data "$EXP06_ROOT/data" --out "$out" \
      --steps 200 --eval-every 100 --amp "$EXP06_AMP" > "$out.log" 2>&1
    rc=$?
    line=$(grep '^FINAL' "$out.log" | tail -1)
    echo "$model $arm rc=$rc ${line#FINAL }"
    [ $rc -eq 0 ] || { fail=1; tail -5 "$out.log"; }
  done
done
# Resume check on the GPU: stop exactly at the replay arm's phase boundary (step 80 of 200), again mid phase 2
# (step 110), then finish; must complete with the same images seen as the uninterrupted run above.
r="$SM/resume"; RA=(--arm sequential_replay --model tiny --seed 1 --data "$EXP06_ROOT/data" --out "$r" --steps 200 --eval-every 100 --amp "$EXP06_AMP")
python "$EXP06_CODE/train.py" "${RA[@]}" --stop-after 80 > "$r.1.log" 2>&1; rc1=$?
python "$EXP06_CODE/train.py" "${RA[@]}" --stop-after 30 > "$r.2.log" 2>&1; rc2=$?
python "$EXP06_CODE/train.py" "${RA[@]}" > "$r.3.log" 2>&1; rc3=$?
same=$(python -c "import json,sys;a,b=(json.load(open(p)) for p in sys.argv[1:]);print(int(all(a[k]==b[k] for k in ('src_images','tgt_images','steps_done'))))" "$r/final.json" "$SM/tiny/sequential_replay/final.json" 2>/dev/null)
echo "resume: rc $rc1 $rc2 (expect 3 3), final rc $rc3 (expect 0), same images as uninterrupted: ${same:-0}"
[ $rc1 -eq 3 ] && [ $rc2 -eq 3 ] && [ $rc3 -eq 0 ] && [ "${same:-0}" = 1 ] || fail=1
python - "$SM" "$S_FULL" "$B" <<'EOF' || fail=1
import json, sys, pathlib
sm, S, B = pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
for model in ('tiny', 'small'):
    ips = [json.loads((p / 'final.json').read_text()) for p in sorted((sm / model).iterdir()) if (p / 'final.json').exists()]
    if ips:
        r = min(f['img_per_s'] for f in ips); mem = max(f['peak_mem_gib'] or 0 for f in ips)
        h = S * B / r / 3600
        print(f'{model}: slowest arm {r:.0f} img/s, peak {mem:.2f} GiB -> full cell ~{h:.1f} h of training '
              f'(+ ~21 evals) at S={S}, B={B}')
        if model == 'small' and h > 24:
            sys.exit('GATE FAIL: projected ViT-Small cell > 24 h (PROTOCOL launch gate)')
EOF
if [ $fail -eq 0 ]; then date > "$SM/PASS"; echo SMOKE PASS; else echo SMOKE FAIL; exit 1; fi
