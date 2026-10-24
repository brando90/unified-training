#!/usr/bin/env bash
# Exp 05 job bodies, called by the per-cluster sbatch wrappers (or directly with P05_LOCAL=1):
#   jobs.sh prep          venv (jr1 lock), jr1 data prepare if missing, p05 data prepare + verify, Pythia
#                         config/tokenizer files at the pinned revisions into $HF_HOME; writes PREP_OK
#   jobs.sh calib         p05.calibrate on every size in $P05_PRECISION -> receipts/calibration/
#   jobs.sh cells v1|v2   one array index ($SLURM_ARRAY_TASK_ID, or $P05_INDEX locally)
set -euo pipefail
P05_SH=${P05_SH:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
source "$P05_SH/env.sh"
source "$P05_SH/common.sh"
cd "$P05_STATE_ROOT"
stage=${1:?usage: jobs.sh prep|calib|cells v1|v2}
case $stage in
  prep)
    jr1_log "prep start ($P05_CLUSTER, $P05_PRECISION, code $P05_CODE_COMMIT)"
    rm -f PREP_OK
    [ "${P05_LOCAL:-0}" = 1 ] || bash "$JR1_SH/setup_env.sh"
    p05_py -m p05.envcheck --out receipts/env_prep.json --require none
    if ! { [ -f "$P05_JR1_DATA/manifest.json" ] && p05_py -m jr1.data verify --root "$P05_JR1_DATA"; }; then
      jr1_log "jr1 data prepare into $P05_JR1_DATA (WikiText-103, GSM8K, UltraFeedback, MATH-500, MMLU, ARC-Easy)"
      HF_HUB_OFFLINE=0 p05_py -m jr1.data prepare --root "$P05_JR1_DATA"
      p05_py -m jr1.data verify --root "$P05_JR1_DATA"
    fi
    jr1_log "p05 data prepare -> $P05_DATA_ROOT"
    HF_HUB_OFFLINE=0 p05_py -m p05.data prepare --jr1-data "$P05_JR1_DATA" --out "$P05_DATA_ROOT"
    p05_py -m p05.data verify --out "$P05_DATA_ROOT"
    HF_HUB_OFFLINE=0 "$P05_PY" - <<'PY'
from huggingface_hub import hf_hub_download
from p05 import spec
for s in spec.SIZES.values():
    for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        hf_hub_download(s["id"], f, revision=s["revision"])
print("pythia config/tokenizer files cached")
PY
    date -Iseconds > PREP_OK
    jr1_log "prep done";;
  calib)
    jr1_require PREP_OK "run prep first"
    p05_env_receipt receipts/env_calib.json
    p05_py -m p05.calibrate --precision "$P05_PRECISION" --data-root "$P05_DATA_ROOT" \
      --out receipts/calibration ${P05_CALIB_ARGS:-}
    jr1_log "calibration: receipts/calibration/calibration.md";;
  cells)
    version=${2:?jobs.sh cells v1|v2}
    jr1_require PREP_OK "run prep first"
    p05_cell "$version" "${SLURM_ARRAY_TASK_ID:-${P05_INDEX:?array index}}";;
  *) echo "unknown stage $stage" >&2; exit 2;;
esac
