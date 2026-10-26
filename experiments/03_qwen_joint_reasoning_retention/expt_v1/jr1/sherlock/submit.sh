#!/usr/bin/env bash
# Submit one JR1 stage with the pinned GPU type, recording every job id in $JR1_STATE_ROOT/jobs.txt.
#   submit.sh prep | calib | dev | schedule04 | cells03 | cells04 [extra sbatch args, e.g. --dependency=afterok:123]
# Environment: GPU_SKU (default L40S; one SKU for all measured cells), JR1_TEST_ONLY=1 (sbatch --test-only, nothing
# submitted), JR1_TIME (override -t), JR1_ARRAY (override the array range, e.g. 7,13 to rerun two cells; they resume).
set -euo pipefail
source "${JR1_SH:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/env.sh"
stage=${1:?usage: submit.sh prep|calib|dev|schedule04|cells03|cells04 [sbatch args]}; shift
case $stage in
  prep|calib|schedule04) array=;;
  dev) array=0-24;;
  cells03) array=0-36;;
  cells04) array=0-5;;
  *) echo "unknown stage $stage" >&2; exit 2;;
esac
array=${JR1_ARRAY:-$array}
args=(-C "GPU_SKU:$GPU_SKU" --export=ALL,JR1_STATE_ROOT=$JR1_STATE_ROOT,GPU_SKU=$GPU_SKU)
[ -n "$array" ] && args+=(--array="$array")
[ -n "${JR1_TIME:-}" ] && args+=(-t "$JR1_TIME")
cd "$JR1_STATE_ROOT"
mkdir -p logs
if [ "${JR1_TEST_ONLY:-0}" = 1 ]; then
  sbatch --test-only "${args[@]}" "$@" "$JR1_SH/$stage.sbatch"
  exit 0
fi
id=$(sbatch --parsable "${args[@]}" "$@" "$JR1_SH/$stage.sbatch")
echo "$(date -Iseconds) $stage $id GPU_SKU=$GPU_SKU array=${array:-none} code=$JR1_CODE_COMMIT args=$*" | tee -a jobs.txt
echo "$id"
