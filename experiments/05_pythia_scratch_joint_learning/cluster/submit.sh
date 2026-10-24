#!/usr/bin/env bash
# Submit one Exp 05 stage on this cluster, recording every job id in $P05_STATE_ROOT/jobs.txt.
#   submit.sh prep | calib | cells_v1 | cells_v2 [extra sbatch args, e.g. --dependency=afterok:123]
# Sherlock -> cluster/sherlock_v100/*.sbatch (V100 32GB, fp16); FarmShare -> cluster/farmshare_l40s/*.sbatch (L40S, bf16).
# Environment: P05_TEST_ONLY=1 (sbatch --test-only: validates the script and constraint, submits nothing),
# P05_ARRAY (override the array, e.g. 3,7 to rerun two cells; they resume), P05_THROTTLE (default 4 at once),
# P05_TIME (override -t), P05_CONSTRAINT (override -C on Sherlock).
set -euo pipefail
source "${P05_SH:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/env.sh"
stage=${1:?usage: submit.sh prep|calib|cells_v1|cells_v2 [sbatch args]}; shift
case $P05_CLUSTER in
  sherlock) dir=$P05_SH/sherlock_v100;;
  farmshare) dir=$P05_SH/farmshare_l40s;;
  *) echo "submit.sh runs on sherlock or farmshare, not $P05_CLUSTER" >&2; exit 2;;
esac
case $stage in
  prep|calib) array=;;
  cells_v1) array=0-8;;
  cells_v2) array=0-17;;
  *) echo "unknown stage $stage" >&2; exit 2;;
esac
args=(--export=ALL,P05_STATE_ROOT=$P05_STATE_ROOT,P05_CLUSTER=$P05_CLUSTER)
if [ -n "$array" ]; then args+=(--array="${P05_ARRAY:-$array}%${P05_THROTTLE:-4}"); fi
[ -n "${P05_TIME:-}" ] && args+=(-t "$P05_TIME")
[ -n "${P05_CONSTRAINT:-}" ] && args+=(-C "$P05_CONSTRAINT")
cd "$P05_STATE_ROOT"
mkdir -p logs
if [ "${P05_TEST_ONLY:-0}" = 1 ]; then
  sbatch --test-only "${args[@]}" "$@" "$dir/$stage.sbatch"
  exit 0
fi
id=$(sbatch --parsable "${args[@]}" "$@" "$dir/$stage.sbatch")
echo "$(date -Iseconds) $stage $id cluster=$P05_CLUSTER precision=$P05_PRECISION array=${array:-none} code=$P05_CODE_COMMIT args=$*" | tee -a jobs.txt
echo "$id"
