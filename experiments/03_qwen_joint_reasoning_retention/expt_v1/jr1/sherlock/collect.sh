#!/usr/bin/env bash
# Pack every small result file of the campaign for fetching (no weights, no optimizer state):
# receipts, manifests, calibration, dev selection, per-cell receipt/log/slurm/eval summaries and per-item files,
# Slurm logs, jobs.txt, and index.json (one row per planned cell over the full 37 + 6 denominators).
#   collect.sh [--light]    (--light leaves out per-item eval files)
# Output: $JR1_STATE_ROOT/collect/jr1_results_<stamp>.tgz and .sha256 (fetch with OnDemand ?download=1).
set -euo pipefail
source "${JR1_SH:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/env.sh"
cd "$JR1_STATE_ROOT"
mkdir -p collect
"$JR1_PY" -m jr1.sherlock.tools index --root . --out collect/index.json
stamp=$(date +%Y%m%d_%H%M%S)
list=collect/files_$stamp.txt
{
  ls -1 PREP_OK jobs.txt collect/index.json 2>/dev/null || true
  find receipts models/model_manifest.json data/manifest.json -maxdepth 1 -type f 2>/dev/null || true
  find logs -type f -name '*.log' 2>/dev/null || true
  find calib -type f \( -name '*.json' -o -name 'log.jsonl' \) -not -path '*/ckpt_*/*' -not -path '*/state/*' \
    -not -name 'items_*' 2>/dev/null || true
  find dev runs -type f \( -name 'receipt.json' -o -name 'log.jsonl' -o -name 'slurm.json' -o -name 'env.json' \
    -o -name 'config.json' -o -name 'protocol_config.json' -o -name 'dev_task.json' -o -name 'CELL_DONE' \
    -o -name 'selection.json' -o -name 'summary.json' -o -name 'jr1_ckpt.json' -o -name 'schedule_*' \) \
    -not -path '*/state/*' 2>/dev/null | grep -v '/ckpt_[0-9]*/config.json$' || true
  if [ "${1:-}" != --light ]; then
    find dev runs -type f -name 'items_*.jsonl' 2>/dev/null || true
  fi
} | sort -u > "$list"
out=collect/jr1_results_$stamp.tgz
tar czf "$out" -T "$list"
sha256sum "$out" | tee "$out.sha256"
echo "files: $(wc -l < "$list")  size: $(du -h "$out" | cut -f1)"
echo "fetch: https://ondemand.sherlock.stanford.edu/pun/sys/dashboard/files/fs/$(pwd)/$out?download=1"
