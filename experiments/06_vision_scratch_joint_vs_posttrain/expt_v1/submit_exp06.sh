#!/bin/bash
# Exp 06 driver. Run each stage in order from the unpacked expt_v1 folder:
#   bash submit_exp06.sh build      # venv (FarmShare: sbatch on normal; Sherlock: login node)
#   bash submit_exp06.sh download   # Sherlock only: fetch both archives on the login node (compute may lack internet)
#   bash submit_exp06.sh prep       # data -> $EXP06_ROOT/data (FarmShare: sbatch; Sherlock: foreground srun -p dev)
#   bash submit_exp06.sh smoke      # 200 steps per arm on one GPU, resume check; writes smoke/PASS
#   bash submit_exp06.sh cells      # 30-task array (%4) + post job (afterany); job ids appended to $EXP06_ROOT/jobs.txt
#   bash submit_exp06.sh resubmit 7 12   # rerun array tasks that ended early (they resume), then a fresh post job
#   bash submit_exp06.sh post       # rerun only the analysis + results tarball
#   bash submit_exp06.sh status     # sacct summary + cells with final.json
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
L="$EXP06_ROOT/logs"
case "${1:-}" in
  build)
    if [ "$CLUSTER" = sherlock ]; then bash env.sh build
    else sbatch --parsable $CPU_FLAGS -c 4 --mem=16G -t 01:00:00 $MAIL_FLAGS -J exp06_build -o "$L/build_%j.out" \
           --wrap "cd $EXP06_CODE && bash env.sh build" | sed 's/^/build /' | tee -a "$EXP06_ROOT/jobs.txt"; fi ;;
  download)
    mkdir -p "$EXP06_ROOT/raw"; cd "$EXP06_ROOT/raw"
    curl -sSL -o cifar-100-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
    curl -sSL -o tiny-imagenet-200.zip https://cs231n.stanford.edu/tiny-imagenet-200.zip
    sha256sum cifar-100-python.tar.gz tiny-imagenet-200.zip ;;
  prep)
    if [ "$CLUSTER" = sherlock ]; then
      PREP_ARGS=--no-download srun -p dev -c 4 --mem=16G -t 01:00:00 bash "$EXP06_CODE/prep.sbatch"
    else
      sbatch --parsable $CPU_FLAGS $MAIL_FLAGS -o "$L/prep_%j.out" "$EXP06_CODE/prep.sbatch" | sed 's/^/prep /' | tee -a "$EXP06_ROOT/jobs.txt"
    fi ;;
  smoke)
    test -f "$EXP06_ROOT/data/manifest.json" || { echo "run prep first"; exit 1; }
    if [ "$CLUSTER" = sherlock ]; then
      srun -p dev --gres=gpu:1 -c 4 --mem=32G -t 00:45:00 bash "$EXP06_CODE/smoke.sbatch" 2>&1 | tee "$L/smoke_dev.out"
    else
      sbatch --parsable $GPU_FLAGS $MAIL_FLAGS -o "$L/smoke_%j.out" "$EXP06_CODE/smoke.sbatch" | sed 's/^/smoke /' | tee -a "$EXP06_ROOT/jobs.txt"
    fi ;;
  cells)
    test -f "$EXP06_ROOT/smoke/PASS" || { echo "smoke has not passed: no $EXP06_ROOT/smoke/PASS"; exit 1; }
    grep -q '^Status: FROZEN' "$EXP06_CODE/PROTOCOL.md" || { echo "PROTOCOL.md is not frozen"; exit 1; }
    arr=$(sbatch --parsable $GPU_FLAGS --array=0-29%4 $MAIL_FLAGS -o "$L/cells_%A_%a.out" "$EXP06_CODE/cells.sbatch")
    post=$(sbatch --parsable $CPU_FLAGS --dependency=afterany:$arr $MAIL_FLAGS -o "$L/post_%j.out" "$EXP06_CODE/post.sbatch")
    echo "cells $arr" | tee -a "$EXP06_ROOT/jobs.txt"
    echo "post $post" | tee -a "$EXP06_ROOT/jobs.txt" ;;
  resubmit)
    shift; [ $# -gt 0 ] || { echo "usage: resubmit <task> [<task> ...]"; exit 1; }
    idx=$(IFS=,; echo "$*")
    arr=$(sbatch --parsable $GPU_FLAGS --array=$idx%4 $MAIL_FLAGS -o "$L/cells_%A_%a.out" "$EXP06_CODE/cells.sbatch")
    post=$(sbatch --parsable $CPU_FLAGS --dependency=afterany:$arr $MAIL_FLAGS -o "$L/post_%j.out" "$EXP06_CODE/post.sbatch")
    echo "resubmit[$idx] $arr" | tee -a "$EXP06_ROOT/jobs.txt"
    echo "post $post" | tee -a "$EXP06_ROOT/jobs.txt" ;;
  post)
    sbatch --parsable $CPU_FLAGS $MAIL_FLAGS -o "$L/post_%j.out" "$EXP06_CODE/post.sbatch" | sed 's/^/post /' | tee -a "$EXP06_ROOT/jobs.txt" ;;
  status)
    cat "$EXP06_ROOT/jobs.txt" 2>/dev/null || true
    ids=$(awk '{print $2}' "$EXP06_ROOT/jobs.txt" 2>/dev/null | paste -sd, -)
    [ -n "$ids" ] && sacct -X -j "$ids" --format=JobID%20,JobName%14,State%12,Elapsed,NodeList%12 | cut -c1-90
    echo "complete cells: $(grep -l '"status": "complete"' "$EXP06_ROOT"/runs/*/*/seed*/final.json 2>/dev/null | wc -l)/30" ;;
  *) sed -n 2,11p "$0"; exit 1 ;;
esac
