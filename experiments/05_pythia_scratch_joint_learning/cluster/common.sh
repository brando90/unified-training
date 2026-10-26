# Shared functions for the Exp 05 job bodies (sourced after env.sh). jr1's common.sh supplies the requeue-on-USR1
# trap, jr1_run (background + wait so the trap fires), jr1_log, jr1_slurm_note, jr1_status_is and jr1_require.
source "$JR1_SH/common.sh"

p05_py() { jr1_run "$P05_PY" "$@"; }

p05_env_receipt() { p05_py -m p05.envcheck --out "$1" --require "$P05_REQUIRE"; }

p05_eval() {   # p05_eval <ckpt> <suite> <out>
  p05_py -m p05.evaluate --ckpt "$1" --suite "$2" --out "$3" --data-root "$P05_DATA_ROOT" --resume ${P05_EVAL_ARGS:-}
}

# scratch_lc at every retained checkpoint (0/25/50/75/100% of charged work) and scratch_full at the untouched start
# and the terminal checkpoint. All evaluation is charged as wall/device time in the receipts, never fed back.
p05_eval_cell() {
  local d=$1 ck
  for ck in 000 025 050 075 100; do
    jr1_log "eval scratch_lc ckpt_$ck"
    p05_eval "$d/ckpt_$ck" scratch_lc "$d/eval/lc_$ck"
  done
  for ck in 000 100; do
    jr1_log "eval scratch_full ckpt_$ck"
    p05_eval "$d/ckpt_$ck" scratch_full "$d/eval/full_$ck"
  done
}

# One measured cell of version $1 at array index $2: train (resumable), evaluate, CELL_DONE.
p05_cell() {
  local version=$1 index=$2 cell size seed method config
  eval "$("$P05_PY" -m p05.cells --version "$version" --index "$index" --precision "$P05_PRECISION")"
  local d=$P05_STATE_ROOT/runs/$cell
  if [ -f "$d/CELL_DONE" ]; then jr1_log "cell $cell already done"; return 0; fi
  mkdir -p "$d"
  jr1_slurm_note "$d"
  p05_env_receipt "$d/env.json"
  jr1_require "$config" "the Exp 05 config for $size/$P05_PRECISION"
  cp -n "$config" "$d/protocol_config.json" 2>/dev/null || true
  jr1_log "train $cell ($method, seed $seed, $size, $P05_PRECISION)"
  p05_py -m p05.train --method "$method" --seed "$seed" --config "$config" --out "$d" \
    --data-root "$P05_DATA_ROOT" --resume ${P05_TRAIN_ARGS:-}
  jr1_status_is "$d/receipt.json" complete || { jr1_log "train did not complete"; return 7; }
  p05_eval_cell "$d"
  date -Iseconds > "$d/CELL_DONE"
  jr1_log "cell $cell done"
}
