"""Exp 05 AT-1085: DPO + RL with the reference set to the post-SFT policy (the usual DPO reference) instead of the random init.

    python -m p05.dpo_refsft --seed S --variant {orig,lowlr} --from <dir written by p05.dpo_ref_check> --out <dir> --config ... --data-root ...

Copies the post-SFT state, replaces ckpt_000 (jr1's frozen DPO reference) with the post-SFT weights, then continues
DPO -> RL to the frozen work target exactly as `p05.diag branch`, with a WikiText NLL after every stage. Nothing else changes.
"""
from __future__ import annotations

import argparse
import shutil
import traceback
from pathlib import Path

from jr1 import common as C

from . import diag as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--variant", choices=["orig", "lowlr"], required=True)
    ap.add_argument("--lr-scale", type=float, default=0.1)
    ap.add_argument("--from", dest="src", type=Path, required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists")
    shutil.copytree(a.src, a.out)
    tr = D.build_trainer(a)
    tr.variant, tr.lr_scale = a.variant, a.lr_scale
    try:
        tr.load_state(a.out / "state")
        assert tr.eff_stage(tr.progress()) == 2, tr.progress()
        shutil.rmtree(a.out / "ckpt_000")
        tr.save_ckpt(0.0)                      # the post-SFT policy is now the frozen DPO reference
        tr.reference = None
        D.run_loop(tr, a.out, a.data_root, stop_at_stage=None)
        C.atomic_json(a.out / "receipt.json", tr.receipt("complete"))
    except BaseException as exc:
        C.atomic_json(a.out / "failed.json", {"error": "".join(traceback.format_exception_only(type(exc), exc)).strip()})
        raise


if __name__ == "__main__":
    main()
