"""Exp 05 AT-1111: post-SFT-reference DPO -> RL, with the RL stage capped at a fixed number of RL optimizer steps.

    python -m p05.dpo_refsft_matched --seed S --variant lowlr --rl-steps N --from <chk dir> --out <dir> --config ... --data-root ...

Same as p05.dpo_refsft (post-SFT policy replaces the random-init DPO reference), but the RL stage stops after N RL optimizer
steps (N = the RL optimizer steps of the random-init-reference run of the same seed/variant), not at the charged-work target.
The learning rate of the RL stage follows the same cosine path as in the random-init-reference run (progress from the RL
boundary to 1 is traversed over N steps instead of over the work budget), so lr per RL step matches. WikiText NLL after each stage.
"""
from __future__ import annotations

import argparse
import json
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
    ap.add_argument("--rl-steps", type=int, required=True)
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
        tr.save_ckpt(0.0)
        tr.reference = None
        rl0 = tr.schedule.boundaries[2]
        n = a.rl_steps
        rl_steps = lambda: tr.exposure["rl"]["optimizer_steps"]
        lr_base, done_base = tr.lr_at, tr.done

        def lr_at(progress):
            if tr.schedule.stage(progress) >= 3:
                progress = rl0 + min(1.0, rl_steps() / n) * (1.0 - rl0)
            return lr_base(progress)

        tr.lr_at = lr_at
        tr.done = lambda: done_base() or rl_steps() >= n
        D.run_loop(tr, a.out, a.data_root, stop_at_stage=None)
        rec = tr.receipt("complete")
        C.atomic_json(a.out / "receipt.json", rec)
        C.atomic_json(a.out / "matched.json", {"rl_steps_cap": n, "rl_optimizer_steps": rl_steps(), "rl_boundary": rl0,
                                               "progress_end": tr.progress(), "work_total": tr.work.total,
                                               "rl_exposure": tr.exposure["rl"]})
    except BaseException as exc:
        C.atomic_json(a.out / "failed.json", {"error": "".join(traceback.format_exception_only(type(exc), exc)).strip()})
        raise


if __name__ == "__main__":
    main()
