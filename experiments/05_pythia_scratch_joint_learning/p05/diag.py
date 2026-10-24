"""Exp 05 diagnosis of the sequential post-training collapse (AT-1024): same code, config and data as the 27 cells.

    python -m p05.diag pt      --seed S --out <root>/s<S>/pt      --config <frozen 14m fp16 json> --data-root DIR
    python -m p05.diag branch  --seed S --out <root>/s<S>/<variant> --pt <root>/s<S>/pt --variant {orig,lowlr,sftonly,ptonly} ...

`pt` trains sequential's PT stage (progress < boundary 0.55) exactly as the cell did, saves the full resumable state, and
reads WikiText-103 test NLL. `branch` copies that state and continues to the frozen work target under a variant:
  orig     the frozen recipe (SFT -> DPO -> RL), read again with a checkpoint and an NLL after every stage
  lowlr    same, but the learning rate is multiplied by --lr-scale (0.1) from the end of PT on
  sftonly  SFT from the end of PT to the target (no DPO, no RL), original learning-rate schedule
  ptonly   control: PT continues to the target (no post-training)
Every stage end writes a fp32 checkpoint and a WikiText NLL (p05.evaluate protocol, HF backend) into <out>/stages/.
The frozen config's sha256 is not checked: this is a diagnostic run, never part of the 27-cell result.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

from jr1 import common as C
from jr1 import train as T

from . import train as P

STAGES = ("pt", "sft", "dpo", "rl")


class DiagTrainer(P.P05Trainer):
    variant = "orig"
    lr_scale = 1.0

    def eff_stage(self, progress: float) -> int:
        if self.variant == "ptonly":
            return 0
        s = self.schedule.stage(progress)
        if self.variant == "sftonly":
            return min(s, 1)
        return s

    def lr_at(self, progress: float) -> float:
        lr = super().lr_at(progress)
        if self.variant == "lowlr" and self.schedule.stage(progress) >= 1:
            lr *= self.lr_scale
        return lr

    def step(self) -> dict:
        # re-route the schedule's weights through the variant's effective stage
        s = self.eff_stage(self.progress())
        sched = self.schedule
        orig = sched.weights
        sched.weights = lambda progress, s=s: [float(i == s) for i in range(len(STAGES))]
        try:
            return super().step()
        finally:
            sched.weights = orig


def read_nll(out: Path) -> dict:
    s = json.loads((out / "summary.json").read_text())
    return {k: s["wikitext"][k] for k in s["wikitext"] if k in ("nll_per_token", "n_tokens", "tokens", "items")}


def stage_ckpt_and_nll(tr: DiagTrainer, name: str, root: Path, data_root: str) -> dict:
    d = root / "stages" / name
    ck = d / "ckpt"
    if ck.exists():
        shutil.rmtree(ck)
    tr.model.save_pretrained(ck, safe_serialization=True)
    tr.tok.save_pretrained(ck)
    C.atomic_json(ck / "jr1_ckpt.json", {"stage": name, "progress": tr.progress(), "work_total": tr.work.total,
                                         "updates": tr.updates, "variant": tr.variant})
    cmd = [sys.executable, "-m", "p05.diag_eval", "--ckpt", str(ck), "--suite", "diag_wiki", "--out", str(d / "eval"),
           "--data-root", data_root]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    nll = json.loads((d / "eval" / "summary.json").read_text())["wikitext"]
    rec = {"stage": name, "progress": tr.progress(), "work_total": tr.work.total, "updates": tr.updates,
           "wikitext": nll, "lr_last": tr.lr_at(min(tr.progress(), 1.0))}
    C.atomic_json(d / "stage.json", rec)
    shutil.rmtree(ck)          # 14m fp32 checkpoint: the NLL is the record; keep disk free
    print(json.dumps({"stage": name, "progress": rec["progress"], "nll": nll.get("nll_per_token", nll)}), flush=True)
    return rec


def build_trainer(a, method="sequential") -> DiagTrainer:
    cfg, sha = T.load_config(a.config)
    cfg["data"]["root"] = a.data_root
    manifest = Path(a.data_root) / "manifest.json"
    tr = DiagTrainer(cfg, a.out, a.seed, method, config_sha=sha,
                     data_manifest=C.sha256(manifest) if manifest.exists() else None, device_info=P.check_device(cfg))
    return tr


def run_loop(tr: DiagTrainer, root: Path, data_root: str, stop_at_stage: int | None) -> None:
    prev = tr.eff_stage(tr.progress())
    while not tr.done():
        tr.step()
        cur = tr.eff_stage(tr.progress()) if not tr.done() else None
        if cur != prev:
            stage_ckpt_and_nll(tr, f"after_{STAGES[prev]}", root, data_root)
            tr.save_state()
            if stop_at_stage is not None and cur is not None and cur >= stop_at_stage:
                return
            prev = cur if cur is not None else prev
    if tr.done() and not (root / "stages" / f"after_{STAGES[prev]}" / "stage.json").exists():
        stage_ckpt_and_nll(tr, f"after_{STAGES[prev]}", root, data_root)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("phase", choices=["pt", "branch"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--pt", type=Path, help="branch: the pt phase out dir")
    ap.add_argument("--variant", choices=["orig", "lowlr", "sftonly", "ptonly"], default="orig")
    ap.add_argument("--lr-scale", type=float, default=0.1)
    ap.add_argument("--work-target", type=float, help="smoke only")
    a = ap.parse_args(argv)
    a.out = Path(a.out)
    if a.phase == "branch":
        if a.out.exists():
            raise SystemExit(f"{a.out} exists")
        a.out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(a.pt, a.out, ignore=shutil.ignore_patterns("stages"))
        (a.out / "stages").mkdir()
        shutil.copytree(a.pt / "stages", a.out / "stages" / "_pt_shared", dirs_exist_ok=True)
    tr = build_trainer(a)
    if a.work_target:
        tr.target = a.work_target
    tr.variant, tr.lr_scale = (a.variant if a.phase == "branch" else "orig"), a.lr_scale
    try:
        if a.phase == "pt":
            tr.save_ckpt(0.0)      # the frozen DPO reference (random init), as jr1's run() writes it
            run_loop(tr, a.out, a.data_root, stop_at_stage=1)
            C.atomic_json(a.out / "pt_done.json", {"progress": tr.progress(), "updates": tr.updates,
                                                   "work_total": tr.work.total, "seed": a.seed})
        else:
            tr.load_state(a.out / "state")
            run_loop(tr, a.out, a.data_root, stop_at_stage=None)
            C.atomic_json(a.out / "receipt.json", tr.receipt("complete"))
    except BaseException as exc:
        C.atomic_json(a.out / "failed.json", {"error": "".join(traceback.format_exception_only(type(exc), exc)).strip()})
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
