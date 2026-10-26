"""Exp 05 calibration: time one update of each objective and the peak memory per model size, then derive the work target.

    python -m p05.calibrate --precision {fp16,bf16} --data-root DIR --out DIR [--sizes 14m,70m,160m]
        [--warmup 2] [--updates 10] [--margin 1.05]

For each size, a fresh random-init model (DRAFT config, seed 9001, never a measured seed) runs `warmup` untimed then
`updates` timed updates of each objective (pt, sft, dpo, rl) through jr1's Trainer.update, measuring per update: wall
seconds (CUDA-synchronized), charged work (jr1 Work ledger, forward-equivalent tokens), observed PT tokens, RL
zero-variance groups, and peak allocated memory per objective.

Derived per method m (jr1 schedule weights w_o(f), update-count shares at work progress f; c_o = charged work per
update; t_o = wall per update; p = PT tokens per PT update):
    PT tokens(W) = W * integral_0^1 w_pt(f) p / sum_o w_o(f) c_o df
    wall(W)      = W * integral_0^1 sum_o w_o(f) t_o / sum_o w_o(f) c_o df
W_needed(m) = 50M / (PT tokens per unit work). The recommended frozen target is the largest W_needed over methods and
sizes times `margin`, rounded up to two significant figures, the same for every cell of both versions. If any peak
memory exceeds 90% of the device, the pre-declared fallback is to halve every objective's batch (PROTOCOL.md) and
calibrate again.
Writes <out>/calibration.json and <out>/calibration.md.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch

from jr1.controllers import OBJECTIVES, make_schedule

from . import spec
from .train import P05Trainer, check_device

CALIB_SEED = 9001
GRID = 2000


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure(trainer: P05Trainer, obj: str, warmup: int, updates: int) -> dict:
    for _ in range(warmup):
        trainer.update(obj, None, "train")
    sync()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    walls, charges, pt_tokens, zero_var, steps = [], [], [], 0, 0
    for _ in range(updates):
        w0, t0, pt0 = trainer.work.total, time.monotonic(), trainer.exposure["pt"]["tokens"]
        rec = trainer.update(obj, None, "train")
        sync()
        walls.append(time.monotonic() - t0)
        charges.append(trainer.work.total - w0)
        pt_tokens.append(trainer.exposure["pt"]["tokens"] - pt0)
        zero_var += int(rec.get("zero_variance_groups", 0))
        steps += int(rec["optimizer_step"])
    out = {"updates": updates, "wall_s_mean": sum(walls) / updates, "wall_s_max": max(walls),
           "charge_mean": sum(charges) / updates, "charge_max": max(charges),
           "pt_tokens_mean": sum(pt_tokens) / updates, "optimizer_steps": steps}
    if obj == "rl":
        out["zero_variance_groups"] = zero_var
        out["groups"] = updates * trainer.cfg["objectives"]["rl"]["prompts"]
    if torch.cuda.is_available():
        out["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated())
    return out


def rates(cfg: dict, per_obj: dict) -> dict:
    out = {}
    c = [per_obj[o]["charge_mean"] for o in OBJECTIVES]
    t = [per_obj[o]["wall_s_mean"] for o in OBJECTIVES]
    p = per_obj["pt"]["pt_tokens_mean"]
    for m in spec.METHODS:
        sched = make_schedule(m, cfg["methods"][m], CALIB_SEED)
        pt_rate = wall_rate = 0.0
        shares = [0.0] * len(OBJECTIVES)
        for i in range(GRID):
            f = (i + 0.5) / GRID
            w = sched.weights(f)
            denom = sum(wi * ci for wi, ci in zip(w, c))
            pt_rate += w[0] * p / denom / GRID
            wall_rate += sum(wi * ti for wi, ti in zip(w, t)) / denom / GRID
            for k in range(len(OBJECTIVES)):
                shares[k] += w[k] * c[k] / denom / GRID
        out[m] = {"pt_tokens_per_work": pt_rate, "wall_s_per_work": wall_rate,
                  "work_share": dict(zip(OBJECTIVES, shares)),
                  "work_needed_for_target": spec.OBSERVED_PT_TARGET / pt_rate if pt_rate else math.inf}
    return out


def round_up(x: float, sig: int = 2) -> float:
    if not math.isfinite(x) or x <= 0:
        return x
    e = math.floor(math.log10(x)) - sig + 1
    return math.ceil(x / 10 ** e) * 10 ** e


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--precision", choices=sorted(spec.PRECISIONS), required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sizes", default="14m,70m,160m")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--updates", type=int, default=10)
    ap.add_argument("--margin", type=float, default=1.05)
    ap.add_argument("--objective-overrides", help="JSON merged into cfg['objectives'] (CPU smoke only)")
    ap.add_argument("--compute", choices=["fp32"], help="CPU smoke only: train in fp32 instead of --precision")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(a.data_root) / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    per_epoch = manifest.get("pt_blocks", {}).get("trainer_tokens_per_epoch")
    result = {"precision": a.precision, "compute_override": a.compute, "warmup": a.warmup, "updates": a.updates, "margin": a.margin,
              "data_root": a.data_root, "trainer_tokens_per_epoch": per_epoch, "sizes": {}}
    for size in a.sizes.split(","):
        version = next(v for v, sizes in spec.VERSIONS.items() if size in sizes)
        cfg = spec.config(size, a.precision, version)
        cfg["data"]["root"] = a.data_root
        if a.compute:
            cfg["precision"]["compute"] = a.compute
        if a.objective_overrides:
            for k, v in json.loads(a.objective_overrides).items():
                cfg["objectives"][k].update(v)
        device = check_device(cfg)
        tmp = Path(tempfile.mkdtemp(prefix=f"p05calib_{size}_", dir=a.out))
        t0 = time.monotonic()
        trainer = P05Trainer(cfg, tmp, CALIB_SEED, "fixed_joint", device_info=device)
        build_s = time.monotonic() - t0
        t0 = time.monotonic()
        trainer.save_ckpt(0.0)   # the random init as ckpt_000: DPO's frozen reference, as in a measured cell
        ckpt_s = time.monotonic() - t0
        per_obj = {}
        for obj in OBJECTIVES:
            per_obj[obj] = measure(trainer, obj, a.warmup, a.updates)
            print(json.dumps({"size": size, "objective": obj, **{k: per_obj[obj][k] for k in
                              ("wall_s_mean", "charge_mean")}, "peak_gb": round(per_obj[obj].get(
                                  "peak_memory_bytes", 0) / 2**30, 2)}), flush=True)
        r = rates(cfg, per_obj)
        peak = max(x.get("peak_memory_bytes", 0) for x in per_obj.values())
        total = device.get("total_memory_bytes")
        result["sizes"][size] = {"device": device, "build_s": build_s, "checkpoint_s": ckpt_s, "objectives": per_obj, "methods": r,
                                 "fp16_skipped_steps": trainer.skipped(),
                                 "parameters": sum(p.numel() for p in trainer.model.parameters()),
                                 "peak_memory_bytes": peak,
                                 "memory_fallback_needed": bool(total and peak > 0.9 * total)}
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        shutil.rmtree(tmp, ignore_errors=True)

    need = max(m["work_needed_for_target"] for s in result["sizes"].values() for m in s["methods"].values())
    target = round_up(need * a.margin)
    result["recommended_work_target"] = target
    for size, s in result["sizes"].items():
        for m, x in s["methods"].items():
            x["pt_tokens_at_target"] = x["pt_tokens_per_work"] * target
            x["pt_epochs_at_target"] = x["pt_tokens_at_target"] / per_epoch if per_epoch else None
            x["wall_h_at_target"] = x["wall_s_per_work"] * target / 3600
        n_seeds = len(spec.SEEDS)
        s["train_gpu_h_all_seeds"] = sum(x["wall_h_at_target"] for x in s["methods"].values()) * n_seeds
    (a.out / "calibration.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [f"# Exp 05 calibration ({a.precision})", "",
             f"Recommended frozen work target: **{target:.3g}** forward-equivalent tokens per cell "
             f"(max W needed for {spec.OBSERVED_PT_TARGET / 1e6:.0f}M observed PT tokens x {a.margin}).", "",
             "| size | params | objective | s/update | charged work/update | peak GB |", "|---|---|---|---|---|---|"]
    for size, s in result["sizes"].items():
        for obj, x in s["objectives"].items():
            lines.append(f"| {size} | {s['parameters'] / 1e6:.1f}M | {obj} | {x['wall_s_mean']:.3f} | "
                         f"{x['charge_mean']:.0f} | {x.get('peak_memory_bytes', 0) / 2**30:.2f} |")
    lines += ["", "| size | method | PT work share | W needed | PT tokens at target | PT epochs | train h/cell |",
              "|---|---|---|---|---|---|---|"]
    for size, s in result["sizes"].items():
        for m, x in s["methods"].items():
            ep = "n/a" if x["pt_epochs_at_target"] is None else f"{x['pt_epochs_at_target']:.2f}"
            lines.append(f"| {size} | {m} | {x['work_share']['pt']:.3f} | {x['work_needed_for_target']:.3g} | "
                         f"{x['pt_tokens_at_target'] / 1e6:.1f}M | {ep} | {x['wall_h_at_target']:.2f} |")
    lines += ["", "Memory fallback needed: " + ", ".join(f"{k}={v['memory_fallback_needed']}"
                                                      for k, v in result["sizes"].items())]
    (a.out / "calibration.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"recommended_work_target": target, "out": str(a.out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
