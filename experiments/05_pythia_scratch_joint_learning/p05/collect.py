"""Collect Exp 05 cells: status of every prospective cell, charged-work accounting, scores and paired gaps.

    python -m p05.collect --runs <state>/runs --out <dir>

Reads <runs>/<version>/<size>/<method>_s<seed>/{receipt.json, slurm.json, eval/full_000|full_100|lc_XXX/summary.json}
for all 27 prospective cells (9 v1 + 18 v2); a missing cell is listed as unstarted, never dropped. Writes collect.json
and collect.md: per-cell status, Slurm job ids, work total / overshoot / work share by objective, observed PT tokens
(and whether >= 50M), fp16 skipped steps, terminal scores (WikiText test NLL, LAMBADA acc and ppl, ARC-Easy acc,
GSM8K accuracy on all 1,319 with parse failures and a 95% Wilson interval), and per size the paired gaps by seed with
their mean for fixed_joint - sequential, smooth_joint - sequential and fixed_joint - smooth_joint (sign convention:
for NLL/ppl, negative = the first method better).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from . import spec

METRICS = {"wikitext_nll": ("wikitext", "nll_per_token"), "lambada_acc": ("lambada", "acc"),
           "lambada_ppl": ("lambada", "ppl"), "arc_easy_acc": ("arc_easy", "acc"), "gsm8k_acc": ("gsm8k", "acc"),
           "pref_acc": ("pref", "acc")}
CONTRASTS = (("fixed_joint", "sequential"), ("smooth_joint", "sequential"), ("fixed_joint", "smooth_joint"))


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if not n:
        return None
    p = k / n
    centre, half = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [(centre - half) / (1 + z * z / n), (centre + half) / (1 + z * z / n)]


def load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def cell_row(runs: Path, c: dict) -> dict:
    d = runs / c["cell"]
    r = load(d / "receipt.json")
    slurm = load(d / "slurm.json") or {"starts": []}
    row = {**c, "status": "unstarted" if r is None else r["status"], "cell_done": (d / "CELL_DONE").exists(),
           "slurm_job_ids": sorted({f"{s.get('slurm_array_job_id') or s.get('slurm_job_id')}_{s.get('slurm_array_task_id')}"
                                    for s in slurm["starts"] if s.get("slurm_job_id")})}
    if r is not None:
        total = r["work_total"] or 1
        row.update(work_total=r["work_total"], work_target=r["target"], work_overshoot=r["work_overshoot"],
                   overshoot_bounded=r["overshoot_bounded"], updates=r["updates"], optimizer_steps=r["optimizer_steps"],
                   wall_h=r["wall_s"] / 3600, peak_memory_gb=(r.get("peak_memory_bytes") or 0) / 2**30,
                   device=r.get("device"), work_by_purpose=r["work_by_purpose"],
                   updates_by_objective={o: e["updates"] for o, e in r["exposure"].items()},
                   tokens_by_objective={o: e["tokens"] for o, e in r["exposure"].items()},
                   rl_zero_variance_groups=r["exposure"]["rl"].get("zero_variance_groups"),
                   rl_informative_groups=r["exposure"]["rl"].get("informative_groups"))
        p = r.get("p05") or {}
        row.update(observed_pt_tokens=p.get("observed_pt_tokens"), reached_pt_target=p.get("reached_pt_target"),
                   fp16_skipped_steps=p.get("fp16_skipped_steps"))
    for tag in ("full_000", "full_100"):
        s = load(d / "eval" / tag / "summary.json")
        if s:
            for name, (bench, key) in METRICS.items():
                if bench in s:
                    row[f"{name}@{tag[-3:]}"] = s[bench].get(key)
            if "gsm8k" in s:
                row[f"gsm8k_n@{tag[-3:]}"] = s["gsm8k"]["n"]
                row[f"gsm8k_parse_fail@{tag[-3:]}"] = s["gsm8k"]["parse_fail"]
                row[f"gsm8k_wilson95@{tag[-3:]}"] = wilson(int(s["gsm8k"]["correct"]), int(s["gsm8k"]["n"]))
    row["lc"] = {ck: load(d / "eval" / f"lc_{ck}" / "summary.json") is not None for ck in ("000", "025", "050", "075", "100")}
    return row


def gaps(rows: list[dict]) -> dict:
    out = {}
    sizes = sorted({r["size"] for r in rows}, key=lambda s: float(s[:-1]))
    for size in sizes:
        out[size] = {}
        for first, second in CONTRASTS:
            name = f"{first}-{second}"
            out[size][name] = {}
            for metric in METRICS:
                key = f"{metric}@100"
                diffs = []
                for seed in spec.SEEDS:
                    a = next((r.get(key) for r in rows if r["size"] == size and r["seed"] == seed and r["method"] == first), None)
                    b = next((r.get(key) for r in rows if r["size"] == size and r["seed"] == seed and r["method"] == second), None)
                    diffs.append(None if a is None or b is None else a - b)
                done = [x for x in diffs if x is not None]
                out[size][name][metric] = {"by_seed": diffs, "n_pairs": len(done),
                                            "mean": sum(done) / len(done) if done else None,
                                            "all_same_sign": bool(done) and (all(x > 0 for x in done) or all(x < 0 for x in done))}
    return out


def fmt(x, nd=3):
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    rows = [cell_row(a.runs, c) for v in spec.VERSIONS for c in spec.cells(v)]
    g = gaps(rows)
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "collect.json").write_text(json.dumps({"cells": rows, "gaps": g, "status_counts": counts,
                                                    "prospective_cells": len(rows)}, indent=2, sort_keys=True) + "\n")
    lines = [f"# Exp 05 cells ({len(rows)} prospective): " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())), "",
             "| cell | status | job ids | work total | overshoot | PT tokens | >=50M | fp16 skipped | WikiText NLL | "
             "LAMBADA acc | LAMBADA ppl | ARC-E acc | GSM8K acc (n) [95% Wilson] |", "|" + "---|" * 13]
    for r in rows:
        lines.append("| " + " | ".join([
            r["cell"], r["status"], " ".join(r["slurm_job_ids"]) or "–", fmt(r.get("work_total")),
            fmt(r.get("work_overshoot")), fmt(r.get("observed_pt_tokens")), fmt(r.get("reached_pt_target")),
            fmt(r.get("fp16_skipped_steps")), fmt(r.get("wikitext_nll@100")), fmt(r.get("lambada_acc@100")),
            fmt(r.get("lambada_ppl@100"), 1), fmt(r.get("arc_easy_acc@100")),
            f"{fmt(r.get('gsm8k_acc@100'))} ({fmt(r.get('gsm8k_n@100'))})" + (
                " [{:.3f}, {:.3f}]".format(*r["gsm8k_wilson95@100"]) if r.get("gsm8k_wilson95@100") else "")]) + " |")
    lines += ["", "## Paired gaps at ckpt_100 (first minus second method, by seed 1/2/3)", "",
              "| size | contrast | metric | seed gaps | mean | same sign |", "|---|---|---|---|---|---|"]
    for size, by in g.items():
        for name, ms in by.items():
            for metric, x in ms.items():
                lines.append(f"| {size} | {name} | {metric} | {', '.join(fmt(d, 4) for d in x['by_seed'])} | "
                             f"{fmt(x['mean'], 4)} | {x['all_same_sign']} |")
    (a.out / "collect.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status_counts": counts, "out": str(a.out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
