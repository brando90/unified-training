"""JR1 Sherlock helpers: environment receipt, model/data preparation checks, cell tables, dev selection, calibration.

The sbatch scripts in this directory call these subcommands; every one also runs on a CPU laptop with the tiny model
(JR1_TINY=1), which is how e2e_cpu_sherlock.sh tests the whole chain. Usage: python -m jr1.sherlock.tools <cmd> --help
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from .. import common as C

HERE = Path(__file__).resolve().parent
JR1 = HERE.parent
DEV_GRID = HERE / "dev_grid.json"
SEEDS03 = tuple(range(1101, 1107))           # PLAN: six paired seeds
SEEDS04 = tuple(range(1101, 1104))           # PLAN: three seeds, paired with Exp 03's first three
ORDERS04 = ("interleaved", "stage_sorted")
QWEN = {"id": "Qwen/Qwen2.5-1.5B", "revision": "8faed761d45a263340a0528343f099c05c9a4323"}
SKU_NAMES = {"L40S": "L40S", "H100_SXM5": "H100", "A100_SXM4": "A100", "V100_SXM2": "V100", "V100S_PCIE": "V100S"}


def dump(path: Path, value) -> None:
    C.atomic_json(Path(path), value)


def load(path) -> dict:
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------- cell tables (the measured denominators)

def cells03() -> list[dict]:
    """Index i -> seed 1101 + i//6, method METHODS[i%6]: every finished block of six is one complete paired seed.
    Index 36 is the untouched-base evaluation (no training)."""
    rows = [{"index": i, "exp": "03", "method": C.METHODS[i % 6], "seed": SEEDS03[i // 6],
             "cell": f"{C.METHODS[i % 6]}_s{SEEDS03[i // 6]}"} for i in range(36)]
    rows.append({"index": 36, "exp": "03", "method": "base", "seed": None, "cell": "base"})
    return rows


def cells04() -> list[dict]:
    return [{"index": i, "exp": "04", "method": f"replay_{ORDERS04[i % 2]}", "order": ORDERS04[i % 2],
             "seed": SEEDS04[i // 2], "cell": f"{ORDERS04[i % 2]}_s{SEEDS04[i // 2]}"} for i in range(6)]


def dev_tasks(grid: dict | None = None) -> list[dict]:
    """Index -> (method, candidate); the last index is the untouched base model on the dev suite."""
    grid = grid or load(DEV_GRID)
    rows = []
    for m in C.METHODS:
        for j, cand in enumerate(grid["candidates"][m]):
            rows.append({"index": len(rows), "method": m, "candidate": cand["id"], "slot": j})
    rows.append({"index": len(rows), "method": "base", "candidate": "base", "slot": None})
    return rows


def cmd_cell(a) -> int:
    table = {"03": cells03, "04": cells04, "dev": dev_tasks}[a.exp]()
    row = table[a.index]
    print(" ".join(f"{k}={'' if v is None else v}" for k, v in row.items()))
    return 0


# ---------------------------------------------------------------- configs

def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def base_config(grid: dict) -> dict:
    if os.environ.get("JR1_TINY") == "1":   # CPU e2e: the tiny model saved by prep, loaded like a pretrained one
        cfg = load(JR1 / "configs" / "tiny_cpu.json")
        cfg["model"] = {"source": "pretrained", "id": "jr1-tiny-qwen2", "revision": None,
                        "local_path": str(Path(os.environ["JR1_STATE_ROOT"]) / "models" / "base"),
                        "attn_implementation": "eager"}
        return cfg
    return load(JR1 / grid["base_config"])


def dev_config(index: int, grid: dict | None = None) -> dict:
    grid = grid or load(DEV_GRID)
    task = dev_tasks(grid)[index]
    if task["method"] == "base":
        raise ValueError("the base dev task trains nothing")
    target = grid["work_target"] if os.environ.get("JR1_TINY") != "1" else grid["tiny_work_target"]
    if not isinstance(target, (int, float)) or target <= 0:
        raise SystemExit("dev_grid.json work_target is not set: AT-571 calibration fills it before the dev array runs")
    cand = next(c for c in grid["candidates"][task["method"]] if c["id"] == task["candidate"])
    over = {k: v for k, v in cand.items() if k != "id"}
    cfg = deep_merge(base_config(grid), over)
    cfg["work"] = {**cfg.get("work", {}), "target": target}
    cfg["protocol"] = f"jr1-dev-{grid['version']}-{task['candidate']} (development selection; tune split only)"
    return cfg


def cmd_dev_config(a) -> int:
    cfg = dev_config(a.index)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    return 0


def cmd_set_dev_target(a) -> int:
    """AT-571: write the calibrated dev work target (the same for every method) into dev_grid.json, once."""
    text = DEV_GRID.read_text()
    grid = json.loads(text)
    if grid["work_target"] is not None and not a.force:
        raise SystemExit(f"dev_grid.json work_target is already {grid['work_target']} (frozen); --force only before any dev job")
    key = '"work_target": null'
    if key not in text:
        raise SystemExit("dev_grid.json has no unset work_target line")
    DEV_GRID.write_text(text.replace(key, f'"work_target": {a.work_target:.6g}', 1))
    print(json.dumps({"work_target": a.work_target, "dev_grid_sha256": C.sha256(DEV_GRID)}))
    return 0


def cmd_freeze_configs(a) -> int:
    """Per-method PROTOCOL v1 configs from the dev selection (AT-572): base config + selected candidate + target."""
    grid, sel = load(DEV_GRID), load(a.selection)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = C.stamp()
    for m in C.METHODS:
        cand = next(c for c in grid["candidates"][m] if c["id"] == sel["selected"][m]["candidate"])
        cfg = deep_merge(base_config(grid), {k: v for k, v in cand.items() if k != "id"})
        cfg["work"] = {**cfg.get("work", {}), "target": a.work_target}
        cfg["protocol"] = f"jr1-protocol-v1-FROZEN {stamp} method={m} candidate={cand['id']}"
        (out / f"{m}.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"wrote": sorted(p.name for p in out.glob("*.json")), "work_target": a.work_target}))
    return 0


def cmd_check_frozen(a) -> int:
    cfg = load(a.config)
    name = str(cfg.get("protocol", ""))
    if os.environ.get("JR1_ALLOW_UNFROZEN") == "1":
        print(f"WARNING: unfrozen config allowed by JR1_ALLOW_UNFROZEN=1 (tests only): {name}")
        return 0
    if "FROZEN" not in name or "UNFROZEN" in name:
        print(f"REFUSED: {a.config} is not a frozen protocol config (protocol={name!r}); PLAN.md requires the freeze "
              "before any measured update", file=sys.stderr)
        return 3
    return 0


# ---------------------------------------------------------------- environment receipt

def cmd_env_receipt(a) -> int:
    import torch
    rec = {"timestamp": C.stamp(), "host": platform.node(), "python": platform.python_version(),
           "torch": torch.__version__, "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
           "slurm": {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
                                                     "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST", "SLURM_RESTART_COUNT")},
           "gpu_sku_requested": os.environ.get("GPU_SKU"), "code_commit": os.environ.get("JR1_CODE_COMMIT")}
    try:
        rec["pip_freeze"] = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True,
                                           timeout=120).stdout.splitlines()
    except Exception as exc:  # pragma: no cover
        rec["pip_freeze_error"] = repr(exc)
    if torch.cuda.is_available():
        d = torch.device("cuda:0")
        p = torch.cuda.get_device_properties(d)
        rec["gpu"] = {"name": p.name, "total_memory_gb": round(p.total_memory / 2**30, 2),
                      "capability": f"{p.major}.{p.minor}", "bf16_supported": torch.cuda.is_bf16_supported()}
        x = torch.randn(1024, 1024, device=d, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        rec["gpu"]["bf16_matmul_ok"] = bool(torch.isfinite(x @ x).all().item())
        try:
            rec["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                                "--format=csv,noheader"], capture_output=True, text=True,
                                               timeout=60).stdout.strip()
        except Exception:
            pass
        want = SKU_NAMES.get(os.environ.get("GPU_SKU", ""), os.environ.get("GPU_SKU", ""))
        rec["gpu_sku_match"] = bool(want) and want.split("_")[0] in p.name
    else:
        rec["gpu"] = None
        rec["gpu_sku_match"] = None
    dump(a.out, rec)
    print(json.dumps({k: rec[k] for k in ("torch", "torch_cuda", "gpu", "gpu_sku_match")}))
    if a.require_gpu and (rec["gpu"] is None or not rec["gpu_sku_match"]):
        print(f"REFUSED: GPU {rec['gpu']} does not match GPU_SKU={os.environ.get('GPU_SKU')}", file=sys.stderr)
        return 4
    return 0


# ---------------------------------------------------------------- data and model preparation

def cmd_check_data(a) -> int:
    """Compare the freshly prepared data with the committed manifest: every file's sha256 and row count."""
    want = load(a.manifest)["files"]
    got = load(Path(a.data) / "manifest.json")["files"]
    bad = {rel: {"want": w, "got": got.get(rel)} for rel, w in want.items()
           if got.get(rel, {}).get("sha256") != w["sha256"] or got.get(rel, {}).get("rows") != w["rows"]}
    extra = sorted(set(got) - set(want))
    rec = {"timestamp": C.stamp(), "files": len(want), "mismatched": bad, "extra": extra, "ok": not bad and not extra,
           "manifest_sha256": C.sha256(Path(a.data) / "manifest.json"), "committed_manifest": str(a.manifest)}
    dump(a.out, rec)
    print(json.dumps({"ok": rec["ok"], "files": len(want), "mismatched": sorted(bad), "extra": extra}))
    return 0 if rec["ok"] else 5


def _git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def cmd_prep_model(a) -> int:
    """Download Qwen2.5-1.5B at the pinned revision into $HF_HOME, sha256 every file, check against the Hub's own
    hashes (LFS sha256, git blob sha1 for small files), and link models/base to the snapshot. JR1_TINY=1 saves the
    offline tiny model instead (same manifest format)."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    link = out / "base"
    rec = {"timestamp": C.stamp()}
    if os.environ.get("JR1_TINY") == "1":
        from ..tiny import tiny_model, tiny_tokenizer
        snap = out / "tiny_base"
        if not snap.exists():
            tmp = out / "tiny_base.tmp"
            shutil.rmtree(tmp, ignore_errors=True)
            tiny_model().save_pretrained(tmp, safe_serialization=True)
            tiny_tokenizer().save_pretrained(tmp)
            tmp.rename(snap)
        rec.update(id="jr1-tiny-qwen2", revision=None, hub_check="n/a (offline tiny model)")
    else:
        from huggingface_hub import HfApi, snapshot_download
        t0 = time.monotonic()
        snap = Path(snapshot_download(QWEN["id"], revision=QWEN["revision"]))
        rec.update(id=QWEN["id"], revision=QWEN["revision"], download_s=round(time.monotonic() - t0, 1))
        info = HfApi().model_info(QWEN["id"], revision=QWEN["revision"], files_metadata=True)
        rec["hub_files"] = {s.rfilename: {"size": s.size, "lfs_sha256": s.lfs.sha256 if s.lfs else None,
                                          "blob_id": s.blob_id} for s in info.siblings}
    files = {}
    for p in sorted(snap.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(snap))
            files[rel] = {"bytes": p.stat().st_size, "sha256": C.sha256(p.resolve())}
    rec["snapshot"] = str(snap)
    rec["files"] = files
    if "hub_files" in rec:
        bad = []
        for rel, h in rec["hub_files"].items():
            if rel not in files:
                bad.append((rel, "missing"))
            elif h["lfs_sha256"] and h["lfs_sha256"] != files[rel]["sha256"]:
                bad.append((rel, "lfs sha256"))
            elif not h["lfs_sha256"] and h["blob_id"] and h["blob_id"] != _git_blob_sha1((snap / rel).resolve()):
                bad.append((rel, "git blob sha1"))
        rec["hub_check"] = "ok" if not bad else bad
    weights = sorted(k for k in files if k.endswith(".safetensors"))
    rec["weights_sha256"] = {k: files[k]["sha256"] for k in weights}
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(snap.resolve())
    dump(out / "model_manifest.json", rec)
    print(json.dumps({"snapshot": str(snap), "files": len(files), "hub_check": rec["hub_check"],
                      "weights_sha256": rec["weights_sha256"]}))
    return 0 if rec["hub_check"] in ("ok", "n/a (offline tiny model)") else 6


def cmd_tiny_data(a) -> int:
    """CPU e2e only: synthetic training pools (jr1.tiny) + the first N rows of every eval/tune file of a real
    prepared data directory, so the evaluator runs its real protocols on real items with the tiny model."""
    from ..tiny import write_synthetic_data
    out = Path(a.out)
    write_synthetic_data(out)
    src = Path(a.eval_from)
    names = ["eval/gsm8k_test", "eval/math500", "eval/mmlu_test", "eval/arc_easy_test", "eval/lc_gsm8k",
             "eval/lc_mmlu", "eval/mmlu_dev", "pt/test", "pref/test", "gsm8k_dev/tune", "pt/tune", "pref/tune"]
    for name in names:
        lines = (src / f"{name}.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)
        n = len(lines) if name == "eval/mmlu_dev" else a.n
        if name in ("pt/test", "pt/tune"):   # WikiText articles are long; keep the tiny CPU run short
            rows = [json.loads(x) for x in lines[:n]]
            lines = [json.dumps({**r, "text": r["text"][:2000]}) + "\n" for r in rows]
        (out / f"{name}.jsonl").parent.mkdir(parents=True, exist_ok=True)
        (out / f"{name}.jsonl").write_text("".join(lines[:n]), encoding="utf-8")
    dump(out / "manifest.json", {"tiny": True, "files": {
        str(p.relative_to(out)): {"rows": sum(1 for _ in p.open()), "sha256": C.sha256(p)}
        for p in sorted(out.rglob("*.jsonl"))}})
    print(json.dumps({"tiny_data": str(out), "eval_rows_each": a.n}))
    return 0


# ---------------------------------------------------------------- dev selection

def _summary(path: Path) -> dict | None:
    p = path / "summary.json"
    return load(p) if p.exists() else None


def select(dev_root: Path, grid: dict) -> dict:
    """The pre-written selection rule (dev_grid.json 'selection'), applied per method."""
    rule = grid["selection"]
    base = _summary(dev_root / "base" / "eval_dev")
    if base is None or base.get("status") != "complete":
        raise SystemExit(f"base dev evaluation missing or incomplete under {dev_root}/base/eval_dev")
    base_nll = base["dev_wikitext"]["nll_per_token"]
    out = {"timestamp": C.stamp(), "grid_version": grid["version"], "grid_sha256": C.sha256(DEV_GRID),
           "rule": rule, "base": {"dev_gsm8k_acc": base["dev_gsm8k"]["acc"], "dev_wikitext_nll": base_nll,
                                  "dev_pref_acc": base["dev_pref"]["acc"]}, "candidates": {}, "selected": {}}
    for m in C.METHODS:
        rows = []
        for slot, cand in enumerate(grid["candidates"][m]):
            d = dev_root / cand["id"]
            rec = load(d / "receipt.json") if (d / "receipt.json").exists() else {}
            s = _summary(d / "eval_dev")
            meta = load(d / "dev_task.json") if (d / "dev_task.json").exists() else {}
            row = {"candidate": cand["id"], "slot": slot, "train_status": rec.get("status"),
                   "within_allowance": meta.get("within_allowance"), "train_wall_s": rec.get("wall_s"),
                   "eval_status": s and s.get("status")}
            if s and s.get("status") == "complete" and rec.get("status") == "complete" and meta.get("within_allowance"):
                row.update(dev_gsm8k_acc=s["dev_gsm8k"]["acc"], dev_pref_acc=s["dev_pref"]["acc"],
                           dev_wikitext_nll=s["dev_wikitext"]["nll_per_token"])
                row["nll_ratio"] = row["dev_wikitext_nll"] / base_nll
                row["retention_ok"] = row["nll_ratio"] <= 1 + rule["max_nll_rise"]
                row["valid"] = True
            else:
                row["valid"] = False
            rows.append(row)
        out["candidates"][m] = rows
        valid = [r for r in rows if r["valid"]]
        ok = [r for r in valid if r["retention_ok"]]
        if ok:
            best = max(ok, key=lambda r: (r["dev_gsm8k_acc"], r["dev_pref_acc"], -r["dev_wikitext_nll"], -r["slot"]))
            how = "max dev_gsm8k_acc among retention-passing candidates"
        elif valid:
            best = min(valid, key=lambda r: (r["nll_ratio"], -r["dev_gsm8k_acc"], r["slot"]))
            how = "no candidate passed retention: smallest NLL rise (method flagged retention_failed)"
        else:
            best = rows[0]
            how = "no valid candidate (all failed or over allowance): slot 0 default (method flagged no_valid_candidate)"
        out["selected"][m] = {"candidate": best["candidate"], "how": how,
                              "retention_failed": bool(valid) and not ok, "no_valid_candidate": not valid}
    return out


def cmd_select(a) -> int:
    res = select(Path(a.dev), load(DEV_GRID))
    dump(a.out, res)
    print(json.dumps({m: v["candidate"] for m, v in res["selected"].items()}))
    return 0


def cmd_dev_mark(a) -> int:
    """Record whether a dev candidate's training stayed within its equal device-time allowance."""
    rp = Path(a.dir) / "receipt.json"
    rec = load(rp) if rp.exists() else {}
    wall = rec.get("wall_s")
    within = a.rc == 0 and rec.get("status") == "complete" and wall is not None and wall <= a.allowance_s
    dump(Path(a.dir) / "dev_task.json", {**dev_tasks()[a.index], "train_rc": a.rc, "train_status": rec.get("status"),
                                         "train_wall_s": wall, "allowance_s": a.allowance_s,
                                         "within_allowance": within, "timestamp": C.stamp()})
    print(json.dumps({"candidate": dev_tasks()[a.index]["candidate"], "within_allowance": within, "wall_s": wall}))
    return 0


# ---------------------------------------------------------------- calibration

def _per_action(log_path: Path) -> list[dict]:
    rows = C.read_jsonl(log_path)
    prev = 0.0
    out = []
    for r in rows:
        out.append({"objective": r.get("objective"), "purpose": r.get("purpose"), "action": r.get("action"),
                    "seconds": r["wall_s"] - prev, "charge": r.get("action_charge", 0)})
        prev = r["wall_s"]
    return out


def calib_report(calib: Path, work_target: float) -> dict:
    rep = {"timestamp": C.stamp(), "calib_dir": str(calib), "work_target_used": work_target,
           "env": load(calib / "env.json") if (calib / "env.json").exists() else None,
           "objectives": {}, "methods": {}, "checkpoint": None, "restart": None, "eval": {}}
    by_obj: dict[str, list[dict]] = {}
    for m in C.METHODS:
        d = calib / "train" / m
        if not (d / "receipt.json").exists():
            rep["methods"][m] = {"status": "missing"}
            continue
        r = load(d / "receipt.json")
        acts = _per_action(d / "log.jsonl")[1:]   # the first action also pays model warm-up
        for x in acts:
            if x["action"] == "update":
                by_obj.setdefault(x["objective"], []).append(x)
        secs = sum(x["seconds"] for x in acts)
        work = sum(x["charge"] for x in acts)
        rate = work / secs if secs else None
        rep["methods"][m] = {"status": r["status"], "actions": len(acts) + 1, "updates": r["updates"],
                             "wall_s": r["wall_s"], "peak_memory_gb": round(r.get("peak_memory_bytes", 0) / 2**30, 3),
                             "work_per_s": rate,
                             "est_train_hours_at_target": (work_target / rate / 3600) if rate else None}
    for o, xs in sorted(by_obj.items()):
        s = [x["seconds"] for x in xs]
        rep["objectives"][o] = {"updates_timed": len(xs), "median_s_per_update": statistics.median(s),
                                "mean_work_per_update": statistics.mean(x["charge"] for x in xs),
                                "work_per_s": sum(x["charge"] for x in xs) / sum(s) if sum(s) else None}
    if (calib / "ckpt_timing.json").exists():
        rep["checkpoint"] = load(calib / "ckpt_timing.json")
    if (calib / "restart.json").exists():
        rep["restart"] = load(calib / "restart.json")
    for suite in ("lc", "full"):
        s = _summary(calib / f"eval_{suite}")
        if s:   # wall time and memory only: calibration never reads scores
            rep["eval"][suite] = {"status": s.get("status"), "wall_s": s.get("wall_s"), "peak_mem_gb": s.get("peak_mem_gb"),
                                  "device": s.get("device"), "backend": s.get("backend"),
                                  "bench_wall_s": {b: s[b]["wall_s"] for b in s.get("benches", []) if b in s}}
    lc = (rep["eval"].get("lc") or {}).get("wall_s") or 0.0
    full = (rep["eval"].get("full") or {}).get("wall_s") or 0.0
    trains = [v["est_train_hours_at_target"] for v in rep["methods"].values() if v.get("est_train_hours_at_target")]
    if trains:
        per_cell = {m: v["est_train_hours_at_target"] + (5 * lc + full) / 3600
                    for m, v in rep["methods"].items() if v.get("est_train_hours_at_target")}
        rep["per_cell_hours"] = per_cell
        mean_cell = statistics.mean(per_cell.values())
        jr1_cell = per_cell.get("jr1", mean_cell)
        rep["totals_gpu_hours"] = {"exp03_36_cells": 6 * sum(per_cell.values()),
                                   "exp03_base_eval": (lc + full) / 3600,
                                   "exp04_schedule_run": jr1_cell, "exp04_6_cells": 6 * mean_cell,
                                   "note": "train estimate from calibrated work/s at work_target_used + 5 lc evals + "
                                           "1 full eval per cell; replay cells use the six-method mean"}
        rep["totals_gpu_hours"]["all"] = sum(v for k, v in rep["totals_gpu_hours"].items() if isinstance(v, float))
    return rep


def cmd_calib_report(a) -> int:
    rep = calib_report(Path(a.calib), a.work_target)
    dump(a.out, rep)
    print(json.dumps({"methods": {m: v.get("est_train_hours_at_target") for m, v in rep["methods"].items()},
                      "totals": rep.get("totals_gpu_hours")}, default=str))
    return 0


def cmd_ckpt_timing(a) -> int:
    import torch
    from transformers import AutoModelForCausalLM
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t0 = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(a.ckpt, dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32)
    model.to(dev)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    load_s = time.monotonic() - t0
    tmp = Path(a.scratch)
    shutil.rmtree(tmp, ignore_errors=True)
    t0 = time.monotonic()
    model.save_pretrained(tmp, safe_serialization=True)
    save_s = time.monotonic() - t0
    size = sum(p.stat().st_size for p in tmp.rglob("*") if p.is_file())
    shutil.rmtree(tmp, ignore_errors=True)
    rec = {"ckpt": a.ckpt, "load_to_device_s": round(load_s, 3), "save_pretrained_s": round(save_s, 3),
           "bytes": size, "device": str(dev)}
    dump(a.out, rec)
    print(json.dumps(rec))
    return 0


# ---------------------------------------------------------------- collection index

def cmd_index(a) -> int:
    """One row per planned cell (full denominators), with the status found on disk."""
    root = Path(a.root)
    rows = []
    for exp, table in (("03", cells03()), ("04", cells04())):
        for c in table:
            d = root / "runs" / exp / c["cell"]
            rec = load(d / "receipt.json") if (d / "receipt.json").exists() else None
            evals = {}
            for e in sorted((d / "eval").glob("*")) if (d / "eval").exists() else []:
                s = _summary(e)
                evals[e.name] = s.get("status") if s else None
            done = (d / "CELL_DONE").exists()
            status = "complete" if done else (rec or {}).get("status") or ("evaluating" if evals else "unstarted")
            rows.append({**c, "status": status, "train_status": (rec or {}).get("status"),
                         "work_total": (rec or {}).get("work_total"), "wall_s": (rec or {}).get("wall_s"),
                         "evals": evals, "slurm": load(d / "slurm.json") if (d / "slurm.json").exists() else None})
    counts = {}
    for exp in ("03", "04"):
        sub = [r for r in rows if r["exp"] == exp]
        counts[exp] = {"planned": len(sub), **{s: sum(r["status"] == s for r in sub)
                                                for s in sorted({r["status"] for r in sub})}}
    out = {"timestamp": C.stamp(), "root": str(root), "counts": counts, "cells": rows}
    dump(a.out, out)
    print(json.dumps(counts))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cell"); p.add_argument("--exp", choices=["03", "04", "dev"], required=True)
    p.add_argument("--index", type=int, required=True); p.set_defaults(fn=cmd_cell)
    p = sub.add_parser("dev-config"); p.add_argument("--index", type=int, required=True)
    p.add_argument("--out", required=True); p.set_defaults(fn=cmd_dev_config)
    p = sub.add_parser("dev-mark"); p.add_argument("--index", type=int, required=True); p.add_argument("--dir", required=True)
    p.add_argument("--rc", type=int, required=True); p.add_argument("--allowance-s", type=int, required=True)
    p.set_defaults(fn=cmd_dev_mark)
    p = sub.add_parser("select"); p.add_argument("--dev", required=True); p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_select)
    p = sub.add_parser("set-dev-target"); p.add_argument("--work-target", type=float, required=True)
    p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_set_dev_target)
    p = sub.add_parser("freeze-configs"); p.add_argument("--selection", required=True)
    p.add_argument("--work-target", type=float, required=True); p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_freeze_configs)
    p = sub.add_parser("check-frozen"); p.add_argument("--config", required=True); p.set_defaults(fn=cmd_check_frozen)
    p = sub.add_parser("env-receipt"); p.add_argument("--out", required=True)
    p.add_argument("--require-gpu", action="store_true"); p.set_defaults(fn=cmd_env_receipt)
    p = sub.add_parser("check-data"); p.add_argument("--data", required=True); p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True); p.set_defaults(fn=cmd_check_data)
    p = sub.add_parser("prep-model"); p.add_argument("--out", required=True); p.set_defaults(fn=cmd_prep_model)
    p = sub.add_parser("tiny-data"); p.add_argument("--out", required=True); p.add_argument("--eval-from", required=True)
    p.add_argument("--n", type=int, default=3); p.set_defaults(fn=cmd_tiny_data)
    p = sub.add_parser("calib-report"); p.add_argument("--calib", required=True)
    p.add_argument("--work-target", type=float, required=True); p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_calib_report)
    p = sub.add_parser("ckpt-timing"); p.add_argument("--ckpt", required=True); p.add_argument("--scratch", required=True)
    p.add_argument("--out", required=True); p.set_defaults(fn=cmd_ckpt_timing)
    p = sub.add_parser("index"); p.add_argument("--root", required=True); p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_index)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
