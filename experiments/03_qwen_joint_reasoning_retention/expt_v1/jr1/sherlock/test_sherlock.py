"""CPU tests for the Sherlock layer: cell tables, the frozen dev grid, the selection rule, the freeze guard and the
sbatch headers. Run: python -m pytest -q jr1/sherlock/test_sherlock.py   (from expt_v1/)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from jr1 import common as C
from jr1 import train as T
from jr1.controllers import make_schedule
from jr1.sherlock import tools

HERE = Path(__file__).resolve().parent


def test_cell_tables_full_denominators():
    c3, c4 = tools.cells03(), tools.cells04()
    assert len(c3) == 37 and c3[-1]["cell"] == "base"
    pairs = {(r["method"], r["seed"]) for r in c3[:36]}
    assert pairs == {(m, s) for m in C.METHODS for s in range(1101, 1107)}
    for b in range(6):   # every block of six is one complete paired seed
        assert {r["method"] for r in c3[6 * b:6 * b + 6]} == set(C.METHODS)
    assert len(c4) == 6 and {(r["order"], r["seed"]) for r in c4} == {
        (o, s) for o in ("interleaved", "stage_sorted") for s in (1101, 1102, 1103)}
    assert len({r["cell"] for r in c3 + c4}) == 43


def test_dev_grid_is_equal_and_valid(monkeypatch):
    grid = json.loads(tools.DEV_GRID.read_text())
    assert grid["dev_seed"] == C.DEV_SEED == 9001
    counts = {m: len(grid["candidates"][m]) for m in C.METHODS}
    assert set(counts.values()) == {grid["allowance"]["candidates_per_method"]}
    ids = [c["id"] for m in C.METHODS for c in grid["candidates"][m]]
    assert len(ids) == len(set(ids)) == 24
    tasks = tools.dev_tasks(grid)
    assert len(tasks) == 25 and tasks[-1]["method"] == "base"
    grid = {**grid, "work_target": 1.0e8}
    for t in tasks[:-1]:
        cfg = T.deep_merge(T.DEFAULTS, tools.dev_config(t["index"], grid))
        assert cfg["work"]["target"] == 1.0e8
        make_schedule(t["method"], cfg["methods"][t["method"]], 9001)   # every candidate is a valid schedule


def test_dev_config_refuses_unset_target():
    grid = json.loads(tools.DEV_GRID.read_text())
    assert grid["work_target"] is None   # filled once by AT-571 calibration
    with pytest.raises(SystemExit):
        tools.dev_config(0, grid)


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_selection_rule(tmp_path):
    grid = json.loads(tools.DEV_GRID.read_text())
    dev = tmp_path / "dev"
    _write(dev / "base" / "eval_dev" / "summary.json", {"status": "complete", "dev_gsm8k": {"acc": 0.5},
                                                        "dev_wikitext": {"nll_per_token": 2.0}, "dev_pref": {"acc": 0.6}})
    plan = {0: (0.60, 2.01), 1: (0.70, 2.10), 2: (0.65, 2.05), 3: (0.60, 2.00)}   # slot: (gsm8k acc, nll)
    for m in C.METHODS:
        for slot, cand in enumerate(grid["candidates"][m]):
            d = dev / cand["id"]
            acc, nll = plan[slot]
            if m == "aioli":
                nll += 0.2          # nobody passes retention -> smallest rise, flagged
            _write(d / "receipt.json", {"status": "complete", "wall_s": 10})
            _write(d / "dev_task.json", {"within_allowance": not (m == "jr1" and slot == 0)})
            _write(d / "eval_dev" / "summary.json", {"status": "complete", "dev_gsm8k": {"acc": acc},
                                                     "dev_wikitext": {"nll_per_token": nll},
                                                     "dev_pref": {"acc": 0.6}})
    sel = tools.select(dev, grid)["selected"]
    # slots 1 and 2 score higher but break the 2% NLL rule; slots 0 and 3 tie on score -> lower nll wins (slot 3)
    assert sel["sequential"]["candidate"] == grid["candidates"]["sequential"][3]["id"]
    assert sel["sequential"]["retention_failed"] is False
    assert sel["aioli"]["retention_failed"] is True
    assert sel["aioli"]["candidate"] == grid["candidates"]["aioli"][3]["id"]
    assert sel["jr1"]["candidate"] == grid["candidates"]["jr1"][3]["id"]   # slot 0 over allowance, not selectable


def test_freeze_guard(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("JR1_ALLOW_UNFROZEN", raising=False)
    unfrozen = Path(T.__file__).parent / "configs" / "qwen15b_v0.json"
    assert tools.main(["check-frozen", "--config", str(unfrozen)]) == 3
    p = tmp_path / "jr1.json"
    p.write_text(json.dumps({"protocol": "jr1-protocol-v1-FROZEN 2026-10-21T00:00:00-07:00 method=jr1"}))
    assert tools.main(["check-frozen", "--config", str(p)]) == 0


def test_sbatch_headers():
    for name in ("prep", "calib", "dev", "schedule04", "cells03", "cells04"):
        text = (HERE / f"{name}.sbatch").read_text()
        heads = re.findall(r"^#SBATCH (.*)$", text, re.M)
        assert "-p gpu" in heads and "-G 1" in heads and "--requeue" in heads, name
        assert "--mail-type=END,FAIL" in heads, name
        assert "--mail-user=brandojazz@gmail.com,brando.science@gmail.com" in heads, name
        assert "--signal=B:USR1@900" in heads, name
        if name in ("calib", "dev", "schedule04", "cells03", "cells04"):
            assert "--resume" in text or "jr1_eval" in text, name
    sub = (HERE / "submit.sh").read_text()
    assert '-C "GPU_SKU:$GPU_SKU"' in sub
    for name, rng in (("dev", "0-24"), ("cells03", "0-36"), ("cells04", "0-5")):
        assert f"{name}) array={rng};;" in sub


def test_lock_is_fully_pinned_and_hashed():
    lines = (HERE / "requirements-sherlock.lock").read_text().splitlines()
    reqs = [l for l in lines if l and not l.startswith("#") and not l.startswith("    --hash")]
    hashes = [l for l in lines if l.startswith("    --hash=sha256:")]
    assert len(reqs) == len(hashes) and all(re.match(r"^[A-Za-z0-9_.\-]+==[^ ]+ \\$", r) for r in reqs)
    pins = {r.split("==")[0].lower(): r.split("==")[1].split()[0] for r in reqs}
    assert pins["torch"] == "2.6.0" and pins["transformers"] == "4.57.6" and pins["math-verify"] == "0.9.0"
