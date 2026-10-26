"""Freeze the Exp 05 configs: write the calibrated work target, mark them frozen and record their sha256.

    python -m p05.freeze --precision {fp16,bf16} --calibration <calibration.json> [--work-target X]

Writes expt_v1/configs/pythia160m_<prec>.json and expt_v2/configs/pythia{14m,70m}_<prec>.json with frozen=true,
protocol "p05-<v>-<size>-<prec>-FROZEN" and work.target = the calibration's recommended_work_target (or the explicit
--work-target, which must not be below it), then expt_vN/configs/FROZEN.sha256 (p05/train.py refuses a frozen config
whose sha256 differs). Commit the configs, FROZEN.sha256 and the PROTOCOL.md freeze table before the first measured
update; the commit order is the evidence that the protocol preceded measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from . import spec

HOME = Path(__file__).resolve().parents[1]


def render(cfg: dict) -> str:
    return json.dumps(cfg, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--precision", choices=sorted(spec.PRECISIONS), required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--work-target", type=float)
    a = ap.parse_args(argv)
    cal = json.loads(a.calibration.read_text())
    if cal["precision"] != a.precision:
        raise SystemExit(f"calibration was measured in {cal['precision']}, not {a.precision}")
    target = a.work_target or cal["recommended_work_target"]
    if target < cal["recommended_work_target"]:
        raise SystemExit("a work target below the calibrated minimum cannot reach 50M PT tokens for every method")
    rows = []
    for version, sizes in spec.VERSIONS.items():
        d = HOME / f"expt_{version}" / "configs"
        lines = []
        for size in sizes:
            cfg = spec.config(size, a.precision, version)
            cfg["protocol"] = f"p05-{version}-{size}-{a.precision}-FROZEN"
            cfg["frozen"] = True
            cfg["work"]["target"] = float(target)
            cfg["p05"]["calibration_sha256"] = hashlib.sha256(a.calibration.read_bytes()).hexdigest()
            text = render(cfg)
            name = spec.config_name(size, a.precision)
            (d / name).write_text(text)
            sha = hashlib.sha256(text.encode()).hexdigest()
            lines.append(f"{sha}  {name}")
            rows.append((version, size, name, sha))
        (d / "FROZEN.sha256").write_text("\n".join(lines) + "\n")
    print(f"work target {target:.4g}")
    for r in rows:
        print("| " + " | ".join(r) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
