"""Exp 05 cell lookup for Slurm arrays: print shell assignments for one array index, or list every cell.

    python -m p05.cells --version {v1,v2} --index I --precision {fp16,bf16}   -> cell=... size=... seed=... ...
    python -m p05.cells --version {v1,v2} --list
"""
from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

from . import spec
from .freeze import HOME


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--version", choices=sorted(spec.VERSIONS), required=True)
    ap.add_argument("--index", type=int)
    ap.add_argument("--precision", choices=sorted(spec.PRECISIONS) + ["fp32"], help="fp32: CPU smoke configs only")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    cells = spec.cells(a.version)
    if a.list:
        for c in cells:
            print(f"{c['index']:>2} {c['cell']}")
        return 0
    if a.index is None or not 0 <= a.index < len(cells) or a.precision is None:
        raise SystemExit(f"--index must be in 0..{len(cells) - 1} and --precision given")
    c = cells[a.index]
    # P05_CONFIG_DIR: CPU smoke only (tiny batches); measured cells read the committed expt_vN/configs.
    base = Path(os.environ["P05_CONFIG_DIR"]) if os.environ.get("P05_CONFIG_DIR") else HOME / f"expt_{a.version}" / "configs"
    cfg = base / spec.config_name(c["size"], a.precision)
    for key, value in (("cell", c["cell"]), ("size", c["size"]), ("seed", c["seed"]), ("method", c["method"]),
                       ("config", cfg)):
        print(f"{key}={shlex.quote(str(value))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
