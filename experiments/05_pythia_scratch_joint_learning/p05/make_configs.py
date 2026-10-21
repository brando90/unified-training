"""Write the DRAFT Exp 05 configs (every size x both precisions) from p05/spec.py.

    python -m p05.make_configs [--check]

expt_v1/configs/pythia160m_{bf16,fp16}.json and expt_v2/configs/pythia{14m,70m}_{bf16,fp16}.json. --check fails if a
committed config differs from spec.py (run by the tests). freeze.py later rewrites the chosen precision as frozen.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import spec
from .freeze import HOME, render


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    stale = []
    for version, sizes in spec.VERSIONS.items():
        d = HOME / f"expt_{version}" / "configs"
        d.mkdir(parents=True, exist_ok=True)
        for size in sizes:
            for precision in sorted(spec.PRECISIONS):
                path = d / spec.config_name(size, precision)
                text = render(spec.config(size, precision, version))
                if a.check:
                    if not path.exists() or (path.read_text() != text and not json.loads(path.read_text()).get("frozen")):
                        stale.append(str(path))
                else:
                    path.write_text(text)
    if stale:
        print("stale configs: " + ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
