"""Environment receipt for an Exp 05 job, refusing the wrong GPU.

    python -m p05.envcheck --out <env.json> [--require {none,v100_32gb,sm80}]

v100_32gb: a V100-family GPU (compute capability 7.0) with at least 30 GiB (the 32GB cards), for the fp16 configs.
sm80: compute capability 8.0 or newer (L40S, A100, H100), for the bf16 configs.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path


def receipt() -> dict:
    import torch
    import transformers
    r = {"host": platform.node(), "python": platform.python_version(), "torch": torch.__version__,
         "cuda_build": torch.version.cuda, "transformers": transformers.__version__, "cuda": torch.cuda.is_available(),
         "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "code_commit": os.environ.get("P05_CODE_COMMIT"),
         "cluster": os.environ.get("P05_CLUSTER"), "precision": os.environ.get("P05_PRECISION")}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        r.update(gpu=p.name, capability=f"{p.major}.{p.minor}", gpu_memory_gib=round(p.total_memory / 2**30, 2),
                 bf16_supported=torch.cuda.is_bf16_supported())
    return r


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--require", choices=["none", "v100_32gb", "sm80"], default="none")
    a = ap.parse_args(argv)
    r = receipt()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(r, indent=2, sort_keys=True) + "\n")
    problem = None
    if a.require != "none" and not r["cuda"]:
        problem = "no CUDA device"
    elif a.require == "v100_32gb" and not (r["capability"] == "7.0" and r["gpu_memory_gib"] >= 30):
        problem = f"need a 32GB V100 (cc 7.0), got {r['gpu']} cc {r['capability']} {r['gpu_memory_gib']} GiB"
    elif a.require == "sm80" and float(r["capability"]) < 8.0:
        problem = f"need compute capability >= 8.0 for bf16, got {r['gpu']} cc {r['capability']}"
    if problem:
        print(f"REFUSED: {problem}", file=sys.stderr)
        return 3
    print(json.dumps({k: r.get(k) for k in ("gpu", "capability", "gpu_memory_gib", "torch", "transformers")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
