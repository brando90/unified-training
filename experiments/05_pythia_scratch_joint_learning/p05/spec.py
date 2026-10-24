"""Exp 05 design constants: sizes, pinned revisions, methods, seeds, cell index and the shared config body.

expt_v1: Pythia-160m, 3 methods x 3 seeds = 9 cells (primary).
expt_v2: Pythia-14m and Pythia-70m, 3 methods x 3 seeds = 18 cells (pre-declared secondary size ladder).
Every cell of every size gets the same charged-work target (forward-equivalent tokens, jr1/common.py MULTIPLIERS),
so all sizes see the same token budget and the joint-minus-sequential gap can be read against model size.
"""
from __future__ import annotations

import copy

METHODS = ("sequential", "fixed_joint", "smooth_joint")
SEEDS = (1, 2, 3)
OBSERVED_PT_TARGET = 50_000_000   # PLAN.md line 73: at least 50M observed PT tokens per seed when possible

# EleutherAI Pythia configs and the GPT-NeoX-20B tokenizer at pinned Hub commits (main on 2026-10-20). Only
# config.json and the tokenizer files are read: the weights are initialized fresh from the config, seeded per cell.
# lr: the Pythia suite's peak learning rates (Biderman et al. 2023: 160m 6e-4, 70m 1e-3; 14m, added to the suite
# later, takes the 70m value). Fixed a priori; no development selection for the scratch study.
SIZES = {
    "14m": {"id": "EleutherAI/pythia-14m", "revision": "cf967c0a9a04383db6f7b1108d86b2962634b4ac", "lr": 1.0e-3},
    "70m": {"id": "EleutherAI/pythia-70m", "revision": "a39f36b100fe8a5377810d56c3f4789b9c53ac42", "lr": 1.0e-3},
    "160m": {"id": "EleutherAI/pythia-160m", "revision": "50f5173d932e8e61f858120bcb800b97af589f46", "lr": 6.0e-4},
}
VERSIONS = {"v1": ("160m",), "v2": ("14m", "70m")}
# Compute precision per GPU: V100 (compute capability 7.0) has no bf16 and no FlashAttention-2, so it trains in
# fp16 autocast with fp32 master weights and a dynamic GradScaler (Pythia itself was trained in fp16). L40S/H100 use
# bf16 autocast. One GPU type (and so one precision) for every measured cell of a version; calibration decides.
PRECISIONS = {"fp16": "V100 (Sherlock GPU_SKU V100S_PCIE|V100_PCIE|V100_SXM2, 32GB)",
              "bf16": "L40S 46GB (FarmShare gpu partition) or another sm80+ GPU"}

PROVISIONAL_WORK_TARGET = 3.0e8   # DRAFT placeholder; calibrate.py computes the frozen value, freeze.py writes it

BASE = {
    "model": {"source": "random", "local_path": None, "init_seed": None, "tiny": None,
              "attn_implementation": "sdpa", "gradient_checkpointing": False},
    "precision": {"master": "fp32", "reference": "compute", "checkpoint": "fp32"},
    "data": {"root": None,
             "splits": {"pt": "pt_blocks/train", "sft": "sft/train", "rl": "rl/train", "dpo": "pref/train",
                        "pt_controller": "pt/controller", "sft_controller": "sft/controller",
                        "rl_controller": "rl/controller", "dpo_controller": "pref/controller"}},
    "optimizer": {"weight_decay": 0.01, "betas": [0.9, 0.95], "eps": 1e-8, "grad_clip": 1.0, "schedule": "cosine",
                  "warmup_fraction": 0.01, "warmup_start_ratio": 0.0, "min_lr_ratio": 0.1},
    "objectives": {"pt": {"batch_size": 32, "max_len": 1024},
                   "sft": {"batch_size": 32, "max_len": 512},
                   "dpo": {"batch_size": 8, "max_len": 1024, "beta": 0.1},
                   "rl": {"prompts": 4, "group_size": 8, "max_new_tokens": 256, "max_prompt_len": 256,
                          "temperature": 1.0, "reward": "gsm8k_exact", "backend": "hf"}},
    # jr1 defaults (attributed to experiments/00_program/controller.py), unchanged so Exp 03 and Exp 05 share them.
    "methods": {"sequential": {"boundaries": [0.55, 0.75, 0.85]},
                "fixed_joint": {"weights": {"pt": 0.55, "sft": 0.20, "dpo": 0.10, "rl": 0.15}},
                "smooth_joint": {"start": {"pt": 0.70, "sft": 0.20, "dpo": 0.05, "rl": 0.05},
                                 "end": {"pt": 0.20, "sft": 0.20, "dpo": 0.15, "rl": 0.45},
                                 "floors": {"pt": 0.05, "dpo": 0.05}}},
    "checkpoints": {"fractions": [0.0, 0.25, 0.5, 0.75, 1.0], "state_every_actions": 1000,
                    "state_every_seconds": 1800},
}


def config(size: str, precision: str, version: str) -> dict:
    """The full Exp 05 config for one size and precision (methods share it; --method picks the schedule)."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}")
    s = SIZES[size]
    cfg = copy.deepcopy(BASE)
    cfg["protocol"] = f"p05-{version}-{size}-{precision}-DRAFT"
    cfg["frozen"] = False
    cfg["p05"] = {"version": version, "size": size, "precision": precision, "methods": list(METHODS),
                  "seeds": list(SEEDS), "observed_pt_target": OBSERVED_PT_TARGET, "gpu": PRECISIONS[precision]}
    cfg["model"].update(id=s["id"], revision=s["revision"])
    cfg["precision"]["compute"] = precision
    cfg["optimizer"]["lr"] = s["lr"]
    cfg["work"] = {"target": PROVISIONAL_WORK_TARGET}
    return cfg


def cells(version: str) -> list[dict]:
    """Array index -> cell. Seed-major, so each finished block of len(sizes) * 3 indices is a complete paired seed."""
    out = []
    for seed in SEEDS:
        for size in VERSIONS[version]:
            for method in METHODS:
                out.append({"index": len(out), "version": version, "size": size, "seed": seed, "method": method,
                            "cell": f"{version}/{size}/{method}_s{seed}"})
    return out


def config_name(size: str, precision: str) -> str:
    return f"pythia{size}_{precision}.json"
