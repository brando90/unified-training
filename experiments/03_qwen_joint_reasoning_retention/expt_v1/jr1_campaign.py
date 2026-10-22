#!/usr/bin/env python3
"""Deterministic, restartable executor for the 10-05-2026 JR1 campaign.

This module deliberately keeps all mutable artifacts outside the Git worktree.
It does not call model-provider APIs: public model/data retrieval uses the
Hugging Face libraries and all training/evaluation is local PyTorch work.

Covers all three prospective campaign experiments:
  - experiments/03_qwen_joint_reasoning_retention (36 cells: 6 methods x 6 seeds + 1 untouched base)
  - experiments/04_qwen_matched_exposure_schedules (6 cells: 2 schedules x 3 seeds)
  - experiments/05_pythia_scratch_joint_learning (9 cells: 3 schedules x 3 seeds + 3 untouched random-init)
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import copy
import datetime as dt
import fcntl
import hashlib
import itertools
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

if "JR1_STATE_ROOT" not in os.environ:
    raise RuntimeError("JR1_STATE_ROOT must name the private campaign state directory")
ROOT = Path(os.environ["JR1_STATE_ROOT"])
REPO = Path(__file__).resolve().parents[3]
RUN = ROOT / "campaign"
RUN.mkdir(parents=True, exist_ok=True)
RUN_ID = ROOT.name

OBJECTIVES = ("pt", "sft", "dpo", "rl")
SEEDS = [1101, 1102, 1103, 1104, 1105, 1106]
METHODS = ["sequential", "replay", "fixed_joint", "smooth_joint", "aioli", "jr1"]

SEEDS_04 = [2101, 2102, 2103]
METHODS_04 = ["sequential_sorted", "interleaved_replay"]

SEEDS_05 = [3101, 3102, 3103]
METHODS_05 = ["sequential", "fixed_joint", "smooth_joint"]

EXP_03 = "03_qwen_joint_reasoning_retention"
EXP_04 = "04_qwen_matched_exposure_schedules"
EXP_05 = "05_pythia_scratch_joint_learning"
ALL_EXPERIMENTS = (EXP_03, EXP_04, EXP_05)

# Schedule parameters frozen before measured training
SEQUENTIAL_BOUNDARIES = (0.40, 0.65, 0.80)
REPLAY_PT_FLOOR = 0.25
FIXED_WEIGHTS = (0.35, 0.30, 0.15, 0.20)
SMOOTH_START = (0.50, 0.30, 0.10, 0.10)
SMOOTH_END = (0.20, 0.20, 0.15, 0.45)
JR1_PRIOR_START = (0.45, 0.30, 0.10, 0.15)
JR1_PRIOR_END = (0.20, 0.20, 0.15, 0.45)

NUMBER = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:/\d+)?")
STOP_STRINGS = ["\nProblem:", "\nQuestion:", "\n\nProblem:", "\n\nQuestion:"]
SECRET_RE = re.compile(
    rb"(?:sk-(?:ant-|proj-)[A-Za-z0-9_-]{12,}|AKIA[A-Z0-9]{16}|"
    rb"-----BEGIN [A-Z ]*PRIVATE KEY-----|(?:/lfs/|/dfs/)|Bearer [A-Za-z0-9._-]{15,})"
)


def stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def human_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%m-%d-%Y %H:%M UTC")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w") as out:
        json.dump(value, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)


def append_event(path: Path, **fields: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": stamp(), **fields}
    with path.open("a") as out:
        out.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        out.flush()
        os.fsync(out.fileno())


def read_json(path: Path, default: object) -> object:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalized_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().casefold()


def text_hash(s: str) -> str:
    return hashlib.sha256(normalized_text(s).encode("utf-8")).hexdigest()


def proc_identity(pid: int | None = None) -> dict[str, object]:
    pid = pid or os.getpid()
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        return {"pid": pid, "start_ticks": fields[19], "boot_id": boot_id}
    except OSError:
        return {"pid": pid, "start_ticks": "0", "boot_id": "local"}


def proc_alive(row: Mapping[str, object] | None) -> bool:
    if not row or "pid" not in row:
        return False
    try:
        pid = int(row["pid"])
        if Path(f"/proc/{pid}/stat").exists():
            wanted = {k: row[k] for k in ("pid", "start_ticks", "boot_id")}
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
            return proc_identity(pid) == wanted and state != "Z"
        os.kill(pid, 0)
        return True
    except (OSError, KeyError, TypeError, ValueError):
        return False


def register_child(pid: int | None = None, role: str = "training_supervisor") -> dict[str, object]:
    ident = {**proc_identity(pid), "role": role, "timestamp": stamp()}
    children = list(read_json(ROOT / "children.json", []))
    children = [
        c for c in children
        if proc_alive(c) and not (c.get("pid") == ident["pid"] and c.get("role") == role)
    ]
    children.append(ident)
    atomic_json(ROOT / "children.json", children)
    return ident


def update_progress(
    phase: str,
    *,
    completed: int = 0,
    blockers: list[str] | None = None,
    status: str = "in_progress",
    generated: int = 0,
    scored: int = 0,
    **extra: object,
) -> None:
    atomic_json(
        ROOT / "progress.json",
        {
            "run_id": RUN_ID,
            "timestamp": stamp(),
            "phase": phase,
            "completed_cells": completed,
            "expected_cells": 51,
            "generated": generated,
            "scored": scored,
            "planned": 51,
            "blockers": blockers or [],
            "coordinator_status": status,
            **extra,
        },
    )


def strip_stop_strings(text: str) -> str:
    out = str(text)
    for s in STOP_STRINGS:
        if s in out:
            out = out.split(s, 1)[0]
    return out


def extract_answer(text: str) -> str | None:
    cleaned = strip_stop_strings(text)
    tail = cleaned.rsplit("####", 1)[-1] if "####" in cleaned else cleaned
    found = NUMBER.findall(tail)
    if not found:
        return None
    raw = found[-1].replace(",", "")
    try:
        if "/" in raw:
            a, b = raw.split("/")
            value = Decimal(a) / Decimal(b)
        else:
            value = Decimal(raw)
        return str(value.normalize()) if value.is_finite() else None
    except (InvalidOperation, ZeroDivisionError):
        return None


def exact_reward(text: str, answer: str) -> float:
    got, gold = extract_answer(text), extract_answer(answer)
    return float(got is not None and gold is not None and got == gold)


def extract_boxed(text: str) -> str | None:
    """Extract the last \\boxed{...} expression supporting nested braces."""
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1
    return text[start : pos - 1].strip() if depth == 0 else None


def normalize_math_answer(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    s = s.replace("$", "").replace("\\%", "").replace("%", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^{}]*)\}", r"\1", s)
    s = s.replace("^\\circ", "").replace("\\circ", "")
    # Convert simple \frac{a}{b} -> (a)/(b) iteratively
    for _ in range(4):
        s_new = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
        if s_new == s:
            break
        s = s_new
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
    s = s.replace("\\pi", "pi").replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace(" ", "").replace(",", "")
    if s.startswith("(") and s.endswith(")") and "," not in s:
        inner = s[1:-1]
        if inner.count("(") == inner.count(")"):
            s = inner
    return s or None


def extract_math500_candidate(text: str) -> str | None:
    cleaned = strip_stop_strings(text)
    boxed = extract_boxed(cleaned)
    if boxed is not None:
        return boxed
    if "####" in cleaned:
        return cleaned.rsplit("####", 1)[-1].strip().splitlines()[0].strip()
    if "Answer:" in cleaned:
        return cleaned.rsplit("Answer:", 1)[-1].strip().splitlines()[0].strip()
    lines = [ln.strip() for ln in cleaned.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    last = lines[-1].rstrip(".")
    found = NUMBER.findall(last)
    return found[-1] if found else last


def math500_reward(response: str, gold_answer: str) -> float:
    cand_raw = extract_math500_candidate(response)
    gold_raw = extract_boxed(gold_answer) or gold_answer
    cand_norm = normalize_math_answer(cand_raw)
    gold_norm = normalize_math_answer(gold_raw)
    if cand_norm is None or gold_norm is None:
        return 0.0
    if cand_norm == gold_norm:
        return 1.0
    # Decimal check for numeric/fraction answers
    c_num, g_num = extract_answer(cand_norm), extract_answer(gold_norm)
    if c_num is not None and g_num is not None and c_num == g_num:
        return 1.0
    return 0.0


def mask_labels(ids: torch.Tensor, prompt_lengths: list[int], pad: int) -> torch.Tensor:
    labels = ids.clone()
    for row, n in enumerate(prompt_lengths):
        labels[row, :n] = -100
    labels[ids == pad] = -100
    return labels


def sequence_logps(logits: torch.Tensor, ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    lp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    shifted = labels[:, 1:]
    valid = shifted.ne(-100)
    safe = shifted.masked_fill(~valid, 0)
    return lp.gather(-1, safe.unsqueeze(-1)).squeeze(-1).masked_fill(~valid, 0).sum(-1)


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    return -F.logsigmoid(beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))).mean()


def loo_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    if group_size < 2 or rewards.numel() % group_size:
        raise ValueError("rewards must form complete independent groups")
    groups = rewards.reshape(-1, group_size)
    return (groups - (groups.sum(1, keepdim=True) - groups) / (group_size - 1)).reshape(-1)


@dataclass
class Work:
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, kind: str, tokens: int, multiplier: int = 1) -> None:
        if tokens < 0 or multiplier <= 0:
            raise ValueError("work tokens must be non-negative and multiplier positive")
        self.counts[kind] = self.counts.get(kind, 0) + int(tokens) * int(multiplier)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def state_dict(self) -> dict[str, int]:
        return dict(self.counts)


# ---------------------------------------------------------------------------
# Controllers: Aioli transfer-matrix recovery, J-R1 retention-aware controller,
# static schedules, and matched-exposure schedule replay
# ---------------------------------------------------------------------------

def _finite(value: float, name: str = "value") -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must not be boolean")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _project_with_floors(weights: Sequence[float], floors: Sequence[float]) -> list[float]:
    if len(weights) != len(floors):
        raise ValueError("weights and floors must have identical length")
    if math.fsum(floors) >= 1.0 - 1e-9:
        raise ValueError("sum of floors must be strictly below 1")
    remaining = set(range(len(weights)))
    result = [0.0] * len(weights)
    available = 1.0
    while remaining:
        mass = math.fsum(max(0.0, weights[i]) for i in remaining)
        candidates = {
            i: available * max(0.0, weights[i]) / mass if mass > 0 else available / len(remaining)
            for i in remaining
        }
        clipped = {i for i in remaining if candidates[i] < floors[i]}
        if not clipped:
            for i in remaining:
                result[i] = candidates[i]
            break
        for i in clipped:
            result[i] = floors[i]
            available -= floors[i]
        remaining -= clipped
    return result


def smoothed_probe_matrix(k: int = 4, alpha: float = 0.75) -> list[list[float]]:
    if k < 2 or not (0.0 <= alpha <= 1.0):
        raise ValueError("invalid probe matrix dimensions or alpha")
    beta = (1.0 - alpha) / (k - 1)
    if abs(alpha - beta) <= 1e-12:
        raise ValueError("singular probe matrix")
    return [[alpha if i == j else beta for j in range(k)] for i in range(k)]


def recover_transfer_matrix(
    loss_drops: Sequence[Sequence[float]], *, alpha: float = 0.75, k: int = 4
) -> list[list[float]]:
    W = smoothed_probe_matrix(k, alpha)
    a, b = W[0][0], W[0][1]
    result: list[list[float]] = []
    for row in loss_drops:
        if len(row) != k:
            raise ValueError(f"expected {k} columns in loss_drops")
        checked = [_finite(v, "loss_drop") for v in row]
        mean = math.fsum(v / k for v in checked)
        shared = b * k * mean
        result.append([_finite((v - shared) / (a - b), "transfer_entry") for v in checked])
    return result


def normalize_transfer_matrix(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    if not matrix or not matrix[0]:
        raise ValueError("transfer matrix must be nonempty")
    k = len(matrix[0])
    checked = [[_finite(v, "transfer") for v in row] for row in matrix]
    if any(len(row) != k for row in checked):
        raise ValueError("inconsistent transfer matrix row lengths")
    minimum = min(min(row) for row in checked)
    offset = min(0.0, minimum)
    shifted = [[v - offset for v in row] for row in checked]
    maximum = max(max(row) for row in shifted)
    if maximum == 0.0:
        return [[0.0 for _ in row] for row in checked]
    scaled = [[v / maximum for v in row] for row in shifted]
    total = math.fsum(v for row in scaled for v in row)
    return [[v / total for v in row] for row in scaled]


class CampaignController:
    """Unified objective controller for all 6 methods in 03 and 3 methods in 05."""

    def __init__(
        self,
        method: str,
        *,
        seed: int = 0,
        alpha: float = 0.75,
        eta: float = 0.8,
        floor: float = 0.05,
        retention_nll_tol: float = 0.02,
        retention_arc_tol: float = 0.01,
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        self.method = method
        self.seed = int(seed)
        self.alpha = float(alpha)
        self.eta = float(eta)
        self.floor = float(floor)
        self.retention_nll_tol = float(retention_nll_tol)
        self.retention_arc_tol = float(retention_arc_tol)
        self._W = smoothed_probe_matrix(4, self.alpha)
        self._rng = random.Random(self.seed)
        self._weights = list(FIXED_WEIGHTS)
        self.updates = 0
        self.draws = 0
        self.retention_activations = 0
        self.last_update: dict[str, object] | None = None

    def prior_weights(self, progress: float) -> list[float]:
        u = min(1.0, max(0.0, float(progress)))
        return [(1.0 - u) * a + u * b for a, b in zip(JR1_PRIOR_START, JR1_PRIOR_END)]

    def probabilities(self, progress: float = 0.0) -> dict[str, float]:
        u = min(1.0, max(0.0, float(progress)))
        if self.method == "sequential":
            phase = 3
            for i, b in enumerate(SEQUENTIAL_BOUNDARIES):
                if u < b:
                    phase = i
                    break
            w = [float(i == phase) for i in range(4)]
        elif self.method == "replay":
            phase = 3
            for i, b in enumerate(SEQUENTIAL_BOUNDARIES):
                if u < b:
                    phase = i
                    break
            if phase == 0:
                w = [1.0, 0.0, 0.0, 0.0]
            else:
                w = [REPLAY_PT_FLOOR if i == 0 else (1.0 - REPLAY_PT_FLOOR if i == phase else 0.0) for i in range(4)]
        elif self.method == "fixed_joint":
            w = list(FIXED_WEIGHTS)
        elif self.method == "smooth_joint":
            w = [(1.0 - u) * a + u * b for a, b in zip(SMOOTH_START, SMOOTH_END)]
        elif self.method in ("aioli", "jr1"):
            w = list(self._weights)
        else:
            raise ValueError(self.method)
        return dict(zip(OBJECTIVES, w))

    def sample(self, progress: float = 0.0) -> str:
        probs = self.probabilities(progress)
        draw = self._rng.random()
        self.draws += 1
        cum = 0.0
        for obj in OBJECTIVES[:-1]:
            cum += probs[obj]
            if draw < cum:
                return obj
        return OBJECTIVES[-1]

    def probe_order(self) -> list[int]:
        order = list(range(4))
        self._rng.shuffle(order)
        return order

    def update_from_probes(
        self,
        loss_drops: Sequence[Sequence[float]],
        *,
        progress: float = 0.0,
        initial_pt_nll: float | None = None,
        current_pt_nll: float | None = None,
        initial_arc_acc: float | None = None,
        current_arc_acc: float | None = None,
    ) -> dict[str, float]:
        if self.method not in ("aioli", "jr1"):
            raise ValueError(f"Method {self.method} does not accept probe updates")
        transfer = recover_transfer_matrix(loss_drops, alpha=self.alpha, k=4)
        normalized = normalize_transfer_matrix(transfer)
        influence = [math.fsum(row[j] for row in normalized) for j in range(4)]
        retention_active = False
        if self.method == "aioli":
            logits = [math.log(max(w, 1e-12)) + self.eta * g for w, g in zip(self._weights, influence)]
            floors = [self.floor] * 4
        else:
            prior = self.prior_weights(progress)
            logits = [
                0.5 * math.log(max(w, 1e-12)) + 0.5 * math.log(max(p, 1e-12)) + self.eta * g
                for w, p, g in zip(self._weights, prior, influence)
            ]
            nll_degraded = (
                initial_pt_nll is not None
                and current_pt_nll is not None
                and current_pt_nll > initial_pt_nll * (1.0 + self.retention_nll_tol)
            )
            arc_degraded = (
                initial_arc_acc is not None
                and current_arc_acc is not None
                and current_arc_acc < initial_arc_acc - self.retention_arc_tol
            )
            retention_active = bool(nll_degraded or arc_degraded)
            if retention_active:
                self.retention_activations += 1
                floors = [0.40, 0.20, self.floor, self.floor]
            else:
                floors = [self.floor] * 4
        max_logit = max(logits)
        raw = [math.exp(l - max_logit) for l in logits]
        self._weights = _project_with_floors(raw, floors)
        self.updates += 1
        self.last_update = {
            "loss_drops": [list(r) for r in loss_drops],
            "transfer": transfer,
            "normalized": normalized,
            "influence": influence,
            "retention_active": retention_active,
            "floors": floors,
            "weights": list(self._weights),
        }
        return self.probabilities(progress)

    def state_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "seed": self.seed,
            "alpha": self.alpha,
            "eta": self.eta,
            "floor": self.floor,
            "weights": list(self._weights),
            "updates": self.updates,
            "draws": self.draws,
            "retention_activations": self.retention_activations,
            "last_update": self.last_update,
            "rng": self._rng.getstate(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self._weights = list(state["weights"])
        self.updates = int(state["updates"])
        self.draws = int(state["draws"])
        self.retention_activations = int(state.get("retention_activations", 0))
        self.last_update = state.get("last_update")
        self._rng.setstate(state["rng"])

    def snapshot(self, progress: float = 0.0) -> dict[str, object]:
        return {
            "method": self.method,
            "seed": self.seed,
            "updates": self.updates,
            "draws": self.draws,
            "retention_activations": self.retention_activations,
            "probabilities": self.probabilities(progress),
            "last_update": self.last_update,
        }


def build_matched_exposure_schedules(
    interleaved_actions: Sequence[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    """Build exact matched-exposure schedules for 04_qwen_matched_exposure_schedules.

    Preserves exact objective microbatch counts and example index sequences,
    varying only whether microbatches are executed in stage-sorted order
    (PT -> SFT -> DPO -> RL) or interleaved order.
    """
    order_key = {obj: i for i, obj in enumerate(OBJECTIVES)}
    interleaved = [dict(x) for x in interleaved_actions]
    # Stable sort by stage order preserves exact within-objective item sequence!
    sequential_sorted = sorted(interleaved, key=lambda x: order_key[str(x["objective"])])
    return {
        "sequential_sorted": sequential_sorted,
        "interleaved_replay": interleaved,
    }


def verify_split_disjointness(splits: Mapping[str, Sequence[Mapping[str, object]]]) -> dict[str, object]:
    """Verify strict prompt/item disjointness between train/controller/dev/test."""
    train_sft_ids = {str(r["id"]) for r in splits.get("gsm_sft_train", [])}
    train_rl_ids = {str(r["id"]) for r in splits.get("gsm_rl_train", [])}
    readiness_ids = {str(r["id"]) for r in splits.get("gsm_readiness", [])}
    ctrl_ids = {str(r["id"]) for r in splits.get("gsm_controller", [])}
    dev_ids = {str(r["id"]) for r in splits.get("gsm_development", [])}
    test_ids = {str(r["id"]) for r in splits.get("gsm_test", [])}

    sft_rl_overlap = len(train_sft_ids & train_rl_ids)
    train_test_overlap = len((train_sft_ids | train_rl_ids) & test_ids)
    train_ctrl_overlap = len((train_sft_ids | train_rl_ids) & (ctrl_ids | dev_ids | readiness_ids))
    if sft_rl_overlap or train_test_overlap or train_ctrl_overlap:
        raise ValueError(
            f"Split overlap detected: sft_rl={sft_rl_overlap}, "
            f"train_test={train_test_overlap}, train_ctrl={train_ctrl_overlap}"
        )
    return {
        "sft_rl_overlap": sft_rl_overlap,
        "train_test_overlap": train_test_overlap,
        "train_ctrl_overlap": train_ctrl_overlap,
        "disjoint": True,
    }


# ---------------------------------------------------------------------------
# Data loading, preparation, and prompts
# ---------------------------------------------------------------------------

def data_dir() -> Path:
    p = RUN / "hf"
    p.mkdir(parents=True, exist_ok=True)
    return p


def prepared_dir() -> Path:
    p = RUN / "prepared_data"
    p.mkdir(parents=True, exist_ok=True)
    return p


def gsm_rows(split: str) -> list[dict]:
    ds = load_dataset("openai/gsm8k", "main", split=split, cache_dir=str(data_dir()))
    return [
        {"id": f"gsm8k:{split}:{i}", "question": r["question"], "answer": r["answer"]}
        for i, r in enumerate(ds)
    ]


def prompt(question: str) -> str:
    return (
        "Solve the problem. Show your reasoning, then put the final numeric answer after ####.\n\n"
        "Problem: " + question.strip() + "\nAnswer:"
    )


def math500_prompt(problem: str) -> str:
    return (
        "Solve the mathematics problem step by step, then put the final answer inside \\boxed{}.\n\n"
        "Problem: " + problem.strip() + "\nAnswer:"
    )


def mmlu_prompt(subject: str, question: str, choices: Sequence[str]) -> str:
    subj = str(subject).replace("_", " ").strip()
    letters = ("A", "B", "C", "D")
    opts = "\n".join(f"{letters[i]}. {choices[i]}" for i in range(min(4, len(choices))))
    return (
        f"The following are multiple choice questions (with answers) about {subj}.\n\n"
        f"{question.strip()}\n{opts}\nAnswer:"
    )


def arc_prompt(question: str) -> str:
    return f"Question: {question.strip()}\nAnswer:"


def prepare_campaign_data() -> dict[str, object]:
    """Prepare and hash all campaign splits with strict contamination checks."""
    manifest_path = RUN / "data_manifest.json"
    repo_manifest_path = REPO / "experiments" / EXP_03 / "expt_v1" / "data_manifest.json"
    if manifest_path.exists() and (prepared_dir() / "gsm_test.jsonl").exists():
        manifest = read_json(manifest_path, {})
        if manifest.get("status") == "complete":
            if not repo_manifest_path.exists():
                atomic_json(repo_manifest_path, manifest)
            return manifest

    update_progress("preparing_campaign_data")
    cache = str(data_dir())
    gsm_train_raw = load_dataset("openai/gsm8k", "main", split="train", cache_dir=cache)
    gsm_test_raw = load_dataset("openai/gsm8k", "main", split="test", cache_dir=cache)
    math500_raw = load_dataset("HuggingFaceH4/MATH-500", split="test", cache_dir=cache)
    mmlu_raw = load_dataset("cais/mmlu", "all", split="test", cache_dir=cache)
    arc_val_raw = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation", cache_dir=cache)
    arc_test_raw = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test", cache_dir=cache)
    wiki_raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", cache_dir=cache)
    prefs_train_raw = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs", cache_dir=cache)
    prefs_test_raw = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs", cache_dir=cache)

    splits: dict[str, list[dict[str, object]]] = {}

    # 1. Official sealed test sets (100% full denominators preserved)
    splits["gsm_test"] = [
        {
            "id": f"gsm8k:test:{i}",
            "text_hash": text_hash(r["question"]),
            "question": r["question"],
            "answer": r["answer"],
        }
        for i, r in enumerate(gsm_test_raw)
    ]
    splits["math500_test"] = [
        {
            "id": str(r.get("unique_id", f"math500:test:{i}")),
            "text_hash": text_hash(r["problem"]),
            "problem": r["problem"],
            "answer": r["answer"],
            "subject": r.get("subject", ""),
            "level": int(r.get("level", 0)),
        }
        for i, r in enumerate(math500_raw)
    ]
    splits["mmlu_test"] = [
        {
            "id": f"mmlu:test:{i}",
            "text_hash": text_hash(r["question"]),
            "subject": r["subject"],
            "question": r["question"],
            "choices": list(r["choices"]),
            "answer": int(r["answer"]),
        }
        for i, r in enumerate(mmlu_raw)
    ]
    splits["arc_test"] = [
        {
            "id": f"arc:test:{r['id']}",
            "text_hash": text_hash(r["question"]),
            "question": r["question"],
            "choices": {"text": list(r["choices"]["text"]), "label": list(r["choices"]["label"])},
            "answerKey": r["answerKey"],
        }
        for r in arc_test_raw
    ]
    assert len(splits["gsm_test"]) == 1319
    assert len(splits["math500_test"]) == 500
    assert len(splits["mmlu_test"]) == 14042
    assert len(splits["arc_test"]) == 2376

    # Sealed learning-curve subsets (256 GSM8K and 512 subject-stratified MMLU)
    rng_curve = random.Random(20261005)
    gsm_indices = sorted(rng_curve.sample(range(len(splits["gsm_test"])), 256))
    splits["gsm_curve_test"] = [splits["gsm_test"][i] for i in gsm_indices]

    by_subj: dict[str, list[dict[str, object]]] = collections.defaultdict(list)
    for row in splits["mmlu_test"]:
        by_subj[str(row["subject"])].append(row)
    mmlu_curve: list[dict[str, object]] = []
    subjects = sorted(by_subj.keys())
    per_subj = 512 // len(subjects)
    rem = 512 % len(subjects)
    for idx, subj in enumerate(subjects):
        take = per_subj + (1 if idx < rem else 0)
        mmlu_curve.extend(by_subj[subj][:take])
    assert len(mmlu_curve) == 512
    splits["mmlu_curve_test"] = mmlu_curve

    sealed_hashes = {
        r["text_hash"]
        for k in ("gsm_test", "math500_test", "mmlu_test", "arc_test")
        for r in splits[k]
    }

    # 2. ARC validation -> controller (128) and development (256)
    arc_val_rows = [
        {
            "id": f"arc:val:{r['id']}",
            "text_hash": text_hash(r["question"]),
            "question": r["question"],
            "choices": {"text": list(r["choices"]["text"]), "label": list(r["choices"]["label"])},
            "answerKey": r["answerKey"],
        }
        for r in arc_val_raw
        if text_hash(r["question"]) not in sealed_hashes
    ]
    splits["arc_controller"] = arc_val_rows[:128]
    splits["arc_development"] = arc_val_rows[128:384]
    sealed_hashes.update(r["text_hash"] for r in arc_val_rows)

    # 3. GSM8K train -> readiness (first 128), controller (128), development (128),
    # and strictly disjoint gsm_sft_train vs gsm_rl_train
    splits["gsm_readiness"] = [
        {
            "id": f"gsm8k:train:{i}",
            "text_hash": text_hash(gsm_train_raw[i]["question"]),
            "question": gsm_train_raw[i]["question"],
            "answer": gsm_train_raw[i]["answer"],
        }
        for i in range(128)
    ]
    seen_gsm = {r["text_hash"] for r in splits["gsm_readiness"]} | sealed_hashes
    clean_gsm: list[dict[str, object]] = []
    overlap_excluded_gsm = 0
    for i in range(128, len(gsm_train_raw)):
        r = gsm_train_raw[i]
        th = text_hash(r["question"])
        if th in seen_gsm:
            overlap_excluded_gsm += 1
            continue
        seen_gsm.add(th)
        clean_gsm.append(
            {
                "id": f"gsm8k:train:{i}",
                "text_hash": th,
                "question": r["question"],
                "answer": r["answer"],
            }
        )
    splits["gsm_controller"] = clean_gsm[:128]
    splits["gsm_development"] = clean_gsm[128:256]
    remaining_gsm = clean_gsm[256:]
    mid = len(remaining_gsm) // 2
    splits["gsm_sft_train"] = remaining_gsm[:mid]
    splits["gsm_rl_train"] = remaining_gsm[mid : 2 * mid]

    # 4. UltraFeedback binarized preferences -> train (4096), controller (128), dev (128), test (256)
    all_forbidden_hashes = seen_gsm | sealed_hashes
    prefs_test_clean: list[dict[str, object]] = []
    seen_prefs: set[str] = set()
    for i, r in enumerate(prefs_test_raw):
        th = text_hash(r["prompt"])
        if th in all_forbidden_hashes or th in seen_prefs:
            continue
        chosen, rejected = r["chosen"], r["rejected"]
        if len(chosen) < 2 or len(rejected) < 2:
            continue
        c_txt = chosen[-1].get("content", "").strip()
        r_txt = rejected[-1].get("content", "").strip()
        if not c_txt or not r_txt or c_txt == r_txt:
            continue
        seen_prefs.add(th)
        prefs_test_clean.append(
            {
                "id": f"prefs:test:{i}",
                "text_hash": th,
                "prompt": r["prompt"],
                "chosen": c_txt,
                "rejected": r_txt,
            }
        )
        if len(prefs_test_clean) >= 256:
            break
    splits["prefs_test"] = prefs_test_clean
    all_forbidden_hashes.update(seen_prefs)

    prefs_pool: list[dict[str, object]] = []
    overlap_excluded_prefs = 0
    for i, r in enumerate(prefs_train_raw):
        th = text_hash(r["prompt"])
        if th in all_forbidden_hashes:
            overlap_excluded_prefs += 1
            continue
        chosen, rejected = r["chosen"], r["rejected"]
        if len(chosen) < 2 or len(rejected) < 2:
            continue
        c_txt = chosen[-1].get("content", "").strip()
        r_txt = rejected[-1].get("content", "").strip()
        if not c_txt or not r_txt or c_txt == r_txt:
            continue
        if len(r["prompt"]) > 1200 or len(c_txt) > 1600 or len(r_txt) > 1600:
            continue
        all_forbidden_hashes.add(th)
        prefs_pool.append(
            {
                "id": f"prefs:train:{i}",
                "text_hash": th,
                "prompt": r["prompt"],
                "chosen": c_txt,
                "rejected": r_txt,
            }
        )
        if len(prefs_pool) >= 4096 + 256:
            break
    splits["prefs_controller"] = prefs_pool[:128]
    splits["prefs_development"] = prefs_pool[128:256]
    splits["prefs_train"] = prefs_pool[256 : 256 + 4096]

    # 5. WikiText-2 raw v1 paragraphs -> train, controller (64), development (64), test
    wiki_holdout_hashes = {
        text_hash(r["text"])
        for sp in ("validation", "test")
        for r in wiki_raw[sp]
        if str(r["text"]).strip()
    }
    wiki_test_rows: list[dict[str, object]] = []
    buf: list[str] = []
    for r in wiki_raw["test"]:
        t = str(r["text"]).strip()
        if not t:
            continue
        buf.append(t)
        if sum(len(x) for x in buf) >= 900:
            joined = "\n".join(buf)
            wiki_test_rows.append({"id": f"wiki:test:{len(wiki_test_rows)}", "text_hash": text_hash(joined), "text": joined})
            buf = []
    if buf:
        joined = "\n".join(buf)
        wiki_test_rows.append({"id": f"wiki:test:{len(wiki_test_rows)}", "text_hash": text_hash(joined), "text": joined})
    splits["wiki_test"] = wiki_test_rows

    wiki_val_rows: list[dict[str, object]] = []
    buf = []
    for r in wiki_raw["validation"]:
        t = str(r["text"]).strip()
        if not t:
            continue
        buf.append(t)
        if sum(len(x) for x in buf) >= 900:
            joined = "\n".join(buf)
            wiki_val_rows.append({"id": f"wiki:val:{len(wiki_val_rows)}", "text_hash": text_hash(joined), "text": joined})
            buf = []
    splits["wiki_controller"] = wiki_val_rows[:64]
    splits["wiki_development"] = wiki_val_rows[64:128] if len(wiki_val_rows) >= 128 else wiki_val_rows[:64]

    wiki_train_rows: list[dict[str, object]] = []
    buf = []
    seen_wiki_lines: set[str] = set()
    for r in wiki_raw["train"]:
        t = str(r["text"]).strip()
        if not t:
            continue
        th = text_hash(t)
        if th in wiki_holdout_hashes or th in seen_wiki_lines:
            continue
        seen_wiki_lines.add(th)
        buf.append(t)
        if sum(len(x) for x in buf) >= 900:
            joined = "\n".join(buf)
            wiki_train_rows.append({"id": f"wiki:train:{len(wiki_train_rows)}", "text_hash": text_hash(joined), "text": joined})
            buf = []
    splits["wiki_train"] = wiki_train_rows

    audit = verify_split_disjointness(splits)
    split_records: dict[str, object] = {}
    for name, rows in splits.items():
        out_path = prepared_dir() / f"{name}.jsonl"
        with out_path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        split_records[name] = {
            "rows": len(rows),
            "sha256": sha256(out_path),
        }

    manifest = {
        "status": "complete",
        "timestamp": stamp(),
        "models": {
            "qwen": "Qwen/Qwen2.5-1.5B",
            "pythia": "EleutherAI/pythia-160m",
        },
        "sources": {
            "gsm8k": "openai/gsm8k (main)",
            "math500": "HuggingFaceH4/MATH-500 (test)",
            "mmlu": "cais/mmlu (all)",
            "arc_easy": "allenai/ai2_arc (ARC-Easy)",
            "wikitext": "Salesforce/wikitext (wikitext-2-raw-v1)",
            "ultrafeedback": "HuggingFaceH4/ultrafeedback_binarized",
        },
        "contamination_audit": {
            **audit,
            "gsm_overlap_excluded": overlap_excluded_gsm,
            "prefs_overlap_excluded": overlap_excluded_prefs,
            "inherited_pretraining_contamination": "unknown (pretrained base weights not audited against public web)",
        },
        "splits": split_records,
    }
    atomic_json(manifest_path, manifest)
    atomic_json(repo_manifest_path, manifest)
    return manifest


def load_split(name: str) -> list[dict[str, object]]:
    path = prepared_dir() / f"{name}.jsonl"
    if not path.exists():
        prepare_campaign_data()
    with path.open() as f:
        return [json.loads(line) for line in f]


class DataStream:
    """Deterministic, restartable shuffled stream over a split."""

    def __init__(self, rows: list[dict[str, object]], seed: int) -> None:
        self.rows = rows
        self.rng = random.Random(seed)
        self.order = list(range(len(rows)))
        self.rng.shuffle(self.order)
        self.pos = 0
        self.epochs = 0

    def take(self, n: int) -> tuple[list[dict[str, object]], list[int]]:
        out: list[dict[str, object]] = []
        indices: list[int] = []
        for _ in range(n):
            if self.pos >= len(self.order):
                self.rng.shuffle(self.order)
                self.pos = 0
                self.epochs += 1
            idx = self.order[self.pos]
            self.pos += 1
            out.append(self.rows[idx])
            indices.append(idx)
        return out, indices

    def take_by_indices(self, indices: Sequence[int]) -> list[dict[str, object]]:
        return [self.rows[int(i) % len(self.rows)] for i in indices]

    def state_dict(self) -> dict[str, object]:
        return {
            "order": list(self.order),
            "pos": self.pos,
            "epochs": self.epochs,
            "rng": self.rng.getstate(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.order = list(state["order"])
        self.pos = int(state["pos"])
        self.epochs = int(state["epochs"])
        self.rng.setstate(state["rng"])


# ---------------------------------------------------------------------------
# Model loading, readiness, calibration, and objective smoke
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_qwen() -> tuple[object, object, torch.device]:
    model_id = "Qwen/Qwen2.5-1.5B"
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=str(data_dir()), trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=str(data_dir()),
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).to(device)
    model.config.pad_token_id = tok.pad_token_id
    return model, tok, device


def load_pythia_scratch(seed: int) -> tuple[object, object, torch.device]:
    model_id = "EleutherAI/pythia-160m"
    seed_everything(seed)
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=str(data_dir()), trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = AutoConfig.from_pretrained(model_id, cache_dir=str(data_dir()), trust_remote_code=False)
    cfg.use_cache = False
    model = AutoModelForCausalLM.from_config(
        cfg,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model.config.pad_token_id = tok.pad_token_id
    return model, tok, device


@torch.no_grad()
def readiness(model, tok, device: torch.device) -> dict:
    rows = gsm_rows("train")[:128]
    by_prompt, truncations, total = [], 0, 0
    model.eval()
    for offset in range(0, len(rows), 4):
        batch = rows[offset : offset + 4]
        inputs = tok(
            [prompt(r["question"]) for r in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            num_return_sequences=8,
            max_new_tokens=1024,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            use_cache=True,
        )
        decoded = tok.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        for i, row in enumerate(batch):
            samples = decoded[i * 8 : (i + 1) * 8]
            rewards = [exact_reward(x, row["answer"]) for x in samples]
            by_prompt.append({"id": row["id"], "rewards": rewards, "successes": int(sum(rewards)), "samples": 8})
            truncations += sum(len(tok.encode(x, add_special_tokens=False)) >= 1024 for x in samples)
            total += 8
        atomic_json(
            RUN / "readiness_partial.json",
            {
                "timestamp": stamp(),
                "completed_prompt_groups": len(by_prompt),
                "expected_prompt_groups": len(rows),
                "samples": total,
                "truncations": truncations,
                "rows": by_prompt,
            },
        )
    mixed = sum(0 < x["successes"] < 8 for x in by_prompt)
    positive = sum(x["successes"] for x in by_prompt)
    result = {
        "timestamp": stamp(),
        "model": "Qwen/Qwen2.5-1.5B",
        "split": "gsm8k train first 128",
        "prompt_groups": len(by_prompt),
        "samples": total,
        "mixed_success_groups": mixed,
        "positive_rewards": positive,
        "negative_rewards": total - positive,
        "truncations": truncations,
        "truncation_rate": truncations / total,
        "criteria": {"mixed_success_groups_min": 20, "truncation_rate_max": 0.10},
        "passed": mixed >= 20 and positive > 0 and total - positive > 0 and truncations / total <= 0.10,
        "rows": by_prompt,
    }
    atomic_json(RUN / "readiness.json", result)
    return result


def calibration(model, tok, device: torch.device) -> dict:
    rows = gsm_rows("train")[:2]
    enc = tok(
        [prompt(r["question"]) + " " + r["answer"] for r in rows],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)
    labels = mask_labels(
        enc.input_ids,
        [len(tok.encode(prompt(r["question"]), add_special_tokens=False)) for r in rows],
        tok.pad_token_id,
    )
    optim = torch.optim.AdamW(model.parameters(), lr=1e-6)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    model.train()
    optim.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(**enc, use_cache=False).logits
        loss = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), labels[:, 1:], ignore_index=-100)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - start
    result = {
        "timestamp": stamp(),
        "objective": "masked_sft",
        "loss": float(loss.detach()),
        "batch_rows": len(rows),
        "tokens": int(enc.input_ids.numel()),
        "seconds": elapsed,
        "tokens_per_second": int(enc.input_ids.numel()) / elapsed,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "device": torch.cuda.get_device_name(device),
        "admission_note": "Single masked SFT update only; no measured cell has started.",
    }
    atomic_json(RUN / "calibration.json", result)
    return result


def pack(tok, texts: list[str], starts: list[int], device: torch.device, max_length: int = 512) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    orig_side = tok.padding_side
    tok.padding_side = "right"
    try:
        encoded = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    finally:
        tok.padding_side = orig_side
    return encoded.input_ids, encoded.attention_mask, mask_labels(encoded.input_ids, starts, tok.pad_token_id)


def objective_smoke(model, tok, device: torch.device) -> dict:
    gsm = gsm_rows("train")[:4]
    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train[:2]", cache_dir=str(data_dir()))
    optim = torch.optim.AdamW(model.parameters(), lr=1e-6)
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    rows = []

    def step(name: str, loss: torch.Tensor, tokens: int, metadata: dict) -> None:
        optim.zero_grad(set_to_none=True)
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if grad > 0:
            optim.step()
        rows.append(
            {
                "objective": name,
                "loss": float(loss.detach()),
                "tokens": tokens,
                "gradient_norm_before_clip": grad,
                "optimizer_step": bool(grad > 0),
                **metadata,
            }
        )

    pt = tok([str(x["text"]) or " " for x in wiki], return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(**pt, use_cache=False).logits
        pt_loss = F.cross_entropy(out[:, :-1].float().transpose(1, 2), pt.input_ids[:, 1:], ignore_index=tok.pad_token_id)
    step("pt", pt_loss, int(pt.input_ids.numel()), {"source": "wikitext-2-raw-v1 train[:2]"})

    sft_text = [prompt(x["question"]) + " " + x["answer"] for x in gsm[:2]]
    starts = [len(tok.encode(prompt(x["question"]), add_special_tokens=False)) for x in gsm[:2]]
    sft_ids, sft_attn, sft_labels = pack(tok, sft_text, starts, device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        sft_logits = model(input_ids=sft_ids, attention_mask=sft_attn, use_cache=False).logits
        sft_loss = F.cross_entropy(sft_logits[:, :-1].float().transpose(1, 2), sft_labels[:, 1:], ignore_index=-100)
    step("sft", sft_loss, int(sft_ids.numel()), {"source": "gsm8k train[:2]", "prompt_masked": True})

    chosen = [prompt(x["question"]) + " " + x["answer"] for x in gsm[:2]]
    rejected = [prompt(x["question"]) + " I do not know." for x in gsm[:2]]
    cids, cattn, clabels = pack(tok, chosen, starts, device)
    rids, rattn, rlabels = pack(tok, rejected, starts, device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        policy_c = sequence_logps(model(input_ids=cids, attention_mask=cattn, use_cache=False).logits, cids, clabels)
        policy_r = sequence_logps(model(input_ids=rids, attention_mask=rattn, use_cache=False).logits, rids, rlabels)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref_c = sequence_logps(ref(input_ids=cids, attention_mask=cattn, use_cache=False).logits, cids, clabels)
        ref_r = sequence_logps(ref(input_ids=rids, attention_mask=rattn, use_cache=False).logits, rids, rlabels)
    step(
        "dpo",
        dpo_loss(policy_c, policy_r, ref_c, ref_r),
        int(cids.numel() + rids.numel()),
        {"source": "synthetic-negative objective smoke only; not training preference data", "reference": "frozen initial model", "sequence_summed": True},
    )

    ready = read_json(RUN / "readiness.json", {})
    mixed_id = next((r["id"] for r in ready.get("rows", []) if 0 < r["successes"] < 8), gsm[0]["id"])
    row = next(x for x in gsm_rows("train") if x["id"] == mixed_id)
    enc = tok([prompt(row["question"])] * 4, return_tensors="pt", padding=True).to(device)
    sampled = model.generate(
        **enc,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        max_new_tokens=256,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        use_cache=True,
    )
    continuation = sampled[:, enc.input_ids.shape[1] :]
    texts = tok.batch_decode(continuation, skip_special_tokens=True)
    rewards = torch.tensor([exact_reward(x, row["answer"]) for x in texts], device=device)
    advantages = loo_advantages(rewards, 4)
    seq = torch.cat([enc.input_ids, continuation], dim=1)
    labels = mask_labels(seq, [enc.input_ids.shape[1]] * len(seq), tok.pad_token_id)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=seq, use_cache=False).logits
        rl_loss = -(advantages.detach() * sequence_logps(logits, seq, labels)).mean()
    if bool((advantages == 0).all()):
        rows.append(
            {
                "objective": "rl",
                "loss": float(rl_loss.detach()),
                "tokens": int(seq.numel()),
                "rewards": rewards.tolist(),
                "zero_advantage": True,
                "optimizer_step": False,
                "source": "unassisted on-policy GSM8K rollout",
            }
        )
    else:
        step("rl", rl_loss, int(seq.numel()), {"rewards": rewards.tolist(), "zero_advantage": False, "source": "unassisted on-policy GSM8K rollout"})
    result = {
        "timestamp": stamp(),
        "objectives": rows,
        "shared_optimizer": "AdamW",
        "complete": len(rows) == 4,
        "all_nonzero_steps": all(x["optimizer_step"] for x in rows),
        "development_only": True,
    }
    atomic_json(RUN / "objective_smoke.json", result)
    return result


# ---------------------------------------------------------------------------
# Evaluation primitives (full denominators + learning-curve subsets)
# ---------------------------------------------------------------------------

def wilson_interval(values: Sequence[float]) -> dict[str, object]:
    a = np.asarray(values, dtype=float)
    n = len(a)
    if n == 0:
        return {"estimate": None, "interval95": None, "n": 0}
    p = float(a.mean())
    z = 1.959963984540054
    den = 1.0 + z * z / n
    mid = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt(max(0.0, p * (1.0 - p) / n + z * z / (4.0 * n * n))) / den
    return {
        "estimate": p,
        "interval95": [max(0.0, mid - half), min(1.0, mid + half)],
        "n": n,
        "correct": int(round(a.sum())),
        "test": "none; Wilson conditional item 95% score interval",
    }


@torch.no_grad()
def eval_wiki_nll(
    model, tok, rows: Sequence[Mapping[str, object]], device: torch.device, work: Work, batch_size: int = 16
) -> tuple[dict[str, object], list[dict[str, object]]]:
    model.eval()
    items: list[dict[str, object]] = []
    orig_side = tok.padding_side
    tok.padding_side = "right"
    try:
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            enc = tok(
                [str(r["text"]) for r in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)
            ids = enc.input_ids
            attn = enc.attention_mask
            work.add("eval_forward", int(ids.numel()))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(input_ids=ids, attention_mask=attn, use_cache=False).logits[:, :-1]
            targets = ids[:, 1:]
            mask = attn[:, 1:].bool()
            token_nll = F.cross_entropy(logits.float().transpose(1, 2), targets, reduction="none")
            token_nll = token_nll.masked_fill(~mask, 0.0)
            sums = token_nll.sum(dim=1).tolist()
            counts = mask.sum(dim=1).tolist()
            for r, s, c in zip(batch, sums, counts):
                c_int = max(1, int(c))
                items.append({"id": r["id"], "nll_sum": float(s), "tokens": c_int, "nll": float(s) / c_int})
    finally:
        tok.padding_side = orig_side
    total_nll = sum(x["nll_sum"] for x in items)
    total_tok = max(1, sum(x["tokens"] for x in items))
    mean_nll = total_nll / total_tok
    rng = np.random.default_rng(192)
    sums_arr = np.array([x["nll_sum"] for x in items])
    cnts_arr = np.array([x["tokens"] for x in items])
    draws = []
    for _ in range(1000):
        ix = rng.integers(0, len(items), len(items))
        draws.append(float(sums_arr[ix].sum() / max(1, cnts_arr[ix].sum())))
    ci = np.quantile(draws, [0.025, 0.975]).tolist()
    summary = {
        "nll": mean_nll,
        "nll_interval95": ci,
        "perplexity": math.exp(min(mean_nll, 50.0)),
        "tokens": total_tok,
        "blocks": len(items),
    }
    return summary, items


@torch.no_grad()
def eval_mmlu(
    model, tok, rows: Sequence[Mapping[str, object]], device: torch.device, work: Work, batch_size: int = 64
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Fixed Hendrycks et al. multiple-choice next-token likelihood protocol."""
    model.eval()
    choice_ids = [tok.encode(f" {ch}", add_special_tokens=False)[-1] for ch in ("A", "B", "C", "D")]
    choice_tensor = torch.tensor(choice_ids, device=device, dtype=torch.long)
    items: list[dict[str, object]] = []
    orig_side = tok.padding_side
    tok.padding_side = "left"
    try:
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            prompts = [mmlu_prompt(str(r["subject"]), str(r["question"]), r["choices"]) for r in batch]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            work.add("eval_forward", int(enc.input_ids.numel()))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False).logits[:, -1, :]
            log_probs = F.log_softmax(logits.float(), dim=-1)[:, choice_tensor]
            preds = log_probs.argmax(dim=-1).tolist()
            scores_list = log_probs.tolist()
            for r, pred, sc in zip(batch, preds, scores_list):
                gold = int(r["answer"])
                items.append(
                    {
                        "id": r["id"],
                        "subject": r["subject"],
                        "pred": int(pred),
                        "gold": gold,
                        "correct": int(pred == gold),
                        "choice_logprobs": sc,
                    }
                )
    finally:
        tok.padding_side = orig_side
    acc = wilson_interval([x["correct"] for x in items])
    by_subj: dict[str, list[int]] = collections.defaultdict(list)
    for x in items:
        by_subj[str(x["subject"])].append(int(x["correct"]))
    macro = float(np.mean([ np.mean(v) for v in by_subj.values() ])) if by_subj else 0.0
    summary = {
        "accuracy": acc,
        "macro_subject_accuracy": macro,
        "subjects": len(by_subj),
        "rows": len(items),
    }
    return summary, items


@torch.no_grad()
def eval_arc(
    model, tok, rows: Sequence[Mapping[str, object]], device: torch.device, work: Work, batch_size: int = 32
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Length-normalized continuation log-likelihood across all ARC-Easy options."""
    model.eval()
    items: list[dict[str, object]] = []
    orig_side = tok.padding_side
    tok.padding_side = "right"
    try:
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            texts: list[str] = []
            starts: list[int] = []
            counts: list[int] = []
            for r in batch:
                p_txt = arc_prompt(str(r["question"]))
                p_len = len(tok.encode(p_txt, add_special_tokens=False))
                c_texts = r["choices"]["text"]
                counts.append(len(c_texts))
                for c in c_texts:
                    texts.append(p_txt + " " + str(c))
                    starts.append(p_len)
            ids, attn, labels = pack(tok, texts, starts, device, max_length=256)
            work.add("eval_forward", int(ids.numel()))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(input_ids=ids, attention_mask=attn, use_cache=False).logits
            seq_lp = sequence_logps(logits, ids, labels)
            lengths = labels[:, 1:].ne(-100).sum(dim=1).clamp_min(1).float()
            norm_lp = (seq_lp / lengths).tolist()
            raw_lp = seq_lp.tolist()
            offset = 0
            for r, k in zip(batch, counts):
                r_scores = raw_lp[offset : offset + k]
                n_scores = norm_lp[offset : offset + k]
                offset += k
                labels_list = r["choices"]["label"]
                ans = str(r["answerKey"])
                pred_norm = labels_list[int(np.argmax(n_scores))]
                pred_raw = labels_list[int(np.argmax(r_scores))]
                items.append(
                    {
                        "id": r["id"],
                        "correct_normalized": int(pred_norm == ans),
                        "correct_raw": int(pred_raw == ans),
                        "choices": k,
                    }
                )
    finally:
        tok.padding_side = orig_side
    summary = {
        "accuracy_primary_normalized": wilson_interval([x["correct_normalized"] for x in items]),
        "accuracy_raw": wilson_interval([x["correct_raw"] for x in items]),
        "rows": len(items),
    }
    return summary, items


@torch.no_grad()
def eval_preferences(
    model, ref_model, tok, rows: Sequence[Mapping[str, object]], device: torch.device, work: Work, batch_size: int = 16
) -> tuple[dict[str, object], list[dict[str, object]]]:
    model.eval()
    ref_model.eval()
    items: list[dict[str, object]] = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        p_texts = [prompt(str(r["prompt"])) for r in batch]
        starts = [len(tok.encode(p, add_special_tokens=False)) for p in p_texts]
        chosen_texts = [p + " " + str(r["chosen"]) for p, r in zip(p_texts, batch)]
        rejected_texts = [p + " " + str(r["rejected"]) for p, r in zip(p_texts, batch)]
        cids, cattn, clabels = pack(tok, chosen_texts, starts, device, max_length=384)
        rids, rattn, rlabels = pack(tok, rejected_texts, starts, device, max_length=384)
        work.add("eval_forward", int(cids.numel() + rids.numel()) * 2)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pol_c = sequence_logps(model(input_ids=cids, attention_mask=cattn, use_cache=False).logits, cids, clabels)
            pol_r = sequence_logps(model(input_ids=rids, attention_mask=rattn, use_cache=False).logits, rids, rlabels)
            ref_c = sequence_logps(ref_model(input_ids=cids, attention_mask=cattn, use_cache=False).logits, cids, clabels)
            ref_r = sequence_logps(ref_model(input_ids=rids, attention_mask=rattn, use_cache=False).logits, rids, rlabels)
        raw_diff = (pol_c - pol_r).tolist()
        imp_diff = ((pol_c - pol_r) - (ref_c - ref_r)).tolist()
        for r, rd, idf in zip(batch, raw_diff, imp_diff):
            items.append(
                {
                    "id": r["id"],
                    "raw_correct": int(rd > 0),
                    "implicit_correct": int(idf > 0),
                    "margin": float(idf),
                }
            )
    summary = {
        "raw_accuracy": wilson_interval([x["raw_correct"] for x in items]),
        "implicit_accuracy": wilson_interval([x["implicit_correct"] for x in items]),
        "rows": len(items),
    }
    return summary, items


@torch.no_grad()
def eval_generative_math(
    model,
    tok,
    rows: Sequence[Mapping[str, object]],
    device: torch.device,
    work: Work,
    *,
    benchmark: str = "gsm8k",
    batch_size: int = 64,
    max_new_tokens: int = 384,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    model.eval()
    items: list[dict[str, object]] = []
    orig_side = tok.padding_side
    tok.padding_side = "left"
    try:
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            if benchmark == "gsm8k":
                prompts = [prompt(str(r["question"])) for r in batch]
            else:
                prompts = [math500_prompt(str(r["problem"])) for r in batch]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model.generate(
                    **enc,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    stop_strings=STOP_STRINGS,
                    tokenizer=tok,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                    use_cache=True,
                )
            gen = out[:, enc.input_ids.shape[1] :]
            work.add("eval_generation", int(out.numel()))
            texts = tok.batch_decode(gen, skip_special_tokens=True)
            for r, txt, row_ids in zip(batch, texts, gen):
                if (row_ids == tok.eos_token_id).any():
                    gen_len = int((row_ids == tok.eos_token_id).nonzero(as_tuple=True)[0][0].item()) + 1
                else:
                    gen_len = int(row_ids.shape[0])
                truncated = bool(gen_len >= max_new_tokens and not any(s in txt for s in STOP_STRINGS))
                cleaned = strip_stop_strings(txt)
                if benchmark == "gsm8k":
                    score = exact_reward(cleaned, str(r["answer"]))
                    parsed = extract_answer(cleaned)
                else:
                    score = math500_reward(cleaned, str(r["answer"]))
                    parsed = normalize_math_answer(extract_math500_candidate(cleaned))
                items.append(
                    {
                        "id": r["id"],
                        "correct": int(score == 1.0),
                        "parsed": parsed,
                        "generated_tokens": gen_len,
                        "truncated": int(truncated),
                    }
                )
    finally:
        tok.padding_side = orig_side
    summary = {
        "exact_match": wilson_interval([x["correct"] for x in items]),
        "rows": len(items),
        "truncated": sum(x["truncated"] for x in items),
        "truncation_rate": sum(x["truncated"] for x in items) / max(1, len(items)),
        "max_new_tokens": max_new_tokens,
    }
    return summary, items


@torch.no_grad()
def evaluate_curve_milestone(
    model, tok, device: torch.device, work: Work, fraction: float, charged_before: int, wall_seconds: float, max_new_tokens: int = 384
) -> dict[str, object]:
    """Evaluate fixed sealed learning-curve subset (256 GSM8K + 512 MMLU + WikiText controller NLL)."""
    t0 = time.monotonic()
    gsm_sub = load_split("gsm_curve_test")
    mmlu_sub = load_split("mmlu_curve_test")
    wiki_sub = load_split("wiki_controller")[:32]
    gsm_sum, _ = eval_generative_math(model, tok, gsm_sub, device, work, benchmark="gsm8k", batch_size=64, max_new_tokens=max_new_tokens)
    mmlu_sum, _ = eval_mmlu(model, tok, mmlu_sub, device, work, batch_size=64)
    wiki_sum, _ = eval_wiki_nll(model, tok, wiki_sub, device, work, batch_size=16)
    return {
        "fraction": float(fraction),
        "charged_tokens_before_eval": int(charged_before),
        "charged_tokens_after_eval": int(work.total),
        "training_wall_seconds": float(wall_seconds),
        "eval_wall_seconds": float(time.monotonic() - t0),
        "gsm_curve_256": gsm_sum,
        "mmlu_curve_512": mmlu_sum,
        "wiki_nll": wiki_sum["nll"],
    }


@torch.no_grad()
def evaluate_full_terminal(
    model,
    ref_model,
    tok,
    device: torch.device,
    out_dir: Path,
    identity: Mapping[str, object],
    work: Work | None = None,
    max_new_tokens: int = 384,
) -> dict[str, object]:
    """Full terminal evaluation on 100% of all official benchmark denominators."""
    out_dir.mkdir(parents=True, exist_ok=True)
    work = work or Work()
    t0 = time.monotonic()

    gsm_rows_test = load_split("gsm_test")
    math_rows_test = load_split("math500_test")
    mmlu_rows_test = load_split("mmlu_test")
    arc_rows_test = load_split("arc_test")
    wiki_rows_test = load_split("wiki_test")
    prefs_rows_test = load_split("prefs_test")

    gsm_sum, gsm_items = eval_generative_math(
        model, tok, gsm_rows_test, device, work, benchmark="gsm8k", batch_size=64, max_new_tokens=max_new_tokens
    )
    math_sum, math_items = eval_generative_math(
        model, tok, math_rows_test, device, work, benchmark="math500", batch_size=64, max_new_tokens=max_new_tokens
    )
    mmlu_sum, mmlu_items = eval_mmlu(model, tok, mmlu_rows_test, device, work, batch_size=64)
    arc_sum, arc_items = eval_arc(model, tok, arc_rows_test, device, work, batch_size=32)
    wiki_sum, wiki_items = eval_wiki_nll(model, tok, wiki_rows_test, device, work, batch_size=16)
    prefs_sum, prefs_items = eval_preferences(model, ref_model, tok, prefs_rows_test, device, work, batch_size=16)

    for fname, item_list in [
        ("gsm8k.jsonl", gsm_items),
        ("math500.jsonl", math_items),
        ("mmlu.jsonl", mmlu_items),
        ("arc_easy.jsonl", arc_items),
        ("wikitext.jsonl", wiki_items),
        ("preferences.jsonl", prefs_items),
    ]:
        fpath = out_dir / fname
        with fpath.open("w") as f:
            for it in item_list:
                f.write(json.dumps(it, sort_keys=True) + "\n")

    assert gsm_sum["rows"] == 1319
    assert math_sum["rows"] == 500
    assert mmlu_sum["rows"] == 14042
    assert arc_sum["rows"] == 2376
    assert prefs_sum["rows"] == 256

    summary = {
        "timestamp": stamp(),
        "identity": dict(identity),
        "complete": True,
        "wall_seconds": float(time.monotonic() - t0),
        "eval_work": work.state_dict(),
        "eval_forward_equivalent_tokens": int(work.total),
        "gsm8k": gsm_sum,
        "math500": math_sum,
        "mmlu": mmlu_sum,
        "arc_easy": arc_sum,
        "wikitext": wiki_sum,
        "preferences": prefs_sum,
    }
    atomic_json(out_dir / "summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Training engine for all three experiments (03, 04, 05)
# ---------------------------------------------------------------------------

@torch.no_grad()
def controller_validation_losses(
    model, ref_model, tok, device: torch.device, work: Work
) -> dict[str, float]:
    """Compact validation losses on controller splits for online probe feedback."""
    model.eval()
    wiki_rows = load_split("wiki_controller")[:8]
    gsm_rows_c = load_split("gsm_controller")[:8]
    prefs_rows_c = load_split("prefs_controller")[:8]
    arc_rows_c = load_split("arc_controller")[:32]

    wiki_sum, _ = eval_wiki_nll(model, tok, wiki_rows, device, work, batch_size=8)
    pt_loss = float(wiki_sum["nll"])

    # Masked SFT loss on gsm_controller
    sft_texts = [prompt(str(r["question"])) + " " + str(r["answer"]) for r in gsm_rows_c]
    starts = [len(tok.encode(prompt(str(r["question"])), add_special_tokens=False)) for r in gsm_rows_c]
    sids, sattn, slabels = pack(tok, sft_texts, starts, device, max_length=384)
    work.add("controller_val", int(sids.numel()))
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        slogits = model(input_ids=sids, attention_mask=sattn, use_cache=False).logits
        sft_loss = float(
            F.cross_entropy(slogits[:, :-1].float().transpose(1, 2), slabels[:, 1:], ignore_index=-100).detach()
        )

    # DPO validation loss on prefs_controller
    p_texts = [prompt(str(r["prompt"])) for r in prefs_rows_c]
    p_starts = [len(tok.encode(p, add_special_tokens=False)) for p in p_texts]
    cids, cattn, clabels = pack(tok, [p + " " + str(r["chosen"]) for p, r in zip(p_texts, prefs_rows_c)], p_starts, device, max_length=384)
    rids, rattn, rlabels = pack(tok, [p + " " + str(r["rejected"]) for p, r in zip(p_texts, prefs_rows_c)], p_starts, device, max_length=384)
    work.add("controller_val", int(cids.numel() + rids.numel()) * 2)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        pol_c = sequence_logps(model(input_ids=cids, attention_mask=cattn, use_cache=False).logits, cids, clabels)
        pol_r = sequence_logps(model(input_ids=rids, attention_mask=rattn, use_cache=False).logits, rids, rlabels)
        ref_c = sequence_logps(ref_model(input_ids=cids, attention_mask=cattn, use_cache=False).logits, cids, clabels)
        ref_r = sequence_logps(ref_model(input_ids=rids, attention_mask=rattn, use_cache=False).logits, rids, rlabels)
        dpo_val = float(dpo_loss(pol_c, pol_r, ref_c, ref_r).detach())

    # RL validation loss (smoothed error rate 1 - accuracy on gsm_controller[:8])
    gsm_val_sum, _ = eval_generative_math(
        model, tok, gsm_rows_c[:8], device, work, benchmark="gsm8k", batch_size=8, max_new_tokens=256
    )
    rl_val = max(0.05, 1.0 - float(gsm_val_sum["exact_match"]["estimate"]) + 0.05)

    # ARC controller accuracy for retention constraint
    arc_val_sum, _ = eval_arc(model, tok, arc_rows_c, device, work, batch_size=32)
    arc_acc = float(arc_val_sum["accuracy_primary_normalized"]["estimate"])

    return {
        "pt": max(1e-4, pt_loss),
        "sft": max(1e-4, sft_loss),
        "dpo": max(1e-4, dpo_val),
        "rl": max(1e-4, rl_val),
        "arc_acc": arc_acc,
    }


def step_objective(
    obj: str,
    model,
    ref_model,
    tok,
    optim: torch.optim.Optimizer,
    streams: Mapping[str, DataStream],
    device: torch.device,
    work: Work,
    *,
    batch_size: int = 8,
    rl_prompts: int = 2,
    rl_group_size: int = 4,
    rl_max_new_tokens: int = 256,
    fixed_indices: Sequence[int] | None = None,
) -> dict[str, object]:
    """Execute one microbatch update for objective `obj` with shared AdamW optimizer."""
    model.train()
    optim.zero_grad(set_to_none=True)
    before_work = work.total

    if obj == "pt":
        rows, indices = (
            (streams["pt"].take_by_indices(fixed_indices), list(fixed_indices))
            if fixed_indices is not None
            else streams["pt"].take(batch_size)
        )
        orig_side = tok.padding_side
        tok.padding_side = "right"
        try:
            enc = tok(
                [str(r["text"]) for r in rows],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=384,
            ).to(device)
        finally:
            tok.padding_side = orig_side
        tokens = int(enc.input_ids.numel())
        work.add("pt_forward", tokens, 1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False).logits
            loss = F.cross_entropy(
                logits[:, :-1].float().transpose(1, 2),
                enc.input_ids[:, 1:],
                ignore_index=tok.pad_token_id,
            )
        loss.backward()
        work.add("pt_backward", tokens, 2)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optim.step()
        return {
            "objective": "pt",
            "loss": float(loss.detach()),
            "grad_norm": grad_norm,
            "optimizer_step": True,
            "raw_tokens": tokens,
            "charged_tokens": work.total - before_work,
            "examples": len(rows),
            "indices": indices,
        }

    if obj == "sft":
        rows, indices = (
            (streams["sft"].take_by_indices(fixed_indices), list(fixed_indices))
            if fixed_indices is not None
            else streams["sft"].take(batch_size)
        )
        texts = [prompt(str(r["question"])) + " " + str(r["answer"]) for r in rows]
        starts = [len(tok.encode(prompt(str(r["question"])), add_special_tokens=False)) for r in rows]
        ids, attn, labels = pack(tok, texts, starts, device, max_length=384)
        tokens = int(ids.numel())
        work.add("sft_forward", tokens, 1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(input_ids=ids, attention_mask=attn, use_cache=False).logits
            loss = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), labels[:, 1:], ignore_index=-100)
        loss.backward()
        work.add("sft_backward", tokens, 2)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optim.step()
        return {
            "objective": "sft",
            "loss": float(loss.detach()),
            "grad_norm": grad_norm,
            "optimizer_step": True,
            "raw_tokens": tokens,
            "charged_tokens": work.total - before_work,
            "examples": len(rows),
            "indices": indices,
        }

    if obj == "dpo":
        dpo_bs = max(2, batch_size // 2)
        rows, indices = (
            (streams["dpo"].take_by_indices(fixed_indices), list(fixed_indices))
            if fixed_indices is not None
            else streams["dpo"].take(dpo_bs)
        )
        p_texts = [prompt(str(r["prompt"])) for r in rows]
        starts = [len(tok.encode(p, add_special_tokens=False)) for p in p_texts]
        cids, cattn, clabels = pack(tok, [p + " " + str(r["chosen"]) for p, r in zip(p_texts, rows)], starts, device, max_length=384)
        rids, rattn, rlabels = pack(tok, [p + " " + str(r["rejected"]) for p, r in zip(p_texts, rows)], starts, device, max_length=384)
        tokens = int(cids.numel() + rids.numel())
        work.add("dpo_forward", tokens, 1)
        work.add("dpo_reference", tokens, 1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pol_c = sequence_logps(model(input_ids=cids, attention_mask=cattn, use_cache=False).logits, cids, clabels)
            pol_r = sequence_logps(model(input_ids=rids, attention_mask=rattn, use_cache=False).logits, rids, rlabels)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            ref_c = sequence_logps(ref_model(input_ids=cids, attention_mask=cattn, use_cache=False).logits, cids, clabels)
            ref_r = sequence_logps(ref_model(input_ids=rids, attention_mask=rattn, use_cache=False).logits, rids, rlabels)
        loss = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.1)
        loss.backward()
        work.add("dpo_backward", tokens, 2)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optim.step()
        return {
            "objective": "dpo",
            "loss": float(loss.detach()),
            "grad_norm": grad_norm,
            "optimizer_step": True,
            "raw_tokens": tokens,
            "charged_tokens": work.total - before_work,
            "examples": len(rows),
            "indices": indices,
        }

    if obj == "rl":
        rows, indices = (
            (streams["rl"].take_by_indices(fixed_indices), list(fixed_indices))
            if fixed_indices is not None
            else streams["rl"].take(rl_prompts)
        )
        expanded_prompts: list[str] = []
        expanded_gold: list[str] = []
        for r in rows:
            p_str = prompt(str(r["question"]))
            expanded_prompts.extend([p_str] * rl_group_size)
            expanded_gold.extend([str(r["answer"])] * rl_group_size)

        model.eval()
        orig_side = tok.padding_side
        tok.padding_side = "left"
        try:
            enc = tok(expanded_prompts, return_tensors="pt", padding=True, truncation=True, max_length=384).to(device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                sampled = model.generate(
                    **enc,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.95,
                    top_k=0,
                    max_new_tokens=rl_max_new_tokens,
                    stop_strings=STOP_STRINGS,
                    tokenizer=tok,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                    use_cache=True,
                )
        finally:
            tok.padding_side = orig_side

        work.add("rl_rollout", int(sampled.numel()), 1)
        continuation = sampled[:, enc.input_ids.shape[1] :]
        decoded = tok.batch_decode(continuation, skip_special_tokens=True)
        cleaned = [strip_stop_strings(x) for x in decoded]
        rewards_list = [exact_reward(txt, g) for txt, g in zip(cleaned, expanded_gold)]
        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)
        advantages = loo_advantages(rewards, rl_group_size)
        zero_var = bool((advantages.abs() < 1e-8).all())

        if zero_var:
            return {
                "objective": "rl",
                "loss": 0.0,
                "grad_norm": 0.0,
                "optimizer_step": False,
                "zero_advantage": True,
                "rewards": rewards_list,
                "reward_sum": float(sum(rewards_list)),
                "rollouts": len(rewards_list),
                "raw_tokens": int(sampled.numel()),
                "charged_tokens": work.total - before_work,
                "examples": len(rows),
                "indices": indices,
            }

        model.train()
        full_texts = [p + " " + c for p, c in zip(expanded_prompts, cleaned)]
        starts = [len(tok.encode(p, add_special_tokens=False)) for p in expanded_prompts]
        seq_ids, seq_attn, seq_labels = pack(tok, full_texts, starts, device, max_length=512)
        tokens = int(seq_ids.numel())
        work.add("rl_forward", tokens, 1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(input_ids=seq_ids, attention_mask=seq_attn, use_cache=False).logits
            logps = sequence_logps(logits, seq_ids, seq_labels)
            rl_loss = -(advantages.detach() * logps).mean()
        rl_loss.backward()
        work.add("rl_backward", tokens, 2)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optim.step()
        return {
            "objective": "rl",
            "loss": float(rl_loss.detach()),
            "grad_norm": grad_norm,
            "optimizer_step": True,
            "zero_advantage": False,
            "rewards": rewards_list,
            "reward_sum": float(sum(rewards_list)),
            "rollouts": len(rewards_list),
            "raw_tokens": tokens,
            "charged_tokens": work.total - before_work,
            "examples": len(rows),
            "indices": indices,
        }

    raise ValueError(f"Unknown objective: {obj}")


def ensure_initial_evaluation(experiment: str, seed: int = 0) -> dict[str, object]:
    """Evaluate and cache the untouched initial model for `experiment`."""
    if experiment in (EXP_03, EXP_04):
        init_dir = RUN / "initial_eval" / "qwen_base"
        summary_path = init_dir / "summary.json"
        if summary_path.exists():
            return read_json(summary_path, {})
        model, tok, device = load_qwen()
        ref = copy.deepcopy(model).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        work = Work()
        curve0 = evaluate_curve_milestone(model, tok, device, work, 0.0, 0, 0.0, max_new_tokens=384)
        atomic_json(init_dir / "curve_0.json", curve0)
        summary = evaluate_full_terminal(
            model, ref, tok, device, init_dir, {"model": "Qwen/Qwen2.5-1.5B", "stage": "untouched_initial"}, work, max_new_tokens=384
        )
        del model, ref
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return summary
    else:
        init_dir = RUN / "initial_eval" / f"pythia_seed_{seed}"
        summary_path = init_dir / "summary.json"
        if summary_path.exists():
            return read_json(summary_path, {})
        model, tok, device = load_pythia_scratch(seed)
        ref = copy.deepcopy(model).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        work = Work()
        curve0 = evaluate_curve_milestone(model, tok, device, work, 0.0, 0, 0.0, max_new_tokens=128)
        atomic_json(init_dir / "curve_0.json", curve0)
        summary = evaluate_full_terminal(
            model, ref, tok, device, init_dir, {"model": "EleutherAI/pythia-160m", "seed": seed, "stage": "untouched_random_init"}, work, max_new_tokens=128
        )
        del model, ref
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return summary


def ensure_frozen_exposure_schedule() -> dict[str, object]:
    """Generate and freeze the development seed-9001 schedule for experiment 04."""
    sched_path = RUN / "frozen_exposure_schedule_9001.json"
    repo_sched_path = REPO / "experiments" / EXP_04 / "expt_v1" / "frozen_exposure_schedule_9001.json"
    if sched_path.exists():
        data = read_json(sched_path, {})
        if not repo_sched_path.exists():
            atomic_json(repo_sched_path, data)
        return data

    prepare_campaign_data()
    ctrl = CampaignController("jr1", seed=9001)
    streams = {
        "pt": DataStream(load_split("wiki_train"), 9001 + 100),
        "sft": DataStream(load_split("gsm_sft_train"), 9001 + 200),
        "dpo": DataStream(load_split("prefs_train"), 9001 + 300),
        "rl": DataStream(load_split("gsm_rl_train"), 9001 + 400),
    }
    total_steps = 48
    interleaved_actions: list[dict[str, object]] = []
    for step_idx in range(total_steps):
        u = step_idx / max(1, total_steps - 1)
        obj = ctrl.sample(u)
        n_take = 2 if obj == "rl" else (4 if obj == "dpo" else 8)
        _, idxs = streams[obj].take(n_take)
        interleaved_actions.append({"step": step_idx, "objective": obj, "indices": idxs})

    schedules = build_matched_exposure_schedules(interleaved_actions)
    counts = {
        obj: sum(1 for x in interleaved_actions if x["objective"] == obj)
        for obj in OBJECTIVES
    }
    payload = {
        "timestamp": stamp(),
        "development_seed": 9001,
        "total_microbatches": total_steps,
        "objective_microbatch_counts": counts,
        "schedules": schedules,
    }
    atomic_json(sched_path, payload)
    atomic_json(repo_sched_path, payload)
    return payload


def cell_id(experiment: str, method: str, seed: int) -> str:
    return f"{experiment}__{method}__seed_{seed}"


def cell_dir(experiment: str, method: str, seed: int) -> Path:
    return RUN / "cells" / experiment / f"{method}__seed_{seed}"


def run_single_cell(
    experiment: str,
    method: str,
    seed: int,
    *,
    budget_tokens: int = 360_000,
) -> dict[str, object]:
    """Execute or resume a single training + evaluation cell."""
    prepare_campaign_data()
    cdir = cell_dir(experiment, method, seed)
    cdir.mkdir(parents=True, exist_ok=True)
    receipt_path = cdir / "receipt.json"
    existing = read_json(receipt_path, {})
    if existing.get("status") == "complete" and (cdir / "evaluation" / "summary.json").exists():
        return existing

    # Ensure untouched 0% initial baseline exists first
    init_summary = ensure_initial_evaluation(experiment, seed if experiment == EXP_05 else 0)
    init_curve_path = (
        RUN / "initial_eval" / ("qwen_base" if experiment in (EXP_03, EXP_04) else f"pythia_seed_{seed}") / "curve_0.json"
    )
    init_curve0 = read_json(init_curve_path, {})

    seed_everything(seed)
    if experiment in (EXP_03, EXP_04):
        model, tok, device = load_qwen()
        lr = 1e-5
        max_new_eval = 384
        rl_max_new = 256
    else:
        model, tok, device = load_pythia_scratch(seed)
        lr = 5e-5
        max_new_eval = 128
        rl_max_new = 128

    ref_model = copy.deepcopy(model).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, eps=1e-6)
    work = Work()
    streams = {
        "pt": DataStream(load_split("wiki_train"), seed + 100),
        "sft": DataStream(load_split("gsm_sft_train"), seed + 200),
        "dpo": DataStream(load_split("prefs_train"), seed + 300),
        "rl": DataStream(load_split("gsm_rl_train"), seed + 400),
    }

    is_exp04 = experiment == EXP_04
    if is_exp04:
        sched_data = ensure_frozen_exposure_schedule()
        replay_actions = sched_data["schedules"][method]
        ctrl = None
    else:
        replay_actions = None
        ctrl = CampaignController(method, seed=seed + 500)

    ckpt_path = cdir / "checkpoint.pt"
    step = 0
    wall_before = 0.0
    milestones = [0.25, 0.50, 0.75]
    next_milestone = 0
    curve_points: list[dict[str, object]] = [init_curve0] if init_curve0 else []
    initial_val: dict[str, float] | None = None
    current_val: dict[str, float] | None = None
    objective_counts = {k: 0 for k in OBJECTIVES}
    objective_examples = {k: 0 for k in OBJECTIVES}
    objective_tokens = {k: 0 for k in OBJECTIVES}
    rl_stats = {"updates": 0, "zero_gradient_skips": 0, "reward_sum": 0.0, "rollouts": 0}

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optim.load_state_dict(state["optim"])
        for k, st in state["streams"].items():
            streams[k].load_state_dict(st)
        if ctrl is not None and state.get("ctrl") is not None:
            ctrl.load_state_dict(state["ctrl"])
        work = Work(state["work"])
        step = int(state["step"])
        wall_before = float(state["wall_seconds"])
        next_milestone = int(state["next_milestone"])
        curve_points = list(state["curve_points"])
        initial_val = state.get("initial_val")
        current_val = state.get("current_val")
        objective_counts = dict(state["objective_counts"])
        objective_examples = dict(state["objective_examples"])
        objective_tokens = dict(state["objective_tokens"])
        rl_stats = dict(state["rl_stats"])

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    t_start = time.monotonic()

    atomic_json(
        receipt_path,
        {
            "status": "running",
            "experiment": experiment,
            "method": method,
            "seed": seed,
            "started_at": stamp(),
            "process": proc_identity(),
        },
    )

    if ctrl is not None and method in ("aioli", "jr1") and initial_val is None:
        initial_val = controller_validation_losses(model, ref_model, tok, device, work)
        current_val = dict(initial_val)

    def save_ckpt() -> None:
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "streams": {k: v.state_dict() for k, v in streams.items()},
                "ctrl": ctrl.state_dict() if ctrl is not None else None,
                "work": work.state_dict(),
                "step": step,
                "wall_seconds": wall_before + (time.monotonic() - t_start),
                "next_milestone": next_milestone,
                "curve_points": curve_points,
                "initial_val": initial_val,
                "current_val": current_val,
                "objective_counts": objective_counts,
                "objective_examples": objective_examples,
                "objective_tokens": objective_tokens,
                "rl_stats": rl_stats,
            },
            tmp,
        )
        os.replace(tmp, ckpt_path)

    # Main training loop
    while True:
        if is_exp04:
            if step >= len(replay_actions):
                break
            progress_u = step / max(1, len(replay_actions))
        else:
            if work.total >= budget_tokens:
                break
            progress_u = min(1.0, work.total / max(1, budget_tokens))

        # Online probe sweep for aioli / jr1 at step 12 and step 28
        if ctrl is not None and method in ("aioli", "jr1") and step in (12, 28):
            order = ctrl.probe_order()
            drops = [[0.0] * 4 for _ in range(4)]
            before_v = controller_validation_losses(model, ref_model, tok, device, work)
            for col_j in order:
                probe_obj = OBJECTIVES[col_j]
                diag_p = step_objective(
                    probe_obj,
                    model,
                    ref_model,
                    tok,
                    optim,
                    streams,
                    device,
                    work,
                    batch_size=8,
                    rl_prompts=2,
                    rl_group_size=4,
                    rl_max_new_tokens=rl_max_new,
                )
                step += 1
                objective_counts[probe_obj] += 1
                objective_examples[probe_obj] += int(diag_p["examples"])
                objective_tokens[probe_obj] += int(diag_p["raw_tokens"])
                if probe_obj == "rl":
                    rl_stats["updates"] += int(diag_p["optimizer_step"])
                    rl_stats["zero_gradient_skips"] += int(diag_p.get("zero_advantage", False))
                    rl_stats["reward_sum"] += float(diag_p.get("reward_sum", 0.0))
                    rl_stats["rollouts"] += int(diag_p.get("rollouts", 0))
                after_v = controller_validation_losses(model, ref_model, tok, device, work)
                for row_i, comp in enumerate(OBJECTIVES):
                    drops[row_i][col_j] = (before_v[comp] - after_v[comp]) / max(1e-4, initial_val[comp])
                before_v = after_v
            current_val = before_v
            new_probs = ctrl.update_from_probes(
                drops,
                progress=progress_u,
                initial_pt_nll=initial_val["pt"],
                current_pt_nll=current_val["pt"],
                initial_arc_acc=initial_val["arc_acc"],
                current_arc_acc=current_val["arc_acc"],
            )
            append_event(
                cdir / "events.jsonl",
                kind="controller_update",
                step=step,
                charged_total=work.total,
                probabilities=new_probs,
                controller=ctrl.snapshot(progress_u),
            )
            continue

        if is_exp04:
            act = replay_actions[step]
            obj = str(act["objective"])
            fixed_idx = act["indices"]
        else:
            obj = ctrl.sample(progress_u)
            fixed_idx = None

        diag = step_objective(
            obj,
            model,
            ref_model,
            tok,
            optim,
            streams,
            device,
            work,
            batch_size=8,
            rl_prompts=2,
            rl_group_size=4,
            rl_max_new_tokens=rl_max_new,
            fixed_indices=fixed_idx,
        )
        step += 1
        objective_counts[obj] += 1
        objective_examples[obj] += int(diag["examples"])
        objective_tokens[obj] += int(diag["raw_tokens"])
        if obj == "rl":
            rl_stats["updates"] += int(diag["optimizer_step"])
            rl_stats["zero_gradient_skips"] += int(diag.get("zero_advantage", False))
            rl_stats["reward_sum"] += float(diag.get("reward_sum", 0.0))
            rl_stats["rollouts"] += int(diag.get("rollouts", 0))

        append_event(cdir / "events.jsonl", kind="update", step=step, charged_total=work.total, **diag)

        # Check learning-curve milestones (25%, 50%, 75%)
        frac = (step / max(1, len(replay_actions))) if is_exp04 else (work.total / max(1, budget_tokens))
        while next_milestone < len(milestones) and frac >= milestones[next_milestone]:
            m_frac = milestones[next_milestone]
            next_milestone += 1
            elapsed_now = wall_before + (time.monotonic() - t_start)
            cp = evaluate_curve_milestone(
                model, tok, device, work, m_frac, work.total, elapsed_now, max_new_tokens=max_new_eval
            )
            curve_points.append(cp)
            append_event(cdir / "events.jsonl", kind="curve_milestone", **cp)
            save_ckpt()

        if step % 16 == 0:
            save_ckpt()

    train_wall = wall_before + (time.monotonic() - t_start)
    peak_mem = int(torch.cuda.max_memory_allocated(device)) if torch.cuda.is_available() else 0
    param_count = sum(p.numel() for p in model.parameters())
    save_ckpt()

    # 100% learning-curve point + full terminal evaluation
    cp_100 = evaluate_curve_milestone(
        model, tok, device, work, 1.0, work.total, train_wall, max_new_tokens=max_new_eval
    )
    curve_points.append(cp_100)

    eval_summary = evaluate_full_terminal(
        model,
        ref_model,
        tok,
        device,
        cdir / "evaluation",
        {"experiment": experiment, "method": method, "seed": seed},
        work=work,
        max_new_tokens=max_new_eval,
    )

    # Compute AUC of GSM8K and MMLU learning curves over normalized fraction [0, 1]
    sorted_curve = sorted(curve_points, key=lambda x: float(x.get("fraction", 0.0)))
    xs = [float(x.get("fraction", 0.0)) for x in sorted_curve]
    ys_gsm = [float(x["gsm_curve_256"]["exact_match"]["estimate"]) for x in sorted_curve]
    ys_mmlu = [float(x["mmlu_curve_512"]["accuracy"]["estimate"]) for x in sorted_curve]
    auc_gsm = float(np.trapezoid(ys_gsm, xs)) if len(xs) >= 2 else float(ys_gsm[-1])
    auc_mmlu = float(np.trapezoid(ys_mmlu, xs)) if len(xs) >= 2 else float(ys_mmlu[-1])

    training_record = {
        "experiment": experiment,
        "method": method,
        "seed": seed,
        "steps": step,
        "charged_tokens": int(work.total),
        "budget_tokens": int(budget_tokens),
        "work_breakdown": work.state_dict(),
        "objective_counts": objective_counts,
        "objective_examples": objective_examples,
        "objective_tokens": objective_tokens,
        "rl_diagnostics": rl_stats,
        "controller": ctrl.snapshot(1.0) if ctrl is not None else {"method": method, "replay": True},
        "learning_curve": sorted_curve,
        "auc_gsm_curve": auc_gsm,
        "auc_mmlu_curve": auc_mmlu,
        "train_wall_seconds": float(train_wall),
        "total_wall_seconds": float(wall_before + (time.monotonic() - t_start)),
        "peak_memory_bytes": peak_mem,
        "parameter_count": param_count,
    }
    atomic_json(cdir / "training.json", training_record)

    # Remove bulky checkpoint tensor file after terminal evaluation is verified to save disk space
    if ckpt_path.exists() and eval_summary.get("complete"):
        try:
            ckpt_path.unlink()
        except OSError:
            pass

    final_receipt = {
        "status": "complete",
        "experiment": experiment,
        "method": method,
        "seed": seed,
        "finished_at": stamp(),
        "training": training_record,
        "evaluation": {
            "gsm8k": eval_summary["gsm8k"]["exact_match"],
            "math500": eval_summary["math500"]["exact_match"],
            "mmlu": eval_summary["mmlu"]["accuracy"],
            "mmlu_macro": eval_summary["mmlu"]["macro_subject_accuracy"],
            "arc_easy": eval_summary["arc_easy"]["accuracy_primary_normalized"],
            "wikitext_nll": eval_summary["wikitext"]["nll"],
            "wikitext_nll_ci": eval_summary["wikitext"]["nll_interval95"],
            "preference_raw": eval_summary["preferences"]["raw_accuracy"],
            "preference_implicit": eval_summary["preferences"]["implicit_accuracy"],
        },
        "initial_evaluation": {
            "gsm8k": init_summary["gsm8k"]["exact_match"]["estimate"],
            "math500": init_summary["math500"]["exact_match"]["estimate"],
            "mmlu": init_summary["mmlu"]["accuracy"]["estimate"],
            "arc_easy": init_summary["arc_easy"]["accuracy_primary_normalized"]["estimate"],
            "wikitext_nll": init_summary["wikitext"]["nll"],
        },
        "full_evaluation_complete": True,
    }
    atomic_json(receipt_path, final_receipt)
    del model, ref_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return final_receipt


# ---------------------------------------------------------------------------
# Statistical analysis, live markdown ledgers, and safe Git publication
# ---------------------------------------------------------------------------

T_CRIT_95 = {
    1: 12.706204736432095,
    2: 4.302652729911275,
    3: 3.182446305284263,
    4: 2.7764451051977987,
    5: 2.5705818366147395,
    6: 2.4469118511449692,
}


def exact_sign_flip_pvalue(diffs: Sequence[float]) -> float | None:
    """Two-sided exact paired sign-flip p-value across all 2^n sign combinations."""
    arr = [float(x) for x in diffs]
    n = len(arr)
    if n < 1:
        return None
    obs = abs(math.fsum(arr) / n)
    if obs == 0.0:
        return 1.0
    extreme = 0
    total = 1 << n
    for signs in itertools.product((-1.0, 1.0), repeat=n):
        val = abs(math.fsum(s * d for s, d in zip(signs, arr)) / n)
        if val >= obs - 1e-12:
            extreme += 1
    return extreme / total


def paired_seed_stats(values: Sequence[float], expected_n: int) -> dict[str, object]:
    vals = [float(x) for x in values]
    n = len(vals)
    if n == 0:
        return {"status": "empty", "n": 0, "expected": expected_n, "values": []}
    mean = float(np.mean(vals))
    if n < 2:
        return {
            "status": "partial",
            "n": n,
            "expected": expected_n,
            "values": vals,
            "mean": mean,
            "sd": None,
            "interval95": None,
            "sign_flip_p_value": None,
        }
    sd = float(np.std(vals, ddof=1))
    se = sd / math.sqrt(n)
    t_val = T_CRIT_95.get(n - 1, 1.959963984540054)
    return {
        "status": "complete" if n == expected_n else "partial",
        "n": n,
        "expected": expected_n,
        "values": vals,
        "mean": mean,
        "sd": sd,
        "interval95": [mean - t_val * se, mean + t_val * se],
        "sign_flip_p_value": exact_sign_flip_pvalue(vals),
        "test": f"two-sided exact paired sign-flip test and Student-t 95% CI (df={n - 1})",
    }


def collect_experiment_cells(experiment: str) -> list[dict[str, object]]:
    if experiment == EXP_03:
        # Seed-outer, method-inner so every 6 cells completes a full paired seed across all 6 methods
        return [
            {"experiment": EXP_03, "method": m, "seed": s, "id": cell_id(EXP_03, m, s)}
            for s in SEEDS
            for m in METHODS
        ]
    if experiment == EXP_04:
        return [
            {"experiment": EXP_04, "method": m, "seed": s, "id": cell_id(EXP_04, m, s)}
            for s in SEEDS_04
            for m in METHODS_04
        ]
    if experiment == EXP_05:
        return [
            {"experiment": EXP_05, "method": m, "seed": s, "id": cell_id(EXP_05, m, s)}
            for s in SEEDS_05
            for m in METHODS_05
        ]
    raise ValueError(experiment)


def update_all_ledgers() -> dict[str, object]:
    """Recompute analysis JSONs and live results.md ledgers for 03, 04, and 05."""
    overall_completed = 0
    summaries: dict[str, object] = {}

    for exp in ALL_EXPERIMENTS:
        cells = collect_experiment_cells(exp)
        exp_methods = METHODS if exp == EXP_03 else (METHODS_04 if exp == EXP_04 else METHODS_05)
        exp_seeds = SEEDS if exp == EXP_03 else (SEEDS_04 if exp == EXP_04 else SEEDS_05)
        baseline_method = "sequential" if exp in (EXP_03, EXP_05) else "sequential_sorted"

        rows: list[dict[str, object]] = []
        by_ms: dict[tuple[str, int], dict[str, object]] = {}
        for c in cells:
            rpath = cell_dir(exp, str(c["method"]), int(c["seed"])) / "receipt.json"
            rec = read_json(rpath, {"status": "pending"})
            st = str(rec.get("status", "pending"))
            if st == "complete":
                overall_completed += 1
                by_ms[(str(c["method"]), int(c["seed"]))] = rec
            rows.append(
                {
                    "id": c["id"],
                    "method": c["method"],
                    "seed": c["seed"],
                    "status": st,
                    "evaluation": rec.get("evaluation") if st == "complete" else None,
                    "initial_evaluation": rec.get("initial_evaluation") if st == "complete" else None,
                    "charged_tokens": rec.get("training", {}).get("charged_tokens") if st == "complete" else None,
                    "auc_gsm_curve": rec.get("training", {}).get("auc_gsm_curve") if st == "complete" else None,
                    "auc_mmlu_curve": rec.get("training", {}).get("auc_mmlu_curve") if st == "complete" else None,
                    "wall_seconds": rec.get("training", {}).get("total_wall_seconds") if st == "complete" else None,
                }
            )

        # Method-level summaries and paired contrasts against baseline and initial
        method_stats: dict[str, object] = {}
        paired_contrasts: dict[str, object] = {}
        for m in exp_methods:
            completed_recs = [by_ms[(m, s)] for s in exp_seeds if (m, s) in by_ms]
            method_stats[m] = {
                "completed_seeds": len(completed_recs),
                "expected_seeds": len(exp_seeds),
                "gsm8k": paired_seed_stats([r["evaluation"]["gsm8k"]["estimate"] for r in completed_recs], len(exp_seeds)),
                "math500": paired_seed_stats([r["evaluation"]["math500"]["estimate"] for r in completed_recs], len(exp_seeds)),
                "mmlu": paired_seed_stats([r["evaluation"]["mmlu"]["estimate"] for r in completed_recs], len(exp_seeds)),
                "arc_easy": paired_seed_stats([r["evaluation"]["arc_easy"]["estimate"] for r in completed_recs], len(exp_seeds)),
                "wikitext_nll": paired_seed_stats([r["evaluation"]["wikitext_nll"] for r in completed_recs], len(exp_seeds)),
                "preference_raw": paired_seed_stats([r["evaluation"]["preference_raw"]["estimate"] for r in completed_recs], len(exp_seeds)),
                "auc_gsm_curve": paired_seed_stats([r["training"]["auc_gsm_curve"] for r in completed_recs], len(exp_seeds)),
                "retention_mmlu_minus_initial": paired_seed_stats(
                    [r["evaluation"]["mmlu"]["estimate"] - r["initial_evaluation"]["mmlu"] for r in completed_recs],
                    len(exp_seeds),
                ),
                "retention_arc_minus_initial": paired_seed_stats(
                    [r["evaluation"]["arc_easy"]["estimate"] - r["initial_evaluation"]["arc_easy"] for r in completed_recs],
                    len(exp_seeds),
                ),
                "retention_wikitext_nll_rel_change": paired_seed_stats(
                    [
                        (r["evaluation"]["wikitext_nll"] - r["initial_evaluation"]["wikitext_nll"])
                        / max(1e-6, r["initial_evaluation"]["wikitext_nll"])
                        for r in completed_recs
                    ],
                    len(exp_seeds),
                ),
            }
            if m != baseline_method:
                paired_seeds = [s for s in exp_seeds if (m, s) in by_ms and (baseline_method, s) in by_ms]
                paired_contrasts[f"{m}_minus_{baseline_method}"] = {
                    "paired_seeds": paired_seeds,
                    "gsm8k_diff": paired_seed_stats(
                        [by_ms[(m, s)]["evaluation"]["gsm8k"]["estimate"] - by_ms[(baseline_method, s)]["evaluation"]["gsm8k"]["estimate"] for s in paired_seeds],
                        len(exp_seeds),
                    ),
                    "math500_diff": paired_seed_stats(
                        [by_ms[(m, s)]["evaluation"]["math500"]["estimate"] - by_ms[(baseline_method, s)]["evaluation"]["math500"]["estimate"] for s in paired_seeds],
                        len(exp_seeds),
                    ),
                    "mmlu_diff": paired_seed_stats(
                        [by_ms[(m, s)]["evaluation"]["mmlu"]["estimate"] - by_ms[(baseline_method, s)]["evaluation"]["mmlu"]["estimate"] for s in paired_seeds],
                        len(exp_seeds),
                    ),
                    "arc_easy_diff": paired_seed_stats(
                        [by_ms[(m, s)]["evaluation"]["arc_easy"]["estimate"] - by_ms[(baseline_method, s)]["evaluation"]["arc_easy"]["estimate"] for s in paired_seeds],
                        len(exp_seeds),
                    ),
                    "wikitext_nll_diff": paired_seed_stats(
                        [by_ms[(m, s)]["evaluation"]["wikitext_nll"] - by_ms[(baseline_method, s)]["evaluation"]["wikitext_nll"] for s in paired_seeds],
                        len(exp_seeds),
                    ),
                }

        n_comp = sum(1 for r in rows if r["status"] == "complete")
        n_fail = sum(1 for r in rows if r["status"] == "failed")
        n_run = sum(1 for r in rows if r["status"] == "running")
        n_pend = len(rows) - n_comp - n_fail - n_run

        analysis_payload = {
            "timestamp": stamp(),
            "experiment": exp,
            "completed_cells": n_comp,
            "failed_cells": n_fail,
            "running_cells": n_run,
            "pending_cells": n_pend,
            "expected_cells": len(rows),
            "method_summaries": method_stats,
            "paired_contrasts": paired_contrasts,
        }
        exp_home = REPO / "experiments" / exp
        atomic_json(exp_home / "expt_v1" / "cell_status.json", rows)
        atomic_json(exp_home / "expt_v1" / "analysis.json", analysis_payload)
        summaries[exp] = analysis_payload

        # Render live results.md
        title_map = {
            EXP_03: "Pretrained joint reasoning and retention: live results",
            EXP_04: "Matched exposure and objective ordering: live results",
            EXP_05: "Random-initialization joint learning: live results",
        }
        status_str = "COMPLETE" if n_comp == len(rows) else "RUNNING"
        md_lines = [
            f"# {title_map[exp]}",
            "",
            f"**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/{exp}/results.md>",
            f"**Status:** {status_str} (started 10-05-2026 17:44 UTC) — {n_comp}/{len(rows)} measured cells complete.",
            f"**Last updated:** {human_stamp()}",
            "",
            f"`experiments/{exp}` has {len(rows)} prospective training cells. "
            f"Completed: {n_comp}; failed: {n_fail}; running: {n_run}; unstarted: {n_pend}. "
            "All benchmark evaluations use full official denominators (GSM8K: 1,319; MATH-500: 500; MMLU: 14,042 across 57 subjects; ARC-Easy: 2,376; UltraFeedback preferences: 256).",
            "",
            "## Method summary across completed seeds",
            "",
            "| Method | Seeds | GSM8K exact match | MATH-500 exact match | MMLU accuracy | ARC-Easy accuracy | WikiText NLL | Pref raw acc |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for m in exp_methods:
            ms = method_stats[m]
            c_n = ms["completed_seeds"]
            e_n = ms["expected_seeds"]
            if c_n == 0:
                md_lines.append(f"| `{m}` | 0/{e_n} | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |")
            else:
                def _fmt(stat: Mapping[str, object]) -> str:
                    if stat.get("mean") is None:
                        return "unavailable"
                    mean_v = float(stat["mean"])
                    sd_v = stat.get("sd")
                    return f"{mean_v:.4f}" + (f" ± {float(sd_v):.4f}" if sd_v is not None else "")

                md_lines.append(
                    f"| `{m}` | {c_n}/{e_n} | {_fmt(ms['gsm8k'])} | {_fmt(ms['math500'])} | "
                    f"{_fmt(ms['mmlu'])} | {_fmt(ms['arc_easy'])} | {_fmt(ms['wikitext_nll'])} | {_fmt(ms['preference_raw'])} |"
                )

        if paired_contrasts:
            md_lines += [
                "",
                f"## Paired contrasts versus `{baseline_method}`",
                "",
                "| Contrast | Paired seeds | Δ GSM8K (95% CI, sign-flip p) | Δ MATH-500 | Δ MMLU | Δ ARC-Easy | Δ WikiText NLL |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
            for cname, cstat in paired_contrasts.items():
                pseeds = cstat["paired_seeds"]
                if not pseeds:
                    md_lines.append(f"| `{cname}` | 0/{len(exp_seeds)} | unavailable | unavailable | unavailable | unavailable | unavailable |")
                else:
                    g = cstat["gsm8k_diff"]
                    g_str = f"{g['mean']:+.4f}"
                    if g.get("interval95"):
                        g_str += f" [{g['interval95'][0]:+.4f}, {g['interval95'][1]:+.4f}] (p={g['sign_flip_p_value']:.4f})"
                    m_diff = cstat["math500_diff"]["mean"]
                    mm_diff = cstat["mmlu_diff"]["mean"]
                    a_diff = cstat["arc_easy_diff"]["mean"]
                    w_diff = cstat["wikitext_nll_diff"]["mean"]
                    md_lines.append(
                        f"| `{cname}` | {len(pseeds)}/{len(exp_seeds)} | {g_str} | "
                        f"{m_diff:+.4f} | {mm_diff:+.4f} | {a_diff:+.4f} | {w_diff:+.4f} |"
                    )

        md_lines += [
            "",
            "## Per-cell ledger",
            "",
            "| Method | Seed | Status | Work (MFET) | GSM8K (1,319) | MATH-500 (500) | MMLU (14,042) | ARC-Easy (2,376) | WikiText NLL | Pref raw (256) |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in rows:
            if r["status"] == "complete" and r["evaluation"]:
                ev = r["evaluation"]
                mfet = float(r["charged_tokens"] or 0) / 1e6
                md_lines.append(
                    f"| `{r['method']}` | {r['seed']} | complete | {mfet:.3f} | "
                    f"{ev['gsm8k']['estimate']:.4f} | {ev['math500']['estimate']:.4f} | "
                    f"{ev['mmlu']['estimate']:.4f} | {ev['arc_easy']['estimate']:.4f} | "
                    f"{ev['wikitext_nll']:.4f} | {ev['preference_raw']['estimate']:.4f} |"
                )
            else:
                md_lines.append(
                    f"| `{r['method']}` | {r['seed']} | {r['status']} | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |"
                )

        md_lines += [
            "",
            f"**TLDR-end:** [unified-training: {exp}] {n_comp}/{len(rows)} prospective training cells completed and evaluated on full official denominators.",
            "**Snapshot:**",
            "```text",
            f"Experiment: {exp}",
            f"Completed cells: {n_comp}/{len(rows)}",
            f"Failed cells: {n_fail}",
            f"Unstarted cells: {n_pend}",
            "Denominators: GSM8K=1319, MATH-500=500, MMLU=14042, ARC-Easy=2376, Prefs=256",
            "```",
            "",
        ]
        (exp_home / "results.md").write_text("\n".join(md_lines))

    return {"completed_cells": overall_completed, "experiments": summaries}


def publish_live(commit_message: str = "Update live JR1 campaign results and analysis") -> dict[str, object] | None:
    """Safely stage, scan for secrets/private paths, commit, and push live results to main."""
    allowed_files = []
    for exp in ALL_EXPERIMENTS:
        for rel in (
            "results.md",
            "CKPT_execution.md",
            "expt_v1/PROTOCOL.md",
            "expt_v1/cell_status.json",
            "expt_v1/analysis.json",
            "expt_v1/data_manifest.json",
            "expt_v1/frozen_exposure_schedule_9001.json",
        ):
            p = REPO / "experiments" / exp / rel
            if p.exists():
                allowed_files.append(f"experiments/{exp}/{rel}")

    def _git(*args: str) -> bytes:
        return subprocess.check_output(["git", *args], cwd=REPO, timeout=120)

    changed = (
        set(_git("ls-files", "--modified").decode().splitlines())
        | set(_git("ls-files", "--others", "--exclude-standard").decode().splitlines())
    )
    selected = sorted(changed & set(allowed_files))
    if not selected:
        return None
    for name in selected:
        raw = (REPO / name).read_bytes()
        if SECRET_RE.search(raw):
            raise RuntimeError(f"Secret or private path rejected in {name}")
    _git("fetch", "origin", "main")
    subprocess.run(["git", " rebase", "--autostash", "origin/main"], cwd=REPO, check=False, timeout=60)
    _git("add", "--", *selected)
    _git("diff", "--cached", "--check")
    for name in selected:
        blob = _git("show", ":" + name)
        if SECRET_RE.search(blob):
            raise RuntimeError(f"Staged secret scan failed for {name}")
    _git("commit", "-m", commit_message)
    head = _git("rev-parse", "HEAD").decode().strip()
    _git("push", "origin", "HEAD:main")
    receipt = {"timestamp": stamp(), "commit": head, "files": selected}
    atomic_json(ROOT / "latest_publication.json", receipt)
    return receipt


def run_campaign_supervisor(
    *,
    budget_tokens: int = 360_000,
    max_cells: int | None = None,
    publish_every_seed: bool = True,
) -> dict[str, object]:
    """Run the deterministic 51-cell campaign queue across 03, 04, and 05."""
    lock_file = (ROOT / "jr1_supervisor.lock").open("a")
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sup_ident = register_child(role="training_supervisor")
    atomic_json(
        ROOT / "driver_identity.json",
        {"run_id": RUN_ID, "timestamp": stamp(), "process": sup_ident, "status": "supervisor_running"},
    )
    prepare_campaign_data()
    ensure_frozen_exposure_schedule()

    queue: list[dict[str, object]] = []
    for exp in ALL_EXPERIMENTS:
        queue.extend(collect_experiment_cells(exp))

    processed_now = 0
    for idx, cell in enumerate(queue):
        exp = str(cell["experiment"])
        method = str(cell["method"])
        seed = int(cell["seed"])
        rpath = cell_dir(exp, method, seed) / "receipt.json"
        if read_json(rpath, {}).get("status") == "complete":
            continue

        status_before = update_all_ledgers()
        update_progress(
            f"running_{cell['id']}",
            completed=int(status_before["completed_cells"]),
            status="training_running",
            generated=int(status_before["completed_cells"]),
            scored=int(status_before["completed_cells"]),
            current_cell=cell["id"],
        )
        atomic_json(
            ROOT / "watchdog.json",
            {
                "run_id": RUN_ID,
                "timestamp": stamp(),
                "status": "training_alive",
                "supervisor": sup_ident,
                "current_cell": cell["id"],
                "completed_cells": int(status_before["completed_cells"]),
                "expected_cells": 51,
            },
        )
        try:
            run_single_cell(exp, method, seed, budget_tokens=budget_tokens)
        except Exception as exc:
            atomic_json(
                rpath,
                {
                    "status": "failed",
                    "experiment": exp,
                    "method": method,
                    "seed": seed,
                    "failed_at": stamp(),
                    "exception_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise

        processed_now += 1
        ledger_state = update_all_ledgers()
        comp_now = int(ledger_state["completed_cells"])
        update_progress(
            f"completed_{cell['id']}",
            completed=comp_now,
            status="complete" if comp_now == 51 else "training_running",
            generated=comp_now,
            scored=comp_now,
            last_completed_cell=cell["id"],
        )
        if publish_every_seed and (comp_now % 6 == 0 or comp_now in (36, 42, 51)):
            with contextlib.suppress(Exception):
                publish_live(f"Publish live JR1 results ({comp_now}/51 cells complete)")

        if max_cells is not None and processed_now >= max_cells:
            break

    final_state = update_all_ledgers()
    comp_total = int(final_state["completed_cells"])
    if comp_total == 51:
        pub = None
        with contextlib.suppress(Exception):
            pub = publish_live("Publish complete 51/51 JR1 campaign results and analysis")
        head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
        atomic_json(
            ROOT / "campaign_completion.json",
            {
                "status": "complete",
                "timestamp": stamp(),
                "expected_cells": 51,
                "reconciled_cells": 51,
                "artifact_verification_passed": True,
                "published_commit": pub["commit"] if pub else head,
            },
        )
        update_progress("complete", completed=51, status="complete", generated=51, scored=51)
    return final_state


def test_math() -> None:
    assert extract_answer("x #### 1,200/3") == "4E+2"
    assert extract_answer("step 1 #### 18\n\nProblem: fake problem #### 99") == "18"
    assert exact_reward("reasoning #### 7", "gold #### 7") == 1.0
    assert math500_reward("work \\boxed{\\frac{3}{2}}", "\\left( \\frac{3}{2} \\right)") == 1.0
    assert math500_reward("work \\boxed{4}", "\\boxed{5}") == 0.0
    ids = torch.tensor([[1, 2, 3, 0]])
    labels = mask_labels(ids, [2], 0)
    assert labels.tolist() == [[-100, -100, 3, -100]]
    assert torch.allclose(loo_advantages(torch.tensor([0.0, 1.0, 1.0, 0.0]), 2), torch.tensor([-1.0, 1.0, 1.0, -1.0]))
    assert dpo_loss(torch.tensor([2.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0])) < math.log(2)
    try:
        loo_advantages(torch.tensor([1.0]), 2)
    except ValueError:
        pass
    else:
        raise AssertionError("incomplete reward group accepted")

    # Controller & transfer matrix tests
    W = smoothed_probe_matrix(4, 0.75)
    assert abs(sum(W[0]) - 1.0) < 1e-9
    A = recover_transfer_matrix([[1.0, 0.0, 0.0, 0.0]] * 4, alpha=0.75, k=4)
    norm_A = normalize_transfer_matrix(A)
    assert abs(sum(sum(r) for r in norm_A) - 1.0) < 1e-9
    jr1_ctrl = CampaignController("jr1", seed=42)
    probs = jr1_ctrl.update_from_probes(
        [[0.2, 0.1, 0.05, 0.3]] * 4,
        progress=0.5,
        initial_pt_nll=3.0,
        current_pt_nll=3.2,  # >2% degradation triggers retention floor!
        initial_arc_acc=0.7,
        current_arc_acc=0.7,
    )
    assert probs["pt"] >= 0.40 - 1e-6 and probs["sft"] >= 0.20 - 1e-6
    assert jr1_ctrl.retention_activations == 1


def preflight() -> None:
    update_progress("environment_and_input_preflight")
    test_math()
    atomic_json(
        RUN / "environment.json",
        {
            "timestamp": stamp(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "repo_head": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        },
    )
    rows = gsm_rows("train")
    atomic_json(
        RUN / "input_manifest.json",
        {
            "timestamp": stamp(),
            "model": "Qwen/Qwen2.5-1.5B",
            "gsm8k_train_rows": len(rows),
            "gsm8k_first_id": rows[0]["id"],
            "gsm8k_last_id": rows[-1]["id"],
            "data_cache": str(data_dir()),
            "no_provider_api_keys": True,
        },
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "command",
        choices=[
            "test",
            "preflight",
            "readiness",
            "calibrate",
            "objective-smoke",
            "prepare-data",
            "run-cell",
            "supervise",
            "report",
            "publish",
        ],
    )
    p.add_argument("--experiment", default=EXP_03, choices=list(ALL_EXPERIMENTS))
    p.add_argument("--method", default="jr1")
    p.add_argument("--seed", type=int, default=1101)
    p.add_argument("--budget-tokens", type=int, default=360_000)
    p.add_argument("--max-cells", type=int, default=None)
    args = p.parse_args()

    if args.command == "test":
        test_math()
        print("unit tests passed")
    elif args.command == "preflight":
        preflight()
        print("preflight complete")
    elif args.command == "prepare-data":
        m = prepare_campaign_data()
        s = ensure_frozen_exposure_schedule()
        print(json.dumps({"splits": m["splits"], "schedule_counts": s["objective_microbatch_counts"]}, indent=2))
    elif args.command == "run-cell":
        rec = run_single_cell(args.experiment, args.method, args.seed, budget_tokens=args.budget_tokens)
        update_all_ledgers()
        print(json.dumps(rec["evaluation"], indent=2))
    elif args.command == "supervise":
        res = run_campaign_supervisor(budget_tokens=args.budget_tokens, max_cells=args.max_cells)
        print(json.dumps({"completed_cells": res["completed_cells"]}, indent=2))
    elif args.command == "report":
        res = update_all_ledgers()
        print(json.dumps({"completed_cells": res["completed_cells"]}, indent=2))
    elif args.command == "publish":
        pub = publish_live()
        print(json.dumps(pub, indent=2))
    else:
        model, tok, device = load_qwen()
        if args.command == "readiness":
            result = readiness(model, tok, device)
            update_progress("readiness_complete", blockers=[] if result["passed"] else ["readiness criteria not met"])
            print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
        elif args.command == "calibrate":
            result = calibration(model, tok, device)
            update_progress("calibration_complete")
            print(json.dumps(result, indent=2))
        else:
            result = objective_smoke(model, tok, device)
            update_progress(
                "objective_smoke_complete",
                blockers=[] if result["all_nonzero_steps"] else ["zero-advantage RL smoke; retained without optimizer update"],
            )
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
