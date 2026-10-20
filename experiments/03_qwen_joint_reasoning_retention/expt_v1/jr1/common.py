"""Shared JR1 utilities: paths, atomic files, seeds and RNG state, the Work ledger and objective math.

Owned by the trainer branch (ut/jr1-trainer); data.py, evaluate.py and sherlock/ import it.

Provenance (attributed copies from repo commit 047c79d; the frozen originals are never edited):
- extract_answer, exact_reward, prompt, mask_labels, dpo_loss, loo_advantages, atomic_json, sha256:
  experiments/03_qwen_joint_reasoning_retention/expt_v1/jr1_campaign.py (unchanged semantics).
- Work ledger, padded-batch construction and the generation forward-count hook:
  experiments/00_program/objectives.py (the ledger now keeps a purpose and frozen multipliers).
- rng_state / restore_rng: experiments/00_program/training.py.
Changes: sequence_logps uses cross entropy instead of a full log-softmax tensor (same values, less memory),
and labels are built from explicit lengths because Qwen's pad token equals its EOS token.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import os
import random
import re
import shutil
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

OBJECTIVES = ("pt", "sft", "dpo", "rl")
METHODS = ("sequential", "replay", "fixed_joint", "smooth_joint", "aioli", "jr1")
SEEDS_03 = (1101, 1102, 1103, 1104, 1105, 1106)
DEV_SEED = 9001
NUMBER = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:/\d+)?")

# Forward-equivalent token multipliers, frozen for every method. One token through one forward pass of the
# policy costs 1. A backward pass costs 2 forwards (the standard 2x estimate). Generation charges every
# token the model processes: the prompt prefill plus one token per row per decoding step (KV cache), padding
# and already-finished rows included, because the device computes them. These are token-operation
# estimates, not measured floating-point operations; device and wall time are reported separately.
MULTIPLIERS = {"forward": 1, "backward": 2, "reference": 1, "generation": 1}
PURPOSES = ("train", "probe", "validation")


def state_root() -> Path:
    if "JR1_STATE_ROOT" not in os.environ:
        raise RuntimeError("JR1_STATE_ROOT must name the private campaign state directory")
    return Path(os.environ["JR1_STATE_ROOT"])


def stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w") as out:
        json.dump(value, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)
    fsync_directory(path.parent)


def read_json(path: Path, default: object = None) -> object:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def read_jsonl(path: Path) -> list[dict]:
    # JSONL is delimited by physical LF only; str.splitlines() would also split on U+2028 inside strings.
    with Path(path).open() as src:
        return [json.loads(line) for line in src.read().split("\n") if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    with Path(path).open("a") as out:
        out.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        out.flush()
        os.fsync(out.fileno())


def truncate_jsonl(path: Path, lines: int) -> None:
    """Keep the first `lines` records (used on resume so replayed updates are not logged twice)."""
    path = Path(path)
    if not path.exists():
        if lines:
            raise RuntimeError(f"{path} is missing but the saved state expects {lines} lines")
        return
    rows = path.read_text().split("\n")
    rows = [r for r in rows if r.strip()]
    if len(rows) < lines:
        raise RuntimeError(f"{path} has {len(rows)} lines, saved state expects {lines}")
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text("".join(r + "\n" for r in rows[:lines]))
    os.replace(tmp, path)


def atomic_directory(final: Path, write) -> None:
    """Write a directory through a temporary sibling, then swap it in; a crash leaves the old or the new copy."""
    final = Path(final)
    tmp = final.with_name(final.name + f".tmp-{os.getpid()}")
    old = final.with_name(final.name + ".old")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    write(tmp)
    for p in tmp.rglob("*"):
        if p.is_file():
            with p.open("rb") as f:
                os.fsync(f.fileno())
    fsync_directory(tmp)
    if final.exists():
        shutil.rmtree(old, ignore_errors=True)
        os.replace(final, old)
    os.replace(tmp, final)
    fsync_directory(final.parent)
    shutil.rmtree(old, ignore_errors=True)


def recover_directory(final: Path) -> Path | None:
    """The directory to resume from: final, or its .old copy if a crash hit between the two renames."""
    final = Path(final)
    if final.exists():
        return final
    old = final.with_name(final.name + ".old")
    return old if old.exists() else None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def extract_answer(text: str) -> str | None:
    tail = text.rsplit("####", 1)[-1] if "####" in text else text
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


def prompt(question: str) -> str:
    return ("Solve the problem. Show your reasoning, then put the final numeric answer after ####.\n\nProblem: "
            + question + "\nAnswer:")


def mask_labels(ids: torch.Tensor, prompt_lengths: list[int], pad: int) -> torch.Tensor:
    labels = ids.clone()
    for row, n in enumerate(prompt_lengths):
        labels[row, :n] = -100
    labels[ids == pad] = -100
    return labels


def pack(sequences: list[list[int]], starts: list[int], pad: int, device, left: bool = False):
    """Pad token-id lists; labels keep positions [start, len) and mask the prompt and padding by length."""
    n = max(map(len, sequences))
    ids = torch.full((len(sequences), n), pad, dtype=torch.long)
    attention = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for i, (seq, start) in enumerate(zip(sequences, starts)):
        offset = n - len(seq) if left else 0
        ids[i, offset:offset + len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention[i, offset:offset + len(seq)] = 1
        labels[i, offset + start:offset + len(seq)] = ids[i, offset + start:offset + len(seq)]
    return ids.to(device), attention.to(device), labels.to(device)


def token_logps(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Shifted per-token log probabilities of labels and their validity mask, shape (B, T-1)."""
    shifted = labels[:, 1:]
    valid = shifted.ne(-100)
    safe = shifted.masked_fill(~valid, 0)
    lp = -F.cross_entropy(logits[:, :-1].float().transpose(1, 2), safe, reduction="none")
    return lp.masked_fill(~valid, 0.0), valid


def sequence_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    lp, _ = token_logps(logits, labels)
    return lp.sum(-1)


def dpo_loss(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor,
             ref_chosen: torch.Tensor, ref_rejected: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    return -F.logsigmoid(beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))).mean()


def loo_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    """Leave-one-out baseline: A_i = r_i - mean_{j != i} r_j within each prompt's group."""
    if group_size < 2 or rewards.numel() % group_size:
        raise ValueError("rewards must form complete independent groups")
    groups = rewards.reshape(-1, group_size)
    return (groups - (groups.sum(1, keepdim=True) - groups) / (group_size - 1)).reshape(-1)


class Work:
    """Forward-equivalent token ledger keyed by purpose (train, probe, validation) and kind."""

    def __init__(self, charged: dict | None = None, raw: dict | None = None):
        self.charged = dict(charged or {})
        self.raw = dict(raw or {})

    def add(self, purpose: str, kind: str, tokens: int) -> int:
        if purpose not in PURPOSES or kind not in MULTIPLIERS:
            raise ValueError(f"unknown work entry {purpose}/{kind}")
        tokens = int(tokens)
        if tokens < 0:
            raise ValueError("negative work")
        key = f"{purpose}/{kind}"
        self.raw[key] = self.raw.get(key, 0) + tokens
        self.charged[key] = self.charged.get(key, 0) + tokens * MULTIPLIERS[kind]
        return tokens * MULTIPLIERS[kind]

    @property
    def total(self) -> int:
        return sum(self.charged.values())

    def by_purpose(self) -> dict:
        out = dict.fromkeys(PURPOSES, 0)
        for key, value in self.charged.items():
            out[key.split("/")[0]] += value
        return out

    def state_dict(self) -> dict:
        return {"charged": dict(self.charged), "raw": dict(self.raw)}

    @classmethod
    def from_state(cls, state: dict) -> "Work":
        return cls(state["charged"], state["raw"])


def autocast(device, compute: str):
    """Autocast context for the configured compute dtype; fp32 means no autocast."""
    if compute == "fp32":
        return contextlib.nullcontext()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[compute]
    return torch.autocast(device_type=torch.device(device).type, dtype=dtype)
