#!/usr/bin/env python3
"""Deterministic, restartable executor for the 10-05-2026 JR1 campaign.

This module deliberately keeps all mutable artifacts outside the Git worktree.
It does not call model-provider APIs: public model/data retrieval uses the
Hugging Face libraries and all training/evaluation is local PyTorch work.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

if "JR1_STATE_ROOT" not in os.environ:
    raise RuntimeError("JR1_STATE_ROOT must name the private campaign state directory")
ROOT = Path(os.environ["JR1_STATE_ROOT"])
REPO = Path(__file__).resolve().parents[3]
RUN = ROOT / "campaign"
RUN.mkdir(parents=True, exist_ok=True)
RUN_ID = ROOT.name
SEEDS = [1101, 1102, 1103, 1104, 1105, 1106]
METHODS = ["sequential", "replay", "fixed_joint", "smooth_joint", "aioli", "jr1"]
NUMBER = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:/\d+)?")


def stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w") as out:
        json.dump(value, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)


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


def update_progress(phase: str, *, completed: int = 0, blockers: list[str] | None = None,
                    status: str = "in_progress", generated: int = 0, scored: int = 0) -> None:
    atomic_json(ROOT / "progress.json", {
        "run_id": RUN_ID, "timestamp": stamp(), "phase": phase,
        "completed_cells": completed, "expected_cells": 51,
        "generated": generated, "scored": scored, "planned": 51,
        "blockers": blockers or [], "coordinator_status": status,
    })


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


def dpo_loss(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor,
             ref_chosen: torch.Tensor, ref_rejected: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    return -F.logsigmoid(beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))).mean()


def loo_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    if group_size < 2 or rewards.numel() % group_size:
        raise ValueError("rewards must form complete independent groups")
    groups = rewards.reshape(-1, group_size)
    return (groups - (groups.sum(1, keepdim=True) - groups) / (group_size - 1)).reshape(-1)


@dataclass
class Work:
    counts: dict[str, int]
    def add(self, kind: str, tokens: int) -> None:
        self.counts[kind] = self.counts.get(kind, 0) + int(tokens)
    @property
    def total(self) -> int:
        return sum(self.counts.values())


def data_dir() -> Path:
    p = RUN / "hf"
    p.mkdir(parents=True, exist_ok=True)
    return p


def gsm_rows(split: str) -> list[dict]:
    # GSM8K released labels are retained verbatim.  A failure is a blocker, not
    # a license to synthesize replacement answers.
    ds = load_dataset("openai/gsm8k", "main", split=split, cache_dir=str(data_dir()))
    return [{"id": f"gsm8k:{split}:{i}", "question": r["question"], "answer": r["answer"]}
            for i, r in enumerate(ds)]


def prompt(question: str) -> str:
    return "Solve the problem. Show your reasoning, then put the final numeric answer after ####.\n\nProblem: " + question + "\nAnswer:"


def load_qwen() -> tuple[object, object, torch.device]:
    model_id = "Qwen/Qwen2.5-1.5B"
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=str(data_dir()), trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    device = torch.device("cuda:0")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=str(data_dir()), torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=False).to(device)
    model.config.pad_token_id = tok.pad_token_id
    return model, tok, device


@torch.no_grad()
def readiness(model, tok, device: torch.device) -> dict:
    rows = gsm_rows("train")[:128]
    by_prompt, truncations, total = [], 0, 0
    model.eval()
    for offset in range(0, len(rows), 4):
        batch = rows[offset:offset + 4]
        inputs = tok([prompt(r["question"]) for r in batch], return_tensors="pt", padding=True,
                     truncation=True, max_length=1024).to(device)
        # Eight independent samples per prompt, with the exact fixed cap.
        out = model.generate(**inputs, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                             num_return_sequences=8, max_new_tokens=1024, pad_token_id=tok.pad_token_id,
                             eos_token_id=tok.eos_token_id, use_cache=True)
        decoded = tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, row in enumerate(batch):
            samples = decoded[i * 8:(i + 1) * 8]
            rewards = [exact_reward(x, row["answer"]) for x in samples]
            by_prompt.append({"id": row["id"], "rewards": rewards,
                              "successes": int(sum(rewards)), "samples": 8})
            truncations += sum(len(tok.encode(x, add_special_tokens=False)) >= 1024 for x in samples)
            total += 8
        # A partial receipt is evidence of work, but is never treated as an
        # admissible readiness decision. It lets a recovery distinguish a
        # stopped rollout from a process that had not begun.
        atomic_json(RUN / "readiness_partial.json", {
            "timestamp": stamp(), "completed_prompt_groups": len(by_prompt),
            "expected_prompt_groups": len(rows), "samples": total,
            "truncations": truncations, "rows": by_prompt,
        })
    mixed = sum(0 < x["successes"] < 8 for x in by_prompt)
    positive = sum(x["successes"] for x in by_prompt)
    result = {"timestamp": stamp(), "model": "Qwen/Qwen2.5-1.5B", "split": "gsm8k train first 128",
              "prompt_groups": len(by_prompt), "samples": total, "mixed_success_groups": mixed,
              "positive_rewards": positive, "negative_rewards": total - positive,
              "truncations": truncations, "truncation_rate": truncations / total,
              "criteria": {"mixed_success_groups_min": 20, "truncation_rate_max": 0.10},
              "passed": mixed >= 20 and positive > 0 and total - positive > 0 and truncations / total <= .10,
              "rows": by_prompt}
    atomic_json(RUN / "readiness.json", result)
    return result


def calibration(model, tok, device: torch.device) -> dict:
    rows = gsm_rows("train")[:2]
    enc = tok([prompt(r["question"]) + " " + r["answer"] for r in rows], return_tensors="pt",
              padding=True, truncation=True, max_length=512).to(device)
    labels = mask_labels(enc.input_ids, [len(tok.encode(prompt(r["question"]), add_special_tokens=False)) for r in rows], tok.pad_token_id)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-6)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    model.train(); optim.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(**enc, use_cache=False).logits
        loss = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), labels[:, 1:], ignore_index=-100)
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optim.step(); torch.cuda.synchronize(device)
    elapsed = time.monotonic() - start
    result = {"timestamp": stamp(), "objective": "masked_sft", "loss": float(loss.detach()),
              "batch_rows": len(rows), "tokens": int(enc.input_ids.numel()), "seconds": elapsed,
              "tokens_per_second": int(enc.input_ids.numel()) / elapsed,
              "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
              "device": torch.cuda.get_device_name(device),
              "admission_note": "Single masked SFT update only; no measured cell has started."}
    atomic_json(RUN / "calibration.json", result)
    return result


def pack(tok, texts: list[str], starts: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    return encoded.input_ids, mask_labels(encoded.input_ids, starts, tok.pad_token_id)


def objective_smoke(model, tok, device: torch.device) -> dict:
    """One local update per objective, with a shared optimizer and frozen ref.

    This is an engineering admission check only. The DPO negative is deliberately
    synthetic and is recorded as such; it is never eligible training evidence or
    a replacement for the planned released preference source.
    """
    gsm = gsm_rows("train")[:4]
    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train[:2]",
                        cache_dir=str(data_dir()))
    optim = torch.optim.AdamW(model.parameters(), lr=1e-6)
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    work, rows = Work({}), []

    def step(name: str, loss: torch.Tensor, tokens: int, metadata: dict) -> None:
        optim.zero_grad(set_to_none=True)
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if grad > 0:
            optim.step()
        rows.append({"objective": name, "loss": float(loss.detach()), "tokens": tokens,
                     "gradient_norm_before_clip": grad, "optimizer_step": bool(grad > 0), **metadata})

    # PT: shifted causal language modelling, with no answer/prompt mask.
    pt = tok([str(x["text"]) or " " for x in wiki], return_tensors="pt", padding=True,
             truncation=True, max_length=256).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(**pt, use_cache=False).logits
        pt_loss = F.cross_entropy(out[:, :-1].float().transpose(1, 2), pt.input_ids[:, 1:],
                                  ignore_index=tok.pad_token_id)
    step("pt", pt_loss, int(pt.input_ids.numel()), {"source": "wikitext-2-raw-v1 train[:2]"})

    sft_text = [prompt(x["question"]) + " " + x["answer"] for x in gsm[:2]]
    starts = [len(tok.encode(prompt(x["question"]), add_special_tokens=False)) for x in gsm[:2]]
    sft_ids, sft_labels = pack(tok, sft_text, starts, device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        sft_logits = model(input_ids=sft_ids, use_cache=False).logits
        sft_loss = F.cross_entropy(sft_logits[:, :-1].float().transpose(1, 2), sft_labels[:, 1:], ignore_index=-100)
    step("sft", sft_loss, int(sft_ids.numel()), {"source": "gsm8k train[:2]", "prompt_masked": True})

    chosen = [prompt(x["question"]) + " " + x["answer"] for x in gsm[:2]]
    rejected = [prompt(x["question"]) + " I do not know." for x in gsm[:2]]
    cids, clabels = pack(tok, chosen, starts, device)
    rids, rlabels = pack(tok, rejected, starts, device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        policy_c = sequence_logps(model(input_ids=cids, use_cache=False).logits, cids, clabels)
        policy_r = sequence_logps(model(input_ids=rids, use_cache=False).logits, rids, rlabels)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref_c = sequence_logps(ref(input_ids=cids, use_cache=False).logits, cids, clabels)
        ref_r = sequence_logps(ref(input_ids=rids, use_cache=False).logits, rids, rlabels)
    step("dpo", dpo_loss(policy_c, policy_r, ref_c, ref_r), int(cids.numel() + rids.numel()),
         {"source": "synthetic-negative objective smoke only; not training preference data", "reference": "frozen initial model", "sequence_summed": True})

    # The on-policy check uses a development prompt known from the readiness
    # receipt to have mixed outcomes. A zero-advantage batch is retained and
    # deliberately does not advance the optimizer.
    ready = read_json(RUN / "readiness.json", {})
    mixed_id = next((r["id"] for r in ready.get("rows", []) if 0 < r["successes"] < 8), gsm[0]["id"])
    row = next(x for x in gsm_rows("train") if x["id"] == mixed_id)
    enc = tok([prompt(row["question"])] * 4, return_tensors="pt", padding=True).to(device)
    sampled = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                             max_new_tokens=256, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                             use_cache=True)
    continuation = sampled[:, enc.input_ids.shape[1]:]
    texts = tok.batch_decode(continuation, skip_special_tokens=True)
    rewards = torch.tensor([exact_reward(x, row["answer"]) for x in texts], device=device)
    advantages = loo_advantages(rewards, 4)
    seq = torch.cat([enc.input_ids, continuation], dim=1)
    labels = mask_labels(seq, [enc.input_ids.shape[1]] * len(seq), tok.pad_token_id)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=seq, use_cache=False).logits
        rl_loss = -(advantages.detach() * sequence_logps(logits, seq, labels)).mean()
    if bool((advantages == 0).all()):
        rows.append({"objective": "rl", "loss": float(rl_loss.detach()), "tokens": int(seq.numel()),
                     "rewards": rewards.tolist(), "zero_advantage": True, "optimizer_step": False,
                     "source": "unassisted on-policy GSM8K rollout"})
    else:
        step("rl", rl_loss, int(seq.numel()), {"rewards": rewards.tolist(), "zero_advantage": False,
                                                 "source": "unassisted on-policy GSM8K rollout"})
    result = {"timestamp": stamp(), "objectives": rows, "shared_optimizer": "AdamW", "complete": len(rows) == 4,
              "all_nonzero_steps": all(x["optimizer_step"] for x in rows), "development_only": True}
    atomic_json(RUN / "objective_smoke.json", result)
    return result


def test_math() -> None:
    assert extract_answer("x #### 1,200/3") == "4E+2"
    assert exact_reward("reasoning #### 7", "gold #### 7") == 1.0
    ids = torch.tensor([[1, 2, 3, 0]])
    labels = mask_labels(ids, [2], 0)
    assert labels.tolist() == [[-100, -100, 3, -100]]
    assert torch.allclose(loo_advantages(torch.tensor([0., 1., 1., 0.]), 2), torch.tensor([-1., 1., 1., -1.]))
    assert dpo_loss(torch.tensor([2.]), torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0.])) < math.log(2)
    try:
        loo_advantages(torch.tensor([1.]), 2)
    except ValueError:
        pass
    else:
        raise AssertionError("incomplete reward group accepted")


def preflight() -> None:
    update_progress("environment_and_input_preflight")
    test_math()
    atomic_json(RUN / "environment.json", {
        "timestamp": stamp(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(), "repo_head": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
    })
    # Resolve public inputs early; actual revision hashes are recorded after cache writes.
    rows = gsm_rows("train")
    atomic_json(RUN / "input_manifest.json", {
        "timestamp": stamp(), "model": "Qwen/Qwen2.5-1.5B", "gsm8k_train_rows": len(rows),
        "gsm8k_first_id": rows[0]["id"], "gsm8k_last_id": rows[-1]["id"],
        "data_cache": str(data_dir()), "no_provider_api_keys": True,
    })


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["test", "preflight", "readiness", "calibrate", "objective-smoke"])
    args = p.parse_args()
    if args.command == "test":
        test_math(); print("unit tests passed")
    elif args.command == "preflight":
        preflight(); print("preflight complete")
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
            update_progress("objective_smoke_complete", blockers=[] if result["all_nonzero_steps"] else ["zero-advantage RL smoke; retained without optimizer update"])
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
