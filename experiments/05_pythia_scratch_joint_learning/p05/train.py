"""Exp 05 cell runner: jr1.train.Trainer (imported, not copied) on a random-init Pythia, plus the V100 fp16 guards.

    python -m p05.train --method {sequential,fixed_joint,smooth_joint} --seed S --config <p05 config json> \
        --out <cell dir> [--data-root DIR] [--resume] [--allow-draft] [--max-updates N] [--work-target X]

What this adds to jr1 (and nothing else):
- the config's data root defaults to $P05_DATA_ROOT (p05/data.py layout: packed GPT-NeoX PT blocks);
- _forward reaches the decoder through base_model_prefix (jr1's model.get_decoder() returns the whole
  GPTNeoXForCausalLM in transformers 4.57); everything else in it is jr1's;
- fp16 on V100: a GradScaler that counts the steps it skipped for inf/nan gradients (jr1 counts every scaler.step
  as an optimizer step, so the receipt reports p05.fp16_skipped_steps and the effective step count); the count is
  part of the resumable state;
- refusals: bf16 or FlashAttention-2 on a GPU below compute capability 8.0, a DRAFT config without --allow-draft,
  a frozen config whose sha256 is not the one recorded by freeze.py;
- receipt.p05: observed PT tokens (exposure.pt.tokens, every token of every PT input) against the 50M target, the
  data manifest sha256 and the GPU's compute capability.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import torch

from jr1 import common as C
from jr1 import train as T

from . import spec


class CountingGradScaler(torch.amp.GradScaler):
    """A dynamic loss scaler that counts skipped steps: the scale only drops when a step found inf/nan gradients."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skipped = 0

    def update(self, new_scale=None):
        before = self.get_scale()
        super().update(new_scale)
        if new_scale is None and self.get_scale() < before:
            self.skipped += 1


def frozen_sha_ok(config_path: str, sha: str) -> bool:
    record = Path(config_path).with_name("FROZEN.sha256")
    if not record.exists():
        return False
    names = dict(line.split()[::-1] for line in record.read_text().splitlines() if line.strip())
    return names.get(Path(config_path).name) == sha


def check_device(cfg: dict) -> dict:
    info = {"cuda": torch.cuda.is_available()}
    if not torch.cuda.is_available():
        return info
    cap = torch.cuda.get_device_capability(0)
    info.update(capability=f"{cap[0]}.{cap[1]}", name=torch.cuda.get_device_name(0),
                total_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
    if cap < (8, 0):
        if cfg["precision"]["compute"] == "bf16" or cfg["precision"]["checkpoint"] == "bf16":
            raise RuntimeError(f"{info['name']} (sm{cap[0]}{cap[1]}) has no bf16: use the fp16 config")
        if cfg["model"]["attn_implementation"] == "flash_attention_2":
            raise RuntimeError("FlashAttention-2 needs sm80+; use sdpa on V100")
    return info


class P05Trainer(T.Trainer):
    def __init__(self, cfg: dict, out: Path, seed: int, method: str, config_sha: str | None = None,
                 data_manifest: str | None = None, device_info: dict | None = None):
        if method not in spec.METHODS:
            raise ValueError(f"Exp 05 runs only {spec.METHODS}, not {method!r}")
        super().__init__(cfg, out, seed, method=method, config_sha=config_sha)
        if self.scaler is not None:
            self.scaler = CountingGradScaler("cuda")
        self.data_manifest, self.device_info = data_manifest, device_info or {}

    def _forward(self, model, seqs, starts, purpose, kind):
        # jr1's _forward with one change: the decoder is getattr(model, base_model_prefix) (as jr1.evaluate.Scorer
        # does), because model.get_decoder() returns the whole GPTNeoXForCausalLM in transformers 4.57, whose output
        # has no last_hidden_state. Same packing, charge and row-wise LM-head log-probs (C.head_token_logps).
        ids, attn, labels = C.pack(seqs, starts, self.pad, self.device)
        with C.autocast(self.device, self.compute):
            decoder = getattr(model, model.base_model_prefix)
            hidden = decoder(input_ids=ids, attention_mask=attn, use_cache=False).last_hidden_state
        self.work.add(purpose, kind, ids.numel())
        dtype = None if self.compute == "fp32" else {"bf16": torch.bfloat16, "fp16": torch.float16}[self.compute]
        lp, valid = C.head_token_logps(hidden, model.get_output_embeddings(), labels, dtype)
        return lp, valid, ids.numel()

    def skipped(self) -> int:
        return self.scaler.skipped if self.scaler is not None else 0

    def trainer_state(self) -> dict:
        state = super().trainer_state()
        state["p05"] = {"fp16_skipped_steps": self.skipped()}
        return state

    def load_state(self, path: Path) -> None:
        super().load_state(path)
        if self.scaler is not None:
            self.scaler.skipped = json.loads((path / "trainer.json").read_text()).get("p05", {}).get(
                "fp16_skipped_steps", 0)

    def receipt(self, status: str, error: str | None = None) -> dict:
        r = super().receipt(status, error)
        pt = self.exposure["pt"]
        r["p05"] = {"observed_pt_tokens": pt["tokens"], "observed_pt_target_tokens": pt["target_tokens"],
                    "observed_pt_target": spec.OBSERVED_PT_TARGET,
                    "reached_pt_target": pt["tokens"] >= spec.OBSERVED_PT_TARGET,
                    "fp16_skipped_steps": self.skipped(),
                    "effective_optimizer_steps": self.optimizer_steps - self.skipped(),
                    "data_manifest_sha256": self.data_manifest, "device": self.device_info,
                    "config": self.cfg.get("p05"), "frozen": bool(self.cfg.get("frozen"))}
        return r


def build(config_path: str, data_root: str | None, work_target: float | None, allow_draft: bool):
    cfg, sha = T.load_config(config_path)
    if cfg.get("frozen"):
        if work_target is not None:
            raise SystemExit("--work-target is refused for a frozen config")
        if not frozen_sha_ok(config_path, sha):
            raise SystemExit(f"{config_path}: sha256 {sha} is not the one in FROZEN.sha256 (edited after freeze?)")
    elif not allow_draft:
        raise SystemExit(f"{config_path} is a DRAFT (frozen=false): measured cells need the frozen config "
                         "(p05/freeze.py); pass --allow-draft only for smoke tests and calibration")
    root = data_root or cfg["data"]["root"] or os.environ.get("P05_DATA_ROOT")
    if not root:
        raise SystemExit("set --data-root or P05_DATA_ROOT (p05/data.py prepare output)")
    cfg["data"]["root"] = root
    if work_target is not None:
        cfg["work"]["target"] = float(work_target)
    manifest = Path(root) / "manifest.json"
    return cfg, sha, (C.sha256(manifest) if manifest.exists() else None)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--method", choices=spec.METHODS, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-root")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-draft", action="store_true", help="smoke tests / calibration only")
    ap.add_argument("--work-target", type=float, help="smoke tests only (refused for frozen configs)")
    ap.add_argument("--max-updates", type=int, help="debug only (receipt says debug_stopped)")
    a = ap.parse_args(argv)
    cfg, sha, manifest = build(a.config, a.data_root, a.work_target, a.allow_draft)
    device = check_device(cfg)
    trainer = P05Trainer(cfg, a.out, a.seed, a.method, config_sha=sha, data_manifest=manifest, device_info=device)
    try:
        receipt = trainer.run(resume=a.resume, max_updates=a.max_updates)
    except BaseException as exc:
        C.atomic_json(trainer.out / "receipt.json", trainer.receipt("failed", "".join(
            traceback.format_exception_only(type(exc), exc)).strip()))
        raise
    print(json.dumps({k: receipt[k] for k in ("status", "method", "seed", "updates", "optimizer_steps", "work_total",
                                              "work_overshoot", "wall_s")} | {"p05": {
        k: receipt["p05"][k] for k in ("observed_pt_tokens", "reached_pt_target", "fp16_skipped_steps")}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
