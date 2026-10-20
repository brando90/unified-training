"""JR1 trainer: the six Exp 03 methods and the Exp 04 schedule replay, with work accounting and atomic resume.

    python -m jr1.train --method {sequential,replay,fixed_joint,smooth_joint,aioli,jr1} \
        --seed S --config <frozen protocol json> --out <cell dir> [--resume] [--record-schedule <path>]
    python -m jr1.train --replay-schedule <schedule.jsonl> --order {interleaved,stage_sorted} \
        --seed S --config <json> --out <cell dir> [--resume]

Contract (PLAN.md "Objectives, optimizer and implementation contract"): one shared AdamW over full weights
for every objective, PT shifted causal cross entropy, SFT with prompt and padding masked, DPO on
sequence-summed response log probabilities against the frozen initial model (<cell>/ckpt_000), unassisted
on-policy GSM8K RL with leave-one-out advantages where a batch whose groups all have zero reward variance
makes no backward pass and no optimizer (or weight-decay) step. Every forward-equivalent token is charged:
training, probes, reference passes, generation (prefill + decoding) and controller validation. The run stops
at the first action boundary at or past the frozen work target, so overshoot is at most one action.
Equations, defaults and the Aioli differences table: jr1/TRAINER.md.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

from . import common as C
from .controllers import K, OBJECTIVES, Interleaver, make_schedule

REPO = Path(__file__).resolve().parents[4]
STAGE_ORDER = {o: i for i, o in enumerate(OBJECTIVES)}
CKPT_NAMES = {0.0: "000", 0.25: "025", 0.5: "050", 0.75: "075", 1.0: "100"}

# Every field is part of the frozen protocol; a config file overrides any subset. Values marked "dev" are
# placeholders until development selection (AT-572) freezes them in PROTOCOL v1.
DEFAULTS = {
    "protocol": "jr1-unfrozen-defaults",
    "model": {"source": "pretrained",          # pretrained | random (from_config, seeded) | tiny (offline tests)
              "id": "Qwen/Qwen2.5-1.5B", "revision": "8faed761d45a263340a0528343f099c05c9a4323", "local_path": None, "init_seed": None,
              "tiny": None, "attn_implementation": "sdpa", "gradient_checkpointing": False},
    "precision": {"master": "fp32",            # fp32 | bf16 (pure bf16 AdamW drops updates below bf16 eps)
                  "compute": "bf16",           # bf16 | fp16 (GradScaler, V100) | fp32
                  "reference": "compute",      # dtype of the frozen DPO reference: compute | master
                  "checkpoint": "bf16"},       # dtype of ckpt_XXX/ (Qwen's released weights are bf16)
    "data": {"root": None,                     # default $JR1_STATE_ROOT/data (jr1/data.py layout)
             "splits": {"pt": "pt/train", "sft": "sft/train", "rl": "rl/train", "dpo": "pref/train",
                        "pt_controller": "pt/controller", "sft_controller": "sft/controller",
                        "rl_controller": "rl/controller", "dpo_controller": "pref/controller"}},
    "optimizer": {"lr": 1e-6, "weight_decay": 0.01, "betas": [0.9, 0.95], "eps": 1e-8, "grad_clip": 1.0,
                  "schedule": "cosine", "warmup_fraction": 0.02, "warmup_start_ratio": 0.1,
                  "min_lr_ratio": 0.1},       # dev; the clock is charged-work progress, identical for all methods
    "objectives": {"pt": {"batch_size": 8, "max_len": 1024},
                   "sft": {"batch_size": 8, "max_len": 1024},
                   "dpo": {"batch_size": 4, "max_len": 1024, "beta": 0.1},
                   "rl": {"prompts": 4, "group_size": 8, "max_new_tokens": 1024, "max_prompt_len": 512,
                          "temperature": 1.0, "reward": "gsm8k_exact", "backend": "hf"}},
    "work": {"target": 2.0e9},                 # dev; forward-equivalent tokens, fixed per cell
    "methods": {
        "sequential": {"boundaries": [0.55, 0.75, 0.85]},
        "replay": {"boundaries": [0.55, 0.75, 0.85], "pt_share": 0.2},
        "fixed_joint": {"weights": {"pt": 0.55, "sft": 0.20, "dpo": 0.10, "rl": 0.15}},
        "smooth_joint": {"start": {"pt": 0.70, "sft": 0.20, "dpo": 0.05, "rl": 0.05},
                         "end": {"pt": 0.20, "sft": 0.20, "dpo": 0.15, "rl": 0.45},
                         "floors": {"pt": 0.05, "dpo": 0.05}},
        "aioli": {"alpha": 0.75, "eta": 1.0, "floor": 0.0, "window": 200, "probe_updates": 12},
        "jr1": {"alpha": 0.75, "eta": 1.0, "floor": 0.0, "window": 200, "probe_updates": 12,
                "prior_start": {"pt": 0.45, "sft": 0.35, "dpo": 0.10, "rl": 0.10},
                "prior_end": {"pt": 0.15, "sft": 0.15, "dpo": 0.10, "rl": 0.60},
                "prior_strength": 0.5, "min_share": 0.02, "pt_floor": 0.05, "dpo_floor": 0.05,
                "retention_threshold": 0.02, "retention_min_pt": 0.55}},
    "controller": {"rows": {"pt": 32, "sft": 32, "dpo": 32, "rl": 32}, "batch_size": 8,
                   "rl_max_new_tokens": 512},  # validation-only feedback for aioli and jr1, charged
    "checkpoints": {"fractions": [0.0, 0.25, 0.5, 0.75, 1.0], "state_every_actions": 1000,
                    "state_every_seconds": 1800},
}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | None) -> tuple[dict, str | None]:
    if path is None:
        return copy.deepcopy(DEFAULTS), None
    text = Path(path).read_text()
    return deep_merge(DEFAULTS, json.loads(text)), C.sha256_text(text)


DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
DTYPE_NAMES = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}
REWARDS = {"gsm8k_exact": C.exact_reward}   # tests register extra reward functions here


def _dtype_kwargs(dtype) -> dict:
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}


class Stream:
    """A pool read in a seeded permutation per epoch; the state is (epoch, position), so it stays small."""

    def __init__(self, rows: list[dict], seed: str):
        if not rows:
            raise ValueError(f"empty training pool for stream {seed}")
        self.rows, self.seed, self.epoch, self.pos = rows, seed, 0, 0
        self._order, self._order_epoch = None, None

    def _order_now(self) -> list[int]:
        if self._order_epoch != self.epoch:
            import random
            self._order = list(range(len(self.rows)))
            random.Random(f"{self.seed}:{self.epoch}").shuffle(self._order)
            self._order_epoch = self.epoch
        return self._order

    def take(self, n: int) -> list[dict]:
        out = []
        for _ in range(n):
            if self.pos == len(self.rows):
                self.epoch, self.pos = self.epoch + 1, 0
            out.append(self.rows[self._order_now()[self.pos]])
            self.pos += 1
        return out

    def unique_seen(self) -> int:
        return len(self.rows) if self.epoch else self.pos

    def state_dict(self) -> dict:
        return {"epoch": self.epoch, "pos": self.pos}

    def load_state_dict(self, state: dict) -> None:
        self.epoch, self.pos = state["epoch"], state["pos"]


class Trainer:
    def __init__(self, cfg: dict, out: Path, seed: int, method: str | None = None, config_sha: str | None = None,
                 replay: Path | None = None, order: str | None = None, record: Path | None = None):
        if (method is None) == (replay is None):
            raise ValueError("give exactly one of a method or a replay schedule")
        if replay is not None and order not in ("interleaved", "stage_sorted"):
            raise ValueError("replay needs --order interleaved or stage_sorted")
        if cfg["objectives"]["rl"]["backend"] != "hf":
            raise NotImplementedError("only the HF generate rollout backend is implemented in the trainer")
        self.cfg, self.out, self.seed, self.config_sha = cfg, Path(out), int(seed), config_sha
        self.method = method or f"replay_{order}"
        self.out.mkdir(parents=True, exist_ok=True)
        self.state_dir, self.log_path = self.out / "state", self.out / "log.jsonl"
        self.record = Path(record) if record else None
        p = cfg["precision"]
        self.compute, self.master = p["compute"], DTYPES[p["master"]]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.scaler = torch.amp.GradScaler("cuda") if self.compute == "fp16" and self.device.type == "cuda" else None

        self.tok = self._build_tokenizer()
        self.pad = self.tok.pad_token_id
        self.eos = self.tok.eos_token_id
        self.model = self._build_model()
        C.seed_everything(self.seed)
        self.reference = None
        o = cfg["optimizer"]
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"],
                                     weight_decay=o["weight_decay"])

        root = Path(cfg["data"]["root"]) if cfg["data"]["root"] else C.state_root() / "data"
        split = lambda key: C.read_jsonl(root / (cfg["data"]["splits"][key] + ".jsonl"))
        self.pools = {obj: split(obj) for obj in OBJECTIVES}
        self.streams = {obj: Stream(self.pools[obj], f"jr1:{self.seed}:{obj}") for obj in OBJECTIVES}
        self.reward_fn = REWARDS[cfg["objectives"]["rl"]["reward"]]

        self.schedule = None
        self.replay_rows, self.replay_progress, self.replay_identity = None, None, None
        if replay is None:
            self.schedule = make_schedule(method, cfg["methods"][method], self.seed)
            if self.schedule.adaptive:
                rows = cfg["controller"]["rows"]
                self.controller_rows = {obj: split(f"{obj}_controller")[:rows[obj]] for obj in OBJECTIVES}
        else:
            recorded = C.read_jsonl(replay)
            self.replay_progress = [r["progress"] for r in recorded]    # positional LR clock, same for both orders
            self.replay_rows = recorded if order == "interleaved" else \
                sorted(recorded, key=lambda r: (STAGE_ORDER[r["objective"]], r["i"]))
            self.by_id = {obj: {r["id"]: r for r in self.pools[obj]} for obj in OBJECTIVES}
            missing = [(r["objective"], i) for r in recorded for i in r["ids"] if i not in self.by_id[r["objective"]]]
            if missing:
                raise ValueError(f"replay schedule names {len(missing)} ids absent from the pools, e.g. {missing[:3]}")
            self.replay_identity = {"schedule_sha256": C.sha256(replay), "order": order, "actions": len(recorded)}
        self.interleaver = Interleaver()
        self.target = float(cfg["work"]["target"])

        self.work = C.Work()
        self.actions = self.updates = self.optimizer_steps = 0
        self.exposure = {obj: {"updates": 0, "optimizer_steps": 0, "examples": 0, "tokens": 0, "target_tokens": 0}
                         for obj in OBJECTIVES}
        self.exposure["rl"].update(rollouts=0, generated_tokens=0, informative_groups=0, zero_variance_groups=0)
        self.probe, self.since_probe = None, None
        self.initial_val, self.latest_val, self.ref_cache = None, None, None
        self.milestones_done: list[float] = []
        self.log_lines = self.schedule_lines = 0
        self.replay_pos = 0
        self.wall_base, self.t0 = 0.0, time.monotonic()
        self.resumes, self.max_action_charge = 0, 0
        self.debug_stop = None

    # ---------------------------------------------------------------- construction
    def _build_tokenizer(self):
        m = self.cfg["model"]
        if m["source"] == "tiny":
            from .tiny import tiny_tokenizer
            return tiny_tokenizer()
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(m["local_path"] or m["id"], revision=m["revision"])
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        return tok

    def _build_model(self):
        m = self.cfg["model"]
        init_seed = self.seed if m["init_seed"] is None else int(m["init_seed"])
        if m["source"] == "tiny":
            from .tiny import tiny_model
            model = tiny_model(m["tiny"], seed=init_seed, dtype=self.master)
        else:
            from transformers import AutoConfig, AutoModelForCausalLM
            path = m["local_path"] or m["id"]
            if m["source"] == "random":
                config = AutoConfig.from_pretrained(path, revision=m["revision"])
                torch.manual_seed(init_seed)
                model = AutoModelForCausalLM.from_config(config, attn_implementation=m["attn_implementation"],
                                                         **_dtype_kwargs(self.master))
            elif m["source"] == "pretrained":
                model = AutoModelForCausalLM.from_pretrained(path, revision=m["revision"],
                                                             attn_implementation=m["attn_implementation"],
                                                             **_dtype_kwargs(self.master))
            else:
                raise ValueError(f"unknown model source {m['source']!r}")
        model.config.use_cache = False
        model.config.pad_token_id = self.tok.pad_token_id
        if m["gradient_checkpointing"]:
            model.gradient_checkpointing_enable()
        if any(p.dtype != self.master for p in model.parameters()):
            raise RuntimeError("master weights are not in the configured dtype")
        return model.to(self.device).train()

    def _load_reference(self):
        from transformers import AutoModelForCausalLM
        if not (self.out / "ckpt_000").exists():
            if self.optimizer_steps:
                raise RuntimeError("ckpt_000 (the frozen initial reference) is missing after training began")
            self.save_ckpt(0.0)
        p = self.cfg["precision"]
        dtype = DTYPES[self.compute] if p["reference"] == "compute" and self.compute != "fp32" else self.master
        ref = AutoModelForCausalLM.from_pretrained(self.out / "ckpt_000", **_dtype_kwargs(dtype))
        ref.config.use_cache = False
        self.reference = ref.to(self.device).eval().requires_grad_(False)

    # ---------------------------------------------------------------- clocks
    def progress(self) -> float:
        if self.replay_rows is not None:
            return self.replay_pos / len(self.replay_rows)
        return min(1.0, self.work.total / self.target)

    def lr_at(self, progress: float) -> float:
        import math
        o = self.cfg["optimizer"]
        w = o["warmup_fraction"]
        if w > 0 and progress < w:
            return o["lr"] * (o["warmup_start_ratio"] + (1 - o["warmup_start_ratio"]) * progress / w)
        if o["schedule"] == "constant":
            return o["lr"]
        if o["schedule"] != "cosine":
            raise ValueError(f"unknown lr schedule {o['schedule']!r}")
        t = (progress - w) / (1 - w) if w < 1 else 1.0
        return o["lr"] * (o["min_lr_ratio"] + (1 - o["min_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))

    def wall(self) -> float:
        return self.wall_base + time.monotonic() - self.t0

    # ---------------------------------------------------------------- tokens and forwards
    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def _prompt_response(self, prompt: str, response: str, max_len: int) -> tuple[list[int], int]:
        p, r = self.encode(prompt), self.encode(response) + [self.eos]
        if len(p) >= max_len:
            p = p[-(max_len // 2):]
        return (p + r)[:max_len], len(p)

    def _forward(self, model, seqs, starts, purpose, kind):
        ids, attn, labels = C.pack(seqs, starts, self.pad, self.device)
        with C.autocast(self.device, self.compute):
            logits = model(input_ids=ids, attention_mask=attn, use_cache=False).logits
        self.work.add(purpose, kind, ids.numel())
        lp, valid = C.token_logps(logits, labels)
        return lp, valid, ids.numel()

    @torch.no_grad()
    def generate(self, prompts: list[list[int]], max_new: int, sample: bool, purpose: str):
        ids, attn, _ = C.pack(prompts, [len(p) for p in prompts], self.pad, self.device, left=True)

        def count(module, args, kwargs):
            x = kwargs.get("input_ids", args[0] if args else None)
            if x is not None:
                self.work.add(purpose, "generation", x.numel())

        hook = self.model.register_forward_pre_hook(count, with_kwargs=True)
        was_training = self.model.training
        self.model.eval()
        kw = ({"do_sample": True, "temperature": self.cfg["objectives"]["rl"]["temperature"], "top_k": 0, "top_p": 1.0}
              if sample else {"do_sample": False, "temperature": None, "top_k": None, "top_p": None})
        try:
            with C.autocast(self.device, self.compute):
                out = self.model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=max_new, use_cache=True,
                                          pad_token_id=self.pad, eos_token_id=self.eos, **kw)
        finally:
            hook.remove()
            self.model.train(was_training)
        generated, truncated = [], []
        for row in out[:, ids.shape[1]:].tolist():
            if self.eos in row:
                row = row[:row.index(self.eos) + 1]
                truncated.append(False)
            else:
                truncated.append(len(row) == max_new)
            generated.append(row)
        return generated, truncated

    # ---------------------------------------------------------------- objectives
    def loss_pt(self, rows, purpose):
        c = self.cfg["objectives"]["pt"]
        seqs = [(self.encode(r["text"][:c["max_len"] * 12]) or [self.eos])[:c["max_len"]] for r in rows]
        seqs = [s if len(s) > 1 else s + [self.eos] for s in seqs]
        lp, valid, n = self._forward(self.model, seqs, [0] * len(seqs), purpose, "forward")
        return -lp.sum() / valid.sum().clamp_min(1), n, {"target_tokens": int(valid.sum())}

    def loss_sft(self, rows, purpose):
        c = self.cfg["objectives"]["sft"]
        pairs = [self._prompt_response(r["prompt"], r["response"], c["max_len"]) for r in rows]
        lp, valid, n = self._forward(self.model, [s for s, _ in pairs], [k for _, k in pairs], purpose, "forward")
        return -lp.sum() / valid.sum().clamp_min(1), n, {"target_tokens": int(valid.sum())}

    def _pref_batch(self, rows, max_len):
        chosen = [self._prompt_response(r["prompt"], r["chosen"], max_len) for r in rows]
        rejected = [self._prompt_response(r["prompt"], r["rejected"], max_len) for r in rows]
        return [s for s, _ in chosen + rejected], [k for _, k in chosen + rejected]

    def _reference_logps(self, seqs, starts, purpose):
        if self.reference is None:
            self._load_reference()
        with torch.no_grad():
            lp, _, _ = self._forward(self.reference, seqs, starts, purpose, "reference")
        return lp.sum(-1)

    def loss_dpo(self, rows, purpose):
        c = self.cfg["objectives"]["dpo"]
        seqs, starts = self._pref_batch(rows, c["max_len"])
        lp, valid, n = self._forward(self.model, seqs, starts, purpose, "forward")
        policy = lp.sum(-1)
        ref = self._reference_logps(seqs, starts, purpose)
        k = len(rows)
        loss = C.dpo_loss(policy[:k], policy[k:], ref[:k], ref[k:], c["beta"])
        margin = c["beta"] * ((policy[:k] - policy[k:]) - (ref[:k] - ref[k:])).detach()
        return loss, n, {"target_tokens": int(valid.sum()), "dpo_margin_mean": float(margin.float().mean()),
                         "dpo_accuracy": float((margin > 0).float().mean())}

    def loss_rl(self, rows, purpose):
        c = self.cfg["objectives"]["rl"]
        g = c["group_size"]
        prompts = [self.encode(r["prompt"])[-c["max_prompt_len"]:] for r in rows for _ in range(g)]
        answers = [r["answer"] for r in rows for _ in range(g)]
        generated, truncated = self.generate(prompts, c["max_new_tokens"], True, purpose)
        texts = self.tok.batch_decode(generated, skip_special_tokens=True)
        rewards = torch.tensor([self.reward_fn(t, a) for t, a in zip(texts, answers)], dtype=torch.float32)
        adv = C.loo_advantages(rewards, g)
        informative = rewards.view(-1, g).var(1, unbiased=False) > 0
        meta = {"rollouts": len(rewards), "reward_mean": float(rewards.mean()), "reward_sum": float(rewards.sum()),
                "groups": len(rows), "informative_groups": int(informative.sum()),
                "zero_variance_groups": int((~informative).sum()), "truncated": int(sum(truncated)),
                "generated_tokens": int(sum(map(len, generated))), "advantage_estimator": "leave_one_out"}
        if not bool(informative.any()):
            return None, 0, meta
        keep = [i for i in range(len(rewards)) if informative[i // g]]
        lp, valid, n = self._forward(self.model, [prompts[i] + generated[i] for i in keep],
                                     [len(prompts[i]) for i in keep], purpose, "forward")
        # Score-function gradient of expected sequence reward; zero-advantage groups contribute exactly zero,
        # so they are skipped, and the sum is divided by all rollouts to keep the estimator's scale.
        loss = -(adv[keep].to(lp.device) * lp.sum(-1)).sum() / len(rewards)
        meta["target_tokens"] = int(valid.sum())
        return loss, n, meta

    # ---------------------------------------------------------------- one update
    def take(self, obj: str) -> list[dict]:
        if obj == "rl":
            return self.streams[obj].take(self.cfg["objectives"]["rl"]["prompts"])
        return self.streams[obj].take(self.cfg["objectives"][obj]["batch_size"])

    def update(self, obj: str, rows: list[dict] | None, purpose: str, progress: float | None = None) -> dict:
        progress = self.progress() if progress is None else progress
        lr = self.lr_at(progress)
        rows = self.take(obj) if rows is None else rows
        before = self.work.total
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        loss, tokens, meta = getattr(self, f"loss_{obj}")(rows, purpose)
        stepped, norm = False, 0.0
        if loss is not None:
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite {obj} loss at update {self.updates}")
            (self.scaler.scale(loss) if self.scaler else loss).backward()
            self.work.add(purpose, "backward", tokens)
            if self.scaler:
                self.scaler.unscale_(self.opt)
            norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["optimizer"]["grad_clip"],
                                                        error_if_nonfinite=self.scaler is None))
            for group in self.opt.param_groups:
                group["lr"] = lr
            if self.scaler:
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            stepped = True
            self.optimizer_steps += 1
        e = self.exposure[obj]
        e["updates"] += 1
        e["optimizer_steps"] += int(stepped)
        e["examples"] += len(rows)
        e["tokens"] += int(tokens)
        e["target_tokens"] += int(meta.get("target_tokens", 0))
        if obj == "rl":
            for key in ("rollouts", "generated_tokens", "informative_groups", "zero_variance_groups"):
                e[key] += meta[key]
        ids = [r["id"] for r in rows]
        if self.record is not None:
            C.append_jsonl(self.record, {"i": self.updates, "objective": obj, "ids": ids, "purpose": purpose,
                                         "progress": progress, "lr": lr})
            self.schedule_lines += 1
        self.updates += 1
        return {"action": "update", "objective": obj, "purpose": purpose, "update": self.updates - 1,
                "loss": None if loss is None else float(loss.detach()), "optimizer_step": stepped,
                "zero_gradient": loss is None, "grad_norm": norm, "lr": lr, "progress": progress,
                "examples": len(rows), "example_ids": ids, "tokens": int(tokens),
                "update_charge": self.work.total - before, **meta}

    # ---------------------------------------------------------------- controller validation (aioli, jr1)
    @torch.no_grad()
    def validate(self) -> dict:
        b = self.cfg["controller"]["batch_size"]
        rows = self.controller_rows
        self.model.eval()
        out = {}
        for obj in ("pt", "sft"):
            total, count = 0.0, 0
            for i in range(0, len(rows[obj]), b):
                chunk = rows[obj][i:i + b]
                if obj == "pt":
                    c = self.cfg["objectives"]["pt"]
                    seqs = [(self.encode(r["text"][:c["max_len"] * 12]) or [self.eos])[:c["max_len"]] for r in chunk]
                    seqs = [s if len(s) > 1 else s + [self.eos] for s in seqs]
                    starts = [0] * len(seqs)
                else:
                    pairs = [self._prompt_response(r["prompt"], r["response"], self.cfg["objectives"]["sft"]["max_len"])
                             for r in chunk]
                    seqs, starts = [s for s, _ in pairs], [k for _, k in pairs]
                lp, valid, _ = self._forward(self.model, seqs, starts, "validation", "forward")
                total += float(-lp.sum())
                count += int(valid.sum())
            out[obj] = total / max(count, 1)
        c = self.cfg["objectives"]["dpo"]
        if self.ref_cache is None:   # the frozen reference on fixed rows is constant: computed and charged once
            self.ref_cache = []
            for i in range(0, len(rows["dpo"]), b):
                seqs, starts = self._pref_batch(rows["dpo"][i:i + b], c["max_len"])
                self.ref_cache.extend(self._reference_logps(seqs, starts, "validation").float().tolist())
        losses, offset = [], 0
        for i in range(0, len(rows["dpo"]), b):
            chunk = rows["dpo"][i:i + b]
            seqs, starts = self._pref_batch(chunk, c["max_len"])
            lp, _, _ = self._forward(self.model, seqs, starts, "validation", "forward")
            policy = lp.sum(-1).float().cpu()
            ref = torch.tensor(self.ref_cache[2 * offset:2 * offset + 2 * len(chunk)])
            k = len(chunk)
            losses.extend((-torch.nn.functional.logsigmoid(c["beta"] * ((policy[:k] - policy[k:]) - (ref[:k] - ref[k:])))).tolist())
            offset += k
        out["dpo"] = sum(losses) / len(losses)
        correct, n = 0.0, 0
        r = self.cfg["objectives"]["rl"]
        for i in range(0, len(rows["rl"]), b):
            chunk = rows["rl"][i:i + b]
            generated, _ = self.generate([self.encode(x["prompt"])[-r["max_prompt_len"]:] for x in chunk],
                                         self.cfg["controller"]["rl_max_new_tokens"], False, "validation")
            texts = self.tok.batch_decode(generated, skip_special_tokens=True)
            correct += sum(self.reward_fn(t, x["answer"]) for t, x in zip(texts, chunk))
            n += len(chunk)
        out["rl"] = (n - correct + 0.5) / (n + 1)    # smoothed greedy failure rate, strictly positive
        self.model.train()
        if self.initial_val is None:
            if not all(v > 0 for v in out.values()):
                raise ValueError(f"initial validation components must be positive: {out}")
            self.initial_val = dict(out)
        self.latest_val = dict(out)
        return out

    # ---------------------------------------------------------------- one action
    def done(self) -> bool:
        if self.debug_stop is not None:
            return True
        if self.replay_rows is not None:
            return self.replay_pos >= len(self.replay_rows)
        return self.work.total >= self.target

    def step(self) -> dict:
        before = self.work.total
        if self.replay_rows is not None:
            row = self.replay_rows[self.replay_pos]
            rows = [self.by_id[row["objective"]][i] for i in row["ids"]]
            rec = self.update(row["objective"], rows, "train", progress=self.replay_progress[self.replay_pos])
            rec["source_update"] = row["i"]
            self.replay_pos += 1
        elif self.schedule.adaptive and (self.probe is not None or self.since_probe is None
                                         or self.since_probe >= self.schedule.window):
            if self.probe is None:
                self.probe = {**self.schedule.probe_plan(), "col": 0, "off": 0, "before": None,
                              "drops": [[0.0] * K for _ in range(K)]}
            p = self.probe
            if p["before"] is None:
                p["before"] = self.validate()
                rec = {"action": "validate", "validation": dict(p["before"]), "probe_order": list(p["order"])}
            else:
                j = p["order"][p["col"]]
                rec = self.update(p["actions"][p["col"]][p["off"]], None, "probe")
                rec["probe_mixture"] = OBJECTIVES[j]
                p["off"] += 1
                if p["off"] == len(p["actions"][p["col"]]):
                    after = self.validate()
                    for i, obj in enumerate(OBJECTIVES):
                        p["drops"][i][j] = (p["before"][obj] - after[obj]) / self.initial_val[obj]
                    rec["validation"] = dict(after)
                    # Nothing trains between this mixture's after and the next mixture's before, so they are
                    # the same measurement (greedy, deterministic); it is reused rather than charged twice.
                    p["col"], p["off"], p["before"] = p["col"] + 1, 0, after
                    if p["col"] == K:
                        self.schedule.feedback(p["drops"], self.progress(), after, self.initial_val)
                        self.probe, self.since_probe = None, 0
                        rec["controller"] = self.schedule.snapshot(self.progress())
        else:
            weights = self.schedule.weights(self.progress())
            rec = self.update(self.interleaver.choose(weights), None, "train")
            rec["weights"] = dict(zip(OBJECTIVES, weights))
            if self.since_probe is not None:
                self.since_probe += 1
        charge = self.work.total - before
        self.max_action_charge = max(self.max_action_charge, charge)
        self.actions += 1
        rec.update(action_index=self.actions - 1, action_charge=charge, work_total=self.work.total,
                   work_by_purpose=self.work.by_purpose(), progress_after=self.progress(),
                   exposure=copy.deepcopy(self.exposure), wall_s=round(self.wall(), 3))
        if self.schedule is not None:
            stage = self.schedule.stage(rec.get("progress", 0.0)) if rec["action"] == "update" else None
            if stage is not None:
                rec["stage"] = OBJECTIVES[stage]
            if self.schedule.adaptive:
                rec["schedule"] = {"weights": dict(zip(OBJECTIVES, self.schedule.weights(self.progress())))}
                if hasattr(self.schedule, "retention_active"):
                    rec["schedule"]["retention_active"] = self.schedule.retention_active()
        C.append_jsonl(self.log_path, rec)
        self.log_lines += 1
        return rec

    # ---------------------------------------------------------------- checkpoints and state
    def save_ckpt(self, fraction: float) -> None:
        name = CKPT_NAMES[fraction]
        dtype = DTYPES[self.cfg["precision"]["checkpoint"]]

        def write(d: Path):
            state = {k: v.detach().to(dtype) if v.is_floating_point() else v for k, v in self.model.state_dict().items()}
            self.model.save_pretrained(d, state_dict=state, safe_serialization=True)
            cfg = json.loads((d / "config.json").read_text())
            cfg["torch_dtype"] = cfg["dtype"] = DTYPE_NAMES[self.cfg["precision"]["checkpoint"]]
            (d / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))
            self.tok.save_pretrained(d)
            C.atomic_json(d / "jr1_ckpt.json", {
                "milestone": fraction, "work_total": self.work.total, "progress": self.progress(),
                "updates": self.updates, "optimizer_steps": self.optimizer_steps, "actions": self.actions,
                "exposure": self.exposure, "work": self.work.state_dict(), "wall_s": self.wall(),
                "method": self.method, "seed": self.seed, "dtype": self.cfg["precision"]["checkpoint"],
                "timestamp": C.stamp()})

        C.atomic_directory(self.out / f"ckpt_{name}", write)
        if fraction not in self.milestones_done:
            self.milestones_done.append(fraction)

    def identity(self) -> dict:
        return {"method": self.method, "seed": self.seed, "config_sha256": self.config_sha,
                "replay": self.replay_identity, "target": self.target}

    def trainer_state(self) -> dict:
        return {"identity": self.identity(), "work": self.work.state_dict(), "actions": self.actions,
                "updates": self.updates, "optimizer_steps": self.optimizer_steps, "exposure": self.exposure,
                "probe": self.probe, "since_probe": self.since_probe, "initial_val": self.initial_val,
                "latest_val": self.latest_val, "ref_cache": self.ref_cache, "interleaver": self.interleaver.state_dict(),
                "schedule": self.schedule.state_dict() if self.schedule else None,
                "streams": {k: s.state_dict() for k, s in self.streams.items()},
                "milestones_done": self.milestones_done, "log_lines": self.log_lines,
                "schedule_lines": self.schedule_lines, "replay_pos": self.replay_pos, "wall_s": self.wall(),
                "resumes": self.resumes, "max_action_charge": self.max_action_charge, "saved_at": C.stamp()}

    def save_state(self) -> None:
        def write(d: Path):
            torch.save(self.model.state_dict(), d / "policy.pt")
            torch.save(self.opt.state_dict(), d / "optimizer.pt")
            torch.save({"rng": C.rng_state(), "scaler": self.scaler.state_dict() if self.scaler else None}, d / "rng.pt")
            (d / "trainer.json").write_text(json.dumps(self.trainer_state(), sort_keys=True, allow_nan=False))
        C.atomic_directory(self.state_dir, write)
        self._last_state_actions, self._last_state_time = self.actions, time.monotonic()

    def load_state(self, path: Path) -> None:
        t = json.loads((path / "trainer.json").read_text())
        if t["identity"] != self.identity():
            raise ValueError(f"saved state belongs to another cell: {t['identity']} != {self.identity()}")
        self.model.load_state_dict(torch.load(path / "policy.pt", map_location=self.device))
        self.opt.load_state_dict(torch.load(path / "optimizer.pt", map_location=self.device))
        extra = torch.load(path / "rng.pt", map_location="cpu", weights_only=False)
        if self.scaler and extra["scaler"]:
            self.scaler.load_state_dict(extra["scaler"])
        self.work = C.Work.from_state(t["work"])
        for key in ("actions", "updates", "optimizer_steps", "exposure", "probe", "since_probe", "initial_val",
                    "latest_val", "ref_cache", "milestones_done", "log_lines", "schedule_lines", "replay_pos",
                    "max_action_charge"):
            setattr(self, key, t[key])
        self.interleaver.load_state_dict(t["interleaver"])
        if self.schedule:
            self.schedule.load_state_dict(t["schedule"])
        for k, s in t["streams"].items():
            self.streams[k].load_state_dict(s)
        self.wall_base, self.t0 = t["wall_s"], time.monotonic()
        self.resumes = t["resumes"] + 1
        C.truncate_jsonl(self.log_path, self.log_lines)
        if self.record is not None:
            C.truncate_jsonl(self.record, self.schedule_lines)
        C.restore_rng(extra["rng"])

    # ---------------------------------------------------------------- receipt and main loop
    def receipt(self, status: str, error: str | None = None) -> dict:
        if self.device.type == "cuda":
            peak = {"device": torch.cuda.get_device_name(self.device),
                    "peak_memory_bytes": int(torch.cuda.max_memory_allocated(self.device))}
        else:
            peak = {"device": f"cpu ({platform.processor() or platform.machine()})",
                    "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024}
        overshoot = self.work.total - self.target if self.replay_rows is None else None
        import transformers
        try:
            head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            head = None
        streams = {k: {"epoch": s.epoch, "pos": s.pos, "pool": len(s.rows), "unique_seen": s.unique_seen()}
                   for k, s in self.streams.items()}
        return {"status": status, "error": error, **self.identity(), "protocol": self.cfg["protocol"],
                "work_total": self.work.total, "work": self.work.state_dict(), "work_by_purpose": self.work.by_purpose(),
                "work_overshoot": overshoot, "max_action_charge": self.max_action_charge,
                "overshoot_bounded": None if overshoot is None else overshoot <= self.max_action_charge,
                "actions": self.actions, "updates": self.updates, "optimizer_steps": self.optimizer_steps,
                "exposure": self.exposure, "streams": streams, "milestones_done": self.milestones_done,
                "controller": self.schedule.snapshot(self.progress()) if self.schedule else None,
                "initial_validation": self.initial_val, "latest_validation": self.latest_val,
                "wall_s": self.wall(), "resumes": self.resumes, "debug_stop": self.debug_stop, **peak,
                "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                             "transformers": transformers.__version__, "repo_head": head},
                "precision": self.cfg["precision"], "timestamp": C.stamp()}

    def run(self, resume: bool = False, max_updates: int | None = None, stop_after_actions: int | None = None) -> dict:
        saved = C.recover_directory(self.state_dir)
        if saved is not None and not resume:
            raise RuntimeError(f"{self.state_dir} exists; pass --resume to continue this cell")
        if saved is not None:
            self.load_state(saved)
        else:
            for stale in (self.log_path, self.record):
                if stale is not None and stale.exists():
                    stale.unlink()
            self.save_ckpt(0.0)
            self.save_state()
        self._last_state_actions, self._last_state_time = self.actions, time.monotonic()
        C.atomic_json(self.out / "receipt.json", self.receipt("running"))
        cp = self.cfg["checkpoints"]
        session_actions = 0
        while not self.done():
            self.step()
            session_actions += 1
            if max_updates is not None and self.updates >= max_updates:
                self.debug_stop = f"max_updates={max_updates}"
            crossed = [f for f in cp["fractions"] if 0 < f < 1 and f not in self.milestones_done and self.progress() >= f]
            for fraction in crossed:
                self.save_ckpt(fraction)
            if crossed or self.actions - self._last_state_actions >= cp["state_every_actions"] \
                    or time.monotonic() - self._last_state_time >= cp["state_every_seconds"]:
                self.save_state()
            if stop_after_actions is not None and session_actions >= stop_after_actions and not self.done():
                self.save_state()
                receipt = self.receipt("interrupted")
                C.atomic_json(self.out / "receipt.json", receipt)
                return receipt
        if 1.0 not in self.milestones_done:
            self.save_ckpt(1.0)
        self.save_state()
        if self.record is not None:
            counts = {o: {"updates": 0, "examples": 0} for o in OBJECTIVES}
            for row in C.read_jsonl(self.record):
                counts[row["objective"]]["updates"] += 1
                counts[row["objective"]]["examples"] += len(row["ids"])
            C.atomic_json(Path(str(self.record) + ".meta.json"), {
                **self.identity(), "rows": self.schedule_lines, "counts": counts, "work_total": self.work.total,
                "work_by_purpose": self.work.by_purpose(), "record_sha256": C.sha256(self.record),
                "note": "train and probe updates are recorded; controller validation passes are not training actions"})
        receipt = self.receipt("debug_stopped" if self.debug_stop else "complete")
        C.atomic_json(self.out / "receipt.json", receipt)
        return receipt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--method", choices=C.METHODS)
    ap.add_argument("--replay-schedule", type=Path)
    ap.add_argument("--order", choices=["interleaved", "stage_sorted"])
    ap.add_argument("--record-schedule", type=Path)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", type=str)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", action="store_true", help="continue from <out>/state if present (safe on requeue)")
    ap.add_argument("--max-updates", type=int, help="debug only: finish after N updates (receipt says debug_stopped)")
    ap.add_argument("--stop-after-actions", type=int, help="test only: save state and exit as if preempted")
    a = ap.parse_args(argv)
    cfg, sha = load_config(a.config)
    trainer = Trainer(cfg, a.out, a.seed, method=a.method, config_sha=sha, replay=a.replay_schedule, order=a.order,
                      record=a.record_schedule)
    try:
        receipt = trainer.run(resume=a.resume, max_updates=a.max_updates, stop_after_actions=a.stop_after_actions)
    except BaseException as exc:
        C.atomic_json(trainer.out / "receipt.json", trainer.receipt("failed", "".join(
            traceback.format_exception_only(type(exc), exc)).strip()))
        raise
    print(json.dumps({k: receipt[k] for k in ("status", "method", "seed", "updates", "optimizer_steps", "work_total",
                                              "work_overshoot", "milestones_done", "wall_s")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
