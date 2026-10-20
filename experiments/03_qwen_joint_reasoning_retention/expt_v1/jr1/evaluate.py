#!/usr/bin/env python3
"""JR1 evaluator (Unified Training Exp 03 and 04): full and learning-curve suites for one checkpoint.

    python -m jr1.evaluate --ckpt <dir or hub id> --suite {full,lc} --out <dir> [--backend {vllm,hf}]

full: GSM8K test (1,319, greedy, jr1 parser), MATH-500 (500, math-verify), MMLU test (all items, 5-shot letter
      likelihood), ARC-Easy test (2,376, likelihood), WikiText-103 test NLL, held-out UltraFeedback preference ordering.
lc:   the sealed learning-curve subsets, 256 GSM8K + 512 subject-stratified MMLU (same protocols).

Inputs are the jr1/data.py files under $JR1_STATE_ROOT/data (or --data-root). Every protocol choice is in PROTOCOL
and copied into summary.json. Outputs: <out>/items_<bench>.jsonl, one row per item (never capped; parse failures and
truncations are rows like any other) and <out>/summary.json {bench: {n, correct, acc, parse_fail, truncated, ...},
wall_s, device, ...}.

Backends: generation (GSM8K, MATH-500) runs on vLLM (inside the Sherlock vLLM Apptainer image) or HF generate; both
get the same token-id prompts, greedy decoding, token cap and stop strings, and their text goes through the same
finish/parse code. Likelihood benchmarks always run through the HF forward pass, so those numbers do not depend on the
backend. Greedy decoding and fixed batch order make a rerun on the same hardware and software reproduce the items.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

# ---------------------------------------------------------------- protocol (frozen before any test output is read)

GSM8K_PREFIX = "Solve the problem. Show your reasoning, then put the final numeric answer after ####.\n\nProblem: "
MATH_PREFIX = "Solve the problem. Show your reasoning, then put the final answer in \\boxed{}.\n\nProblem: "
STOP = ["\nProblem:", "\nSolve the problem."]  # a base model otherwise writes a next made-up problem (and its ####)
MATH_VERIFY_PINS = {"math-verify": "0.9.0", "latex2sympy2_extended": "1.11.0"}  # a different install is refused
MMLU_SHOTS = 5
LETTERS = "ABCDEFGHIJ"

PROTOCOL = {
    "gsm8k": {"prompt": "jr1_campaign.prompt(question) (identical to jr1.data.prompt)", "decoding": "greedy",
              "max_new_tokens": 1024, "stop": STOP + ["<eos>"],
              "parser": "jr1_campaign.extract_answer: last number after the last '####', else last number in the text; "
                        "Decimal-normalized; gold parsed the same way from the released '#### n' answer"},
    "math500": {"prompt": "MATH_PREFIX + problem + '\\nAnswer:'", "decoding": "greedy", "max_new_tokens": 2048,
                "stop": STOP + ["<eos>"],
                "checker": f"math-verify=={MATH_VERIFY_PINS['math-verify']} (latex2sympy2_extended=="
                           f"{MATH_VERIFY_PINS['latex2sympy2_extended']}): verify(parse('\\\\boxed{{gold}}'), parse(completion)) "
                           "with default Latex+Expr extraction (boxed first); secondary column: Hendrycks et al. 2021 "
                           "is_equiv on the last \\boxed{}"},
    "mmlu": {"protocol": "lm-evaluation-harness v0.4 'mmlu' task (Hendrycks et al. 2021 format), 5-shot from the dev "
                         "split (first 5 per subject in release order), log-likelihood of ' A'..' D' after 'Answer:'",
             "metric": "acc (argmax log-likelihood; ties -> first choice); macro_acc over subjects also reported",
             "max_len": 4096, "truncation": "left (counted as truncated)"},
    "arc_easy": {"protocol": "lm-evaluation-harness v0.4 'arc_easy' task, 0-shot, 'Question: {q}\\nAnswer:' + "
                             "' {choice}' log-likelihood",
                 "metric": "acc = acc_norm (argmax log-likelihood / len(choice) in characters); unnormalized acc_raw kept",
                 "max_len": 4096},
    "wikitext": {"protocol": "rolling log-likelihood of every token of each test article exactly once; document "
                             "prefixed with <|endoftext|>; windows of 1024 tokens, the first 256 of each later window "
                             "are context only", "max_len": 1024, "context": 256,
                 "metric": "nll_per_token (nats), ppl, bits_per_byte"},
    "pref": {"protocol": "trainer encoding encode(prompt) + encode(response) + [eos], prompt kept at most max_len//2 "
                         "from the end when it alone fills max_len, then right-truncated to max_len; summed log-prob "
                         "of response tokens", "max_len": 1024,
             "metric": "acc = P(sum logp chosen > sum logp rejected); acc_norm with per-token means. Items keep the "
                       "log-probs, so DPO reference-adjusted accuracy is computed against the ckpt_000 run's items"},
}
for _b in ("lc_gsm8k",):
    PROTOCOL[_b] = PROTOCOL["gsm8k"]
PROTOCOL["lc_mmlu"] = PROTOCOL["mmlu"]

FILES = {"gsm8k": "eval/gsm8k_test", "math500": "eval/math500", "mmlu": "eval/mmlu_test", "arc_easy": "eval/arc_easy_test",
         "wikitext": "pt/test", "pref": "pref/test", "lc_gsm8k": "eval/lc_gsm8k", "lc_mmlu": "eval/lc_mmlu",
         "mmlu_dev": "eval/mmlu_dev"}
EXPECTED = {"gsm8k": 1319, "math500": 500, "mmlu": 14042, "arc_easy": 2376, "lc_gsm8k": 256, "lc_mmlu": 512,
            "mmlu_dev": 285}
SUITES = {"full": ["gsm8k", "math500", "mmlu", "arc_easy", "wikitext", "pref"], "lc": ["lc_gsm8k", "lc_mmlu"]}
GENERATIVE = {"gsm8k", "lc_gsm8k", "math500"}
MMLU_DEV = {"repo": "cais/mmlu", "file": "all/dev-00000-of-00001.parquet",
            "revision": "c30699e8356da336a370243923dbaf21066bb9fe"}  # same pinned revision as jr1/data.py SOURCES


def stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def log(msg: str) -> None:
    print(f"[jr1.evaluate {dt.datetime.now().strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- GSM8K: the jr1 parser
# The trainer's RL reward and this evaluator must parse identically, so the merged package uses jr1/common.py
# (trainer branch). On this branch alone, an attributed copy of jr1_campaign.py prompt(), NUMBER, extract_answer()
# and exact_reward() stands in; test_evaluate.py checks every available implementation agrees.
NUMBER = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:/\d+)?")


def _local_gsm8k_prompt(question: str) -> str:
    return GSM8K_PREFIX + question + "\nAnswer:"


def _local_extract_answer(text: str) -> str | None:
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


def _local_exact_reward(text: str, answer: str) -> float:
    got, gold = _local_extract_answer(text), _local_extract_answer(answer)
    return float(got is not None and gold is not None and got == gold)


try:
    from .common import exact_reward, extract_answer, prompt as gsm8k_prompt
    PARSER_SOURCE = "jr1.common (the trainer's RL reward functions)"
except ImportError:
    extract_answer, exact_reward, gsm8k_prompt = _local_extract_answer, _local_exact_reward, _local_gsm8k_prompt
    PARSER_SOURCE = "jr1.evaluate attributed copy of jr1_campaign.extract_answer (jr1.common absent)"


def score_gsm8k(row: dict, gen: dict) -> dict:
    pred, gold = extract_answer(gen["text"]), extract_answer(row["answer"])
    if gold is None:
        raise ValueError(f"{row['id']}: released GSM8K answer has no number")
    return {"id": row["id"], "pred": pred, "gold": gold, "parse_fail": pred is None,
            "correct": bool(exact_reward(gen["text"], row["answer"])), **gen}


# ---------------------------------------------------------------- MATH-500: math-verify + Hendrycks is_equiv
# Hendrycks et al. 2021 normalization, attributed copy of hendrycks/math modeling/math_equivalence.py and
# last_boxed_only_string / remove_boxed (MIT License). Kept as a secondary column to validate the primary checker.

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if substr and substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a, b = substr[0], substr[1]
                if b != "{":
                    new_str += "{" + a + "}{" + b + "}" + substr[2:]
                else:
                    new_str += "{" + a + "}" + b + substr[2:]
    return new_str


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        a, b = int(a), int(b)
        assert string == f"{a}/{b}"
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except (ValueError, AssertionError):
        return string


def _remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def _strip_string(string):
    string = string.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "").replace("%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    return _fix_a_slash_b(string)


def hendrycks_equiv(str1: str | None, str2: str | None) -> bool:
    if str1 is None or str2 is None:
        return False
    try:
        return _strip_string(str1) == _strip_string(str2)
    except Exception:  # the reference implementation falls back to raw comparison on any normalization error
        return str1 == str2


def last_boxed_only_string(string: str) -> str | None:
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    depth, i = 0, idx
    while i < len(string):
        if string[i] == "{":
            depth += 1
        elif string[i] == "}":
            depth -= 1
            if depth == 0:
                return string[idx:i + 1]
        i += 1
    return None


def remove_boxed(s: str | None) -> str | None:
    for left in ("\\boxed{", "\\fbox{"):
        if s and s.startswith(left) and s.endswith("}"):
            return s[len(left):-1]
    return None


def math_verify_module():
    import math_verify
    from importlib.metadata import version
    for pkg, pin in MATH_VERIFY_PINS.items():
        if version(pkg) != pin:
            raise RuntimeError(f"{pkg} {version(pkg)} installed; the protocol pins {pin}")
    return math_verify


def math_prompt(problem: str) -> str:
    return MATH_PREFIX + problem + "\nAnswer:"


def score_math(row: dict, gen: dict) -> dict:
    mv = math_verify_module()
    gold = mv.parse("\\boxed{" + row["answer"] + "}")
    pred = mv.parse(gen["text"])
    correct = bool(gold and pred and mv.verify(gold, pred))
    boxed = remove_boxed(last_boxed_only_string(gen["text"]))
    return {"id": row["id"], "pred": str(pred[1]) if len(pred) > 1 else (str(pred[0]) if pred else None),
            "gold": row["answer"], "gold_parse_fail": not gold, "parse_fail": not pred, "correct": correct,
            "boxed": boxed, "hendrycks_correct": hendrycks_equiv(boxed, row["answer"]), **gen}


# ---------------------------------------------------------------- generation: shared finish rule + two backends

def finish(text: str, n_tokens: int, hit_eos: bool, max_new: int, stop: list[str] = STOP) -> dict:
    """Cut at the first stop string; a completion is truncated only if it reached the cap with no EOS or stop."""
    cut = min((i for i in (text.find(s) for s in stop) if i >= 0), default=-1)
    if cut >= 0:
        return {"text": text[:cut], "gen_tokens": n_tokens, "finish": "stop", "truncated": False}
    if hit_eos:
        return {"text": text, "gen_tokens": n_tokens, "finish": "eos", "truncated": False}
    return {"text": text, "gen_tokens": n_tokens, "finish": "length" if n_tokens >= max_new else "other",
            "truncated": n_tokens >= max_new}


class HFGenerator:
    def __init__(self, model, tok, device, batch: int):
        self.model, self.tok, self.device, self.batch = model, tok, device, batch

    def generate(self, prompts: list[list[int]], max_new: int) -> list[dict]:
        import torch
        tok, eos, pad = self.tok, self.tok.eos_token_id, self.tok.pad_token_id
        order = sorted(range(len(prompts)), key=lambda i: (-len(prompts[i]), i))
        out: list[dict | None] = [None] * len(prompts)
        for s in range(0, len(order), self.batch):
            idx = order[s:s + self.batch]
            width = max(len(prompts[i]) for i in idx)
            ids = torch.tensor([[pad] * (width - len(prompts[i])) + prompts[i] for i in idx], device=self.device)
            attn = torch.tensor([[0] * (width - len(prompts[i])) + [1] * len(prompts[i]) for i in idx],
                                device=self.device)
            with torch.no_grad():
                gen = self.model.generate(input_ids=ids, attention_mask=attn, do_sample=False, temperature=None,
                                          top_p=None, top_k=None, max_new_tokens=max_new, pad_token_id=pad,
                                          eos_token_id=eos, stop_strings=STOP, tokenizer=tok, use_cache=True)
            for row, i in zip(gen[:, width:].tolist(), idx):
                hit = eos in row
                row = row[:row.index(eos) + 1] if hit else row
                text = tok.decode([t for t in row if t != eos], skip_special_tokens=True)
                out[i] = finish(text, len(row), hit, max_new)
            log(f"generated {min(s + self.batch, len(order))}/{len(order)}")
        return out


class VLLMGenerator:
    def __init__(self, llm):
        self.llm = llm

    @classmethod
    def load(cls, ckpt: str, tokenizer: str, mem: float, max_len: int):
        from vllm import LLM
        return cls(LLM(model=ckpt, tokenizer=tokenizer, dtype="bfloat16", seed=0, gpu_memory_utilization=mem,
                       max_model_len=max_len, trust_remote_code=False))

    def generate(self, prompts: list[list[int]], max_new: int) -> list[dict]:
        from vllm import SamplingParams
        params = SamplingParams(temperature=0.0, max_tokens=max_new, stop=STOP, seed=0, skip_special_tokens=True)
        res = self.llm.generate([{"prompt_token_ids": p} for p in prompts], params, use_tqdm=False)
        return [vllm_finish(r.outputs[0], max_new) for r in res]


def vllm_finish(o, max_new: int) -> dict:
    # vLLM drops the stop string from .text and says why it finished; map it onto the HF rule.
    n = len(o.token_ids)
    if o.finish_reason == "stop" and isinstance(o.stop_reason, str):
        return {"text": o.text, "gen_tokens": n, "finish": "stop", "truncated": False}
    return finish(o.text, n, o.finish_reason == "stop", max_new)


# ---------------------------------------------------------------- likelihood scorer (HF forward)

def encode_pair(tok, context: str, continuation: str) -> tuple[list[int], list[int]]:
    """lm-evaluation-harness v0.4 HFLM._encode_pair: trailing context spaces move to the continuation."""
    n = len(context) - len(context.rstrip())
    if n:
        continuation, context = context[-n:] + continuation, context[:-n]
    whole = tok.encode(context + continuation, add_special_tokens=False)
    ctx = tok.encode(context, add_special_tokens=False)
    return ctx, whole[len(ctx):]


class Scorer:
    """Sum of log p(continuation | context) for many (context ids, continuation ids) requests.

    Requests whose continuation is one token and whose (left-truncated) context is identical share one forward pass
    (MMLU's four letters). Otherwise one sequence per request. Inputs longer than max_len lose tokens on the left.
    """

    def __init__(self, model, device, max_tokens: int, max_batch: int = 64, head_chunk: int = 2048):
        # head_chunk: scored positions per output-head call. Full-vocab fp32 log-probs for every position of a
        # 32k-token WikiText batch would be ~20 GB (152k vocab); chunking keeps it at ~1.2 GB, numbers unchanged.
        self.model, self.device, self.max_tokens, self.max_batch = model, device, max_tokens, max_batch
        self.head_chunk = head_chunk
        self.base = getattr(model, model.base_model_prefix)
        self.head = model.get_output_embeddings()
        self.pad = 0

    def units(self, reqs: list[tuple[list[int], list[int]]], max_len: int):
        units, shared = [], {}
        trunc = [False] * len(reqs)
        for r, (ctx, cont) in enumerate(reqs):
            if not ctx or not cont:
                raise ValueError("every request needs a non-empty context and continuation")
            if len(cont) > max_len:
                raise ValueError(f"continuation of {len(cont)} tokens exceeds max_len {max_len}")
            if len(cont) == 1:
                inp = ctx[-max_len:]
                trunc[r] = len(ctx) > max_len
                key = tuple(inp)
                if key not in shared:
                    shared[key] = len(units)
                    units.append((inp, []))
                units[shared[key]][1].append((len(inp) - 1, cont[0], r))
            else:
                seq = (ctx + cont)[:-1]
                drop = max(0, len(seq) - max_len)
                trunc[r] = drop > 0
                start = len(ctx) - 1 - drop
                units.append((seq[drop:], [(start + j, t, r) for j, t in enumerate(cont)]))
        return units, trunc

    def loglik(self, reqs: list[tuple[list[int], list[int]]], max_len: int) -> list[dict]:
        import torch
        import torch.nn.functional as F
        units, trunc = self.units(reqs, max_len)
        lp = [0.0] * len(reqs)
        greedy = [True] * len(reqs)
        order = sorted(range(len(units)), key=lambda u: (-len(units[u][0]), u))
        s, every, nxt = 0, max(1, len(order) // 10), 0
        while s < len(order):
            if s >= nxt and len(order) > 1:
                log(f"scored {s}/{len(order)} sequences")
                nxt = s + every
            width = len(units[order[s]][0])
            k = max(1, min(self.max_batch, self.max_tokens // max(width, 1)))
            idx = order[s:s + k]
            s += k
            ids = torch.tensor([units[u][0] + [self.pad] * (width - len(units[u][0])) for u in idx], device=self.device)
            attn = torch.tensor([[1] * len(units[u][0]) + [0] * (width - len(units[u][0])) for u in idx],
                                device=self.device)
            rows, pos, tgt, req = [], [], [], []
            for b, u in enumerate(idx):
                for p, t, r in units[u][1]:
                    rows.append(b)
                    pos.append(p)
                    tgt.append(t)
                    req.append(r)
            with torch.no_grad():
                h = self.base(input_ids=ids, attention_mask=attn, use_cache=False).last_hidden_state
                got, top = [], []
                for c in range(0, len(rows), self.head_chunk):
                    sl = slice(c, c + self.head_chunk)
                    logits = self.head(h[torch.tensor(rows[sl], device=self.device),
                                         torch.tensor(pos[sl], device=self.device)])
                    logp = F.log_softmax(logits.float(), dim=-1)
                    t = torch.tensor(tgt[sl], device=self.device)
                    got += logp.gather(-1, t[:, None]).squeeze(-1).tolist()
                    top += logp.argmax(-1).eq(t).tolist()
                    del logits, logp
            for r, g, ok in zip(req, got, top):
                lp[r] += g
                greedy[r] = greedy[r] and ok
        return [{"logprob": lp[r], "tokens": len(reqs[r][1]), "greedy": greedy[r], "truncated": trunc[r]}
                for r in range(len(reqs))]


def argmax_first(xs: list[float]) -> int:
    return max(range(len(xs)), key=lambda i: (xs[i], -i))


# ---------------------------------------------------------------- likelihood benchmarks

def mmlu_example(row: dict) -> str:
    return row["question"].strip() + "".join(f"\n{LETTERS[i]}. {c}" for i, c in enumerate(row["choices"])) + "\nAnswer:"


def mmlu_context(row: dict, shots: list[dict]) -> str:
    head = f"The following are multiple choice questions (with answers) about {row['subject'].replace('_', ' ')}.\n\n"
    body = "".join(mmlu_example(s) + " " + LETTERS[s["answer_idx"]] + "\n\n" for s in shots)
    return head + body + mmlu_example(row)


def mmlu_shots(dev: list[dict], k: int = MMLU_SHOTS) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = {}
    for r in dev:
        by.setdefault(r["subject"], []).append(r)
    return {s: rows[:k] for s, rows in by.items()}


def eval_mmlu(rows, dev, tok, scorer) -> list[dict]:
    shots = mmlu_shots(dev)
    reqs, spans = [], []
    for r in rows:
        if r["subject"] not in shots or len(shots[r["subject"]]) < MMLU_SHOTS:
            raise ValueError(f"no {MMLU_SHOTS} dev shots for subject {r['subject']}")
        ctx = mmlu_context(r, shots[r["subject"]])
        spans.append(len(reqs))
        reqs += [encode_pair(tok, ctx, " " + LETTERS[i]) for i in range(len(r["choices"]))]
    res = scorer.loglik(reqs, PROTOCOL["mmlu"]["max_len"])
    out = []
    for r, s in zip(rows, spans):
        part = res[s:s + len(r["choices"])]
        lps = [p["logprob"] for p in part]
        pred = argmax_first(lps)
        out.append({"id": r["id"], "subject": r["subject"], "logprobs": lps, "pred": pred, "gold": r["answer_idx"],
                    "correct": pred == r["answer_idx"], "parse_fail": False,
                    "truncated": any(p["truncated"] for p in part), "ctx_tokens": len(reqs[s][0])})
    return out


def eval_arc(rows, tok, scorer) -> list[dict]:
    reqs, spans = [], []
    for r in rows:
        spans.append(len(reqs))
        reqs += [encode_pair(tok, f"Question: {r['question']}\nAnswer:", " " + c) for c in r["choices"]]
    res = scorer.loglik(reqs, PROTOCOL["arc_easy"]["max_len"])
    out = []
    for r, s in zip(rows, spans):
        part = res[s:s + len(r["choices"])]
        lps = [p["logprob"] for p in part]
        norm = [lp / len(c) for lp, c in zip(lps, r["choices"])]
        raw, pred = argmax_first(lps), argmax_first(norm)
        out.append({"id": r["id"], "logprobs": lps, "char_lens": [len(c) for c in r["choices"]], "pred": pred,
                    "pred_raw": raw, "gold": r["answer_idx"], "correct": pred == r["answer_idx"],
                    "correct_raw": raw == r["answer_idx"], "parse_fail": False,
                    "truncated": any(p["truncated"] for p in part)})
    return out


def rolling_requests(ids: list[int], eos: int, max_len: int, context: int) -> list[tuple[list[int], list[int]]]:
    """Requests that score every token of ids exactly once: (up to `context` previous tokens, next window)."""
    stream = [eos] + ids
    step = max_len - context
    reqs, start = [], 1
    while start < len(stream):
        end = min(len(stream), start + step)
        reqs.append((stream[max(0, start - context):start], stream[start:end]))
        start = end
    return reqs


def eval_wikitext(rows, tok, scorer) -> list[dict]:
    p = PROTOCOL["wikitext"]
    reqs, spans = [], []
    for r in rows:
        spans.append((len(reqs), None))
        reqs += rolling_requests(tok.encode(r["text"], add_special_tokens=False), tok.eos_token_id,
                                 p["max_len"], p["context"])
        spans[-1] = (spans[-1][0], len(reqs))
    res = scorer.loglik(reqs, p["max_len"])
    out = []
    for r, (a, b) in zip(rows, spans):
        part = res[a:b]
        out.append({"id": r["id"], "tokens": sum(x["tokens"] for x in part), "bytes": len(r["text"].encode("utf-8")),
                    "nll_sum": -sum(x["logprob"] for x in part), "windows": b - a, "parse_fail": False,
                    "truncated": any(x["truncated"] for x in part)})
    return out


def pref_pair(tok, prompt: str, response: str, max_len: int) -> tuple[list[int], list[int], bool]:
    """Same token sequence as jr1/train.py Trainer._prompt_response (the DPO objective's encoding)."""
    p = tok.encode(prompt, add_special_tokens=False)
    r = tok.encode(response, add_special_tokens=False) + [tok.eos_token_id]
    cut = len(p) >= max_len
    if cut:
        p = p[-(max_len // 2):]
    seq = (p + r)[:max_len]
    return p, seq[len(p):], cut or len(p) + len(r) > max_len


def eval_pref(rows, tok, scorer) -> list[dict]:
    m = PROTOCOL["pref"]["max_len"]
    reqs, cut = [], []
    for r in rows:
        for key in ("chosen", "rejected"):
            p, c, t = pref_pair(tok, r["prompt"], r[key], m)
            reqs.append((p, c))
            cut.append(t)
    res = scorer.loglik(reqs, m)
    out = []
    for i, r in enumerate(rows):
        c, j = res[2 * i], res[2 * i + 1]
        out.append({"id": r["id"], "lp_chosen": c["logprob"], "lp_rejected": j["logprob"], "n_chosen": c["tokens"],
                    "n_rejected": j["tokens"], "correct": c["logprob"] > j["logprob"],
                    "correct_norm": c["logprob"] / c["tokens"] > j["logprob"] / j["tokens"], "parse_fail": False,
                    "truncated": cut[2 * i] or cut[2 * i + 1]})
    return out


# ---------------------------------------------------------------- summaries

def summarize(bench: str, items: list[dict], wall: float) -> dict:
    n = len(items)
    s = {"n": n, "expected_n": EXPECTED.get(bench), "complete": EXPECTED.get(bench) in (None, n),
         "parse_fail": sum(bool(x.get("parse_fail")) for x in items),
         "truncated": sum(bool(x.get("truncated")) for x in items), "wall_s": round(wall, 3)}
    if bench == "wikitext":
        tokens, nll, nbytes = sum(x["tokens"] for x in items), sum(x["nll_sum"] for x in items), sum(x["bytes"] for x in items)
        s.update(correct=None, acc=None, tokens=tokens, bytes=nbytes, nll_per_token=nll / max(tokens, 1),
                 ppl=math.exp(nll / max(tokens, 1)), bits_per_byte=nll / math.log(2) / max(nbytes, 1))
        return s
    correct = sum(bool(x["correct"]) for x in items)
    s.update(correct=correct, acc=correct / n if n else None)
    if bench in GENERATIVE:
        s["mean_gen_tokens"] = sum(x["gen_tokens"] for x in items) / n if n else None
    if bench == "math500":
        s["hendrycks_correct"] = sum(x["hendrycks_correct"] for x in items)
        s["gold_parse_fail"] = sum(x["gold_parse_fail"] for x in items)
        s["checkers_agree"] = sum(x["hendrycks_correct"] == x["correct"] for x in items)
    if bench in ("mmlu", "lc_mmlu"):
        by: dict[str, list[bool]] = {}
        for x in items:
            by.setdefault(x["subject"], []).append(x["correct"])
        s["subjects"] = len(by)
        s["macro_acc"] = sum(sum(v) / len(v) for v in by.values()) / len(by) if by else None
        s["subject_acc"] = {k: sum(v) / len(v) for k, v in sorted(by.items())}
    if bench == "arc_easy":
        s["acc_raw"] = sum(x["correct_raw"] for x in items) / n if n else None
    if bench == "pref":
        s["acc_norm"] = sum(x["correct_norm"] for x in items) / n if n else None
    return s


# ---------------------------------------------------------------- I/O

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(root: Path, name: str) -> list[dict]:
    with open(root / f"{FILES[name]}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_items(path: Path, items: list[dict]) -> None:
    atomic_write(path, "".join(json.dumps(x, sort_keys=True, ensure_ascii=False) + "\n" for x in items))


def write_summary(path: Path, summary: dict) -> None:
    atomic_write(path, json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")


def ensure_mmlu_dev(root: Path) -> dict:
    """eval/mmlu_dev from jr1/data.py when present; otherwise the pinned cais/mmlu dev parquet, written beside it."""
    path = root / f"{FILES['mmlu_dev']}.jsonl"
    if path.exists():
        return {"source": "jr1/data.py", "sha256": sha256_file(path)}
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    local = hf_hub_download(MMLU_DEV["repo"], MMLU_DEV["file"], repo_type="dataset", revision=MMLU_DEV["revision"])
    table = pq.read_table(local).to_pylist()
    rows = [{"id": f"mmlu:dev:{i:05d}", "subject": r["subject"], "question": r["question"],
             "choices": list(r["choices"]), "answer_idx": int(r["answer"])} for i, r in enumerate(table)]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_items(path, rows)
    return {"source": f"{MMLU_DEV['repo']}@{MMLU_DEV['revision']}/{MMLU_DEV['file']}", "sha256": sha256_file(path)}


def ckpt_fingerprint(ckpt: str) -> dict:
    p = Path(ckpt)
    if not p.is_dir():
        return {"hub_id": ckpt}
    files = sorted(p.glob("*.safetensors")) + sorted(p.glob("config.json"))
    return {"path": str(p.resolve()), "files": {f.name: sha256_file(f) for f in files}}


# ---------------------------------------------------------------- main

def load_hf(ckpt: str, tokenizer: str, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=dtype, trust_remote_code=False).to(device).eval()
    return model, tok


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True, help="HF save_pretrained directory or hub id (untouched start model)")
    ap.add_argument("--suite", choices=sorted(SUITES), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--data-root", help="default $JR1_STATE_ROOT/data")
    ap.add_argument("--tokenizer", help="default: the checkpoint's own tokenizer files, else Qwen/Qwen2.5-1.5B")
    ap.add_argument("--device", help="default cuda:0 if available, else cpu")
    ap.add_argument("--gen-batch", type=int, default=64, help="HF generate batch size (no effect on vLLM)")
    ap.add_argument("--lik-tokens", type=int, default=32768, help="padded tokens per likelihood forward batch")
    ap.add_argument("--head-chunk", type=int, default=2048,
                    help="scored positions per output-head call (memory only; numbers unchanged)")
    ap.add_argument("--vllm-mem", type=float, default=0.45, help="vLLM GPU memory share; HF scorer uses the rest")
    ap.add_argument("--resume", action="store_true", help="keep benchmarks already finished in <out>/summary.json")
    ap.add_argument("--only", help="comma list: run just these benchmarks of the suite now (status stays 'partial' "
                                   "until --resume runs have filled every benchmark of the suite)")
    a = ap.parse_args(argv)

    import torch
    t0 = time.monotonic()
    if a.data_root:
        root = Path(a.data_root)
    elif "JR1_STATE_ROOT" in os.environ:
        root = Path(os.environ["JR1_STATE_ROOT"]) / "data"
    else:
        raise RuntimeError("pass --data-root or set JR1_STATE_ROOT")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    tokenizer = a.tokenizer or (a.ckpt if Path(a.ckpt, "tokenizer_config.json").exists() or not Path(a.ckpt).is_dir()
                                else "Qwen/Qwen2.5-1.5B")
    benches = SUITES[a.suite]
    only = set(a.only.split(",")) if a.only else set(benches)
    if not only <= set(benches):
        raise ValueError(f"--only {sorted(only - set(benches))} not in suite {a.suite}: {benches}")
    if "math500" in only:
        math_verify_module()  # fail before any compute if the pinned checker is missing

    data = {b: read_rows(root, b) for b in benches}
    files = {b: sha256_file(root / f"{FILES[b]}.jsonl") for b in benches}
    if any(b in ("mmlu", "lc_mmlu") for b in benches):
        files["mmlu_dev"] = ensure_mmlu_dev(root)["sha256"]
        data["mmlu_dev"] = read_rows(root, "mmlu_dev")

    sp = out / "summary.json"
    prev = json.loads(sp.read_text()) if a.resume and sp.exists() else {}
    if prev and (prev.get("ckpt") != a.ckpt or prev.get("data_sha256") != files or prev.get("protocol") != PROTOCOL):
        raise RuntimeError("--resume found a summary for a different checkpoint, data or protocol")
    import transformers
    versions = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__}
    try:
        from importlib.metadata import version
        versions.update({k: version(k) for k in MATH_VERIFY_PINS})
        if a.backend == "vllm":
            versions["vllm"] = version("vllm")
    except Exception:
        pass
    meta = Path(a.ckpt, "jr1_ckpt.json")
    summary = {"suite": a.suite, "parser_source": PARSER_SOURCE,
               "ckpt_meta": json.loads(meta.read_text()) if meta.exists() else None, "ckpt": a.ckpt,
               "ckpt_fingerprint": prev.get("ckpt_fingerprint") or ckpt_fingerprint(a.ckpt),
               "tokenizer": tokenizer, "backend": a.backend, "data_root": str(root), "data_sha256": files,
               "protocol": PROTOCOL, "versions": versions, "started": prev.get("started", stamp()),
               "device": (torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "cpu")
               + f" ({device})", "status": "running", "benches": benches}
    for b in benches:
        if b in prev and (out / f"items_{b}.jsonl").exists():
            summary[b] = prev[b]
    write_summary(sp, summary)

    todo = [b for b in benches if b not in summary and b in only]
    gen_todo = [b for b in todo if b in GENERATIVE]
    model = tok = None
    if gen_todo and a.backend == "vllm":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=False)
        generator = VLLMGenerator.load(a.ckpt, tokenizer, a.vllm_mem, max_len=4096)
    if (todo and a.backend == "hf") or any(b not in GENERATIVE for b in todo):
        log(f"loading {a.ckpt} on {device}")
        model, tok = load_hf(a.ckpt, tokenizer, device)
    if gen_todo and a.backend == "hf":
        generator = HFGenerator(model, tok, device, a.gen_batch)
    scorer = Scorer(model, device, a.lik_tokens, head_chunk=a.head_chunk) if model is not None else None
    if scorer is not None:
        scorer.pad = tok.pad_token_id

    for b in todo:
        tb = time.monotonic()
        rows = data[b]
        log(f"{b}: {len(rows)} items")
        if b in ("gsm8k", "lc_gsm8k"):
            prompts = [tok.encode(gsm8k_prompt(r["question"]), add_special_tokens=False) for r in rows]
            gens = generator.generate(prompts, PROTOCOL["gsm8k"]["max_new_tokens"])
            items = [score_gsm8k(r, g) | {"prompt_tokens": len(p)} for r, g, p in zip(rows, gens, prompts)]
        elif b == "math500":
            prompts = [tok.encode(math_prompt(r["problem"]), add_special_tokens=False) for r in rows]
            gens = generator.generate(prompts, PROTOCOL["math500"]["max_new_tokens"])
            items = [score_math(r, g) | {"prompt_tokens": len(p)} for r, g, p in zip(rows, gens, prompts)]
        elif b in ("mmlu", "lc_mmlu"):
            items = eval_mmlu(rows, data["mmlu_dev"], tok, scorer)
        elif b == "arc_easy":
            items = eval_arc(rows, tok, scorer)
        elif b == "wikitext":
            items = eval_wikitext(rows, tok, scorer)
        elif b == "pref":
            items = eval_pref(rows, tok, scorer)
        else:
            raise ValueError(b)
        if len(items) != len(rows):
            raise AssertionError(f"{b}: {len(items)} items for {len(rows)} rows")
        write_items(out / f"items_{b}.jsonl", items)
        summary[b] = summarize(b, items, time.monotonic() - tb)
        write_summary(sp, summary)
        print(json.dumps({"bench": b, **{k: summary[b].get(k) for k in ("n", "acc", "parse_fail", "truncated", "wall_s")}}),
              flush=True)

    done = all(b in summary for b in benches)
    summary.update(status="complete" if done else "partial", finished=stamp(),
                   wall_s=round(time.monotonic() - t0 + (prev.get("wall_s") or 0.0), 3),
                   complete=done and all(summary[b]["complete"] for b in benches))
    if device.type == "cuda":
        summary["peak_mem_gb"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
    write_summary(sp, summary)
    return summary


if __name__ == "__main__":
    main()
