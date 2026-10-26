"""Exp 05 scratch evaluation: jr1.evaluate's protocols on the p05 data root, plus LAMBADA, in fp32 with HF only.

    python -m p05.evaluate --ckpt <dir> --suite {scratch_full,scratch_lc} --out <dir> [--data-root DIR] [--resume]

scratch_full (ckpt_000 and ckpt_100): WikiText-103 test NLL (jr1 rolling 1024/256 protocol), LAMBADA (OpenAI
  variant, 5,153), ARC-Easy (2,376, 0-shot likelihood), GSM8K test (all 1,319, greedy, jr1 parser; reported even when
  every item is wrong, since scratch math at a floor is the expected language-learning result), UltraFeedback
  held-out preference ordering.
scratch_lc (ckpt_000/025/050/075/100): WikiText test NLL, LAMBADA, ARC-Easy and the sealed 256-item GSM8K subset.
MMLU and MATH-500 are not run: a 14m-160m model trained from scratch on ~10^8 tokens is at chance on both, and the
GSM8K floor already carries the reasoning endpoint.

LAMBADA protocol (lm-evaluation-harness v0.4 'lambada_openai'): context = text up to the last space, continuation =
' ' + last word; acc = every continuation token is the greedy argmax; ppl = exp(-mean summed log-prob).
Models load in fp32 (no bf16 on V100, and fp32 removes fp16 overflow from the numbers); generation uses HF generate
(jr1.evaluate.HFGenerator), never vLLM.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

from jr1 import evaluate as E

E.FILES.update({"lambada": "eval/lambada"})
E.EXPECTED.update({"lambada": 5153})
E.PROTOCOL["lambada"] = {"protocol": "lm-evaluation-harness v0.4 lambada_openai: context = text.rsplit(' ', 1)[0], "
                                     "continuation = ' ' + last word, log-likelihood", "max_len": 2048,
                         "metric": "acc (all continuation tokens greedy), ppl = exp(-mean logprob)"}
E.SUITES.update({"scratch_full": ["wikitext", "arc_easy", "gsm8k", "pref"],
                 "scratch_lc": ["wikitext", "arc_easy", "lc_gsm8k"]})
LAMBADA_SUITES = {"scratch_full", "scratch_lc"}


def _load_hf_fp32(ckpt: str, tokenizer: str, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    tok = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.float32, trust_remote_code=False).to(device).eval()
    return model, tok


E.load_hf = _load_hf_fp32


def lambada_requests(tok, rows):
    reqs = []
    for r in rows:
        ctx, word = r["text"].rsplit(" ", 1)
        reqs.append(E.encode_pair(tok, ctx, " " + word))
    return reqs


def eval_lambada(ckpt: str, out: Path, root: Path, device_name: str | None, resume: bool, lik_tokens: int) -> dict:
    import torch
    sp = out / "summary.json"
    summary = json.loads(sp.read_text())
    if resume and "lambada" in summary and (out / "items_lambada.jsonl").exists():
        return summary
    t0 = time.monotonic()
    rows = E.read_rows(root, "lambada")
    device = torch.device(device_name or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model, tok = _load_hf_fp32(ckpt, summary["tokenizer"], device)
    scorer = E.Scorer(model, device, lik_tokens)
    scorer.pad = tok.pad_token_id
    res = scorer.loglik(lambada_requests(tok, rows), E.PROTOCOL["lambada"]["max_len"])
    items = [{"id": r["id"], "logprob": x["logprob"], "tokens": x["tokens"], "correct": x["greedy"],
              "parse_fail": False, "truncated": x["truncated"]} for r, x in zip(rows, res)]
    E.write_items(out / "items_lambada.jsonl", items)
    s = E.summarize("lambada", items, time.monotonic() - t0)
    s["ppl"] = math.exp(-sum(x["logprob"] for x in items) / max(len(items), 1))
    s["nll_per_token"] = -sum(x["logprob"] for x in items) / max(sum(x["tokens"] for x in items), 1)
    summary = json.loads(sp.read_text())
    summary["lambada"] = s
    summary["data_sha256"]["lambada"] = E.sha256_file(root / "eval/lambada.jsonl")
    summary["benches"] = list(summary["benches"]) + ["lambada"]
    summary["complete"] = bool(summary.get("complete")) and s["complete"]
    E.write_summary(sp, summary)
    print(json.dumps({"bench": "lambada", "n": s["n"], "acc": s["acc"], "ppl": round(s["ppl"], 3)}), flush=True)
    return summary


def main(argv: list[str] | None = None) -> dict:
    import argparse
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--ckpt")
    ap.add_argument("--suite")
    ap.add_argument("--out")
    ap.add_argument("--data-root")
    ap.add_argument("--device")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--lik-tokens", type=int, default=32768)
    known, _ = ap.parse_known_args(argv)
    if "--backend" in argv:
        raise SystemExit("p05.evaluate always uses the HF backend")
    if known.resume and known.suite in LAMBADA_SUITES:
        # jr1's resume check compares the benchmark list; take LAMBADA out before it runs, put it back after.
        sp = Path(known.out) / "summary.json"
        if sp.exists():
            s = json.loads(sp.read_text())
            if "lambada" in s.get("data_sha256", {}):
                s["data_sha256"].pop("lambada")
                sp.write_text(json.dumps(s, indent=2, sort_keys=True) + "\n")
    summary = E.main(argv + ["--backend", "hf"])
    if known.suite in LAMBADA_SUITES:
        root = Path(known.data_root) if known.data_root else Path(summary["data_root"])
        summary = eval_lambada(known.ckpt, Path(known.out), root, known.device, known.resume, known.lik_tokens)
    return summary


if __name__ == "__main__":
    main()
