"""Deterministic CPU tests for jr1/evaluate.py (no network, no downloads): parsers, finish rules, the likelihood
scorer against a brute-force forward, rolling NLL coverage, prompt formats and an end-to-end run on a tiny model.

    cd experiments/03_qwen_joint_reasoning_retention/expt_v1 && python -m pytest -q jr1/test_evaluate.py
"""
from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from jr1 import evaluate as E

EOS = "<|endoftext|>"
HERE = Path(__file__).resolve().parent


# Same construction as jr1/tiny.py on the trainer branch: byte-level BPE without merges, random tiny Qwen2.
def tiny_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    vocab = {ch: i for i, ch in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    vocab[EOS] = len(vocab)
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens([EOS])
    return PreTrainedTokenizerFast(tokenizer_object=tok, eos_token=EOS, pad_token=EOS)


def tiny_model(seed: int = 0):
    from transformers import AutoModelForCausalLM, Qwen2Config
    tok = tiny_tokenizer()
    cfg = Qwen2Config(vocab_size=260, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=8192, tie_word_embeddings=True, use_cache=True)
    cfg.eos_token_id = cfg.pad_token_id = cfg.bos_token_id = tok.eos_token_id
    torch.manual_seed(seed)
    return AutoModelForCausalLM.from_config(cfg, attn_implementation="eager").eval(), tok


# ---------------------------------------------------------------- GSM8K parser

CASES = ["#### 1,234", "so 3 + 4 = 7\n#### 7", "x is 2 then #### 5 and later 9", "the answer is 3/4", "-0.50",
         "no numbers here", "#### ", "1/0", "12.", ".5 apples", "#### 18 dollars\n#### 19", "1e3"]


def test_gsm8k_parser_cases():
    got = [E.extract_answer(c) for c in CASES]
    assert got == ["1234", "7", "9", "0.75", "-0.5", None, None, None, "12", "0.5", "19", "3"]
    assert E.exact_reward("We get 72.\n#### 72", "... #### 72") == 1.0
    assert E.exact_reward("#### 72.0", "#### 72") == 1.0
    assert E.exact_reward("I don't know", "#### 72") == 0.0


def load_campaign(tmp_path, monkeypatch):
    monkeypatch.setenv("JR1_STATE_ROOT", str(tmp_path / "state"))
    spec = importlib.util.spec_from_file_location("jr1_campaign_copy", HERE.parent / "jr1_campaign.py")
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mod)  # its dataclass looks itself up there
    try:
        spec.loader.exec_module(mod)
    except ImportError as e:  # needs datasets
        pytest.skip(f"jr1_campaign not importable here: {e}")
    return mod


def test_parser_and_prompt_match_jr1_campaign(tmp_path, monkeypatch):
    camp = load_campaign(tmp_path, monkeypatch)
    rng = random.Random(0)
    pieces = ["####", " ", "1,000", "3/4", "-2", ".5", "x", "\n", "7.", "1/0", "dollars", "#"]
    texts = CASES + ["".join(rng.choice(pieces) for _ in range(rng.randint(0, 12))) for _ in range(500)]
    for t in texts:
        assert E.extract_answer(t) == camp.extract_answer(t) == E._local_extract_answer(t), t
    assert E.gsm8k_prompt("Q?") == camp.prompt("Q?") == E._local_gsm8k_prompt("Q?")


def test_parser_matches_package_modules_when_present():
    for name, fns in (("jr1.common", ("extract_answer", "prompt")), ("jr1.data", ("prompt",))):
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue  # merged in the Sherlock packaging branch; checked there
        if "extract_answer" in fns:
            assert all(mod.extract_answer(t) == E.extract_answer(t) for t in CASES)
        assert mod.prompt("Q?") == E.gsm8k_prompt("Q?")


# ---------------------------------------------------------------- finish rules (shared by both backends)

def test_finish_rules():
    assert E.finish("#### 4\nProblem: next", 9, False, 1024) == {"text": "#### 4", "gen_tokens": 9, "finish": "stop",
                                                                 "truncated": False}
    assert E.finish("#### 4", 5, True, 1024)["finish"] == "eos"
    echo = E.finish("so 540 meters.\n\n\nSolve the problem. Show your reasoning, then ... ####.\n", 81, True, 1024)
    assert echo["text"] == "so 540 meters.\n\n" and E.extract_answer(echo["text"]) == E.extract_answer("#### 540")
    t = E.finish("going on", 1024, False, 1024)
    assert t["truncated"] and t["finish"] == "length"
    assert not E.finish("short", 3, False, 1024)["truncated"]


def test_vllm_finish_maps_onto_hf_rule():
    out = lambda text, n, reason, stop: SimpleNamespace(text=text, token_ids=list(range(n)), finish_reason=reason,
                                                        stop_reason=stop)
    assert E.vllm_finish(out("#### 4", 6, "stop", "\nProblem:"), 1024)["finish"] == "stop"
    assert E.vllm_finish(out("#### 4", 6, "stop", None), 1024)["finish"] == "eos"
    long = E.vllm_finish(out("abc", 1024, "length", None), 1024)
    assert long["truncated"] and long["finish"] == "length"
    hf = E.finish("#### 4\nProblem: x", 6, False, 1024)
    vl = E.vllm_finish(out("#### 4", 6, "stop", "\nProblem:"), 1024)
    assert hf["text"] == vl["text"] and hf["truncated"] == vl["truncated"]


# ---------------------------------------------------------------- MATH checker

def test_math_checker():
    pytest.importorskip("math_verify")
    E.math_verify_module()
    gen = lambda t: {"text": t, "gen_tokens": 1, "finish": "eos", "truncated": False}
    row = {"id": "m0", "problem": "p", "answer": "\\frac{14}{3}"}
    good = E.score_math(row, gen("We get $\\boxed{\\dfrac{14}{3}}$."))
    assert good["correct"] and good["hendrycks_correct"] and not good["parse_fail"]
    assert E.score_math(row, gen("The answer is 14/3"))["correct"]  # unboxed fallback (a trailing "." makes it 14/3.0)
    bad = E.score_math(row, gen("The answer is $\\boxed{5}$."))
    assert not bad["correct"] and not bad["hendrycks_correct"]
    none = E.score_math(row, gen("I cannot solve this."))
    assert none["parse_fail"] and not none["correct"]
    assert E.score_math({"id": "m1", "problem": "p", "answer": "\\text{(C)}"}, gen("\\boxed{\\text{(C)}}"))["correct"]


def test_hendrycks_equiv():
    assert E.hendrycks_equiv("\\frac12", "\\frac{1}{2}")
    assert E.hendrycks_equiv("0.5", "\\frac{1}{2}")
    assert E.hendrycks_equiv("x = 3", "3")
    assert E.hendrycks_equiv("\\dfrac{3}{4}", "3/4")
    assert not E.hendrycks_equiv("2", "3") and not E.hendrycks_equiv(None, "3")
    assert E.remove_boxed(E.last_boxed_only_string("a \\boxed{1} then \\boxed{\\frac{a}{b}} end")) == "\\frac{a}{b}"
    assert E.last_boxed_only_string("no box") is None


# ---------------------------------------------------------------- prompts

def test_mmlu_prompt_format():
    shot = {"subject": "high_school_physics", "question": " What? ", "choices": ["a", "b", "c", "d"], "answer_idx": 2}
    row = {"subject": "high_school_physics", "question": "Why?", "choices": ["w", "x", "y", "z"], "answer_idx": 0}
    assert E.mmlu_context(row, [shot, shot]) == (
        "The following are multiple choice questions (with answers) about high school physics.\n\n"
        "What?\nA. a\nB. b\nC. c\nD. d\nAnswer: C\n\n"
        "What?\nA. a\nB. b\nC. c\nD. d\nAnswer: C\n\n"
        "Why?\nA. w\nB. x\nC. y\nD. z\nAnswer:")
    dev = [dict(shot, question=f"q{i}") for i in range(7)]
    assert [s["question"] for s in E.mmlu_shots(dev)["high_school_physics"]] == ["q0", "q1", "q2", "q3", "q4"]


def test_encode_pair_moves_trailing_space():
    tok = tiny_tokenizer()
    ctx, cont = E.encode_pair(tok, "Answer: ", "B")
    assert tok.decode(ctx) == "Answer:" and tok.decode(cont) == " B"
    ctx, cont = E.encode_pair(tok, "Answer:", " B")
    assert tok.decode(ctx + cont) == "Answer: B"


def test_pref_pair_is_trainer_encoding():
    tok = tiny_tokenizer()
    p, c, cut = E.pref_pair(tok, "hello", " world", 64)
    assert tok.decode(p) == "hello" and c == tok.encode(" world") + [tok.eos_token_id] and not cut
    p, c, cut = E.pref_pair(tok, "x" * 100, " yz", 64)  # prompt alone fills max_len: keep its last 32 tokens
    assert len(p) == 32 and len(p) + len(c) <= 64 and cut
    p, c, cut = E.pref_pair(tok, "ab", " " + "r" * 100, 64)  # response right-truncated
    assert len(p) + len(c) == 64 and cut and tok.eos_token_id not in c


# ---------------------------------------------------------------- likelihood scorer

def brute(model, reqs, max_len):
    out = []
    for ctx, cont in reqs:
        seq = (ctx + cont)[:-1][-max_len:]
        with torch.no_grad():
            lp = torch.log_softmax(model(input_ids=torch.tensor([seq])).logits[0].float(), -1)
        out.append(sum(lp[len(seq) - len(cont) + j, t].item() for j, t in enumerate(cont)))
    return out


def random_reqs(seed=0, n=40):
    rng = random.Random(seed)
    reqs = []
    for i in range(n):
        ctx = [rng.randrange(256) for _ in range(rng.randint(1, 90))]
        if i % 3 == 0:  # MMLU-like: shared context, single-token continuations
            reqs += [(ctx, [t]) for t in (32, 65, 66, 67)]
        else:
            reqs.append((ctx, [rng.randrange(256) for _ in range(rng.randint(1, 20))]))
    return reqs


def test_scorer_matches_brute_force_and_batching():
    model, tok = tiny_model()
    reqs = random_reqs()
    for max_len in (64, 128):
        want = brute(model, reqs, max_len)
        for budget, head_chunk in [(b, c) for b in (64, 300, 100000) for c in (3, 2048)]:  # chunked head: same
            sc = E.Scorer(model, torch.device("cpu"), budget, head_chunk=head_chunk)
            sc.pad = tok.pad_token_id
            got = sc.loglik(reqs, max_len)
            assert [g["tokens"] for g in got] == [len(c) for _, c in reqs]
            assert max(abs(g["logprob"] - w) for g, w in zip(got, want)) < 1e-4
            assert [g["truncated"] for g in got] == [len(c + x) - 1 > max_len for c, x in reqs]
    sc = E.Scorer(model, torch.device("cpu"), 300)
    assert sc.loglik(reqs, 64) == sc.loglik(reqs, 64)  # deterministic rerun


def test_rolling_windows_cover_every_token_once():
    model, tok = tiny_model()
    ids = [random.Random(1).randrange(256) for _ in range(500)]
    reqs = E.rolling_requests(ids, tok.eos_token_id, 128, 32)
    assert sum(len(c) for _, c in reqs) == len(ids) and [t for _, c in reqs for t in c] == ids
    assert all(len(x) + len(c) <= 128 for x, c in reqs)
    sc = E.Scorer(model, torch.device("cpu"), 10000)
    one = sum(r["logprob"] for r in sc.loglik(E.rolling_requests(ids, tok.eos_token_id, 1024, 32), 1024))
    full = brute(model, [([tok.eos_token_id], ids)], 1024)[0]
    assert abs(one - full) < 1e-3


# ---------------------------------------------------------------- end to end (tiny model, synthetic files)

def write_data(root: Path, n: int = 3) -> None:
    subjects = ["abstract_algebra", "virology"]
    tables = {
        "eval/gsm8k_test": [{"id": f"g{i}", "question": f"What is {i} plus 2?", "answer": f"{i}+2={i + 2}\n#### {i + 2}"}
                            for i in range(n)],
        "eval/math500": [{"id": f"m{i}", "problem": f"Compute {i}/2.", "answer": f"\\frac{{{i}}}{{2}}"} for i in range(n)],
        "eval/mmlu_test": [{"id": f"u{i}", "subject": subjects[i % 2], "question": f"Pick {i}?",
                            "choices": ["one", "two", "three", "four"], "answer_idx": i % 4} for i in range(n)],
        "eval/mmlu_dev": [{"id": f"d{s}{i}", "subject": s, "question": f"Dev {i}?", "choices": ["a", "b", "c", "d"],
                           "answer_idx": i % 4} for s in subjects for i in range(5)],
        "eval/arc_easy_test": [{"id": f"a{i}", "subject": "arc", "question": f"Which is warm {i}?",
                                "choices": ["ice", "the sun", "snow"], "answer_idx": 1} for i in range(n)],
        "pt/test": [{"id": f"w{i}", "text": f" = Article {i} =\n The quick brown fox jumps. " * (i + 1)} for i in range(n)],
        "pref/test": [{"id": f"p{i}", "prompt": f"Say hi {i}.", "chosen": " Hi there!", "rejected": " No."}
                      for i in range(n)],
    }
    tables["eval/lc_gsm8k"] = tables["eval/gsm8k_test"][:2]
    tables["eval/lc_mmlu"] = tables["eval/mmlu_test"][:2]
    for name, rows in tables.items():
        p = root / f"{name}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_end_to_end_hf(tmp_path, monkeypatch):
    pytest.importorskip("math_verify")
    model, tok = tiny_model()
    ckpt = tmp_path / "ckpt"
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)
    write_data(tmp_path / "data")
    monkeypatch.setattr(E, "PROTOCOL", json.loads(json.dumps(E.PROTOCOL)))
    for b in ("gsm8k", "math500"):  # keep CPU test time small; the real caps are checked below
        E.PROTOCOL[b]["max_new_tokens"] = 24
    E.PROTOCOL["lc_gsm8k"] = E.PROTOCOL["gsm8k"]
    args = ["--ckpt", str(ckpt), "--backend", "hf", "--data-root", str(tmp_path / "data"), "--device", "cpu",
            "--gen-batch", "2", "--lik-tokens", "512"]
    s = E.main(args + ["--suite", "full", "--out", str(tmp_path / "o1")])
    assert s["status"] == "complete" and s["complete"] is False  # 3 synthetic rows, not the release counts
    for b in E.SUITES["full"]:
        items = (tmp_path / "o1" / f"items_{b}.jsonl").read_text().splitlines()
        assert len(items) == s[b]["n"] == 3
        assert {"n", "correct", "acc", "parse_fail", "truncated", "wall_s"} <= set(s[b])
    assert s["gsm8k"]["truncated"] + sum(json.loads(x)["finish"] != "length"
                                         for x in (tmp_path / "o1" / "items_gsm8k.jsonl").read_text().splitlines()) == 3
    assert s["wikitext"]["tokens"] > 0 and s["wikitext"]["bits_per_byte"] > 0
    again = E.main(args + ["--suite", "full", "--out", str(tmp_path / "o2")])
    for b in E.SUITES["full"]:
        strip = lambda p: [{k: v for k, v in json.loads(x).items()} for x in p.read_text().splitlines()]
        assert strip(tmp_path / "o1" / f"items_{b}.jsonl") == strip(tmp_path / "o2" / f"items_{b}.jsonl")
    lc = E.main(args + ["--suite", "lc", "--out", str(tmp_path / "lc")])
    assert lc["lc_gsm8k"]["n"] == 2 and lc["lc_mmlu"]["n"] == 2
    resumed = E.main(args + ["--suite", "lc", "--out", str(tmp_path / "lc"), "--resume"])
    assert resumed["lc_mmlu"] == lc["lc_mmlu"]
    part = E.main(args + ["--suite", "lc", "--out", str(tmp_path / "lc2"), "--only", "lc_mmlu"])
    assert part["status"] == "partial" and "lc_gsm8k" not in part
    whole = E.main(args + ["--suite", "lc", "--out", str(tmp_path / "lc2"), "--resume"])
    assert whole["status"] == "complete" and whole["lc_mmlu"] == part["lc_mmlu"]
    for b in ("lc_gsm8k", "lc_mmlu"):
        assert (tmp_path / "lc" / f"items_{b}.jsonl").read_text() == (tmp_path / "lc2" / f"items_{b}.jsonl").read_text()


def test_real_protocol_caps():
    assert E.PROTOCOL["gsm8k"]["max_new_tokens"] == 1024 and E.PROTOCOL["math500"]["max_new_tokens"] == 2048
    assert E.EXPECTED == {"gsm8k": 1319, "math500": 500, "mmlu": 14042, "arc_easy": 2376, "lc_gsm8k": 256,
                          "lc_mmlu": 512, "mmlu_dev": 285}
