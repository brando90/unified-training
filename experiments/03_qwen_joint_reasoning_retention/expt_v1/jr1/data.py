#!/usr/bin/env python3
"""JR1 campaign data: pinned public sources, split manifests and contamination audit.

    python -m jr1.data prepare [--root DIR]    download pinned sources, build every split, write manifest.json
    python -m jr1.data verify  [--root DIR]    re-hash every file against manifest.json and re-check split integrity

DIR defaults to $JR1_STATE_ROOT/data; Hub files are cached under $HF_HOME (default $JR1_STATE_ROOT/hf).
Run from experiments/03_qwen_joint_reasoning_retention/expt_v1 (Exp 04 imports this package; no copy).

Layout written under DIR (one JSON object per line, keys in the order below):
  pt/{train,validation,tune,controller,test}.jsonl   {"id","text"}  WikiText-103-raw articles; tune/controller are
                                                     disjoint halves of validation; test is the LM-NLL eval corpus
  sft/train.jsonl                                    {"id","prompt","response"}  GSM8K train rationale + "#### answer"
  rl/train.jsonl                                     {"id","prompt","answer"}    GSM8K train, final answer only,
                                                     disjoint from sft/train
  gsm8k_dev/{readiness,tune,controller}.jsonl        {"id","question","prompt","response","answer"}; disjoint from
                                                     both pools. sft/{tune,controller} and rl/{tune,controller} are
                                                     the same tune/controller rows in the sft and rl schemas
  pref/{train,tune,controller,test}.jsonl            {"id","prompt","chosen","rejected"}  UltraFeedback binarized;
                                                     test = official test_prefs (held-out ordering accuracy)
  eval/gsm8k_test.jsonl    {"id","question","answer"}                          1,319 (answer verbatim, "#### n")
  eval/math500.jsonl       {"id","problem","answer"}                           500
  eval/mmlu_test.jsonl     {"id","subject","question","choices","answer_idx"}  full release (14,042)
  eval/arc_easy_test.jsonl same schema as mmlu_test                            2,376
  eval/mmlu_dev.jsonl      same schema; cais/mmlu 'all' dev, 285 = 5 per subject, release order: the published
                           5-shot exemplars (few-shot context only, not an audited eval set)
  eval/lc_gsm8k.jsonl      256 rows of gsm8k_test; eval/lc_mmlu.jsonl 512 subject-stratified rows of mmlu_test.
                           Sealed learning-curve subsets: never used for controller or development signals.
  wikitext_test is pt/test.jsonl (alias in manifest).
  audit_matches.jsonl      every contamination hit (pool, row id, eval set, eval id, kind)

Prompts: prompt = PROMPT_PREFIX + question + "\\nAnswer:" (verbatim copy of jr1_campaign.prompt()); an SFT response
starts with a space and carries no EOS (the trainer appends it). Pref prompts and responses are the raw single-turn
UltraFeedback texts; the trainer applies its own template.

Contamination audit: every train/development pool (pt train/validation, GSM8K train rows, UltraFeedback train_prefs)
against every eval set (GSM8K test, MATH-500, MMLU test, ARC-Easy test, WikiText test paragraphs, UltraFeedback
test_prefs prompts) with three kinds of overlap: exact (raw question/prompt/paragraph string equal), normalized
(casefold + collapsed whitespace equal) and embedded (word-normalized eval text of at least MIN_EMBED_CHARS characters
appears inside any text field of the train row, e.g. a benchmark question inside a preference prompt or response;
stems shared by two items of one set, such as MMLU's "Which of the following statements is true?", are skipped, and
WikiText paragraphs count only from MIN_PARAGRAPH_CHARS).
Overlapping train rows are dropped; eval rows never are. The already-used readiness rows are kept and only reported.
Contamination inherited from Qwen2.5 pre-training is unknown and not measured here.

Reuses, by attributed adaptation, the normalization, provenance and embedded-substring ideas of
experiments/00_program/prepare_data.py and substring_audit.py (frozen; not imported).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import platform
import re
import sys
from pathlib import Path

SALT = "jr1-data-v1"
MIN_EMBED_CHARS = 40
MIN_PARAGRAPH_CHARS = 80   # WikiText paragraph identities; shorter lines are boilerplate (" Key ", " <formula> ")
SIZES = {"readiness": 128, "tune": 512, "controller": 512, "pref_tune": 512, "pref_controller": 512,
         "lc_gsm8k": 256, "lc_mmlu": 512}
EXPECTED_EVAL_ROWS = {"gsm8k_test": 1319, "math500": 500, "arc_easy_test": 2376, "mmlu_test": 14042, "mmlu_dev": 285}

# Pinned Hub revisions (same commits as experiments/00_program/data_manifest.json where the source is shared).
SOURCES = {
    "wikitext": {"repo": "Salesforce/wikitext", "config": "wikitext-103-raw-v1",
                 "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
                 "files": {"train": ["wikitext-103-raw-v1/train-00000-of-00002.parquet",
                                     "wikitext-103-raw-v1/train-00001-of-00002.parquet"],
                           "validation": ["wikitext-103-raw-v1/validation-00000-of-00001.parquet"],
                           "test": ["wikitext-103-raw-v1/test-00000-of-00001.parquet"]}},
    "gsm8k": {"repo": "openai/gsm8k", "config": "main", "revision": "740312add88f781978c0658806c59bc2815b9866",
              "files": {"train": ["main/train-00000-of-00001.parquet"], "test": ["main/test-00000-of-00001.parquet"]}},
    "ultrafeedback": {"repo": "HuggingFaceH4/ultrafeedback_binarized", "config": None,
                      "revision": "3949bf5f8c17c394422ccfab0c31ea9c20bdeb85",
                      "files": {"train_prefs": ["data/train_prefs-00000-of-00001.parquet"],
                                "test_prefs": ["data/test_prefs-00000-of-00001.parquet"]}},
    "math500": {"repo": "HuggingFaceH4/MATH-500", "config": None, "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
                "files": {"test": ["test.jsonl"]}},
    "mmlu": {"repo": "cais/mmlu", "config": "all", "revision": "c30699e8356da336a370243923dbaf21066bb9fe",
             "files": {"test": ["all/test-00000-of-00001.parquet"], "dev": ["all/dev-00000-of-00001.parquet"]}},
    "arc_easy": {"repo": "allenai/ai2_arc", "config": "ARC-Easy", "revision": "210d026faf9955653af8916fad021475a3f00453",
                 "files": {"test": ["ARC-Easy/test-00000-of-00001.parquet"]}},
}

SCHEMAS = {
    "pt": ["id", "text"],
    "sft": ["id", "prompt", "response"],
    "rl": ["id", "prompt", "answer"],
    "gsm8k_dev": ["id", "question", "prompt", "response", "answer"],
    "pref": ["id", "prompt", "chosen", "rejected"],
    "eval/gsm8k_test": ["id", "question", "answer"],
    "eval/lc_gsm8k": ["id", "question", "answer"],
    "eval/math500": ["id", "problem", "answer"],
    "eval/mmlu_test": ["id", "subject", "question", "choices", "answer_idx"],
    "eval/lc_mmlu": ["id", "subject", "question", "choices", "answer_idx"],
    "eval/mmlu_dev": ["id", "subject", "question", "choices", "answer_idx"],
    "eval/arc_easy_test": ["id", "subject", "question", "choices", "answer_idx"],
}

# Verbatim copy of jr1_campaign.prompt() (that module needs torch and a state root at import time).
PROMPT_PREFIX = "Solve the problem. Show your reasoning, then put the final numeric answer after ####.\n\nProblem: "


def prompt(question: str) -> str:
    return PROMPT_PREFIX + question + "\nAnswer:"


def schema_for(name: str) -> list[str]:
    return SCHEMAS[name] if name in SCHEMAS else SCHEMAS[name.split("/")[0]]


# ---------------------------------------------------------------- text identities

def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().casefold()


def norm_words(s: str) -> str:
    return " ".join(re.findall(r"\w+", s.casefold()))


def salted(tag: str, row_id: str) -> str:
    return hashlib.sha256(f"{SALT}:{tag}:{row_id}".encode()).hexdigest()


def strip_calculator(answer: str) -> str:
    """GSM8K rationales carry <<48/2=24>> calculator annotations; the SFT target drops them."""
    return re.sub(r"<<[^<>]*>>", "", answer)


def final_answer(answer: str) -> str:
    return answer.rsplit("####", 1)[1].strip()


def paragraphs(text: str, min_chars: int = 0) -> list[str]:
    """WikiText body lines (headings and blank lines excluded) with at least min_chars word-normalized characters."""
    return [ln.strip() for ln in text.split("\n") if ln.strip() and not re.fullmatch(r"\s*=+ .* =+\s*", ln)
            and len(norm_words(ln)) >= min_chars]


# ---------------------------------------------------------------- source rows -> records

def wikitext_articles(lines: list[str], split: str) -> list[dict]:
    """Group WikiText-103 raw lines into articles; an article starts at a level-1 heading line ' = Title = '."""
    arts, cur = [], []
    for ln in lines:
        if re.fullmatch(r" = [^=].* = \n", ln) and cur:
            arts.append(cur)
            cur = []
        cur.append(ln)
    if cur:
        arts.append(cur)
    out = []
    for i, a in enumerate(arts):
        text = "".join(a)
        if text.strip():
            out.append({"id": f"wikitext103:{split}:{i:06d}", "text": text})
    return out


def gsm8k_records(rows: list[dict], split: str) -> list[dict]:
    return [{"id": f"gsm8k:{split}:{i}", "index": i, "question": r["question"], "answer": r["answer"]}
            for i, r in enumerate(rows)]


def mc_records(rows: list[dict], prefix: str, kind: str) -> list[dict]:
    out = []
    for i, r in enumerate(rows):
        if kind == "mmlu":
            out.append({"id": f"{prefix}:{i:05d}", "subject": r["subject"], "question": r["question"],
                        "choices": list(r["choices"]), "answer_idx": int(r["answer"])})
        else:
            labels = list(r["choices"]["label"])
            out.append({"id": f"{prefix}:{r['id']}", "subject": "ARC-Easy", "question": r["question"],
                        "choices": list(r["choices"]["text"]), "answer_idx": labels.index(r["answerKey"])})
    return out


def pref_records(rows: list[dict], split: str, filters: collections.Counter) -> list[dict]:
    out, seen = [], set()
    for r in rows:
        ch, rj = list(r["chosen"]), list(r["rejected"])
        if (len(ch) != 2 or len(rj) != 2 or ch[0]["role"] != "user" or ch[1]["role"] != "assistant"
                or rj[1]["role"] != "assistant" or ch[0]["content"] != r["prompt"]):
            filters[f"{split}_not_single_turn"] += 1
            continue
        c, j = ch[1]["content"], rj[1]["content"]
        if not r["prompt"].strip() or not c.strip() or not j.strip() or c == j:
            filters[f"{split}_empty_or_identical"] += 1
            continue
        key = norm_ws(r["prompt"])
        if key in seen:
            filters[f"{split}_duplicate_prompt"] += 1
            continue
        seen.add(key)
        out.append({"id": f"ultrafeedback:{split}:{r['prompt_id']}", "prompt": r["prompt"], "chosen": c, "rejected": j})
    return out


# ---------------------------------------------------------------- splits

def split_gsm8k(rows: list[dict], sizes: dict = SIZES) -> dict[str, list[dict]]:
    """readiness = original indices < sizes['readiness'] (the prompts already used); the rest is ordered by a salted
    hash of the id: tune, controller, then the remainder halved into sft_train and rl_train (disjoint pools)."""
    ready = [r for r in rows if r["index"] < sizes["readiness"]]
    rest = sorted((r for r in rows if r["index"] >= sizes["readiness"]), key=lambda r: salted("gsm8k", r["id"]))
    t, c = sizes["tune"], sizes["controller"]
    pool = rest[t + c:]
    half = (len(pool) + 1) // 2
    return {"readiness": ready, "tune": rest[:t], "controller": rest[t:t + c], "sft_train": pool[:half],
            "rl_train": pool[half:]}


def split_pref(rows: list[dict], sizes: dict = SIZES) -> dict[str, list[dict]]:
    rows = sorted(rows, key=lambda r: salted("pref", r["id"]))
    t, c = sizes["pref_tune"], sizes["pref_controller"]
    return {"tune": rows[:t], "controller": rows[t:t + c], "train": rows[t + c:]}


def split_wiki_validation(rows: list[dict]) -> dict[str, list[dict]]:
    rows = sorted(rows, key=lambda r: salted("wiki", r["id"]))
    half = (len(rows) + 1) // 2
    return {"tune": rows[:half], "controller": rows[half:]}


def stratified_quota(counts: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder allocation of `total` across strata, at least one per stratum, ties by name."""
    n = sum(counts.values())
    raw = {k: total * v / n for k, v in counts.items()}
    quota = {k: max(1, min(counts[k], int(raw[k]))) for k in counts}
    order = sorted(counts, key=lambda k: (-(raw[k] - int(raw[k])), k))
    i = 0
    while sum(quota.values()) < total:
        k = order[i % len(order)]
        if quota[k] < counts[k]:
            quota[k] += 1
        i += 1
    while sum(quota.values()) > total:
        k = max((k for k in quota if quota[k] > 1), key=lambda k: (quota[k] - raw[k], k))
        quota[k] -= 1
    return quota


def lc_subsets(gsm_test: list[dict], mmlu: list[dict], sizes: dict = SIZES) -> tuple[list[dict], list[dict]]:
    lc_g = sorted(gsm_test, key=lambda r: salted("lc_gsm8k", r["id"]))[:sizes["lc_gsm8k"]]
    by = collections.defaultdict(list)
    for r in mmlu:
        by[r["subject"]].append(r)
    quota = stratified_quota({k: len(v) for k, v in by.items()}, sizes["lc_mmlu"])
    lc_m = [r for k in sorted(by) for r in sorted(by[k], key=lambda r: salted("lc_mmlu", r["id"]))[:quota[k]]]
    order = {r["id"]: i for i, r in enumerate(gsm_test)}
    order_m = {r["id"]: i for i, r in enumerate(mmlu)}
    return sorted(lc_g, key=lambda r: order[r["id"]]), sorted(lc_m, key=lambda r: order_m[r["id"]])


# ---------------------------------------------------------------- contamination audit

class EvalIndex:
    """Exact, normalized and embedded (Aho-Corasick over word-normalized text) identities of every eval text."""

    def __init__(self, targets: dict[str, list[tuple[str, str]]]):
        import ahocorasick
        self.exact = collections.defaultdict(set)
        self.normal = collections.defaultdict(set)
        self.auto = ahocorasick.Automaton()
        embed = collections.defaultdict(set)
        for name, items in targets.items():
            for eid, text in items:
                self.exact[text.strip()].add((name, eid))
                self.normal[norm_ws(text)].add((name, eid))
                w = norm_words(text)
                if len(w) >= MIN_EMBED_CHARS:
                    embed[f" {w} "].add((name, eid))
        self.generic_patterns = 0
        for pat, hits in list(embed.items()):
            # A stem shared by several items of one set ("Which of the following statements is true?") names no
            # item; it is kept for exact/normalized identity but not searched inside train text.
            if max(collections.Counter(n for n, _ in hits).values()) > 1:
                self.generic_patterns += 1
                del embed[pat]
                continue
            self.auto.add_word(pat, frozenset(hits))
        self.embedded_patterns = len(embed)
        if embed:
            self.auto.make_automaton()

    def hits(self, keys: list[str], texts: list[str]) -> set[tuple[str, str, str]]:
        """keys: question-like fields (exact/normalized identity); texts: every text field (embedded search)."""
        out = set()
        for k in keys:
            for name, eid in self.exact.get(k.strip(), ()):
                out.add((name, eid, "exact"))
            for name, eid in self.normal.get(norm_ws(k), ()):
                out.add((name, eid, "normalized"))
        if self.embedded_patterns:
            for t in texts:
                for _, found in self.auto.iter(f" {norm_words(t)} "):
                    for name, eid in found:
                        out.add((name, eid, "embedded"))
        return out


def audit_pool(pool: str, rows: list[dict], index: EvalIndex, keys, texts, report: dict, matches: list,
               keep_ids: frozenset = frozenset()) -> list[dict]:
    """Drop every row with any overlap (rows in keep_ids are only reported). Counts are rows, per eval set and kind."""
    kept, stats = [], report.setdefault(pool, {"rows_audited": 0, "rows_dropped": 0, "rows_flagged_kept": 0,
                                               "by_eval": {}})
    for r in rows:
        stats["rows_audited"] += 1
        found = index.hits(keys(r), texts(r))
        if not found:
            kept.append(r)
            continue
        for name, kind in {(n, k) for n, _, k in found}:
            e = stats["by_eval"].setdefault(name, {"exact": 0, "normalized": 0, "embedded": 0, "rows": 0})
            e[kind] += 1
        for name in {n for n, _, _ in found}:
            stats["by_eval"][name]["rows"] += 1
        for name, eid, kind in sorted(found):
            matches.append({"pool": pool, "id": r["id"], "eval": name, "eval_id": eid, "kind": kind})
        if r["id"] in keep_ids:
            stats["rows_flagged_kept"] += 1
            kept.append(r)
        else:
            stats["rows_dropped"] += 1
    return kept


# ---------------------------------------------------------------- build

def build(tables: dict, sizes: dict = SIZES, expected: dict | None = EXPECTED_EVAL_ROWS) -> tuple[dict, dict, dict]:
    """Pure transformation of loaded source tables into (files {relpath: rows}, splits manifest, audit manifest)."""
    filters = collections.Counter()
    files: dict[str, list[dict]] = {}

    gsm_test = [{"id": r["id"], "question": r["question"], "answer": r["answer"]}
                for r in gsm8k_records(tables["gsm8k"]["test"], "test")]
    math500 = [{"id": f"math500:{r['unique_id']}", "problem": r["problem"], "answer": r["answer"]}
               for r in tables["math500"]["test"]]
    mmlu = mc_records(tables["mmlu"]["test"], "mmlu:test", "mmlu")
    arc = mc_records(tables["arc_easy"]["test"], "arc_easy:test", "arc")
    mmlu_dev = mc_records(tables["mmlu"]["dev"], "mmlu:dev", "mmlu")
    if expected:
        got = {"gsm8k_test": len(gsm_test), "math500": len(math500), "arc_easy_test": len(arc), "mmlu_test": len(mmlu),
               "mmlu_dev": len(mmlu_dev)}
        bad = {k: (got[k], v) for k, v in expected.items() if got[k] != v}
        if bad:
            raise RuntimeError(f"eval release counts differ from the pinned release (got, expected): {bad}")
    lc_g, lc_m = lc_subsets(gsm_test, mmlu, sizes)
    files.update({"eval/gsm8k_test.jsonl": gsm_test, "eval/math500.jsonl": math500, "eval/mmlu_test.jsonl": mmlu,
                  "eval/arc_easy_test.jsonl": arc, "eval/lc_gsm8k.jsonl": lc_g, "eval/lc_mmlu.jsonl": lc_m,
                  "eval/mmlu_dev.jsonl": mmlu_dev})

    wiki = {s: wikitext_articles(tables["wikitext"].pop(s), s) for s in ("train", "validation", "test")}
    log(f"wikitext articles: { {s: len(v) for s, v in wiki.items()} }")
    pref_test = pref_records(tables["ultrafeedback"]["test_prefs"], "test_prefs", filters)
    targets = {
        "gsm8k_test": [(r["id"], r["question"]) for r in gsm_test],
        "math500": [(r["id"], r["problem"]) for r in math500],
        "mmlu_test": [(r["id"], r["question"]) for r in mmlu],
        "arc_easy_test": [(r["id"], r["question"]) for r in arc],
        "wikitext_test": [(r["id"], p) for r in wiki["test"] for p in paragraphs(r["text"], MIN_PARAGRAPH_CHARS)],
        "pref_test": [(r["id"], r["prompt"]) for r in pref_test],
    }
    index = EvalIndex(targets)
    log(f"eval index: {index.embedded_patterns} embedded patterns")
    report, matches = {}, []

    # GSM8K train: dedupe by normalized question (first occurrence wins), audit, split.
    seen, gsm_rows = set(), []
    for r in gsm8k_records(tables["gsm8k"]["train"], "train"):
        k = norm_ws(r["question"])
        if k in seen:
            filters["gsm8k_train_duplicate_question"] += 1
            continue
        seen.add(k)
        gsm_rows.append(r)
    ready_ids = frozenset(r["id"] for r in gsm_rows if r["index"] < sizes["readiness"])
    gsm_rows = audit_pool("gsm8k_train", gsm_rows, index, lambda r: [r["question"]],
                          lambda r: [r["question"], r["answer"]], report, matches, keep_ids=ready_ids)
    gs = split_gsm8k(gsm_rows, sizes)
    log("gsm8k audited and split")
    files["sft/train.jsonl"] = [{"id": r["id"], "prompt": prompt(r["question"]),
                                 "response": " " + strip_calculator(r["answer"])} for r in gs["sft_train"]]
    files["rl/train.jsonl"] = [{"id": r["id"], "prompt": prompt(r["question"]), "answer": final_answer(r["answer"])}
                               for r in gs["rl_train"]]
    for s in ("tune", "controller"):  # the same development rows in the trainer's sft/rl schemas
        files[f"sft/{s}.jsonl"] = [{"id": r["id"], "prompt": prompt(r["question"]),
                                    "response": " " + strip_calculator(r["answer"])} for r in gs[s]]
        files[f"rl/{s}.jsonl"] = [{"id": r["id"], "prompt": prompt(r["question"]), "answer": final_answer(r["answer"])}
                                  for r in gs[s]]
    for s in ("readiness", "tune", "controller"):
        files[f"gsm8k_dev/{s}.jsonl"] = [{"id": r["id"], "question": r["question"], "prompt": prompt(r["question"]),
                                          "response": " " + strip_calculator(r["answer"]),
                                          "answer": final_answer(r["answer"])} for r in gs[s]]

    # UltraFeedback: structural filters, drop train prompts that are official test prompts, audit, split.
    pref_train = pref_records(tables["ultrafeedback"]["train_prefs"], "train_prefs", filters)
    test_keys = {norm_ws(r["prompt"]) for r in pref_test}
    before = len(pref_train)
    pref_train = [r for r in pref_train if norm_ws(r["prompt"]) not in test_keys]
    filters["train_prefs_prompt_in_test_prefs"] = before - len(pref_train)
    pref_train = audit_pool("pref_train", pref_train, index, lambda r: [r["prompt"]],
                            lambda r: [r["prompt"], r["chosen"], r["rejected"]], report, matches)
    ps = split_pref(pref_train, sizes)
    log("ultrafeedback audited and split")
    files.update({"pref/train.jsonl": ps["train"], "pref/tune.jsonl": ps["tune"],
                  "pref/controller.jsonl": ps["controller"], "pref/test.jsonl": pref_test})

    # WikiText-103: articles; train and validation audited (paragraph identities + embedded eval texts).
    for s in ("train", "validation"):
        wiki[s] = audit_pool(f"pt_{s}", wiki[s], index, lambda r: paragraphs(r["text"], MIN_PARAGRAPH_CHARS),
                             lambda r: [r["text"]],
                             report, matches)
    wv = split_wiki_validation(wiki["validation"])
    log("wikitext audited and split")
    files.update({"pt/train.jsonl": wiki["train"], "pt/validation.jsonl": wiki["validation"],
                  "pt/tune.jsonl": sorted(wv["tune"], key=lambda r: r["id"]),
                  "pt/controller.jsonl": sorted(wv["controller"], key=lambda r: r["id"]), "pt/test.jsonl": wiki["test"]})

    for k in ("sft/train.jsonl", "rl/train.jsonl", "sft/tune.jsonl", "sft/controller.jsonl", "rl/tune.jsonl",
              "rl/controller.jsonl", "gsm8k_dev/tune.jsonl", "gsm8k_dev/controller.jsonl", "gsm8k_dev/readiness.jsonl"):
        files[k].sort(key=lambda r: int(r["id"].rsplit(":", 1)[1]))
    for k in ("pref/train.jsonl", "pref/tune.jsonl", "pref/controller.jsonl"):
        files[k].sort(key=lambda r: r["id"])
    files["audit_matches.jsonl"] = sorted(matches, key=lambda m: (m["pool"], m["id"], m["eval"], m["eval_id"], m["kind"]))

    splits = {
        "gsm8k": {s: [r["id"] for r in files[f"gsm8k_dev/{s}.jsonl"]] for s in ("readiness", "tune", "controller")},
        "lc_gsm8k": [r["id"] for r in lc_g],
        "lc_mmlu": [r["id"] for r in lc_m],
        "lc_mmlu_quota": stratified_quota(dict(collections.Counter(r["subject"] for r in mmlu)), sizes["lc_mmlu"]),
        "pref": {s: [r["id"] for r in files[f"pref/{s}.jsonl"]] for s in ("tune", "controller")},
        "pt": {s: [r["id"] for r in files[f"pt/{s}.jsonl"]] for s in ("tune", "controller")},
        "sft_train_ids_sha256": ids_digest(files["sft/train.jsonl"]),
        "rl_train_ids_sha256": ids_digest(files["rl/train.jsonl"]),
        "pref_train_ids_sha256": ids_digest(files["pref/train.jsonl"]),
        "pt_train_ids_sha256": ids_digest(files["pt/train.jsonl"]),
    }
    audit = {"method": {"exact": "raw question/prompt/paragraph string equality (stripped)",
                        "normalized": "casefold + collapsed whitespace equality",
                        "embedded": f"word-normalized eval text (>= {MIN_EMBED_CHARS} chars, not a stem shared by "
                                    "two items of one eval set) inside any train text field",
                        "eval_text_fields": {"gsm8k_test": "question", "math500": "problem", "mmlu_test": "question",
                                             "arc_easy_test": "question",
                                             "wikitext_test": f"body paragraphs >= {MIN_PARAGRAPH_CHARS} chars",
                                             "pref_test": "prompt"},
                        "train_text_fields": {"gsm8k_train": "question (+answer for embedded)",
                                              "pref_train": "prompt (+chosen, rejected for embedded)",
                                              "pt_train/pt_validation": "body paragraphs (+full article for embedded)"},
                        "policy": "overlapping train/dev rows dropped; eval rows never dropped; readiness rows kept "
                                  "and reported (already used before this audit)",
                        "pretraining_contamination": "unknown (Qwen2.5 pre-training data not public)"},
             "eval_targets": {k: len(v) for k, v in targets.items()},
             "embedded_patterns": index.embedded_patterns, "generic_patterns_skipped": index.generic_patterns,
             "pools": report,
             "filters": dict(sorted(filters.items()))}
    check_integrity(files, sizes)
    return files, splits, audit


def log(msg: str) -> None:
    print("BUILD", msg, flush=True)


def ids_digest(rows: list[dict]) -> str:
    return hashlib.sha256("\n".join(r["id"] for r in rows).encode()).hexdigest()


def check_integrity(files: dict, sizes: dict = SIZES) -> None:
    """Disjointness of every GSM8K-train split (by id and normalized question), pref and pt splits, LC subsets inside
    their eval sets, and every row's keys equal to its schema."""
    def ids(k):
        return [r["id"] for r in files[k]]
    gsm = ["sft/train.jsonl", "rl/train.jsonl", "gsm8k_dev/readiness.jsonl", "gsm8k_dev/tune.jsonl",
           "gsm8k_dev/controller.jsonl"]
    for group in (gsm, ["pref/train.jsonl", "pref/tune.jsonl", "pref/controller.jsonl", "pref/test.jsonl"],
                  ["pt/train.jsonl", "pt/tune.jsonl", "pt/controller.jsonl", "pt/test.jsonl"]):
        all_ids = [i for k in group for i in ids(k)]
        if len(all_ids) != len(set(all_ids)):
            raise AssertionError(f"split ids overlap within {group}")
    qs = [norm_ws(r["prompt"]) for k in gsm for r in files[k]]
    if len(qs) != len(set(qs)):
        raise AssertionError("GSM8K splits share a normalized question")
    pq = [norm_ws(r["prompt"]) for k in ("pref/train.jsonl", "pref/tune.jsonl", "pref/controller.jsonl",
                                         "pref/test.jsonl") for r in files[k]]
    if len(pq) != len(set(pq)):
        raise AssertionError("pref splits share a normalized prompt")
    if set(ids("pt/tune.jsonl")) | set(ids("pt/controller.jsonl")) != set(ids("pt/validation.jsonl")):
        raise AssertionError("pt tune + controller must partition validation")
    if not set(ids("eval/lc_gsm8k.jsonl")) <= set(ids("eval/gsm8k_test.jsonl")):
        raise AssertionError("lc_gsm8k outside gsm8k_test")
    if not set(ids("eval/lc_mmlu.jsonl")) <= set(ids("eval/mmlu_test.jsonl")):
        raise AssertionError("lc_mmlu outside mmlu_test")
    want = {"gsm8k_dev/tune.jsonl": sizes["tune"], "gsm8k_dev/controller.jsonl": sizes["controller"],
            "pref/tune.jsonl": sizes["pref_tune"], "pref/controller.jsonl": sizes["pref_controller"],
            "eval/lc_gsm8k.jsonl": sizes["lc_gsm8k"], "eval/lc_mmlu.jsonl": sizes["lc_mmlu"]}
    for v in ("sft", "rl"):
        for s in ("tune", "controller"):
            if ids(f"{v}/{s}.jsonl") != ids(f"gsm8k_dev/{s}.jsonl"):
                raise AssertionError(f"{v}/{s} is not the gsm8k_dev/{s} rows")
    for k, n in want.items():
        if len(files[k]) != n:
            raise AssertionError(f"{k}: {len(files[k])} rows, expected {n}")
    for k, rows in files.items():
        if k == "audit_matches.jsonl":
            continue
        keys = schema_for(k[:-len(".jsonl")])
        for r in rows:
            if list(r) != keys:
                raise AssertionError(f"{k}: row {r.get('id')} keys {list(r)} != {keys}")


# ---------------------------------------------------------------- I/O

def state_root() -> Path:
    if "JR1_STATE_ROOT" not in os.environ:
        raise RuntimeError("JR1_STATE_ROOT must name the private campaign state directory")
    return Path(os.environ["JR1_STATE_ROOT"])


def data_root(root: str | None = None) -> Path:
    return Path(root) if root else state_root() / "data"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load(name: str, root: str | None = None) -> list[dict]:
    """Rows of one prepared split, e.g. load('sft/train'), load('eval/gsm8k_test'), load('wikitext_test')."""
    name = {"wikitext_test": "pt/test"}.get(name, name)
    with open(data_root(root) / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def manifest(root: str | None = None) -> dict:
    return json.loads((data_root(root) / "manifest.json").read_text())


def fetch(cache_dir: Path) -> tuple[dict, dict]:
    """Download every pinned file at its commit; return (tables, provenance)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    api, tables, prov = HfApi(), {}, {}
    for key, src in SOURCES.items():
        info = api.dataset_info(src["repo"], revision=src["revision"])
        card = info.card_data.to_dict() if info.card_data else {}
        entry = {"repo": src["repo"], "config": src["config"], "revision": src["revision"],
                 "license": card.get("license", "see source card"),
                 "url": f"https://huggingface.co/datasets/{src['repo']}/tree/{src['revision']}",
                 "files": {}, "rows": {}}
        tables[key] = {}
        for split, paths in src["files"].items():
            rows = []
            for p in paths:
                local = Path(hf_hub_download(src["repo"], p, repo_type="dataset", revision=src["revision"],
                                             cache_dir=str(cache_dir)))
                entry["files"][p] = sha256_file(local)
                if p.endswith(".parquet"):
                    cols = {"wikitext": ["text"], "ultrafeedback": ["prompt", "prompt_id", "chosen", "rejected"]}
                    t = pq.read_table(local, columns=cols.get(key))
                    rows.extend(t.column("text").to_pylist() if key == "wikitext" else t.to_pylist())
                    del t
                else:
                    rows.extend(json.loads(x) for x in local.read_text(encoding="utf-8").splitlines() if x.strip())
            tables[key][split] = rows
            entry["rows"][split] = len(rows)
        prov[key] = entry
        print("SOURCE", key, src["revision"], entry["rows"], flush=True)
    return tables, prov


def prepare(root: str | None = None) -> dict:
    out = data_root(root)
    cache = Path(os.environ.get("HF_HOME", str(out.parent / "hf"))) / "hub"
    tables, prov = fetch(cache)
    files, splits, audit = build(tables)
    out.mkdir(parents=True, exist_ok=True)
    listing = {}
    for rel in sorted(files):
        write_jsonl(out / rel, files[rel])
        listing[rel] = {"rows": len(files[rel]), "bytes": (out / rel).stat().st_size, "sha256": sha256_file(out / rel)}
        print("WROTE", rel, listing[rel]["rows"], flush=True)
    import huggingface_hub
    import pyarrow
    man = {"schema_version": 1, "producer": "experiments/03_qwen_joint_reasoning_retention/expt_v1/jr1/data.py prepare",
           "code_sha256": sha256_file(Path(__file__)), "salt": SALT, "sizes": SIZES,
           "environment": {"python": platform.python_version(), "pyarrow": pyarrow.__version__,
                           "huggingface_hub": huggingface_hub.__version__},
           "sources": prov, "prompt_prefix": PROMPT_PREFIX, "schemas": SCHEMAS,
           "aliases": {"wikitext_test": "pt/test.jsonl"}, "files": listing, "audit": audit, "splits": splits}
    write_json(out / "manifest.json", man)
    return man


def verify(root: str | None = None) -> dict:
    out, man = data_root(root), manifest(root)
    bad = {rel: "sha256" for rel, m in man["files"].items() if sha256_file(out / rel) != m["sha256"]}
    files = {rel: load(rel[:-len(".jsonl")], root) for rel in man["files"]}
    bad.update({rel: "rows" for rel, m in man["files"].items() if len(files[rel]) != m["rows"]})
    if bad:
        raise AssertionError(f"manifest mismatch: {bad}")
    check_integrity(files, man["sizes"])
    for k, n in EXPECTED_EVAL_ROWS.items():
        assert man["files"][f"eval/{k}.jsonl"]["rows"] == n, k
    return {"files": len(man["files"]), "ok": True}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("command", choices=["prepare", "verify"])
    p.add_argument("--root", help="output directory (default $JR1_STATE_ROOT/data)")
    a = p.parse_args(argv)
    if a.command == "prepare":
        man = prepare(a.root)
        print(json.dumps({k: v["rows"] for k, v in man["files"].items()}, indent=1))
    else:
        print(json.dumps(verify(a.root)))


if __name__ == "__main__":
    sys.exit(main())
