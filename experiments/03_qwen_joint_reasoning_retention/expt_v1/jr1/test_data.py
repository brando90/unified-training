"""Deterministic CPU tests for jr1.data: schemas, counts, split disjointness and the contamination audit.

    cd experiments/03_qwen_joint_reasoning_retention/expt_v1 && python -m unittest jr1.test_data -v

The synthetic tests need no network. RealManifestTest re-hashes a prepared directory when one exists
($JR1_DATA_ROOT, else $JR1_STATE_ROOT/data) and is skipped otherwise.
"""
import collections
import importlib.util
import json
import os
import random
import tempfile
import unittest
from pathlib import Path

from jr1 import data as d

SMALL = {"readiness": 8, "tune": 10, "controller": 10, "pref_tune": 5, "pref_controller": 5, "lc_gsm8k": 6,
         "lc_mmlu": 12}
MMLU_Q = "Which of the following best describes the role of mitochondria in eukaryotic cells?"
GSM_Q = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May."


def synthetic_tables(seed: int = 0) -> dict:
    rnd = random.Random(seed)

    def wiki(prefix, n):
        lines = [""]
        for a in range(n):
            lines += [f" = {prefix} Article {a} = \n", "\n",
                      f" The {prefix} article {a} paragraph about topic {rnd.random():.6f} is long enough to count . \n",
                      " = = Section = = \n", f" Another {prefix} body line {a} with text {rnd.random():.6f} . \n"]
        return lines

    gsm_train = [{"question": f"Train problem {i}: Tom has {i} apples and buys {i + 3} more apples today.",
                  "answer": f"He has {i}+{i + 3}=<<{i}+{i + 3}={2 * i + 3}>>{2 * i + 3} apples.\n#### {2 * i + 3}"}
                 for i in range(80)]
    gsm_train[20]["question"] = GSM_Q                         # exact overlap with a test question
    gsm_train[21]["question"] = "  TEST PROBLEM 0: A TRAIN TRAVELS 0 MILES EACH HOUR FOR 1 HOURS   TOTAL. "  # normalized
    gsm_train[22]["question"] = gsm_train[23]["question"]     # duplicate (dropped before splitting)
    gsm_train[3]["question"] = "Readiness copy: " + MMLU_Q    # readiness row: embedded overlap, flagged, kept
    gsm_test = [{"question": GSM_Q, "answer": "She sold 72 clips.\n#### 72"}] + [
        {"question": f"Test problem {i}: a train travels {i} miles each hour for {i + 1} hours total.",
         "answer": f"#### {i * (i + 1)}"} for i in range(19)]
    subjects = ["anatomy", "biology", "law"]
    mmlu = [{"question": MMLU_Q if i == 0 else f"MMLU question {i}?", "subject": subjects[i % 3 if i < 30 else 2],
             "choices": ["a", "b", "c", "d"], "answer": i % 4} for i in range(40)]
    arc = [{"id": f"ARC_{i}", "question": f"ARC question {i}?",
            "choices": {"text": ["x", "y", "z"], "label": ["A", "B", "C"] if i % 2 else ["1", "2", "3"]},
            "answerKey": "B" if i % 2 else "3"} for i in range(9)]
    math = [{"unique_id": f"test/algebra/{i}.json", "problem": f"Compute ${i}+{i}$.", "answer": str(2 * i)}
            for i in range(7)]

    def pair(p, pid, c="good answer", r="bad answer"):
        return {"prompt": p, "prompt_id": pid,
                "chosen": [{"role": "user", "content": p}, {"role": "assistant", "content": c}],
                "rejected": [{"role": "user", "content": p}, {"role": "assistant", "content": r}]}

    train_prefs = [pair(f"Pref prompt number {i} asks something.", f"p{i:03d}") for i in range(40)]
    train_prefs.append(pair("Please solve: " + MMLU_Q + " Explain.", "embed"))        # embedded benchmark prompt
    train_prefs.append(pair("Held out preference prompt 0.", "dup_test"))             # also in test_prefs
    train_prefs.append(pair("Pref prompt number 1 asks something.", "dup_train"))     # duplicate prompt
    train_prefs.append(pair("Identical pair prompt.", "ident", "same", "same"))       # identical responses
    multi = pair("Multi turn prompt.", "multi")
    multi["chosen"] = multi["chosen"] * 2
    train_prefs.append(multi)
    test_prefs = [pair(f"Held out preference prompt {i}.", f"t{i}") for i in range(6)]
    mmlu_dev = [{"question": f"Dev question {i}?", "subject": subjects[i % 3], "choices": ["a", "b", "c", "d"],
                 "answer": 0} for i in range(15)]
    return {"wikitext": {"train": wiki("train", 12), "validation": wiki("valid", 6), "test": wiki("test", 3)},
            "gsm8k": {"train": gsm_train, "test": gsm_test},
            "ultrafeedback": {"train_prefs": train_prefs, "test_prefs": test_prefs},
            "math500": {"test": math}, "mmlu": {"test": mmlu, "dev": mmlu_dev}, "arc_easy": {"test": arc}}


class TextAndSplitTest(unittest.TestCase):
    def test_prompt_matches_campaign_copy(self):
        self.assertEqual(d.prompt("Q?"), d.PROMPT_PREFIX + "Q?\nAnswer:")
        path = Path(__file__).resolve().parents[1] / "jr1_campaign.py"
        src = path.read_text()
        self.assertIn('return "' + d.PROMPT_PREFIX.replace("\n", "\\n") + '" + question + "\\nAnswer:"', src)

    def test_normalization_and_answers(self):
        self.assertEqual(d.norm_ws("  A\tB\n c "), "a b c")
        self.assertEqual(d.norm_words("What's 2+2?"), "what s 2 2")
        self.assertEqual(d.strip_calculator("48/2=<<48/2=24>>24\n#### 72"), "48/2=24\n#### 72")
        self.assertEqual(d.final_answer("x\n#### 1,000"), "1,000")
        self.assertEqual(d.paragraphs(" = T = \n\n body . \n = = S = = \n"), ["body ."])

    def test_wikitext_articles(self):
        arts = d.wikitext_articles(["", " = A = \n", " x \n", " = = s = = \n", " y \n", " = B = \n", " z \n"], "train")
        self.assertEqual([a["id"] for a in arts], ["wikitext103:train:000001", "wikitext103:train:000002"])
        self.assertEqual(arts[0]["text"], " = A = \n x \n = = s = = \n y \n")

    def test_gsm8k_split_disjoint_sized_and_order_free(self):
        rows = d.gsm8k_records([{"question": f"q{i}", "answer": "#### 1"} for i in range(200)], "train")
        s = d.split_gsm8k(rows, SMALL)
        self.assertEqual([r["index"] for r in s["readiness"]], list(range(8)))
        self.assertEqual((len(s["tune"]), len(s["controller"])), (10, 10))
        self.assertEqual(len(s["sft_train"]) + len(s["rl_train"]), 200 - 28)
        self.assertLessEqual(abs(len(s["sft_train"]) - len(s["rl_train"])), 1)
        ids = [r["id"] for v in s.values() for r in v]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 200)
        shuffled = rows[:]
        random.Random(1).shuffle(shuffled)
        s2 = d.split_gsm8k(shuffled, SMALL)
        for k in s:
            self.assertEqual(sorted(r["id"] for r in s[k]), sorted(r["id"] for r in s2[k]), k)

    def test_stratified_quota(self):
        q = d.stratified_quota({"a": 1000, "b": 100, "c": 3}, 50)
        self.assertEqual(sum(q.values()), 50)
        self.assertGreaterEqual(min(q.values()), 1)
        self.assertGreater(q["a"], q["b"])
        real = d.stratified_quota({f"s{i}": 100 + 7 * i for i in range(57)}, 512)
        self.assertEqual(sum(real.values()), 512)


class AuditTest(unittest.TestCase):
    def test_kinds_threshold_and_keep(self):
        idx = d.EvalIndex({"bench": [("b1", MMLU_Q), ("b2", "Short q?")]})
        self.assertEqual({k for _, _, k in idx.hits([MMLU_Q], [MMLU_Q])}, {"exact", "normalized", "embedded"})
        self.assertEqual({k for _, _, k in idx.hits([MMLU_Q.upper() + "  "], [])}, {"normalized"})
        self.assertEqual({k for _, _, k in idx.hits(["other"], ["Intro. " + MMLU_Q.lower() + " Thanks"])}, {"embedded"})
        self.assertEqual(idx.hits(["x"], ["The answer to Short q? is here"]), set())   # below MIN_EMBED_CHARS
        self.assertEqual(idx.hits(["x"], [MMLU_Q[:-1] + "s"]), set())                  # word boundary respected
        generic = "Which of the following statements is true about this case?"
        idx2 = d.EvalIndex({"bench": [("g1", generic), ("g2", generic), ("u", MMLU_Q)]})
        self.assertEqual(idx2.generic_patterns, 1)
        self.assertEqual(idx2.hits(["x"], ["Intro: " + generic]), set())
        self.assertEqual({k for _, _, k in idx2.hits([generic], [])}, {"exact", "normalized"})
        self.assertEqual(d.paragraphs(" short line \n " + "long words " * 10 + "\n", d.MIN_PARAGRAPH_CHARS),
                         [("long words " * 10).strip()])
        report, matches = {}, []
        rows = [{"id": "r0", "t": MMLU_Q}, {"id": "r1", "t": "clean"}, {"id": "r2", "t": MMLU_Q}]
        kept = d.audit_pool("pool", rows, idx, lambda r: [r["t"]], lambda r: [r["t"]], report, matches,
                            keep_ids=frozenset({"r2"}))
        self.assertEqual([r["id"] for r in kept], ["r1", "r2"])
        self.assertEqual(report["pool"]["rows_dropped"], 1)
        self.assertEqual(report["pool"]["rows_flagged_kept"], 1)
        self.assertEqual(report["pool"]["by_eval"]["bench"]["rows"], 2)


class BuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files, cls.splits, cls.audit = d.build(synthetic_tables(), SMALL, expected=None)

    def test_eval_sets_complete_and_never_dropped(self):
        f = self.files
        self.assertEqual(len(f["eval/gsm8k_test.jsonl"]), 20)
        self.assertEqual(len(f["eval/mmlu_test.jsonl"]), 40)
        self.assertEqual(len(f["eval/arc_easy_test.jsonl"]), 9)
        self.assertEqual(len(f["eval/math500.jsonl"]), 7)
        self.assertEqual(f["eval/mmlu_dev.jsonl"][0]["id"], "mmlu:dev:00000")
        self.assertEqual(len(f["pref/test.jsonl"]), 6)
        self.assertEqual(len(f["pt/test.jsonl"]), 3)
        self.assertEqual([r["answer_idx"] for r in f["eval/arc_easy_test.jsonl"][:2]], [2, 1])
        self.assertEqual(len(f["eval/lc_gsm8k.jsonl"]), 6)
        lc = collections.Counter(r["subject"] for r in f["eval/lc_mmlu.jsonl"])
        self.assertEqual(sum(lc.values()), 12)
        self.assertEqual(set(lc), {"anatomy", "biology", "law"})

    def test_contaminated_train_rows_dropped(self):
        f, pools = self.files, self.audit["pools"]
        train_q = {r["prompt"] for k in ("sft/train.jsonl", "rl/train.jsonl", "gsm8k_dev/tune.jsonl",
                                          "gsm8k_dev/controller.jsonl") for r in f[k]}
        self.assertFalse(any(GSM_Q.casefold() in q.casefold() for q in train_q))
        self.assertEqual(pools["gsm8k_train"]["rows_dropped"], 2)
        self.assertEqual(pools["gsm8k_train"]["rows_flagged_kept"], 1)
        self.assertIn(d.prompt("Readiness copy: " + MMLU_Q), [r["prompt"] for r in f["gsm8k_dev/readiness.jsonl"]])
        self.assertEqual(self.audit["filters"]["gsm8k_train_duplicate_question"], 1)
        pref_ids = {r["id"] for k in ("pref/train.jsonl", "pref/tune.jsonl", "pref/controller.jsonl") for r in f[k]}
        for bad in ("embed", "dup_test", "dup_train", "ident", "multi"):
            self.assertNotIn(f"ultrafeedback:train_prefs:{bad}", pref_ids)
        self.assertEqual(len(pref_ids), 40)
        self.assertEqual(pools["pref_train"]["by_eval"]["mmlu_test"]["embedded"], 1)
        self.assertEqual(self.audit["filters"]["train_prefs_prompt_in_test_prefs"], 1)

    def test_schemas_sizes_disjoint(self):
        d.check_integrity(self.files, SMALL)
        f = self.files
        self.assertEqual(len(f["gsm8k_dev/readiness.jsonl"]), 8)
        self.assertEqual(len(f["sft/train.jsonl"]) + len(f["rl/train.jsonl"]), 80 - 1 - 2 - 28)
        sft = f["sft/train.jsonl"][0]
        self.assertTrue(sft["response"].startswith(" ") and "<<" not in sft["response"] and "####" in sft["response"])
        self.assertNotIn("####", f["rl/train.jsonl"][0]["answer"])
        self.assertEqual(len(f["pt/tune.jsonl"]) + len(f["pt/controller.jsonl"]), len(f["pt/validation.jsonl"]))

    def test_integrity_check_catches_overlap(self):
        bad = {k: list(v) for k, v in self.files.items()}
        bad["rl/train.jsonl"] = bad["rl/train.jsonl"] + [{"id": bad["sft/train.jsonl"][0]["id"], "prompt": "p",
                                                          "answer": "1"}]
        with self.assertRaises(AssertionError):
            d.check_integrity(bad, SMALL)

    def test_deterministic(self):
        again = d.build(synthetic_tables(), SMALL, expected=None)
        self.assertEqual(json.dumps(again, sort_keys=True), json.dumps((self.files, self.splits, self.audit),
                                                                       sort_keys=True))

    def test_expected_release_counts_enforced(self):
        with self.assertRaises(RuntimeError):
            d.build(synthetic_tables(), SMALL)

    def test_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            d.write_jsonl(Path(tmp) / "eval/gsm8k_test.jsonl", self.files["eval/gsm8k_test.jsonl"])
            self.assertEqual(d.load("eval/gsm8k_test", tmp), self.files["eval/gsm8k_test.jsonl"])


def _prepared_root():
    root = os.environ.get("JR1_DATA_ROOT") or (os.environ.get("JR1_STATE_ROOT") and
                                               os.path.join(os.environ["JR1_STATE_ROOT"], "data"))
    return root if root and os.path.exists(os.path.join(root, "manifest.json")) else None


@unittest.skipUnless(_prepared_root() and importlib.util.find_spec("ahocorasick"), "no prepared data directory")
class RealManifestTest(unittest.TestCase):
    def test_verify(self):
        root = _prepared_root()
        self.assertTrue(d.verify(root)["ok"])
        man = d.manifest(root)
        self.assertEqual(man["files"]["eval/mmlu_test.jsonl"]["rows"], 14042)
        self.assertEqual(man["files"]["eval/mmlu_dev.jsonl"]["rows"], 285)
        self.assertEqual(len(man["splits"]["gsm8k"]["readiness"]), 128)
        for src in man["sources"].values():
            self.assertRegex(src["revision"], r"^[0-9a-f]{40}$")


if __name__ == "__main__":
    unittest.main()
