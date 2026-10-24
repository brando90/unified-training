"""Synthetic jr1-layout data for the Exp 05 CPU smoke test (schemas only; never evidence).

    python -m p05.synthetic --out DIR [--n 24]

Writes jr1.tiny.write_synthetic_data's pools plus the files p05/data.py copies and the evaluator reads: pt/{test,
validation,tune}, pref/test, eval/{gsm8k_test,lc_gsm8k,arc_easy_test}.jsonl and a LAMBADA-shaped parquet.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from jr1.tiny import write_synthetic_data


def write(root: Path, rel: str, rows: list[dict]) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args(argv)
    root = a.out
    write_synthetic_data(root, n=a.n)
    rng = random.Random(1)
    words = "the river city music war season album game church ship king team film road station".split()
    arts = [{"id": f"wikitext103:train:{i:06d}",
             "text": f" = Article {i} = \n\n " + " ".join(rng.choice(words) for _ in range(rng.randint(150, 900))) + " \n"}
            for i in range(a.n)]
    write(root, "pt/train.jsonl", arts)
    for split in ("test", "validation", "tune"):
        write(root, f"pt/{split}.jsonl", arts[:3])
    write(root, "pref/test.jsonl", [r for r in map(json.loads, (root / "pref/train.jsonl").read_text().splitlines())][:4])
    gsm = [{"id": f"gsm8k:test:{i}", "question": f"What is {i} plus {i + 1}?", "answer": f"{i}+{i + 1}={2 * i + 1}\n#### {2 * i + 1}"}
           for i in range(3)]
    write(root, "eval/gsm8k_test.jsonl", gsm)
    write(root, "eval/lc_gsm8k.jsonl", gsm[:2])
    write(root, "eval/arc_easy_test.jsonl", [{"id": f"arc_easy:test:{i}", "subject": "arc_easy",
                                              "question": f"Which is a {w}?", "choices": [w, "rock", "blue", "seven"],
                                              "answer_idx": 0} for i, w in enumerate(words[:4])])
    import pyarrow as pa
    import pyarrow.parquet as pq
    lam = [" ".join(rng.choice(words) for _ in range(40)) for _ in range(5)]
    pq.write_table(pa.table({"text": lam}), root / "lambada.parquet")
    print(json.dumps({"out": str(root)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
