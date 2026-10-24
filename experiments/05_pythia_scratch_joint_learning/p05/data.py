#!/usr/bin/env python3
"""Exp 05 data root: GPT-NeoX-packed WikiText-103 PT blocks plus the jr1 pools and eval sets, with a hashed manifest.

    python -m p05.data prepare --jr1-data DIR --out DIR [--block-len 1024]
    python -m p05.data verify  --out DIR

Input is a jr1/data.py root (prepared and contamination-audited by `python -m jr1.data prepare`; its manifest.json
lists every file's sha256). Output layout under --out (one JSON object per line):

  pt_blocks/train.jsonl   {"id","text","articles"}  jr1 pt/train (WikiText-103-raw train articles, audited), each
                          article followed by "<|endoftext|>", tokenized with the pinned GPT-NeoX tokenizer and cut
                          into consecutive windows of exactly --block-len tokens (the last partial window is
                          dropped); text is the character span of the window, so the trainer re-encodes it to
                          (almost always exactly) block_len tokens. The manifest reports the re-encoded counts.
  pt/{test,validation,tune,controller}.jsonl, sft/train, rl/train, pref/{train,test}, eval/{gsm8k_test,lc_gsm8k,
                          arc_easy_test}.jsonl   byte copies of the jr1 files (sha256 checked against jr1's manifest)
  eval/lambada.jsonl      {"id","text"}  EleutherAI/lambada_openai 'default' test (5,153), pinned revision
  manifest.json           sources, tokenizer, block statistics, every file's rows / bytes / sha256

Contamination: the PT pool is jr1's audited pt/train (rows overlapping any eval set were already dropped there).
LAMBADA passages come from BookCorpus novels, not Wikipedia; an exact 50-character substring check of every LAMBADA
target passage against the packed PT text is recorded in the manifest (hits are reported, never dropped from eval).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from . import spec

EOT = "<|endoftext|>"
LAMBADA = {"repo": "EleutherAI/lambada_openai", "file": "default/test/default.parquet",
           "revision": "900124bf3b8235c6daf21033af9948b3f07346c4", "rows": 5153}
COPIES = ["pt/test.jsonl", "pt/validation.jsonl", "pt/tune.jsonl", "pt/controller.jsonl", "sft/train.jsonl",
          "rl/train.jsonl", "pref/train.jsonl", "pref/test.jsonl", "eval/gsm8k_test.jsonl", "eval/lc_gsm8k.jsonl",
          "eval/arc_easy_test.jsonl"]
TOKENIZER_SIZE = "160m"   # every Pythia size ships the same GPT-NeoX-20B tokenizer; the 160m revision is the pin


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f.read().split("\n") if line.strip()]


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def load_tokenizer(name: str | None = None):
    from transformers import AutoTokenizer
    s = spec.SIZES[TOKENIZER_SIZE]
    return AutoTokenizer.from_pretrained(name or s["id"], revision=None if name else s["revision"])


def pack_blocks(articles: list[dict], tok, block_len: int, batch: int = 512):
    """Yield {"id","text","articles"} windows of exactly block_len tokens of the stream article EOT article EOT ...

    Windows are character spans: a window covering tokens [a, b) of one article is text[off[a][0]:off[b-1][1]];
    a window crossing articles joins the spans with the EOT string (which the tokenizer maps to its single token).
    """
    pieces, count, k = [], 0, 0

    def emit():
        nonlocal pieces, count, k
        text = "".join(p for p, _ in pieces)
        ids = sorted({a for _, a in pieces if a is not None})
        row = {"id": f"p05pt:{k:07d}", "text": text, "articles": ids}
        pieces, count, k = [], 0, k + 1
        return row

    for i in range(0, len(articles), batch):
        chunk = articles[i:i + batch]
        enc = tok([a["text"] for a in chunk], add_special_tokens=False, return_offsets_mapping=True)
        for a, offs in zip(chunk, enc["offset_mapping"]):
            text, n, pos = a["text"], len(offs), 0
            while pos < n:
                take = min(block_len - count, n - pos)
                pieces.append((text[offs[pos][0]:offs[pos + take - 1][1]], a["id"]))
                count += take
                pos += take
                if count == block_len:
                    yield emit()
            pieces.append((EOT, None))
            count += 1
            if count == block_len:
                yield emit()


def lambada_rows(path: str) -> list[dict]:
    import pyarrow.parquet as pq
    texts = pq.read_table(path).column("text").to_pylist()
    return [{"id": f"lambada_openai:test:{i:05d}", "text": t} for i, t in enumerate(texts)]


def prepare(jr1_root: Path, out: Path, block_len: int, lambada_path: str | None = None,
            check_manifest: bool = True, expect_lambada: int | None = LAMBADA["rows"]) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    jr1_manifest_path = jr1_root / "manifest.json"
    jr1_manifest = json.loads(jr1_manifest_path.read_text()) if jr1_manifest_path.exists() else None
    if check_manifest and jr1_manifest is None:
        raise SystemExit(f"{jr1_manifest_path} missing: run `python -m jr1.data prepare` first")
    listing = {}
    for rel in COPIES:
        src = jr1_root / rel
        if not src.exists():
            if check_manifest:
                raise SystemExit(f"missing jr1 file {src}")
            continue
        digest = sha256_file(src)
        if check_manifest and jr1_manifest["files"][rel]["sha256"] != digest:
            raise SystemExit(f"{src} does not match jr1 manifest sha256")
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out / rel)
        listing[rel] = {"rows": len(read_jsonl(out / rel)), "bytes": (out / rel).stat().st_size, "sha256": digest}

    tok = load_tokenizer()
    articles = read_jsonl(jr1_root / "pt/train.jsonl")
    blocks = list(pack_blocks(articles, tok, block_len))
    if not blocks:
        raise SystemExit("no full PT block: the PT pool is shorter than one block")
    write_jsonl(out / "pt_blocks/train.jsonl", blocks)
    re_lens = []
    for i in range(0, len(blocks), 512):
        re_lens += [len(x) for x in tok([b["text"] for b in blocks[i:i + 512]], add_special_tokens=False)["input_ids"]]
    over = sum(n > block_len for n in re_lens)
    stats = {"block_len": block_len, "blocks": len(blocks), "packed_tokens": len(blocks) * block_len,
             "articles_in": len(articles), "reencoded_tokens_sum": sum(re_lens),
             "reencoded_min": min(re_lens), "reencoded_max": max(re_lens), "reencoded_over_block_len": over,
             "reencoded_exact": sum(n == block_len for n in re_lens),
             "trainer_tokens_per_epoch": sum(min(n, block_len) for n in re_lens),
             "note": "the trainer keeps the first block_len tokens of each row (jr1 loss_pt), so the observed tokens "
                     "per epoch are trainer_tokens_per_epoch"}
    listing["pt_blocks/train.jsonl"] = {"rows": len(blocks), "bytes": (out / "pt_blocks/train.jsonl").stat().st_size,
                                        "sha256": sha256_file(out / "pt_blocks/train.jsonl")}

    if lambada_path is None:
        from huggingface_hub import hf_hub_download
        lambada_path = hf_hub_download(LAMBADA["repo"], LAMBADA["file"], repo_type="dataset",
                                       revision=LAMBADA["revision"])
    lam = lambada_rows(lambada_path)
    if expect_lambada is not None and len(lam) != expect_lambada:
        raise SystemExit(f"LAMBADA has {len(lam)} rows, expected {expect_lambada}")
    write_jsonl(out / "eval/lambada.jsonl", lam)
    listing["eval/lambada.jsonl"] = {"rows": len(lam), "bytes": (out / "eval/lambada.jsonl").stat().st_size,
                                     "sha256": sha256_file(out / "eval/lambada.jsonl")}
    hits = lambada_overlap(lam, blocks)

    s = spec.SIZES[TOKENIZER_SIZE]
    tok_files = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        try:
            from huggingface_hub import hf_hub_download
            tok_files[name] = sha256_file(Path(hf_hub_download(s["id"], name, revision=s["revision"])))
        except Exception as exc:   # offline smoke with a local tokenizer
            tok_files[name] = f"unavailable: {type(exc).__name__}"
    manifest = {"created_by": "p05/data.py prepare", "block_len": block_len,
                "tokenizer": {"repo": s["id"], "revision": s["revision"], "files_sha256": tok_files,
                              "eos_token_id": tok.eos_token_id, "eot_string": EOT},
                "jr1_data": {"root": str(jr1_root),
                             "manifest_sha256": sha256_file(jr1_manifest_path) if jr1_manifest else None,
                             "pt_train_sha256": sha256_file(jr1_root / "pt/train.jsonl")},
                "lambada": {**LAMBADA, "parquet_sha256": sha256_file(Path(lambada_path)),
                            "pt_overlap_50char_hits": len(hits), "pt_overlap_examples": hits[:20]},
                "pt_blocks": stats, "files": listing,
                "observed_pt_target": spec.OBSERVED_PT_TARGET,
                "epochs_for_target": spec.OBSERVED_PT_TARGET / max(stats["trainer_tokens_per_epoch"], 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def lambada_overlap(lam: list[dict], blocks: list[dict], width: int = 50) -> list[str]:
    """LAMBADA ids whose final 50 characters (target sentence + word) occur verbatim in the packed PT text."""
    try:
        import ahocorasick
    except ImportError:
        return ["skipped: pyahocorasick not installed"]
    auto = ahocorasick.Automaton()
    for r in lam:
        tail = r["text"][-width:]
        if len(tail) == width:
            auto.add_word(tail, r["id"])
    auto.make_automaton()
    found = set()
    for b in blocks:
        for _, rid in auto.iter(b["text"]):
            found.add(rid)
    return sorted(found)


def verify(out: Path) -> dict:
    m = json.loads((out / "manifest.json").read_text())
    bad = {rel: e["sha256"] for rel, e in m["files"].items() if sha256_file(out / rel) != e["sha256"]}
    if bad:
        raise SystemExit(f"sha256 mismatch: {sorted(bad)}")
    print(json.dumps({"ok": True, "files": len(m["files"]), "blocks": m["pt_blocks"]["blocks"],
                      "trainer_tokens_per_epoch": m["pt_blocks"]["trainer_tokens_per_epoch"]}))
    return m


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--jr1-data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--block-len", type=int, default=spec.BASE["objectives"]["pt"]["max_len"])
    p.add_argument("--lambada-parquet", help="local copy of the pinned parquet (default: download it)")
    p.add_argument("--no-manifest-check", action="store_true", help="smoke tests on synthetic data only")
    v = sub.add_parser("verify")
    v.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        m = prepare(a.jr1_data, a.out, a.block_len, a.lambada_parquet, check_manifest=not a.no_manifest_check,
                    expect_lambada=None if a.no_manifest_check else LAMBADA["rows"])
        print(json.dumps({"blocks": m["pt_blocks"]["blocks"],
                          "trainer_tokens_per_epoch": m["pt_blocks"]["trainer_tokens_per_epoch"],
                          "epochs_for_target": round(m["epochs_for_target"], 3),
                          "lambada_overlap": m["lambada"]["pt_overlap_50char_hits"]}))
    else:
        verify(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
