"""Offline tiny model and tokenizer for CPU tests and end-to-end checks (no network, no downloads).

The tokenizer is byte-level BPE with no merges (256 byte symbols + <|endoftext|> as EOS and pad), so any text
round-trips. The model is a randomly initialized Qwen2 architecture small enough for CPU unit tests.
"""
from __future__ import annotations

import torch

EOS = "<|endoftext|>"
TINY_QWEN2 = {"vocab_size": 260, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
              "num_attention_heads": 4, "num_key_value_heads": 2, "max_position_embeddings": 2048,
              "tie_word_embeddings": True, "attention_dropout": 0.0, "use_cache": False}


def tiny_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {ch: i for i, ch in enumerate(alphabet)}
    vocab[EOS] = len(vocab)
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens([EOS])
    return PreTrainedTokenizerFast(tokenizer_object=tok, eos_token=EOS, pad_token=EOS)


def tiny_model(overrides: dict | None = None, seed: int = 0, dtype=torch.float32):
    from transformers import AutoModelForCausalLM, Qwen2Config
    config = Qwen2Config(**{**TINY_QWEN2, **(overrides or {})})
    tok = tiny_tokenizer()
    config.eos_token_id = config.pad_token_id = config.bos_token_id = tok.eos_token_id
    torch.manual_seed(seed)
    return AutoModelForCausalLM.from_config(config, dtype=dtype, attn_implementation="eager")


def write_synthetic_data(root, n: int = 24) -> None:
    """Small arithmetic pools in the jr1/data.py layout (schemas only; never training evidence)."""
    import json
    import random
    from pathlib import Path

    from .common import prompt
    rng = random.Random(0)
    rows = {}
    for split in ("train", "controller"):
        pt, sft, rl, pref = [], [], [], []
        for i in range(n):
            a, b = rng.randint(1, 50), rng.randint(1, 50)
            q = f"What is {a} plus {b}?"
            pt.append({"id": f"pt:{split}:{i}", "text": f"The number {a} is followed by {a + 1}. " * rng.randint(1, 4)})
            sft.append({"id": f"sft:{split}:{i}", "prompt": prompt(q), "response": f" {a}+{b}={a + b}. #### {a + b}"})
            rl.append({"id": f"rl:{split}:{i}", "prompt": prompt(q), "answer": str(a + b)})
            pref.append({"id": f"pref:{split}:{i}", "prompt": f"Say something kind about {a}.",
                         "chosen": f" {a} is a fine number.", "rejected": f" {a} is bad."})
        rows.update({f"pt/{split}": pt, f"sft/{split}": sft, f"rl/{split}": rl, f"pref/{split}": pref})
    for name, data in rows.items():
        path = Path(root) / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in data))
