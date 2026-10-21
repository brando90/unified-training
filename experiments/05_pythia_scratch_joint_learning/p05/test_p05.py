"""Exp 05 unit tests (CPU, offline). Run from experiments/05_pythia_scratch_joint_learning with
PYTHONPATH=.:../03_qwen_joint_reasoning_retention/expt_v1  python -m pytest -q p05/test_p05.py
The packing test needs the pinned GPT-NeoX tokenizer in $HF_HOME and is skipped otherwise."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from p05 import collect, make_configs, spec
from p05.calibrate import rates, round_up
from p05.data import EOT, pack_blocks
from p05.train import CountingGradScaler, build


def test_cells_are_complete_paired_blocks():
    for version, sizes in spec.VERSIONS.items():
        cells = spec.cells(version)
        assert len(cells) == 3 * 3 * len(sizes)
        assert len({c["cell"] for c in cells}) == len(cells)
        block = 3 * len(sizes)
        for start in range(0, len(cells), block):
            chunk = cells[start:start + block]
            assert len({c["seed"] for c in chunk}) == 1
            assert {(c["size"], c["method"]) for c in chunk} == {(s, m) for s in sizes for m in spec.METHODS}
    assert len(spec.cells("v1")) + len(spec.cells("v2")) == 27


def test_committed_configs_match_spec():
    assert make_configs.main(["--check"]) == 0


def test_configs_are_v100_safe_and_matched():
    for version, sizes in spec.VERSIONS.items():
        for size in sizes:
            fp16, bf16 = spec.config(size, "fp16", version), spec.config(size, "bf16", version)
            assert fp16["precision"]["compute"] == "fp16" and fp16["precision"]["checkpoint"] != "bf16"
            assert fp16["model"]["attn_implementation"] != "flash_attention_2"
            assert fp16["objectives"]["rl"]["backend"] == "hf"
            assert fp16["model"]["source"] == "random" and fp16["model"]["init_seed"] is None   # fresh init per seed
            fp16["precision"].pop("compute"), bf16["precision"].pop("compute")
            for c in (fp16, bf16):
                c.pop("protocol"), c["p05"].pop("precision"), c["p05"].pop("gpu")
            assert fp16 == bf16   # precision is the only difference
    targets = {spec.config(s, "fp16", v)["work"]["target"] for v, ss in spec.VERSIONS.items() for s in ss}
    assert len(targets) == 1   # one charged-work target for every size


def test_counting_grad_scaler_counts_skips():
    w = torch.nn.Parameter(torch.ones(3))
    opt = torch.optim.SGD([w], lr=0.1)
    scaler = CountingGradScaler("cpu", init_scale=2.0 ** 4)
    for bad in (False, True, False, True):
        opt.zero_grad()
        loss = (w * (float("inf") if bad else 1.0)).sum()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    assert scaler.skipped == 2


def test_frozen_config_needs_matching_record(tmp_path):
    cfg = spec.config("14m", "fp16", "v2")
    cfg["frozen"] = True
    path = tmp_path / "pythia14m_fp16.json"
    text = json.dumps(cfg)
    path.write_text(text)
    with pytest.raises(SystemExit, match="FROZEN.sha256"):
        build(str(path), str(tmp_path), None, False)
    import hashlib
    (tmp_path / "FROZEN.sha256").write_text(f"{hashlib.sha256(text.encode()).hexdigest()}  {path.name}\n")
    got, _, _ = build(str(path), str(tmp_path), None, False)
    assert got["frozen"] and got["data"]["root"] == str(tmp_path)
    with pytest.raises(SystemExit, match="refused"):
        build(str(path), str(tmp_path), 123.0, False)
    draft = tmp_path / "draft.json"
    draft.write_text(json.dumps(spec.config("14m", "fp16", "v2")))
    with pytest.raises(SystemExit, match="DRAFT"):
        build(str(draft), str(tmp_path), None, False)


def test_rates_sequential_pt_share_is_stage_length():
    per = {o: {"charge_mean": c, "wall_s_mean": 1.0, "pt_tokens_mean": 0} for o, c in
           {"pt": 300.0, "sft": 100.0, "dpo": 400.0, "rl": 50.0}.items()}
    per["pt"]["pt_tokens_mean"] = 100.0
    r = rates(spec.config("160m", "fp16", "v1"), per)
    assert abs(r["sequential"]["work_share"]["pt"] - 0.55) < 1e-3
    assert abs(r["sequential"]["pt_tokens_per_work"] - 0.55 / 3) < 1e-3
    fj = 0.55 * 300 / (0.55 * 300 + 0.2 * 100 + 0.1 * 400 + 0.15 * 50)
    assert abs(r["fixed_joint"]["work_share"]["pt"] - fj) < 1e-6
    assert round_up(2.731e8) == 2.8e8


def test_collect_lists_unstarted_cells(tmp_path):
    out = tmp_path / "out"
    assert collect.main(["--runs", str(tmp_path / "runs"), "--out", str(out)]) == 0
    data = json.loads((out / "collect.json").read_text())
    assert data["prospective_cells"] == 27 and data["status_counts"] == {"unstarted": 27}


def _tokenizer():
    try:
        from p05.data import load_tokenizer
        return load_tokenizer()
    except Exception:
        return None


@pytest.mark.skipif(_tokenizer() is None, reason="pinned GPT-NeoX tokenizer not cached")
def test_pack_blocks_exact_windows():
    tok = _tokenizer()
    arts = [{"id": f"a{i}", "text": f" = Title {i} = \n\n The quick brown fox number {i} jumps. " * (i + 3)} for i in range(12)]
    blocks = list(pack_blocks(arts, tok, 32, batch=5))
    stream = []
    for a in arts:
        stream += tok.encode(a["text"], add_special_tokens=False) + [tok.eos_token_id]
    assert len(blocks) == len(stream) // 32
    for b in blocks:
        n = len(tok.encode(b["text"], add_special_tokens=False))
        assert abs(n - 32) <= 2, (n, b["text"])
    assert any(EOT in b["text"] for b in blocks)
    assert tok.encode(EOT, add_special_tokens=False) == [tok.eos_token_id]


def test_forward_on_gpt_neox_matches_full_logits():
    """P05Trainer._forward (decoder via base_model_prefix, row-wise LM head) on a tiny random GPT-NeoX equals
    token log-probs from the full logits; jr1's model.get_decoder() path does not apply to GPT-NeoX."""
    from types import SimpleNamespace
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    from jr1 import common as C
    from p05.train import P05Trainer
    torch.manual_seed(0)
    model = GPTNeoXForCausalLM(GPTNeoXConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                                             intermediate_size=64, max_position_embeddings=64)).eval()
    assert not hasattr(model.get_decoder()(input_ids=torch.ones(1, 3, dtype=torch.long)), "last_hidden_state")
    charged = []
    fake = SimpleNamespace(pad=0, device=torch.device("cpu"), compute="fp32",
                           work=SimpleNamespace(add=lambda *a: charged.append(a)))
    seqs, starts = [[5, 6, 7, 8, 9], [10, 11, 12]], [2, 1]
    lp, valid, n = P05Trainer._forward(fake, model, seqs, starts, "train", "forward")
    ids, attn, labels = C.pack(seqs, starts, 0, torch.device("cpu"))
    ref, ref_valid = C.token_logps(model(input_ids=ids, attention_mask=attn).logits, labels)
    assert torch.equal(valid, ref_valid) and torch.allclose(lp, ref, atol=1e-5)
    assert n == ids.numel() and charged == [("train", "forward", ids.numel())]
