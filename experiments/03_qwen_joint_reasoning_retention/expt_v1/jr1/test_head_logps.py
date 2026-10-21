"""head_token_logps (row-wise, checkpointed LM head) equals token_logps on full logits, in value and gradient."""
import torch

from jr1 import common as C
from jr1.tiny import tiny_model


def _batch(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    v = model.config.vocab_size
    seqs = [torch.randint(0, v, (n,), generator=g).tolist() for n in (9, 5, 12)]
    return C.pack(seqs, [2, 0, 4], 0, "cpu")


def _both(model, dtype):
    ids, attn, labels = _batch(model)
    model.zero_grad(set_to_none=True)
    with C.autocast("cpu", "bf16" if dtype is not None else "fp32"):
        logits = model(input_ids=ids, attention_mask=attn, use_cache=False).logits
    lp_old, v_old = C.token_logps(logits, labels)
    lp_old.sum().backward()
    g_old = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    with C.autocast("cpu", "bf16" if dtype is not None else "fp32"):
        hidden = model.get_decoder()(input_ids=ids, attention_mask=attn, use_cache=False).last_hidden_state
    lp_new, v_new = C.head_token_logps(hidden, model.get_output_embeddings(), labels, dtype)
    lp_new.sum().backward()
    g_new = [p.grad.clone() for p in model.parameters()]
    return lp_old, v_old, g_old, lp_new, v_new, g_new


def test_head_logps_matches_fp32():
    model = tiny_model(seed=0).train()
    lp_old, v_old, g_old, lp_new, v_new, g_new = _both(model, None)
    assert torch.equal(v_old, v_new)
    torch.testing.assert_close(lp_new, lp_old, rtol=1e-5, atol=1e-5)
    for a, b in zip(g_old, g_new):
        torch.testing.assert_close(b, a, rtol=1e-4, atol=1e-6)


def test_head_logps_matches_bf16_autocast():
    model = tiny_model(seed=1).train()
    lp_old, v_old, g_old, lp_new, v_new, g_new = _both(model, torch.bfloat16)
    assert torch.equal(v_old, v_new)
    torch.testing.assert_close(lp_new, lp_old, rtol=2e-2, atol=2e-2)


def test_head_logps_no_grad():
    model = tiny_model(seed=2).eval()
    ids, attn, labels = _batch(model, 3)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=attn, use_cache=False).logits
        hidden = model.get_decoder()(input_ids=ids, attention_mask=attn, use_cache=False).last_hidden_state
        a, _ = C.token_logps(logits, labels)
        b, _ = C.head_token_logps(hidden, model.get_output_embeddings(), labels)
    torch.testing.assert_close(b, a, rtol=1e-5, atol=1e-5)
