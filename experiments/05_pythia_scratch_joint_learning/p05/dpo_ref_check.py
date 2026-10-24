"""Exp 05 AT-1085: is the DPO stage's initial loss of 11-21 a reference mismatch or an implementation bug?

    python -m p05.dpo_ref_check --seed S --pt <root>/s<S>/pt --variant {orig,lowlr} --out <dir> --config ... --data-root ...

Copies the PT state (as `p05.diag branch`), trains the SFT stage only (stop at the SFT -> DPO boundary), then scores the
first --n-batches DPO training batches (the frozen recipe's batch size) at the post-SFT policy, with three references:
  random_init   ckpt_000, the frozen recipe's reference
  self          a frozen copy of the post-SFT policy (step 0 must give loss ln 2 = 0.6931, margin 0)
  after_pt      the end-of-PT model (reference one stage back)
Also reports mean sequence log probabilities and the gradient norm of one batch under each. Writes <out>/dpo_ref_check.json.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
from pathlib import Path

import torch
from jr1 import common as C
from jr1 import train as T

from . import diag as D


def score(tr, ref_model, rows_batches):
    tr.reference = ref_model
    out = {"loss": [], "margin": [], "acc": [], "pol_c": [], "pol_r": [], "ref_c": [], "ref_r": []}
    c = tr.cfg["objectives"]["dpo"]
    for rows in rows_batches:
        seqs, starts = tr._pref_batch(rows, c["max_len"])
        with torch.no_grad():
            lp, _, _ = tr._forward(tr.model, seqs, starts, "validation", "forward")
        pol = lp.sum(-1)
        ref = tr._reference_logps(seqs, starts, "validation")
        k = len(rows)
        loss = C.dpo_loss(pol[:k], pol[k:], ref[:k], ref[k:], c["beta"])
        m = c["beta"] * ((pol[:k] - pol[k:]) - (ref[:k] - ref[k:]))
        out["loss"].append(float(loss)); out["margin"] += m.float().tolist(); out["acc"] += (m > 0).float().tolist()
        out["pol_c"] += pol[:k].float().tolist(); out["pol_r"] += pol[k:].float().tolist()
        out["ref_c"] += ref[:k].float().tolist(); out["ref_r"] += ref[k:].float().tolist()
    mean = lambda x: sum(x) / len(x)
    # gradient norm of the first batch
    tr.model.zero_grad(set_to_none=True)
    tr.model.train()
    loss, _, _ = tr.loss_dpo(rows_batches[0], "validation")
    loss.backward()
    gn = float(torch.nn.utils.clip_grad_norm_(tr.model.parameters(), 1e9))
    tr.model.zero_grad(set_to_none=True)
    return {"batches": len(rows_batches), "pairs": len(out["margin"]), "loss_mean": mean(out["loss"]),
            "loss_min": min(out["loss"]), "loss_max": max(out["loss"]), "margin_mean": mean(out["margin"]),
            "acc_mean": mean(out["acc"]), "policy_chosen_logp": mean(out["pol_c"]), "policy_rejected_logp": mean(out["pol_r"]),
            "ref_chosen_logp": mean(out["ref_c"]), "ref_rejected_logp": mean(out["ref_r"]),
            "grad_norm_batch0_unclipped": gn}


def frozen_copy(tr, src_model, dtype):
    m = copy.deepcopy(src_model).to(dtype).eval().requires_grad_(False)
    m.config.use_cache = False
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--pt", type=Path, required=True)
    ap.add_argument("--variant", choices=["orig", "lowlr"], required=True)
    ap.add_argument("--lr-scale", type=float, default=0.1)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-batches", type=int, default=16)
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists")
    shutil.copytree(a.pt, a.out, ignore=shutil.ignore_patterns("stages"))
    (a.out / "stages").mkdir()
    shutil.copytree(a.pt / "stages", a.out / "stages" / "_pt_shared", dirs_exist_ok=True)
    tr = D.build_trainer(a)
    tr.variant, tr.lr_scale = a.variant, a.lr_scale
    tr.load_state(a.out / "state")
    pt_model = copy.deepcopy(tr.model).eval().requires_grad_(False)       # end-of-PT weights
    D.run_loop(tr, a.out, a.data_root, stop_at_stage=2)                    # SFT only; state saved at the boundary
    assert tr.eff_stage(tr.progress()) == 2, tr.progress()
    p = tr.cfg["precision"]
    rdtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[tr.compute] if p["reference"] == "compute" else tr.master
    batch = tr.cfg["objectives"]["dpo"]["batch_size"]
    pool = tr.pools["dpo"]
    batches = [pool[i * batch:(i + 1) * batch] for i in range(a.n_batches)]
    tr.model.eval()
    res = {"seed": a.seed, "variant": a.variant, "progress_at_check": tr.progress(), "sft_updates_total": tr.updates,
           "beta": tr.cfg["objectives"]["dpo"]["beta"], "ln2": math.log(2), "reference_dtype": str(rdtype)}
    tr.reference = None
    tr._load_reference()
    res["random_init"] = score(tr, tr.reference, batches)
    res["self"] = score(tr, frozen_copy(tr, tr.model, rdtype), batches)
    res["after_pt"] = score(tr, frozen_copy(tr, pt_model, rdtype), batches)
    C.atomic_json(a.out / "dpo_ref_check.json", res)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
