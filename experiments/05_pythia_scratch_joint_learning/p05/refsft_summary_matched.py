#!/usr/bin/env python3
"""AT-1111: stage NLL + DPO/RL stats for refsft_* runs vs orig/lowlr (stdlib). usage: refsft_summary.py <runs root>"""
import json, sys, statistics as st
from pathlib import Path
root = Path(sys.argv[1]); out = {}
def nll(p):
    try: return json.loads(p.read_text())["wikitext"]["nll_per_token"]
    except Exception: return None
f = lambda x: round(st.mean(x), 4) if x else None
for sd in sorted(root.glob("s*")):
    for v in ("lowlr", "refsft_lowlr", "refsft_lowlr_matched"):
        d = sd / v
        if not (d / "receipt.json").exists(): continue
        r = {"nll_" + sg: nll(d / f"stages/after_{sg}/stage.json") for sg in ("sft", "dpo", "rl")}
        stats = {}
        for line in (d / "log.jsonl").open():
            rec = json.loads(line)
            if rec.get("action") != "update" or rec.get("stage") is None: continue
            a = stats.setdefault(rec["stage"], {"loss": [], "gn": [], "margin": [], "acc": [], "rew": []})
            for k, kk in (("loss", "loss"), ("gn", "grad_norm"), ("margin", "dpo_margin_mean"), ("acc", "dpo_accuracy"), ("rew", "reward_mean")):
                if rec.get(kk) is not None: a[k].append(rec[kk])
        for k in ("dpo", "rl"):
            a = stats.get(k)
            if a: r[k] = {"n": len(a["loss"]), "loss_first20_last20": [f(a["loss"][:20]), f(a["loss"][-20:])], "gn_mean": f(a["gn"]),
                          "margin_f20_l20": [f(a["margin"][:20]), f(a["margin"][-20:])] if a["margin"] else None,
                          "acc_f20_l20": [f(a["acc"][:20]), f(a["acc"][-20:])] if a["acc"] else None, "rl_reward_mean": f(a["rew"])}
        out.setdefault(sd.name[1:], {})[v] = r
print(json.dumps(out, indent=1))
