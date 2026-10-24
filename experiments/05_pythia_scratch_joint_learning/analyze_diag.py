#!/usr/bin/env python3
"""AT-1024: collect stage-by-stage WikiText NLL and per-stage training stats of the Exp 05 diagnosis runs (stdlib only).
usage: analyze_diag.py <runs root> > summary.json   (prints JSON; also markdown table on stderr)"""
import json, sys, statistics as st
from pathlib import Path
root = Path(sys.argv[1])
out = {}
def nll(p):
    try: return json.loads(p.read_text())["wikitext"]["nll_per_token"]
    except Exception: return None
for sd in sorted(root.glob("s*")):
    s = sd.name[1:]
    pt = nll(sd / "pt/stages/after_pt/stage.json")
    out.setdefault(s, {})["after_pt_stage0"] = pt
    for v in ("orig", "lowlr", "sftonly", "ptonly"):
        d = sd / v
        if not d.exists(): continue
        r = {}
        for sg in ("sft", "dpo", "rl", "pt"):
            p = d / f"stages/after_{sg}/stage.json"
            if p.exists(): r["nll_after_" + sg] = nll(p)
        log = d / "log.jsonl"
        stats = {}
        if log.exists():
            for line in log.open():
                rec = json.loads(line)
                if rec.get("action") != "update" or rec.get("stage") is None: continue
                k = rec["stage"]; a = stats.setdefault(k, {"n": 0, "loss": [], "gn": [], "lr": [], "margin": [], "acc": [], "rew": []})
                a["n"] += 1
                if rec.get("loss") is not None: a["loss"].append(rec["loss"])
                if rec.get("grad_norm") is not None: a["gn"].append(rec["grad_norm"])
                a["lr"].append(rec["lr"])
                if "dpo_margin_mean" in rec: a["margin"].append(rec["dpo_margin_mean"]); a["acc"].append(rec["dpo_accuracy"])
                if "reward_mean" in rec: a["rew"].append(rec["reward_mean"])
        def summ(a):
            f = lambda x: (round(st.mean(x), 4) if x else None)
            q = lambda x, n: (f(x[:n]), f(x[-n:])) if x else None
            return {"updates": a["n"], "loss_first20_last20": q(a["loss"], 20), "grad_norm_mean": f(a["gn"]),
                    "lr_first_last": (a["lr"][0], a["lr"][-1]), "dpo_margin_first20_last20": q(a["margin"], 20),
                    "dpo_acc_first20_last20": q(a["acc"], 20), "rl_reward_mean": f(a["rew"])}
        r["stage_training"] = {k: summ(a) for k, a in stats.items()}
        rc = d / "receipt.json"
        if rc.exists():
            j = json.loads(rc.read_text()); r["status"] = j.get("status"); r["fp16_skipped"] = j.get("p05", {}).get("fp16_skipped_steps")
            r["wall_s"] = j.get("wall_s")
        out[s][v] = r
print(json.dumps(out, indent=1))
rows = []
for s, d in out.items():
    for v in ("orig", "lowlr", "sftonly", "ptonly"):
        if v in d:
            x = d[v]; rows.append(f"| {s} | {v} | {d['after_pt_stage0']} | {x.get('nll_after_sft')} | {x.get('nll_after_dpo')} | {x.get('nll_after_rl')} | {x.get('nll_after_pt')} |")
print("| seed | variant | after PT | after SFT | after DPO | after RL | PT-only end |\n|---|---|---|---|---|---|---|\n" + "\n".join(rows), file=sys.stderr)
