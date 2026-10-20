"""Deterministic CPU tests for the JR1 trainer on a tiny random Qwen2 (no network, no GPU).

    cd experiments/03_qwen_joint_reasoning_retention/expt_v1 && python -m unittest jr1.test_train -v
"""
import copy
import json
import math
import os
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(tempfile.mkdtemp(prefix="jr1-train-test-"))
os.environ.setdefault("JR1_STATE_ROOT", str(ROOT))

import torch

torch.set_num_threads(1)

from jr1 import common as C
from jr1 import controllers as S
from jr1 import train as T
from jr1.tiny import write_synthetic_data

DATA = ROOT / "data"
write_synthetic_data(DATA)
TINY = T.load_config(str(Path(__file__).parent / "configs" / "tiny_cpu.json"))[0]
TINY = T.deep_merge(TINY, {"data": {"root": str(DATA)},
                           "methods": {"aioli": {"alpha": 0.5, "probe_updates": 6, "window": 3},
                                       "jr1": {"alpha": 0.5, "probe_updates": 6, "window": 3}}})
# Random byte text never answers arithmetic, so RL tests use explicit reward rules.
T.REWARDS["test_mixed"] = lambda text, answer: float(sum(map(ord, text)) % 2)
T.REWARDS["test_zero"] = lambda text, answer: 0.0
COUNTER = iter(range(10 ** 6))


def config(**over):
    return T.deep_merge(TINY, over)


def trainer(method="fixed_joint", cfg=None, seed=1101, out=None, **kw):
    out = out or ROOT / "cells" / f"{method}-{next(COUNTER)}"
    return T.Trainer(cfg or config(), out, seed, method=None if "replay" in kw else method, **kw)


def params(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def strip(row):
    return {k: v for k, v in row.items() if k != "wall_s"}


class Objectives(unittest.TestCase):
    def test_parser_and_reward(self):
        self.assertEqual(C.extract_answer("so #### 1,200/3"), "4E+2")
        self.assertEqual(C.exact_reward("x #### 7", "7"), 1.0)
        self.assertEqual(C.exact_reward("x #### 8", "7"), 0.0)

    def test_sft_masks_prompt_and_padding(self):
        t = trainer()
        rows = t.pools["sft"][:2]
        pairs = [t._prompt_response(r["prompt"], r["response"], 512) for r in rows]
        seqs, starts = [s for s, _ in pairs], [k for _, k in pairs]
        ids, attn, labels = C.pack(seqs, starts, t.pad, "cpu")
        for i, (s, k) in enumerate(zip(seqs, starts)):
            self.assertTrue((labels[i, :k] == -100).all())                      # prompt masked
            self.assertEqual(labels[i, k:len(s)].tolist(), s[k:])                # response + EOS kept
            self.assertTrue((labels[i, len(s):] == -100).all())                  # padding masked
            self.assertEqual(s[-1], t.eos)
        self.assertEqual(len(t.encode(rows[0]["prompt"])), starts[0])
        short, k = t._prompt_response(rows[0]["prompt"], rows[0]["response"], 64)   # long prompt: keep its tail
        self.assertLessEqual(len(short), 64)
        self.assertEqual(k, 32)
        self.assertEqual(short[:k], t.encode(rows[0]["prompt"])[-32:])

    def test_pt_loss_is_shifted_cross_entropy(self):
        t = trainer()
        rows = t.pools["pt"][:2]
        loss, n, meta = t.loss_pt(rows, "train")
        seqs = [t.encode(r["text"])[:48] for r in rows]
        ids, attn, labels = C.pack(seqs, [0, 0], t.pad, "cpu")
        logits = t.model(input_ids=ids, attention_mask=attn).logits
        ref = torch.nn.functional.cross_entropy(logits[:, :-1].transpose(1, 2), labels[:, 1:], ignore_index=-100)
        self.assertAlmostEqual(float(loss), float(ref), places=5)
        self.assertEqual(meta["target_tokens"], sum(len(s) - 1 for s in seqs))

    def test_dpo_sign_and_direction(self):
        self.assertLess(float(C.dpo_loss(torch.tensor([2.]), torch.tensor([0.]), torch.tensor([0.]), torch.tensor([0.]))),
                        math.log(2))
        self.assertGreater(float(C.dpo_loss(torch.tensor([0.]), torch.tensor([2.]), torch.tensor([0.]), torch.tensor([0.]))),
                           math.log(2))
        t = trainer(cfg=config(optimizer={"lr": 0.01}))
        rows = t.pools["dpo"][:2]
        loss0, _, meta0 = t.loss_dpo(rows, "train")
        self.assertAlmostEqual(meta0["dpo_margin_mean"], 0.0, places=5)          # policy == frozen initial reference
        self.assertAlmostEqual(float(loss0), math.log(2), places=5)
        for _ in range(3):
            t.update("dpo", rows, "train")
        _, _, meta = t.loss_dpo(rows, "train")
        self.assertGreater(meta["dpo_margin_mean"], 0.0)                         # chosen pulled above rejected

    def test_rl_nonzero_gradient_steps(self):
        t = trainer(cfg=config(objectives={"rl": {"reward": "test_mixed"}}))
        before = params(t.model)
        rec = t.update("rl", t.pools["rl"][:2], "train")
        self.assertGreater(rec["informative_groups"], 0)
        self.assertTrue(rec["optimizer_step"])
        self.assertGreater(rec["grad_norm"], 0)
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in t.model.state_dict().items()))

    def test_rl_zero_variance_makes_no_update(self):
        t = trainer(cfg=config(objectives={"rl": {"reward": "test_zero"}}, optimizer={"weight_decay": 0.1}))
        t.update("pt", None, "train")                                           # optimizer has moments now
        before, opt_before = params(t.model), copy.deepcopy(t.opt.state_dict())
        rec = t.update("rl", t.pools["rl"][:2], "train")
        self.assertTrue(rec["zero_gradient"])
        self.assertFalse(rec["optimizer_step"])
        self.assertEqual(rec["zero_variance_groups"], 2)
        for k, v in t.model.state_dict().items():
            self.assertTrue(torch.equal(before[k], v), k)                       # no weight decay either
        for pid, s in t.opt.state_dict()["state"].items():
            self.assertTrue(torch.equal(s["step"], opt_before["state"][pid]["step"]))
        self.assertEqual(t.work.raw.get("train/backward", 0), t.exposure["pt"]["tokens"])   # no RL backward
        self.assertGreater(t.work.raw["train/generation"], 0)                   # rollouts are still charged

    def test_loo_advantages(self):
        adv = C.loo_advantages(torch.tensor([1., 0., 0., 0., 1., 1., 1., 1.]), 4)
        self.assertTrue(torch.allclose(adv[:4], torch.tensor([1., -1 / 3, -1 / 3, -1 / 3])))
        self.assertTrue(torch.equal(adv[4:], torch.zeros(4)))
        with self.assertRaises(ValueError):
            C.loo_advantages(torch.tensor([1., 0., 1.]), 2)


class Schedules(unittest.TestCase):
    def test_sequential_and_replay(self):
        seq = S.Sequential({"boundaries": [0.5, 0.7, 0.9]}, 0)
        self.assertEqual([OBJ for OBJ in (S.OBJECTIVES[seq.stage(f)] for f in (0, .49, .5, .69, .7, .95, 1.0))],
                         ["pt", "pt", "sft", "sft", "dpo", "rl", "rl"])
        rep = S.Replay({"boundaries": [0.5, 0.7, 0.9], "pt_share": 0.25}, 0)
        self.assertEqual(rep.weights(0.1), [1, 0, 0, 0])
        self.assertEqual(rep.weights(0.6), [0.25, 0.75, 0, 0])
        self.assertEqual(rep.weights(0.95), [0.25, 0, 0, 0.75])

    def test_fixed_and_smooth(self):
        fixed = S.FixedJoint({"weights": {"pt": 2, "sft": 1, "dpo": 1, "rl": 1}}, 0)
        self.assertEqual(fixed.weights(0.3), [0.4, 0.2, 0.2, 0.2])
        sm = S.SmoothJoint({"start": {"pt": .7, "sft": .2, "dpo": .05, "rl": .05},
                            "end": {"pt": .0, "sft": .2, "dpo": .0, "rl": .8}, "floors": {"pt": .05, "dpo": .05}}, 0)
        self.assertAlmostEqual(sm.weights(0)[0], 0.7)
        end = sm.weights(1.0)
        self.assertAlmostEqual(sum(end), 1.0)
        self.assertGreaterEqual(end[0], 0.05 - 1e-12)
        self.assertGreaterEqual(end[2], 0.05 - 1e-12)
        self.assertGreater(end[3], sm.weights(0.5)[3])

    def test_interleaver_tracks_weights(self):
        it, w = S.Interleaver(), [0.55, 0.2, 0.1, 0.15]
        counts = Counter(it.choose(w) for _ in range(1000))
        for o, x in zip(S.OBJECTIVES, w):
            self.assertLessEqual(abs(counts[o] - 1000 * x), 1)
        it = S.Interleaver()                                                    # stage switch: no stale picks
        picks = [it.choose([0.3, 0.7, 0, 0]) for _ in range(7)] + [it.choose([0.3, 0, 0.7, 0]) for _ in range(10)]
        self.assertNotIn("sft", picks[7:])

    def test_aioli_matches_frozen_program_math(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "experiments" / "00_program"))
        import aioli as frozen
        a = S.Aioli({"alpha": .75, "eta": 1.0, "window": 10, "probe_updates": 12}, 7)
        plan = a.probe_plan()
        self.assertEqual(sorted(plan["order"]), [0, 1, 2, 3])
        for j, actions in zip(plan["order"], plan["actions"]):
            c = Counter(actions)
            self.assertEqual(c[S.OBJECTIVES[j]], 9)
            self.assertEqual(sorted(c.values()), [1, 1, 1, 9])
        drops = [[.03, -.01, .02, .0], [.01, .04, -.02, .01], [.0, .0, .05, -.01], [-.02, .01, .0, .06]]
        ref = frozen.AioliController(alpha=.75, eta=1.0, seed=0)
        expected = ref.update(drops)
        a.feedback(drops, 0.1, {"pt": 1.0}, {"pt": 1.0})
        for i, o in enumerate(S.OBJECTIVES):
            self.assertAlmostEqual(a.weights(0.1)[i], expected[o], places=12)

    def test_jr1_prior_and_retention(self):
        cfg = copy.deepcopy(T.DEFAULTS["methods"]["jr1"])
        j = S.JR1(cfg, 0)
        early, late = j.weights(0.0), j.weights(1.0)
        self.assertGreater(early[0] + early[1], late[0] + late[1])               # PT/SFT-heavy early
        self.assertGreater(late[3], early[3])                                    # RL-heavy late
        for w in (early, late):
            self.assertAlmostEqual(sum(w), 1.0)
            self.assertTrue(all(x >= cfg["min_share"] - 1e-12 for x in w))
            self.assertGreaterEqual(w[0], cfg["pt_floor"] - 1e-12)
            self.assertGreaterEqual(w[2], cfg["dpo_floor"] - 1e-12)
        zero = [[0.0] * 4 for _ in range(4)]
        j.feedback(zero, 1.0, {"pt": 1.01}, {"pt": 1.0})                         # within 2%: no constraint
        self.assertFalse(j.retention_active())
        j.feedback(zero, 1.0, {"pt": 1.03}, {"pt": 1.0})                         # 3% worse on validation PT
        self.assertTrue(j.retention_active())
        self.assertGreaterEqual(j.weights(1.0)[0], cfg["retention_min_pt"] - 1e-12)


class Runs(unittest.TestCase):
    def test_methods_follow_their_schedule(self):
        t = trainer("sequential", cfg=config(work={"target": 6000},
                                             methods={"sequential": {"boundaries": [0.4, 0.6, 0.8]}},
                                             objectives={"rl": {"reward": "test_mixed"}}))
        t.run()
        rows = C.read_jsonl(t.log_path)
        stages = [S.OBJECTIVES.index(r["objective"]) for r in rows]
        self.assertEqual(stages, sorted(stages))                                 # PT -> SFT -> DPO -> RL
        self.assertEqual(set(stages), {0, 1, 2, 3})
        for r in rows:
            self.assertEqual(r["objective"], r["stage"])
        t = trainer("replay", cfg=config(work={"target": 6000}, objectives={"rl": {"reward": "test_mixed"}},
                                         methods={"replay": {"boundaries": [0.3, 0.6, 0.8], "pt_share": 0.3}}))
        t.run()
        rows = C.read_jsonl(t.log_path)
        later = [r["objective"] for r in rows if r["stage"] != "pt"]
        self.assertIn("pt", later)                                               # corpus replay in later stages
        self.assertTrue({"sft", "dpo"} <= set(later))

    def test_work_accounting_and_bounded_overshoot(self):
        t = trainer()
        t._load_reference()                                                     # writes ckpt_000, as run() does
        rec = t.update("pt", None, "train")
        self.assertEqual(rec["update_charge"], 3 * rec["tokens"])                # forward + 2x backward
        rec = t.update("dpo", None, "train")
        self.assertEqual(rec["update_charge"], 4 * rec["tokens"])                # + frozen reference forward
        t = trainer(cfg=config(objectives={"rl": {"reward": "test_mixed"}}))
        rows = t.pools["rl"][:2]
        prompts = [t.encode(r["prompt"])[-48:] for r in rows for _ in range(4)]
        seen, generate = [], t.generate
        t.generate = lambda *a, **k: seen.append(generate(*a, **k)) or seen[-1]
        rec = t.update("rl", rows, "train")
        steps = max(map(len, seen[0][0]))                                       # decoding stops when all rows end
        prefill = len(prompts) * max(map(len, prompts))
        self.assertEqual(t.work.raw["train/generation"], prefill + len(prompts) * (steps - 1))
        self.assertEqual(rec["update_charge"],
                         t.work.raw["train/generation"] + 3 * t.work.raw.get("train/forward", 0))
        t = trainer("jr1", cfg=config(work={"target": 9000}, objectives={"rl": {"reward": "test_mixed"}}))
        receipt = t.run()
        rows = C.read_jsonl(t.log_path)
        self.assertEqual(sum(r["action_charge"] for r in rows), receipt["work_total"])
        self.assertEqual(receipt["work_total"], t.work.total)
        self.assertLess(rows[-1]["work_total"] - rows[-1]["action_charge"], 9000)  # last action began below target
        self.assertGreaterEqual(receipt["work_total"], 9000)
        self.assertTrue(receipt["overshoot_bounded"])
        self.assertGreater(receipt["work_by_purpose"]["validation"], 0)          # controller feedback is charged
        self.assertGreater(receipt["work_by_purpose"]["probe"], 0)
        self.assertGreater(t.work.raw["validation/reference"], 0)
        self.assertEqual(rows[0]["action"], "validate")

    def test_checkpoints_and_reference(self):
        t = trainer("fixed_joint", cfg=config(work={"target": 5000}))
        initial = params(t.model)
        t.run()
        works = []
        for name, frac in (("000", 0), ("025", .25), ("050", .5), ("075", .75), ("100", 1)):
            meta = json.loads((t.out / f"ckpt_{name}" / "jr1_ckpt.json").read_text())
            self.assertEqual(meta["milestone"], frac)
            self.assertGreaterEqual(meta["work_total"], frac * 5000)
            works.append(meta["work_total"])
        self.assertEqual(works, sorted(works))
        self.assertEqual(works[0], 0)
        if t.reference is None:
            t._load_reference()
        for k, v in t.reference.state_dict().items():
            self.assertTrue(torch.equal(v, initial[k]), k)                      # DPO reference = initial weights

    def test_resume_reproduces_trajectory(self):
        cfg = config(work={"target": 30000}, objectives={"rl": {"reward": "test_mixed"}})
        full = trainer("jr1", cfg=cfg, out=ROOT / "cells" / "resume-full")
        full.run()
        log = C.read_jsonl(full.log_path)
        fed = [i for i, r in enumerate(log) if "controller" in r]
        self.assertTrue(fed and fed[0] < len(log) - 1)                          # a controller update, then production
        self.assertIn("train", {r.get("purpose") for r in log[fed[0] + 1:]})
        out = ROOT / "cells" / "resume-cut"
        part = trainer("jr1", cfg=cfg, out=out)
        r1 = part.run(stop_after_actions=7)                                     # inside the first probe sweep
        self.assertEqual(r1["status"], "interrupted")
        with open(part.log_path, "a") as f:                                    # a line written after the save
            f.write(json.dumps({"stale": True}) + "\n")
        part2 = trainer("jr1", cfg=cfg, out=out)
        r2 = part2.run(stop_after_actions=fed[0] - 7 + 2, resume=True)              # just past the first feedback
        self.assertEqual(r2["status"], "interrupted")
        part3 = trainer("jr1", cfg=cfg, out=out)
        r3 = part3.run(resume=True)
        self.assertEqual(r3["status"], "complete")
        self.assertEqual(r3["resumes"], 2)
        a, b = C.read_jsonl(full.log_path), C.read_jsonl(part3.log_path)
        self.assertEqual([strip(x) for x in a], [strip(x) for x in b])
        for k, v in full.model.state_dict().items():
            self.assertTrue(torch.equal(v, part3.model.state_dict()[k]), k)
        self.assertEqual(full.work.state_dict(), part3.work.state_dict())
        self.assertEqual(full.schedule.state_dict(), part3.schedule.state_dict())
        with self.assertRaises(ValueError):                                     # another cell's state is refused
            trainer("jr1", cfg=cfg, out=out, seed=1102).run(resume=True)
        with self.assertRaises(RuntimeError):                                   # no silent overwrite
            trainer("jr1", cfg=cfg, out=out).run()

    def test_record_and_replay_orders(self):
        cfg = config(work={"target": 6000}, objectives={"rl": {"reward": "test_mixed"}})
        rec_path = ROOT / "schedule.jsonl"
        src = trainer("fixed_joint", cfg=cfg, seed=9001, record=rec_path)
        src.run()
        schedule = C.read_jsonl(rec_path)
        meta = json.loads(Path(str(rec_path) + ".meta.json").read_text())
        self.assertEqual(len(schedule), src.updates)
        self.assertEqual(meta["counts"]["rl"]["updates"], src.exposure["rl"]["updates"])
        logs = {}
        for order in ("interleaved", "stage_sorted"):
            t = T.Trainer(cfg, ROOT / "cells" / f"replay-{order}", 1101, replay=rec_path, order=order)
            receipt = t.run()
            self.assertEqual(receipt["status"], "complete")
            logs[order] = C.read_jsonl(t.log_path)
            self.assertEqual(len(logs[order]), len(schedule))
            for o in S.OBJECTIVES:                                              # identical exposure per objective
                self.assertEqual(Counter(i for r in logs[order] if r["objective"] == o for i in r["example_ids"]),
                                 Counter(i for r in schedule if r["objective"] == o for i in r["ids"]))
                self.assertEqual(t.exposure[o]["examples"], meta["counts"][o]["examples"])
            self.assertEqual([r["lr"] for r in logs[order]], [r["lr"] for r in schedule])   # positional LR clock
        self.assertEqual([r["objective"] for r in logs["interleaved"]], [r["objective"] for r in schedule])
        self.assertEqual([r["example_ids"] for r in logs["interleaved"]], [r["ids"] for r in schedule])
        sorted_stages = [S.OBJECTIVES.index(r["objective"]) for r in logs["stage_sorted"]]
        self.assertEqual(sorted_stages, sorted(sorted_stages))
        self.assertGreater(len(set(sorted_stages)), 2)


if __name__ == "__main__":
    unittest.main()
