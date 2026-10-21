# Random-initialization joint learning: live results

**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/05_pythia_scratch_joint_learning/results.md>
**Status:** READY FOR CALIBRATION — measured training has not started.
**Last updated:** 10-20-2026

`experiments/05_pythia_scratch_joint_learning` has 27 prospective training cells: expt_v1 (Pythia-160m) 9 and expt_v2
(Pythia-14m and 70m) 18. Completed: 0; failed: 0; interrupted: 0; unstarted: 27. The charged-work target is pending
calibration. This denominator must remain visible if any condition is gated or removed prospectively.

| Phase | Status | Evidence |
|---|---|---|
| Design | DONE | [Campaign plan](../03_qwen_joint_reasoning_retention/PLAN.md); [v1](expt_v1/PROTOCOL.md) and [v2](expt_v2/PROTOCOL.md) protocols (DRAFT) |
| Code and CPU smoke | DONE | `smoke_cpu.sh`: 3 methods × 3 sizes complete, one cell through the cluster job body with evaluation, calibration |
| Calibration | PENDING | `SUBMIT.md` step 3 |
| Protocol and input freeze | PENDING | No measured cell admitted |
| Training/full evaluation | PENDING | 0/27 prospective cells |
| Final verification/publication | PENDING | Results unavailable |

Pre-measurement finding: the audited WikiText-103 train split packs into 115,840 GPT-NeoX blocks of 1024 tokens
(118.6M trainer tokens per epoch), so the 50M observed PT token target is 0.42 epoch.

No reasoning, retention or efficiency claim is established by this new experiment. The completed older pilot is
separate evidence and is not counted as execution of this plan.

**TLDR-end:** [unified-training: prospective execution] `experiments/05_pythia_scratch_joint_learning` is ready for
calibration; 27 training cells (9 at 160m, 18 at 14m/70m) await the frozen work target.
**Snapshot:**
```text
Prospective cells: 27 (v1 9, v2 18)
Completed: 0
Unstarted: 27
Code + CPU smoke: done
Measured training: not started
```
