# Pretrained joint reasoning and retention: live results

**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/03_qwen_joint_reasoning_retention/results.md>
**Status:** RUNNING (started 10-05-2026 17:44 UTC) — development/readiness only; measured training has not started.
**Last updated:** 10-05-2026 17:49 UTC

`experiments/03_qwen_joint_reasoning_retention` has 36 prospective training cells. Completed: 0; failed: 0; interrupted: 0; unstarted: 36. Admission and exact resource targets are pending calibration. This denominator must remain visible if any condition is gated or removed prospectively.

| Phase | Status | Evidence |
|---|---|---|
| Design | DONE | [Campaign plan](../03_qwen_joint_reasoning_retention/PLAN.md) |
| Initialization/readiness/calibration | IN PROGRESS | Public GSM8K inputs resolved: train 7,473; test 1,319. One real masked-SFT development update: 236 tokens in 1.393 s, 15.53 GB peak allocation. |
| Protocol and input freeze | PENDING | No measured cell admitted |
| Training/full evaluation | PENDING | 0/36 prospective cells |
| Final verification/publication | PENDING | Results unavailable |

The exact Qwen2.5-1.5B public artifact has been resolved into campaign-local storage. The required 128-prompt × 8-sample unassisted GSM8K reward-informativeness measurement is in progress. The reported update is a preparation calibration, not a prospective training cell and not evidence of reasoning, retention, or efficiency. The completed older pilot is separate evidence and is not counted as execution of this plan.

**TLDR-end:** [unified-training: prospective execution] `experiments/03_qwen_joint_reasoning_retention` is planned; 36 training cells await resource and readiness admission.
**Snapshot:**
```text
Prospective cells: 36
Completed: 0
Unstarted: 36
Calibration: 1 masked-SFT development update, 236 tokens / 1.393 s
Measured training: not started
```
