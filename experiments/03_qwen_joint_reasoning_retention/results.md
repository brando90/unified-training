# Pretrained joint reasoning and retention: live results

**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/03_qwen_joint_reasoning_retention/results.md>
**Status:** RUNNING (started 10-05-2026 17:44 UTC) — development/readiness only; measured training has not started.
**Last updated:** 10-05-2026 18:12 UTC

`experiments/03_qwen_joint_reasoning_retention` has 36 prospective training cells. Completed: 0; failed: 0; interrupted: 0; unstarted: 36. Admission and exact resource targets are pending calibration. This denominator must remain visible if any condition is gated or removed prospectively.

| Phase | Status | Evidence |
|---|---|---|
| Design | DONE | [Campaign plan](../03_qwen_joint_reasoning_retention/PLAN.md) |
| Initialization/readiness/calibration | DONE | GSM8K readiness: 105/128 mixed groups, 268 positive / 756 negative rewards, 34/1,024 truncations (3.32%). Four one-device objective updates all produced nonzero gradients. |
| Protocol and input freeze | PENDING | No measured cell admitted |
| Training/full evaluation | PENDING | 0/36 prospective cells |
| Final verification/publication | PENDING | Results unavailable |

The exact Qwen2.5-1.5B public artifact has been resolved into campaign-local storage. The required 128-prompt × 8-sample unassisted GSM8K reward-informativeness measurement passed its predeclared engineering gate: 105 mixed-success groups (minimum 20), both reward signs, and 3.32% truncation (maximum 10%). Development-only PT, SFT, DPO and unassisted on-policy RL updates each had a nonzero optimizer step; the DPO technical smoke's synthetic negative is explicitly not a substitute for the planned released preference data. These are admission measurements, not prospective cells or evidence of reasoning, retention, or efficiency. The completed older pilot is separate evidence and is not counted as execution of this plan.

**TLDR-end:** [unified-training: prospective execution] `experiments/03_qwen_joint_reasoning_retention` is planned; 36 training cells await resource and readiness admission.
**Snapshot:**
```text
Prospective cells: 36
Completed: 0
Unstarted: 36
Calibration: 1 masked-SFT development update, 236 tokens / 1.393 s
Readiness: 105/128 mixed groups; 268 positive / 756 negative; 34/1,024 truncated
Objective smokes: PT/SFT/DPO/RL all nonzero optimizer steps
Measured training: not started
```
