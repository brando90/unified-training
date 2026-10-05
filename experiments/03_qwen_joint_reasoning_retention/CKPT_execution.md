# Pretrained joint reasoning and retention: execution checkpoint

**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/03_qwen_joint_reasoning_retention/CKPT_execution.md>
Created: 10-05-2026 10:25 PDT
Last updated: 10-05-2026 17:49 UTC
Status: INITIALIZATION IN PROGRESS

- Canonical home: `experiments/03_qwen_joint_reasoning_retention`.
- Read the [campaign plan](../03_qwen_joint_reasoning_retention/PLAN.md) and [execution runbook](../03_qwen_joint_reasoning_retention/expt_v1/cc.md).
- Preserve all historical pilot artifacts. No new measured run has started.
- Inputs resolved: public GSM8K has 7,473 training and 1,319 test rows. The exact Qwen2.5-1.5B artifact is campaign-local.
- Calibration: one masked-SFT development update completed with 236 tokens in 1.393 seconds and 15.53 GB peak allocation on one A100 80 GB device. This is not a prospective cell.
- In progress: 128 development prompts × eight unassisted samples at a 1,024-token cap. Its mixed-success, reward and truncation counts decide whether the Qwen matrix is admitted.
- Next: finish readiness, verify all four objective updates and schedule/controller/recovery tests, freeze resource accounting and then start the durable supervisor only for admitted cells.
- Resource limit is a shared new 96 device-hour campaign ceiling, never per experiment or per worker.
- Actual host/process/run identity and private storage are in the dispatch receipt; update this checkpoint after acknowledgement and every meaningful phase.
