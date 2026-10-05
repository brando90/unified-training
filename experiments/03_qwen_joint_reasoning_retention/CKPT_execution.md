# Pretrained joint reasoning and retention: execution checkpoint

**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/03_qwen_joint_reasoning_retention/CKPT_execution.md>
Created: 10-05-2026 10:25 PDT
Last updated: 10-05-2026 18:12 UTC
Status: READINESS PASSED; ADMISSION FREEZE IN PROGRESS

LANDED 53539dc52889f58aaa5e19616d70dfec38ad2697 https://github.com/brando90/unified-training/pull/5 10-05-2026 17:55 UTC
LANDED 94282326b8b34a66de1a4ecd7e4943a294a14739 https://github.com/brando90/unified-training/pull/7 10-05-2026 18:16 UTC

- Canonical home: `experiments/03_qwen_joint_reasoning_retention`.
- Read the [campaign plan](../03_qwen_joint_reasoning_retention/PLAN.md) and [execution runbook](../03_qwen_joint_reasoning_retention/expt_v1/cc.md).
- Preserve all historical pilot artifacts. No new measured run has started.
- Inputs resolved: public GSM8K has 7,473 training and 1,319 test rows. The exact Qwen2.5-1.5B artifact is campaign-local.
- Calibration: one masked-SFT development update completed with 236 tokens in 1.393 seconds and 15.53 GB peak allocation on one A100 80 GB device. This is not a prospective cell.
- Readiness passed: 105/128 mixed-success groups (threshold 20), 268 positive and 756 negative rewards, 34/1,024 truncations (3.32%; threshold 10%).
- PT, SFT, DPO and unassisted on-policy RL development updates all had nonzero gradients with the shared optimizer. The DPO smoke's synthetic negative is technical-only, never released preference training evidence.
- Device mapping: the completed preparation used physical GPU 0 through `CUDA_VISIBLE_DEVICES=0`; a later fresh probe found all devices idle and establishes the dispatch-provisional physical GPU 1 (`CUDA_VISIBLE_DEVICES=1`, PyTorch logical `cuda:0`) for subsequent campaign work.
- Next: resolve released preference provenance, verify schedule/controller/recovery behavior, freeze resource accounting and then start the durable supervisor only for admitted cells on physical GPU 1.
- Resource limit is a shared new 96 device-hour campaign ceiling, never per experiment or per worker.
- Actual host/process/run identity and private storage are in the dispatch receipt; update this checkpoint after acknowledgement and every meaningful phase.
