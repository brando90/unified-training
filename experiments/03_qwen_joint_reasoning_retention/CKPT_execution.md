# Pretrained joint reasoning and retention: execution checkpoint

**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/03_qwen_joint_reasoning_retention/CKPT_execution.md>
Created: 10-05-2026 10:25 PDT
Last updated: 11-02-2026 06:53 PST
Status: MEASURED CELLS SUBMITTED, QUEUED ON SHERLOCK

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
- Rechecked GPU 1 with `CUDA_VISIBLE_DEVICES=1`: a repeat four-objective development smoke completed in 26.08 wall seconds, with all four nonzero optimizer steps. The released UltraFeedback binarized preference schema is locally resolved (`prompt`, `chosen`, `rejected`, scores); preference split selection is still not frozen.
- Next: freeze released-data splits, schedule/controller/recovery behavior and resource accounting; then start the durable supervisor only for admitted cells on physical GPU 1.
- Resource limit is a shared new 96 device-hour campaign ceiling, never per experiment or per worker.
- Actual host/process/run identity and private storage are in the dispatch receipt; update this checkpoint after acknowledgement and every meaningful phase.

## Measured cells submitted (11-02-2026 06:53 PST)

- Protocol v1 frozen (commit 7b391ce). cells03 array 0-36 submitted as Slurm job 50782976 on H100_SXM5 (36 cells + untouched base at task 36), code 22e2a426a4bd. All 37 tasks PENDING; first start estimate 11-02-2026 18:39 PST.
- Per-cell job ids are `50782976_<i>` (table in [results.md](results.md)); ids also in `/scratch/users/brando9/ut_jr1/jobs.txt`. Squeue snapshot saved.
- Next: collect (AT-574) when tasks end; rerun a failed or timed-out task once with `JR1_ARRAY=<i>`; never delete a cell directory.
