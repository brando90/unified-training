# Matched exposure and objective ordering: execution checkpoint

**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/04_qwen_matched_exposure_schedules/CKPT_execution.md>
Created: 10-05-2026 10:25 PDT
Last updated: 11-02-2026 06:53 PST
Status: MEASURED CELLS SUBMITTED, QUEUED ON SHERLOCK

- Canonical home: `experiments/04_qwen_matched_exposure_schedules`.
- Read the [campaign plan](../03_qwen_joint_reasoning_retention/PLAN.md) and [execution runbook](../03_qwen_joint_reasoning_retention/expt_v1/cc.md).
- Preserve all historical pilot artifacts. No new measured run has started.
- Shared Qwen readiness passed (105/128 mixed-success groups; 3.32% truncation), but no exposure schedule has been selected or frozen.
- Next: wait for the development-only Qwen readiness gate, then freeze and replay the selected schedule only if the shared matrix is admitted.
- Resource limit is a shared new 96 device-hour campaign ceiling, never per experiment or per worker.
- Actual host/process/run identity and private storage are in the dispatch receipt; update this checkpoint after acknowledgement and every meaningful phase.

## Measured cells submitted (11-02-2026 06:53 PST)

- Protocol v1 frozen (commit 7b391ce); schedule hash db3c4c6e448d673851dcd1aef0ec6b34d5480b34edb7b6e9be27210d05357192 (job 50524408). cells04 array 0-5 submitted as Slurm job 50782977 on H100_SXM5, code 22e2a426a4bd. All 6 tasks PENDING; first start estimate 11-02-2026 18:39 PST.
- Per-cell job ids are `50782977_<i>` (table in [results.md](results.md)).
- Next: collect (AT-574) when tasks end; a failed task is rerun once with `JR1_ARRAY=<i>`.
