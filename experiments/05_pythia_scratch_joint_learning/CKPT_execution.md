# Random-initialization joint learning: execution checkpoint

**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/05_pythia_scratch_joint_learning/CKPT_execution.md>
Created: 10-05-2026 10:25 PDT
Last updated: 10-21-2026
Status: COMPLETE: all 27 cells finished 10-22-2026 (Sherlock V100 fp16); final readout in results.md; collect output in results_final/

- Canonical home: `experiments/05_pythia_scratch_joint_learning`. Read the [campaign plan](../03_qwen_joint_reasoning_retention/PLAN.md),
  the protocols ([v1](expt_v1/PROTOCOL.md), [v2](expt_v2/PROTOCOL.md)) and [SUBMIT.md](SUBMIT.md).
- Preserve all historical pilot artifacts. Measured cells are submitted (27, none finished at the time of this edit).
- Done: p05 package on jr1 (imported trainer), DRAFT configs for bf16 (FarmShare L40S, main) and fp16 (Sherlock V100,
  fallback), Slurm scripts for both clusters, CPU smoke passing.
- Next: upload the bundle, prep, calibrate on the measurement GPU, freeze `W` and the data hashes, then cells_v1 and
  cells_v2 (SUBMIT.md steps 1–5).
- Open checks: the Sherlock `-C` constraint syntax (`P05_TEST_ONLY=1`), the FarmShare storage quota (~80 GB
  estimated), and whether the calibrated hours fit the campaign budget (14 device hours were planned for v1).
