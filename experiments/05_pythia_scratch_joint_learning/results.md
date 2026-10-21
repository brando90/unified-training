# Random-initialization joint learning: live results

**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/05_pythia_scratch_joint_learning/results.md>
**Status:** RUNNING — all 27 cells submitted 10-21-2026 00:18 PDT on Sherlock V100 32GB (fp16); none started yet (queued).
**Last updated:** 10-21-2026

`experiments/05_pythia_scratch_joint_learning` has 27 prospective training cells: expt_v1 (Pythia-160m) 9 and expt_v2
(Pythia-14m and 70m) 18. Completed: 0; failed: 0; interrupted: 0; unstarted/queued: 27. The charged-work target is frozen at W = 2.9e8
forward-equivalent tokens per cell (calibration job 48759212). This denominator must remain visible if any condition is gated or removed prospectively.

| Phase | Status | Evidence |
|---|---|---|
| Design | DONE | [Campaign plan](../03_qwen_joint_reasoning_retention/PLAN.md); [v1](expt_v1/PROTOCOL.md) and [v2](expt_v2/PROTOCOL.md) protocols (DRAFT) |
| Code and CPU smoke | DONE | `smoke_cpu.sh`: 3 methods × 3 sizes complete, one cell through the cluster job body with evaluation, calibration |
| Calibration | DONE | Sherlock job 48759212 (V100-PCIE-32GB, fp16); [calibration](calibration/sherlock_v100_fp16/) |
| Protocol and input freeze | DONE | v1 and v2 FROZEN, configs sha256 in `FROZEN.sha256`, code e0f9839 |
| Training/full evaluation | RUNNING | 0/27 complete; jobs below (48765768 v1 array 0-8%4, 48765798 v2 array 0-17%4 afterany v1) |
| Final verification/publication | PENDING | Results unavailable |

## Jobs (Sherlock, code e0f9839, 12 h per cell, requeue+resume, mail END,FAIL)

| Cell | Array task | Job id | Status |
|---|---|---|---|
| v1/160m/sequential_s1 | 48765768_0 | 48765768 | RUNNING (queued) |
| v1/160m/fixed_joint_s1 | 48765768_1 | 48765768 | RUNNING (queued) |
| v1/160m/smooth_joint_s1 | 48765768_2 | 48765768 | RUNNING (queued) |
| v1/160m/sequential_s2 | 48765768_3 | 48765768 | RUNNING (queued) |
| v1/160m/fixed_joint_s2 | 48765768_4 | 48765768 | RUNNING (queued) |
| v1/160m/smooth_joint_s2 | 48765768_5 | 48765768 | RUNNING (queued) |
| v1/160m/sequential_s3 | 48765768_6 | 48765768 | RUNNING (queued) |
| v1/160m/fixed_joint_s3 | 48765768_7 | 48765768 | RUNNING (queued) |
| v1/160m/smooth_joint_s3 | 48765768_8 | 48765768 | RUNNING (queued) |
| v2/14m/sequential_s1 | 48765798_0 | 48765798 | RUNNING (queued, after v1) |
| v2/14m/fixed_joint_s1 | 48765798_1 | 48765798 | RUNNING (queued, after v1) |
| v2/14m/smooth_joint_s1 | 48765798_2 | 48765798 | RUNNING (queued, after v1) |
| v2/70m/sequential_s1 | 48765798_3 | 48765798 | RUNNING (queued, after v1) |
| v2/70m/fixed_joint_s1 | 48765798_4 | 48765798 | RUNNING (queued, after v1) |
| v2/70m/smooth_joint_s1 | 48765798_5 | 48765798 | RUNNING (queued, after v1) |
| v2/14m/sequential_s2 | 48765798_6 | 48765798 | RUNNING (queued, after v1) |
| v2/14m/fixed_joint_s2 | 48765798_7 | 48765798 | RUNNING (queued, after v1) |
| v2/14m/smooth_joint_s2 | 48765798_8 | 48765798 | RUNNING (queued, after v1) |
| v2/70m/sequential_s2 | 48765798_9 | 48765798 | RUNNING (queued, after v1) |
| v2/70m/fixed_joint_s2 | 48765798_10 | 48765798 | RUNNING (queued, after v1) |
| v2/70m/smooth_joint_s2 | 48765798_11 | 48765798 | RUNNING (queued, after v1) |
| v2/14m/sequential_s3 | 48765798_12 | 48765798 | RUNNING (queued, after v1) |
| v2/14m/fixed_joint_s3 | 48765798_13 | 48765798 | RUNNING (queued, after v1) |
| v2/14m/smooth_joint_s3 | 48765798_14 | 48765798 | RUNNING (queued, after v1) |
| v2/70m/sequential_s3 | 48765798_15 | 48765798 | RUNNING (queued, after v1) |
| v2/70m/fixed_joint_s3 | 48765798_16 | 48765798 | RUNNING (queued, after v1) |
| v2/70m/smooth_joint_s3 | 48765798_17 | 48765798 | RUNNING (queued, after v1) |

Cluster and precision are one for every measured cell: Sherlock `gpu` partition, V100 32GB, fp16. v1 is
estimated at about 23 GPU-h (the PLAN budgeted 14 h), v2 about 32 GPU-h, at most 4 GPUs at once.

Pre-measurement finding: the audited WikiText-103 train split packs into 115,840 GPT-NeoX blocks of 1024 tokens
(118.6M trainer tokens per epoch), so the 50M observed PT token target is 0.42 epoch.

No reasoning, retention or efficiency claim is established by this new experiment. The completed older pilot is
separate evidence and is not counted as execution of this plan.

**TLDR-end:** [unified-training: prospective execution] `experiments/05_pythia_scratch_joint_learning` is running: all
27 training cells (9 at 160m, 18 at 14m/70m) are submitted on Sherlock V100s (jobs 48765768 and 48765798), none finished.
**Snapshot:**
```text
Prospective cells: 27 (v1 9, v2 18)
Completed: 0
Queued/running: 27
Code + CPU smoke: done
Measured training: submitted 10-21-2026 00:18 PDT
```
