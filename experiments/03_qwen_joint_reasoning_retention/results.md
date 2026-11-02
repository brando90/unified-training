# Pretrained joint reasoning and retention: live results

**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/03_qwen_joint_reasoning_retention/results.md>
**Status:** RUNNING: all 37 array tasks submitted on Sherlock 11-02-2026 06:53 PST (job 50782976, H100_SXM5, frozen protocol v1, code 22e2a426a4bd); every task is still PENDING, none has started. First task start estimated 11-02-2026 18:39 PST. Real start times are filled in as tasks start.
**Last updated:** 11-02-2026 06:53 PST

`experiments/03_qwen_joint_reasoning_retention` has 36 prospective training cells plus 1 untouched base evaluation (37 array tasks). Completed: 0; failed: 0; interrupted: 0; running: 0; queued: 37. The denominator 36 + 1 stays visible whatever happens to a cell: a failed, interrupted or unstarted cell stays counted.

| Phase | Status | Evidence |
|---|---|---|
| Design | DONE | [Campaign plan](PLAN.md) |
| Initialization/readiness/calibration | DONE | Calibration: `expt_v1/jr1/sherlock/CALIBRATION.md` (H100_SXM5 chosen, W = 3.2e7 per cell) |
| Protocol and input freeze | DONE | [PROTOCOL.md](expt_v1/PROTOCOL.md) v1, configs frozen 2026-10-31T18:41:16Z, commit 7b391ce |
| Training/full evaluation | QUEUED | 0/36 cells done, 37 array tasks pending (job 50782976, submitted 11-02-2026 06:53 PST) |
| Final verification/publication | PENDING | Results unavailable |

Submit command: `JR1_SCHEDULE04_SHA256=db3c4c6e448d673851dcd1aef0ec6b34d5480b34edb7b6e9be27210d05357192 GPU_SKU=H100_SXM5 bash jr1/sherlock/submit.sh cells03` from `/scratch/users/brando9/ut_jr1` (ids in `jobs.txt`). Index map: seed 1101 + i // 6, method (sequential, replay, fixed_joint, smooth_joint, aioli, jr1)[i % 6]; task 36 is the untouched base. A failed or timed-out task is rerun once, resuming (`JR1_ARRAY=<i>`). Squeue snapshot: `squeue_1102_0653.txt` on Sherlock, copied to `expt_v1/jr1/sherlock/squeue_cells_11-02-2026_0653.txt`.

Expected finish: if the queue starts all tasks together at the 18:39 PST estimate, about 11-03-2026 00:00 PST (jr1 cell about 2.0 h, sequential about 1.0 h, plus evaluation); if the queue releases tasks in pieces (Brando's account has about 30 other jobs pending), it runs into 11-04 to 11-05. Total about 60.9 H100 hours.

## Cells

| Array task | Seed | Method | Slurm job id | State |
|---|---|---|---|---|
| 0 | 1101 | sequential | 50782976_0 | PENDING |
| 1 | 1101 | replay | 50782976_1 | PENDING |
| 2 | 1101 | fixed_joint | 50782976_2 | PENDING |
| 3 | 1101 | smooth_joint | 50782976_3 | PENDING |
| 4 | 1101 | aioli | 50782976_4 | PENDING |
| 5 | 1101 | jr1 | 50782976_5 | PENDING |
| 6 | 1102 | sequential | 50782976_6 | PENDING |
| 7 | 1102 | replay | 50782976_7 | PENDING |
| 8 | 1102 | fixed_joint | 50782976_8 | PENDING |
| 9 | 1102 | smooth_joint | 50782976_9 | PENDING |
| 10 | 1102 | aioli | 50782976_10 | PENDING |
| 11 | 1102 | jr1 | 50782976_11 | PENDING |
| 12 | 1103 | sequential | 50782976_12 | PENDING |
| 13 | 1103 | replay | 50782976_13 | PENDING |
| 14 | 1103 | fixed_joint | 50782976_14 | PENDING |
| 15 | 1103 | smooth_joint | 50782976_15 | PENDING |
| 16 | 1103 | aioli | 50782976_16 | PENDING |
| 17 | 1103 | jr1 | 50782976_17 | PENDING |
| 18 | 1104 | sequential | 50782976_18 | PENDING |
| 19 | 1104 | replay | 50782976_19 | PENDING |
| 20 | 1104 | fixed_joint | 50782976_20 | PENDING |
| 21 | 1104 | smooth_joint | 50782976_21 | PENDING |
| 22 | 1104 | aioli | 50782976_22 | PENDING |
| 23 | 1104 | jr1 | 50782976_23 | PENDING |
| 24 | 1105 | sequential | 50782976_24 | PENDING |
| 25 | 1105 | replay | 50782976_25 | PENDING |
| 26 | 1105 | fixed_joint | 50782976_26 | PENDING |
| 27 | 1105 | smooth_joint | 50782976_27 | PENDING |
| 28 | 1105 | aioli | 50782976_28 | PENDING |
| 29 | 1105 | jr1 | 50782976_29 | PENDING |
| 30 | 1106 | sequential | 50782976_30 | PENDING |
| 31 | 1106 | replay | 50782976_31 | PENDING |
| 32 | 1106 | fixed_joint | 50782976_32 | PENDING |
| 33 | 1106 | smooth_joint | 50782976_33 | PENDING |
| 34 | 1106 | aioli | 50782976_34 | PENDING |
| 35 | 1106 | jr1 | 50782976_35 | PENDING |
| 36 | none | untouched base evaluation (no training) | 50782976_36 | PENDING |

**TLDR-end:** [unified-training: Exp 03] 36 cells + base submitted on Sherlock (job 50782976), all queued, 0/36 done.
**Snapshot:**
```text
Prospective cells: 36 (+1 untouched base)
Submitted: 37 (job 50782976, H100_SXM5)
Running: 0
Completed: 0
Queued: 37
First start estimate: 11-02-2026 18:39 PST
```
