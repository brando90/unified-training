# Pretrained joint reasoning and retention: live results

**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/03_qwen_joint_reasoning_retention/results.md>
**Status:** RUNNING: 2 of 36 training cells done (sequential and replay, seed 1101); 35 array tasks of job 50782976 (indexes 2-36, including the base) are still PENDING (Priority) as of 11-17-2026 10:03 PST (no new Exp 03 cell since 11-11). Frozen protocol v1, H100_SXM5, code 22e2a426a4bd.
**Last updated:** 11-17-2026 10:05 PST

`experiments/03_qwen_joint_reasoning_retention` has 36 prospective training cells plus 1 untouched base evaluation (37 array tasks). Completed: 2; failed: 0; interrupted: 0; running: 0; unstarted (queued): 35 (of 36 training + 1 base). The denominator 36 + 1 stays visible whatever happens to a cell: a failed, interrupted or unstarted cell stays counted.

| Phase | Status | Evidence |
|---|---|---|
| Design | DONE | [Campaign plan](PLAN.md) |
| Initialization/readiness/calibration | DONE | Calibration: `expt_v1/jr1/sherlock/CALIBRATION.md` (H100_SXM5 chosen, W = 3.2e7 per cell) |
| Protocol and input freeze | DONE | [PROTOCOL.md](expt_v1/PROTOCOL.md) v1, configs frozen 2026-10-31T18:41:16Z, commit 7b391ce |
| Training/full evaluation | RUNNING | 2/36 cells done (seed 1101 sequential, replay), 35 array tasks pending (job 50782976) |
| Final verification/publication | PENDING | Results unavailable |

Submit command: `JR1_SCHEDULE04_SHA256=db3c4c6e448d673851dcd1aef0ec6b34d5480b34edb7b6e9be27210d05357192 GPU_SKU=H100_SXM5 bash jr1/sherlock/submit.sh cells03` from `/scratch/users/brando9/ut_jr1` (ids in `jobs.txt`). Index map: seed 1101 + i // 6, method (sequential, replay, fixed_joint, smooth_joint, aioli, jr1)[i % 6]; task 36 is the untouched base. A failed or timed-out task is rerun once, resuming (`JR1_ARRAY=<i>`). Squeue snapshot: `squeue_1102_0653.txt` on Sherlock, copied to `expt_v1/jr1/sherlock/squeue_cells_11-02-2026_0653.txt`.

Expected finish: if the queue starts all tasks together at the 18:39 PST estimate, about 11-03-2026 00:00 PST (jr1 cell about 2.0 h, sequential about 1.0 h, plus evaluation); if the queue releases tasks in pieces (Brando's account has about 30 other jobs pending), it runs into 11-04 to 11-05. Total about 60.9 H100 hours.

## Cells

| Array task | Seed | Method | Slurm job id | State |
|---|---|---|---|---|
| 0 | 1101 | sequential | 50782976_0 (Slurm 51000694, 11-04-2026 15:08 to 16:09 PST) | COMPLETED (verified from artifacts: train and 6 evals complete) |
| 1 | 1101 | replay | 50782976_1 (Slurm 51556666, 11-10-2026 11:38 to 12:36 PST) | COMPLETED (verified from artifacts: train and 6 evals complete) |
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

## Results so far (2 of 36 cells, seed 1101 only)

| Cell | GSM8K acc (n=1319) | MATH500 acc (n=500) | MMLU acc (n=14042) | ARC-E acc (n=2376) | WikiText NLL/token | WikiText bits/byte | Pref acc (n=1985) | GSM8K learning-curve AUC (trapezoid, 5 milestones, 256-item subset) |
|---|---|---|---|---|---|---|---|---|
| sequential s1101 | 0.6133 | 0.3760 | 0.6090 | 0.7340 | 2.1096 | 0.7066 | 0.4771 | 0.5151 |
| replay s1101 | 0.6528 | 0.3740 | 0.6069 | 0.7252 | 2.0885 | 0.6995 | 0.4791 | 0.5825 |

Source: Sherlock collect archive `jr1_results_20261111_084827.tgz` (sha256 fb2ecb823dbf5307cbe399c409e9271c974013800f5a6bc45471c5e3ed081485), fetched 11-11-2026 08:50 PST; per-cell `eval/full_100/summary.json` and `eval/lc_*/summary.json`. Single seed (1101) per cell so far: no paired comparison, interval or p-value is possible yet. Model/harness: Qwen2.5-1.5B, custom jr1 trainer and HF-backend evaluator, H100_SXM5, frozen protocol v1, code 22e2a426a4bd.
Reading (descriptive, one seed): replay is ahead of sequential on GSM8K (0.6528 vs 0.6133, +3.9 pts) and learning-curve AUC (0.5825 vs 0.5151), and about level on MATH500 and MMLU (-0.2 pts); ARC-E is 0.9 pts lower and WikiText NLL 0.02 lower. Sequential's GSM8K learning curve dips mid-training (256-item subset: 0.62 at start, 0.38 at 50%, 0.60 at end) and recovers.

**TLDR-end:** [unified-training: Exp 03] 2/36 cells done (seed 1101 sequential, replay), 35 queued; no paired analysis until more seeds finish.
**Snapshot:**
```text
Prospective cells: 36 (+1 untouched base)
Submitted: 37 (job 50782976, H100_SXM5)
Running: 0
Completed: 2
Failed: 0
Interrupted: 0
Unstarted/queued: 35
First start estimate: 11-02-2026 18:39 PST
```
