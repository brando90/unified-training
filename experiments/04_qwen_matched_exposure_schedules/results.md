# Matched exposure and objective ordering: live results

**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/04_qwen_matched_exposure_schedules/results.md>
**Status:** RUNNING: 1 of 6 cells done (interleaved, seed 1101); 5 array tasks of job 50782977 (indexes 1-5) are still PENDING (Priority) as of 11-11-2026 08:44 PST. Schedule sha256 db3c4c6e448d673851dcd1aef0ec6b34d5480b34edb7b6e9be27210d05357192, frozen protocol v1, H100_SXM5.
**Last updated:** 11-11-2026 08:55 PST

`experiments/04_qwen_matched_exposure_schedules` has 6 prospective training cells (2 orders x 3 seeds). Completed: 1; failed: 0; interrupted: 0; running: 0; unstarted (queued): 5 (of 6). The denominator 6 stays visible whatever happens to a cell.

| Phase | Status | Evidence |
|---|---|---|
| Design | DONE | [Campaign plan](../03_qwen_joint_reasoning_retention/PLAN.md) |
| Initialization/readiness/calibration | DONE | Schedule recorded by job 50524408 (COMPLETED, 00:59:09), 1279 rows |
| Protocol and input freeze | DONE | [PROTOCOL.md](expt_v1/PROTOCOL.md) v1, commit 7b391ce |
| Training/full evaluation | RUNNING | 1/6 cells done (job 50782977, submitted 11-02-2026 06:53 PST) |
| Final verification/publication | PENDING | Results unavailable |

Submit command: `JR1_SCHEDULE04_SHA256=db3c4c6e... GPU_SKU=H100_SXM5 bash jr1/sherlock/submit.sh cells04` (the schedule job had already finished, so no dependency). Index map: seed 1101 + i // 2, order (interleaved, stage_sorted)[i % 2]. Expected finish: about 10 H100 hours in total, so about 11-03-2026 00:00 PST if started at the 18:39 PST estimate, later if the queue releases tasks in pieces.

## Cells

| Array task | Seed | Order | Slurm job id | State |
|---|---|---|---|---|
| 0 | 1101 | interleaved | 50782977_0 (Slurm 51023930, 11-04-2026 17:36 to 18:57 PST) | COMPLETED (verified from artifacts: train and 6 evals complete) |
| 1 | 1101 | stage_sorted | 50782977_1 | PENDING |
| 2 | 1102 | interleaved | 50782977_2 | PENDING |
| 3 | 1102 | stage_sorted | 50782977_3 | PENDING |
| 4 | 1103 | interleaved | 50782977_4 | PENDING |
| 5 | 1103 | stage_sorted | 50782977_5 | PENDING |

## Results so far (1 of 6 cells, seed 1101 only)

| Cell | GSM8K acc (n=1319) | MATH500 acc (n=500) | MMLU acc (n=14042) | ARC-E acc (n=2376) | WikiText NLL/token | WikiText bits/byte | Pref acc (n=1985) | GSM8K learning-curve AUC (trapezoid, 5 milestones, 256-item subset) |
|---|---|---|---|---|---|---|---|---|
| interleaved s1101 | 0.6368 | 0.3620 | 0.6084 | 0.6877 | 2.1036 | 0.7046 | 0.4831 | 0.6099 |

Source: Sherlock collect archive `jr1_results_20261111_084827.tgz` (sha256 fb2ecb823dbf5307cbe399c409e9271c974013800f5a6bc45471c5e3ed081485), fetched 11-11-2026 08:50 PST; per-cell `eval/full_100/summary.json` and `eval/lc_*/summary.json`. Single seed (1101) per cell so far: no paired comparison, interval or p-value is possible yet. Model/harness: Qwen2.5-1.5B, custom jr1 trainer and HF-backend evaluator, H100_SXM5, frozen protocol v1, code 22e2a426a4bd.
One cell, no stage_sorted counterpart yet: no comparison possible.

**TLDR-end:** [unified-training: Exp 04] 1/6 cells done (interleaved, seed 1101), 5 queued.
**Snapshot:**
```text
Prospective cells: 6
Submitted: 6 (job 50782977, H100_SXM5)
Running: 0
Completed: 1
Failed: 0
Interrupted: 0
Unstarted/queued: 5
```
