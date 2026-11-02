# Matched exposure and objective ordering: live results

**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/04_qwen_matched_exposure_schedules/results.md>
**Status:** RUNNING: all 6 array tasks submitted on Sherlock 11-02-2026 06:53 PST (job 50782977, H100_SXM5, frozen protocol v1, schedule sha256 db3c4c6e448d673851dcd1aef0ec6b34d5480b34edb7b6e9be27210d05357192 from job 50524408); every task is still PENDING, none has started. First task start estimated 11-02-2026 18:39 PST. Real start times are filled in as tasks start.
**Last updated:** 11-02-2026 06:53 PST

`experiments/04_qwen_matched_exposure_schedules` has 6 prospective training cells (2 orders x 3 seeds). Completed: 0; failed: 0; interrupted: 0; running: 0; queued: 6. The denominator 6 stays visible whatever happens to a cell.

| Phase | Status | Evidence |
|---|---|---|
| Design | DONE | [Campaign plan](../03_qwen_joint_reasoning_retention/PLAN.md) |
| Initialization/readiness/calibration | DONE | Schedule recorded by job 50524408 (COMPLETED, 00:59:09), 1279 rows |
| Protocol and input freeze | DONE | [PROTOCOL.md](expt_v1/PROTOCOL.md) v1, commit 7b391ce |
| Training/full evaluation | QUEUED | 0/6 cells done (job 50782977, submitted 11-02-2026 06:53 PST) |
| Final verification/publication | PENDING | Results unavailable |

Submit command: `JR1_SCHEDULE04_SHA256=db3c4c6e... GPU_SKU=H100_SXM5 bash jr1/sherlock/submit.sh cells04` (the schedule job had already finished, so no dependency). Index map: seed 1101 + i // 2, order (interleaved, stage_sorted)[i % 2]. Expected finish: about 10 H100 hours in total, so about 11-03-2026 00:00 PST if started at the 18:39 PST estimate, later if the queue releases tasks in pieces.

## Cells

| Array task | Seed | Order | Slurm job id | State |
|---|---|---|---|---|
| 0 | 1101 | interleaved | 50782977_0 | PENDING |
| 1 | 1101 | stage_sorted | 50782977_1 | PENDING |
| 2 | 1102 | interleaved | 50782977_2 | PENDING |
| 3 | 1102 | stage_sorted | 50782977_3 | PENDING |
| 4 | 1103 | interleaved | 50782977_4 | PENDING |
| 5 | 1103 | stage_sorted | 50782977_5 | PENDING |

**TLDR-end:** [unified-training: Exp 04] 6 cells submitted on Sherlock (job 50782977), all queued, 0/6 done.
**Snapshot:**
```text
Prospective cells: 6
Submitted: 6 (job 50782977, H100_SXM5)
Running: 0
Completed: 0
Queued: 6
```
