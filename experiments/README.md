# Unified-training experiment index

**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/README.md>

**TLDR:** Two bounded pilot comparisons are frozen and admitted: random initialization and an early pretrained checkpoint. A durable one-device queue executes all 48 cells and publishes verified results incrementally. Follow each folder’s results.md for measured progress, including failures and missing cells.

| Canonical folder | Question/setup | Current status |
|---|---|---|
| [00_program](00_program/PLAN.md) | Shared literature, methods, reviewed plan, orchestration | Required reviews reconciled; 63 tests pass; setup and immutable freeze landed on main |
| [01_scratch_joint_objectives](01_scratch_joint_objectives/README.md) | Randomly initialized Pythia-70m; eight schedules × three seeds | Running; first complete seed and full terminal evaluation published on 10-04-2026 |
| [02_early_checkpoint_joint_objectives](02_early_checkpoint_joint_objectives/README.md) | Pythia-160m step10000; eight schedules × three seeds | Admitted and queued after the scratch matrix; see live cell ledger |

Methods, revisions, data hashes, and exact limits were frozen before measurement. Live ledgers distinguish prepared, launched, completed, and reviewed results. `00_program` owns the common harness used by both experiments; their configurations and outputs remain separate. The current full-program count is in [FINAL_STATUS.md](00_program/FINAL_STATUS.md); neither a completed cell nor a running queue means the full program is complete.
