# Unified-training experiment index

**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/README.md>

**TLDR:** Two bounded pilot comparisons are being initialized: random initialization and an early pretrained checkpoint. Follow each folder’s results.md for measured progress, including failures and missing cells.

| Canonical folder | Question/setup | Current status |
|---|---|---|
| [00_program](00_program/PLAN.md) | Shared literature, methods, reviewed plan, orchestration | Plan review being prepared; source-only initial repository |
| [01_scratch_joint_objectives](01_scratch_joint_objectives/README.md) | Randomly initialized Pythia-70m; eight schedules × three seeds | Planned; 0/24 training cells started |
| [02_early_checkpoint_joint_objectives](02_early_checkpoint_joint_objectives/README.md) | Pythia-160m early checkpoint; eight schedules × three seeds | Planned; 0/24 training cells started |

Methods, revisions, data hashes, and exact limits freeze before each measured matrix. Live ledgers distinguish prepared, launched, completed, and reviewed results. `00_program` owns the common harness used by both experiments; their configurations and outputs remain separate.
