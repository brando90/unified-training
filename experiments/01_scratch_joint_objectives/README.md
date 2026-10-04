# From-scratch joint objectives

**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/01_scratch_joint_objectives/README.md>

**TLDR:** Test eight objective schedules with three seeds from random Pythia-70m architecture initialization. Required reviews are complete and reconciled; the matrix is frozen and running, with the first complete seed published on 10-04-2026. This folder is the canonical home for its configuration and results.

## Question and decision

Does cost-aware objective selection improve the declared primary endpoint compared with staged training and matched Aioli adaptation, using random Pythia-70m architecture initialization? The pilot estimates the tradeoff; three seeds do not justify an all-methods superiority claim. Scratch primarily measures language-modeling loss; the early checkpoint primarily measures ARC-Easy accuracy. Unassisted math is sparse and secondary. Settings use development data only, never test outcomes. See [program plan](../00_program/PLAN.md).

## Method

1. Reconcile the requested independent plan review and implement all eight declared conditions.
2. Pin initial model/tokenizer/data revisions and train/validation/test identities; measure a non-inferential throughput smoke run.
3. Freeze `expt_v1/PROTOCOL.md`, resolved configuration and the full 24-cell manifest before measured training.
4. Run every method/seed cell durably on one available device with inclusive compute accounting.
5. Evaluate full official benchmark test splits only after choices freeze, report all seeds and failures, then publish verified summaries.

## Files and dependencies

```text
01_scratch_joint_objectives/
  README.md
  results.md
  expt_v1/
    PROTOCOL.md          # created and frozen after implementation review/preflight
    cc.md
    manifest.json        # all 24 cells and resolved settings
    analysis.json        # aggregate statistics when available
    runtime/             # ignored item evidence and logs
    checkpoints/         # ignored full model/optimizer state
```

Shared source and literature: `experiments/00_program/`. Training requires PyTorch, Transformers, Datasets, public pinned data/model downloads, and cluster storage. No model-provider keys. The shared harness is an explicit colocation exception because both experiments use the same implementation. Data licenses and immutable revisions are recorded in the program data manifest; private storage receipts remain outside Git.

## Status

| Phase | Status | Evidence |
|---|---|---|
| Plan | Revised after review | Shared PLAN.md |
| Requested Opus 5.5 maximum-effort reviews | Complete; original FAIL findings reconciled | Shared review reports and reconciliation |
| Implementation and tests | Implemented; 63 deterministic tests pass | Shared harness and test suite |
| Frozen inputs and smoke | Complete | Shared frozen_program.json and calibration.json; all measured updates follow admission |
| Full 24-cell matrix | Running | See results.md for verified completed, failed and pending cells |
| Full test evaluation and report | Incremental | First complete cell published; full matrix and initial diagnostics remain pending |

The [first-cell math diagnostic](first_cell_diagnostic.md) documents collapsed
answers in sequential seed 0. Its nonzero exact-match score must not be interpreted
as successful reasoning. Language-modeling loss remains the frozen scratch primary.
