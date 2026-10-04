# Early-checkpoint joint objectives

**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/02_early_checkpoint_joint_objectives/README.md>

**TLDR:** Test eight objective schedules with three seeds from immutable early Pythia-160m checkpoint (step10000). Required reviews are complete and reconciled; full-matrix resource admission is being finalized; this folder is the canonical home for its configuration and results.

## Question and decision

Does cost-aware objective selection improve the declared primary endpoint compared with staged training and matched Aioli adaptation, using immutable early Pythia-160m checkpoint (step10000)? The pilot estimates the tradeoff; three seeds do not justify an all-methods superiority claim. Scratch primarily measures language-modeling loss; the early checkpoint primarily measures ARC-Easy accuracy. Unassisted math is sparse and secondary. Settings use development data only, never test outcomes. See [program plan](../00_program/PLAN.md).

## Method

1. Reconcile the requested independent plan review and implement all eight declared conditions.
2. Pin initial model/tokenizer/data revisions and train/validation/test identities; measure a non-inferential throughput smoke run.
3. Freeze `expt_v1/PROTOCOL.md`, resolved configuration and the full 24-cell manifest before measured training.
4. Run every method/seed cell durably on one available device with inclusive compute accounting.
5. Evaluate full official benchmark test splits only after choices freeze, report all seeds and failures, then publish verified summaries.

## Files and dependencies

```text
02_early_checkpoint_joint_objectives/
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

Shared source and literature: `experiments/00_program/`. Training requires PyTorch, Transformers, Datasets, public pinned data/model downloads, and cluster storage. No model-provider keys. The shared harness is an explicit colocation exception because both experiments use the same implementation. Runtime storage pointers and data licenses will be written before admission.

## Status

| Phase | Status | Evidence |
|---|---|---|
| Plan | Revised after review | Shared PLAN.md |
| Requested Opus 5.5 maximum-effort reviews | Complete; original FAIL findings reconciled | Shared review reports and reconciliation |
| Implementation and tests | Implemented; 63 deterministic tests pass | Shared harness and test suite |
| Frozen inputs and smoke | Pending | No measured training |
| Full 24-cell matrix | Pending | 0/24 started |
| Full test evaluation and report | Pending | No results yet |
