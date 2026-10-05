# From-scratch joint objectives

**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/01_scratch_joint_objectives/README.md>

Complete: eight objective schedules × three seeds from random Pythia-70m initialization, plus three untouched initial evaluations; zero failed or interrupted cells. The proposed controller improved language loss over sequential training, but did not outperform Aioli or the best observed baseline means.

**Status:** COMPLETE — 24/24 training cells and 3/3 initial evaluations.

**Last updated:** 10-04-2026 23:11 PDT.

## Question and decision

Does cost-aware objective selection improve the declared primary endpoint compared with staged training and matched Aioli adaptation, using random Pythia-70m architecture initialization? The pilot estimates the tradeoff; three seeds do not justify an all-methods superiority claim. The frozen primary is WikiText negative log likelihood (NLL), lower is better. Unassisted math is sparse and secondary. Settings used development data only, never test outcomes. See [program plan](../00_program/PLAN.md) and [final report](../00_program/FINAL_REPORT.md).

The proposed validation-progress controller achieved **6.7329 [6.5893, 6.8766] NLL** across three seeds (sample standard deviation 0.0578). Its paired difference from sequential training was **−1.8000 [−2.5334, −1.0666]**, and from Aioli **+0.0199 [−0.0108, +0.0507]**. These are 95% Student's t intervals across three seeds, with two degrees of freedom; p-val=n/a throughout. The Aioli comparison does not establish an advantage. All 24 cells reduced language loss from their untouched random initializations; the proposed controller's three accuracy-change intervals all span zero. See [results](results.md) for initial-change values and the preserved per-cell table.

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
    PROTOCOL.md          # frozen before measured training
    cc.md
    manifest.json        # all 24 cells and resolved settings
    analysis.json        # complete aggregates, paired and initial comparisons
    runtime/             # ignored item evidence and logs
    checkpoints/         # ignored full model/optimizer state
```

Shared source and literature: `experiments/00_program/`. Training requires PyTorch, Transformers, Datasets, public pinned data/model downloads, and cluster storage. No model-provider keys. The shared harness is an explicit colocation exception because both experiments use the same implementation. Data licenses and immutable revisions are recorded in the program data manifest; private storage receipts remain outside Git.

**Canonical-path dependency:** `experiments/01_scratch_joint_objectives/` is retained at the user's requested canonical location because the frozen harness configuration, `expt_v1/manifest.json`, and recorded evidence identities depend on this path. The experiment is complete, not active; moving it would require a separate coordinated migration that preserves those frozen identities.

## Status

| Phase | Status | Evidence |
|---|---|---|
| Plan | Revised after review | Shared PLAN.md |
| Requested Opus 5.5 maximum-effort reviews | Original FAIL; implementer FIXED | [Original reports and reconciliation](../00_program/REVIEW_RECONCILIATION.md); no post-fix reviewer PASS claimed |
| Implementation and tests | Implemented; 63 deterministic tests pass | Shared harness and test suite |
| Frozen inputs and smoke | Complete | Shared frozen_program.json and calibration.json; all measured updates follow admission |
| Full 24-cell matrix | Complete | 24/24 complete; zero failed or interrupted cells |
| Untouched initial diagnostics | Complete | 3/3 initial evaluations; all 24 paired initial comparisons in analysis.json |
| Full test evaluation and report | Complete | [results.md](results.md), [analysis.json](expt_v1/analysis.json), and [final report](../00_program/FINAL_REPORT.md) |

The [first-cell math diagnostic](first_cell_diagnostic.md) documents collapsed
answers in sequential seed 0. Its nonzero exact-match score must not be interpreted
as successful reasoning. Language-modeling loss remains the frozen scratch primary.
