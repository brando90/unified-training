# Early-checkpoint joint objectives

**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/02_early_checkpoint_joint_objectives/README.md>

Complete: eight objective schedules × three seeds from immutable Pythia-160m step10000, plus its untouched initial evaluation; zero failed or interrupted cells. The proposed controller improved accuracy over sequential training, but did not establish an advantage over Aioli, and every trained cell scored below the untouched checkpoint.

**Status:** COMPLETE — 24/24 training cells and 1/1 initial evaluation.

**Last updated:** 10-04-2026 23:11 PDT.

## Question and decision

Does cost-aware objective selection improve the declared primary endpoint compared with staged training and matched Aioli adaptation, using immutable early Pythia-160m checkpoint (step10000)? The frozen primary is normalized ARC-Easy (AI2 Reasoning Challenge, Easy subset) accuracy. Three seeds do not justify an all-methods superiority claim. Unassisted math is sparse and secondary. Settings used development data only, never test outcomes. See [program plan](../00_program/PLAN.md) and [final report](../00_program/FINAL_REPORT.md).

The proposed validation-progress controller achieved **34.2593% [30.5290%, 37.9896%]** across three seeds (sample standard deviation 1.5016 percentage points). Its paired difference from sequential training was **+5.8361 [+2.6886, +8.9837] percentage points**, and from Aioli **+0.0561 [−1.8210, +1.9332] percentage points**. These are 95% Student's t intervals across three seeds, with two degrees of freedom; p-val=n/a throughout. The Aioli comparison does not establish an advantage. The untouched checkpoint scored **37.2475% [35.3256%, 39.2106%]** (95% Wilson item interval; 2,376 items; p-val=n/a), above every trained cell's observed accuracy. See [results](results.md) for paired initial changes and the preserved per-cell table.

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
    PROTOCOL.md          # frozen before measured training
    cc.md
    manifest.json        # all 24 cells and resolved settings
    analysis.json        # complete aggregates, paired and initial comparisons
    runtime/             # ignored item evidence and logs
    checkpoints/         # ignored full model/optimizer state
```

Shared source and literature: `experiments/00_program/`. Training requires PyTorch, Transformers, Datasets, public pinned data/model downloads, and cluster storage. No model-provider keys. The shared harness is an explicit colocation exception because both experiments use the same implementation. Data licenses and immutable revisions are recorded in the program data manifest; private storage receipts remain outside Git.

**Canonical-path dependency:** `experiments/02_early_checkpoint_joint_objectives/` is retained at the user's requested canonical location because the frozen harness configuration, `expt_v1/manifest.json`, and recorded evidence identities depend on this path. The experiment is complete, not active; moving it would require a separate coordinated migration that preserves those frozen identities.

## Status

| Phase | Status | Evidence |
|---|---|---|
| Plan | Revised after review | Shared PLAN.md |
| Requested Opus 5.5 maximum-effort reviews | Original FAIL; implementer FIXED | [Original reports and reconciliation](../00_program/REVIEW_RECONCILIATION.md); no post-fix reviewer PASS claimed |
| Implementation and tests | Implemented; 63 deterministic tests pass | Shared harness and test suite |
| Frozen inputs and smoke | Complete | Shared frozen_program.json and calibration.json |
| Full 24-cell matrix | Complete | 24/24 complete; zero failed or interrupted cells |
| Untouched initial diagnostic | Complete | 1/1 shared initial evaluation; all 24 paired initial comparisons in analysis.json |
| Full test evaluation and report | Complete | [results.md](results.md), [analysis.json](expt_v1/analysis.json), and [final report](../00_program/FINAL_REPORT.md) |

Historical reporting note: before execution, the results template incorrectly labeled
the admitted queue as "Not admitted; review and resource gates remain pending."
Both matrices had passed the same [acceptance](../00_program/acceptance.json) and
[freeze](../00_program/frozen_program.json) before the first measured update.
The frozen implementation was preserved throughout measured work; this historical
template defect did not describe the admission state and does not change the
completed 24/24-cell result.
