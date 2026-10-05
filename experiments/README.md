# Unified-training experiment index

**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/README.md>

Both bounded pilots are complete: 48/48 training cells and 4/4 untouched initial evaluations, with zero failed or interrupted cells. The proposed controller improved over sequential training on the primary endpoints, but did not establish an advantage over Aioli or all baselines; early-checkpoint accuracy declined relative to the untouched checkpoint.

**Status:** COMPLETE — execution and final deterministic verification.

**Last updated:** 10-04-2026 23:11 PDT.

| Canonical folder | Question/setup | Current status |
|---|---|---|
| [experiments/00_program/](00_program/FINAL_REPORT.md) | Shared literature, methods, reviewed plan, orchestration | Complete: 48/48 cells and 4/4 initial evaluations; original reviewer FAIL findings marked FIXED by the implementer; 63 deterministic tests passed |
| [experiments/01_scratch_joint_objectives/](01_scratch_joint_objectives/README.md) | Randomly initialized Pythia-70m; eight schedules × three seeds | Complete: 24/24 cells and 3/3 initial evaluations; proposed language loss 6.7329 [6.5893, 6.8766] |
| [experiments/02_early_checkpoint_joint_objectives/](02_early_checkpoint_joint_objectives/README.md) | Pythia-160m step10000; eight schedules × three seeds | Complete: 24/24 cells and 1/1 initial evaluation; proposed accuracy 34.2593% [30.5290%, 37.9896%] |

Table intervals are 95% Student's t intervals across three seeds, with two degrees of freedom; p-val=n/a (no hypothesis test). Language loss is WikiText negative log likelihood, lower is better; accuracy is normalized ARC-Easy (AI2 Reasoning Challenge, Easy subset), higher is better. The untouched early checkpoint scored 37.2475% [35.3256%, 39.2106%] (95% Wilson item interval; p-val=n/a), above every trained cell's observed accuracy. All 24 scratch cells reduced language loss from their untouched random initializations. Full paired comparisons, seed variability and initial changes are in the [final report](00_program/FINAL_REPORT.md) and each experiment's analysis.

Methods, revisions, data hashes, and exact limits were frozen before measurement. The canonical folders are retained at the user's requested paths because the frozen harness configuration, manifests and evidence identities depend on them; their lifecycle status is COMPLETE, not active. `experiments/00_program/` owns the common harness used by both experiments. See [FINAL_STATUS.md](00_program/FINAL_STATUS.md) for the complete denominator and [review reconciliation](00_program/REVIEW_RECONCILIATION.md) for the original FAIL and implementer FIXED record; no post-fix reviewer PASS is claimed. Baseline adaptations are not full-paper reproductions, and the exact CHERRY-RL reference remains unresolved.
