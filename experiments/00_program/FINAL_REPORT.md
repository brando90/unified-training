# Final joint-training pilot results

**Status:** COMPLETE — 48/48 training cells and 4/4 untouched initial evaluations verified.
**Last updated:** 10-04-2026 23:13 PDT
**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/00_program/FINAL_REPORT.md>

The cost-aware controller improves on the staged baseline in this bounded pilot, but does not establish an advantage over Aioli. The early-checkpoint experiment supplies an accuracy signal: all 24 trained cells have lower observed accuracy than the untouched checkpoint. These results do not support the proposed claim of beating all previous methods.

## Scope and complete denominator

[experiments/01_scratch_joint_objectives](../01_scratch_joint_objectives/README.md) starts a Pythia-70m architecture randomly. Its primary endpoint is WikiText token negative log likelihood (NLL), lower is better.
[experiments/02_early_checkpoint_joint_objectives](../02_early_checkpoint_joint_objectives/README.md) starts Pythia-160m at pretraining step 10000, immutable revision `bb9bd9573b772c899bda5206af551645138ceed8`. Its primary endpoint is normalized accuracy on the AI2 Reasoning Challenge (ARC)-Easy benchmark, higher is better. That checkpoint inherits 20,971,520,000 training tokens, about 7% of its pretraining trajectory; it is an early-checkpoint study, not training from scratch.

Eight methods × three seeds × two setups produced all 48 declared cells. **Failed: 0; interrupted: 0; missing/unstarted: 0.** Untouched evaluations are three independent scratch initializations and one shared early checkpoint. All 52 evaluations contain exactly 564 WikiText blocks (288,158 scored tokens), 2,376 ARC-Easy questions, 1,319 Grade School Math 8K (GSM8K) test questions, and 256 bounded UltraFeedback preference pairs. The preference subset is not the full 2,000-pair release.

## Primary endpoints

Every row contains all three seeds. Intervals below are 95% Student t intervals across seeds, two degrees of freedom; SD is sample standard deviation. The normal-seed assumption is weak with three seeds. **p-val=n/a throughout: no formal hypothesis test or multiplicity-adjusted superiority decision was specified.**

| Method | Scratch NLL [95% interval] | SD | Early accuracy % [95% interval] | SD, percentage points |
|---|---:|---:|---:|---:|
| Sequential | 8.5329 [7.8374, 9.2284] | 0.2800 | 28.4231 [23.9771, 32.8691] | 1.7898 |
| Parallel joint | 6.2345 [6.1965, 6.2725] | 0.0153 | 34.7643 [34.1822, 35.3464] | 0.2343 |
| Fixed joint | 6.0281 [5.9815, 6.0748] | 0.0188 | 34.9186 [32.1341, 37.7032] | 1.1209 |
| Smooth joint | 6.2080 [6.1864, 6.2297] | 0.0087 | 34.9046 [32.9477, 36.8615] | 0.7878 |
| Aioli adaptation | 6.7130 [6.5395, 6.8866] | 0.0699 | 34.2031 [31.1562, 37.2501] | 1.2266 |
| Proposed validation progress | 6.7329 [6.5893, 6.8766] | 0.0578 | 34.2593 [30.5290, 37.9896] | 1.5016 |
| CHORD adaptation | 6.3994 [6.3294, 6.4695] | 0.0282 | 33.0247 [31.7175, 34.3319] | 0.5262 |
| RPT-inspired adaptation | 6.0029 [5.9419, 6.0638] | 0.0245 | 34.4697 [30.2215, 38.7179] | 1.7101 |

## Proposed minus each baseline

Negative loss differences and positive accuracy differences favor the proposed method. Sequential and Aioli are the declared primary contrasts; the other contrasts are secondary. The same three-seed Student t uncertainty above applies. Scratch paired comparisons average disjoint block-level differences, whereas the endpoint table weights by token count; their numerical estimates differ slightly. Neither estimand supports superiority over Aioli.

| Baseline | Scratch block NLL difference [95% interval] | Early accuracy difference, percentage points [95% interval] |
|---|---:|---:|
| Sequential | -1.8000 [-2.5334, -1.0666] | 5.8361 [2.6886, 8.9837] |
| Parallel joint | 0.4985 [0.3899, 0.6071] | -0.5051 [-4.1328, 3.1227] |
| Fixed joint | 0.7048 [0.5211, 0.8885] | -0.6594 [-5.5748, 4.2560] |
| Smooth joint | 0.5249 [0.3827, 0.6671] | -0.6453 [-3.9997, 2.7090] |
| Aioli adaptation | 0.0199 [-0.0108, 0.0507] | 0.0561 [-1.8210, 1.9332] |
| CHORD adaptation | 0.3335 [0.2592, 0.4078] | 1.2346 [-3.2725, 5.7416] |
| RPT-inspired adaptation | 0.7301 [0.5319, 0.9282] | -0.2104 [-3.1750, 2.7541] |

The proposed-minus-Aioli loss difference is +0.0199 [−0.0108, +0.0507]; the accuracy difference is +0.0561 [−1.8210, +1.9332] percentage points. The proposed method is not the best observed method on either primary endpoint. The published analyses also retain conditional item intervals and descriptive two-stage seed/item bootstraps; those do not replace the weak three-seed uncertainty or create a formal superiority test.

## Untouched-checkpoint comparison

Scratch language loss decreases in all 24 cells. Proposed seed changes in block NLL are −4.1266 [−4.1549, −4.0991], −4.2486 [−4.2784, −4.2206], and −4.2069 [−4.2355, −4.1798].

The untouched early checkpoint scores **37.2475% [35.3256%, 39.2106%]** on 2,376 ARC-Easy questions, a 95% Wilson item interval. All 24 trained cells have lower observed accuracy. The proposed method's three changes are **−1.2626 [−3.0724, +0.3788]**, **−3.9983 [−5.8081, −2.2727]**, and **−3.7037 [−5.5135, −1.9360] percentage points**. Change intervals are 95% paired item/block bootstraps conditional on those models (2,376 questions or 564 blocks), not seed intervals; p-val=n/a.

This distinguishes improved language modeling from retained benchmark capability. It motivates testing better capability preservation in a separately specified future experiment; it does not justify changing or repeating this frozen pilot.

## Resources, admission, and verification

One total **48-device-hour** ceiling covered preparation, training, evaluation and finalization. Admission reserved 172,296.509749 seconds against 172,800 seconds, including every retained preparation attempt. Final conservative accounting is **38,468.953629 seconds (10.68582 device-hours)**: 3,109.509749 preparation seconds plus 35,359.443880 seconds of the entire single-device supervisor lifetime. This includes idle time, evaluation, report generation and publication, so it is an upper bound on active device use rather than a utilization measurement. All task training/evaluation processes exited normally. CPU-only data preparation has no separate elapsed-time subtotal and consumes no device allocation; device preparation is covered by calibration/cell timers. The two early failed smoke attempts use retained log timestamps, not hardware timers.

Each cell had the same 19,900,000 forward-equivalent token target, with at most one-update overshoot; maximum observed overshoot was 79,320 against the frozen 217,088 bound. Probes, validation, reference passes and generated tokens were charged. The unchanged minimum was three adaptive sweeps. Infrastructure timeouts differed by measured method/model cost with the same safety margin; scientific budgets did not differ. Original failed admission forecasts remain in the calibration record.

The [final verification receipt](final_verification.json) records a read-only audit of all 48 completed checkpoints/summary receipts and all 52 evaluation filesets. It verified exact unique item identities/counts, 208 item-file hashes, summary hashes, and recomputed language, preference, choice accuracy and parsed math correctness metrics. It also verified 18 prepared splits, 10 cached model files, 37 frozen files, 35 landed source files and five review/calibration evidence hashes. **Zero errors.** The original unchanged implementation passed **63/63 deterministic tests**; this final audit did not rerun models or add training.

Final generated analyses landed directly on main at [94881e8](https://github.com/brando90/unified-training/commit/94881e8c8d174793f537b06e942d66e060c401e6). The audit began against the preceding reporting commit, whose scientific files were identical; the final analysis hashes are separately bound in the receipt. Source setup, immutable freeze and incremental worker publications are preserved without history rewriting. This final documentation pull request records the completed verification.

## Requested review disposition and limitations

Both requested Claude Code `claude-opus-5-5` / `max` reviews completed with original **FAIL** verdicts. The [plan report](PLAN_REVIEW.md), [implementation report](IMPLEMENTATION_REVIEW.md), [finding-by-finding reconciliation](REVIEW_RECONCILIATION.md), [acceptance](acceptance.json), and [63-test receipt](verification.json) are unchanged. The disposition is **implementer FIXED**, with deterministic repairs and explicit scientific narrowing, not independent post-fix reviewer acceptance. No additional review round was launched.

All four critical and nine major plan findings and the implementation correctness, durability, statistics and hygiene findings have recorded dispositions. Correctness repairs include valid Aioli matrix recovery before any proposed production-cost adjustment, actual sampled reward feedback, reference alignment, zero-advantage optimizer skipping, precision, hash-bound admission, full evaluation denominators and bounded recovery. Endpoint floors and sparse reward were narrowed as limitations; they were not declared solved.

Aioli is mandatory and included. CHORD (Controllable Harmonization of On- and Off-Policy Reinforcement Learning via Dynamic Weighting) and RPT (Reinforcement Pre-Training) conditions are disclosed objective adaptations, not tuned reproductions. The exact **CHERRY-RL reference remains unresolved**; no substitute is claimed to be that paper. TRAPO and other related work are discussed but not all implemented. Therefore “all previous methods” is outside this experiment's comparator coverage.

This is a small, untuned pilot with only three seeds and one early checkpoint age. Math rewards were sparse before freeze; some generated answers collapse to constants, and the fixed generation cap limits interpretation. Nonzero exact match is not robust reasoning evidence. WikiText is not broad retention. Conditional intervals do not model dependence beyond the chosen item/block units. GPU bit-for-bit recovery was never claimed. No settings or cells were changed or rerun to improve outcomes.

Korbak, Shi, Chen, Bhalerao, Buckley, Phang, Bowman and Perez's *Pretraining Language Models with Human Preferences* motivates learning preferences during pretraining, because preference information can shape behavior before a later alignment stage. That motivation does not prove this particular four-objective cost-aware controller wins. See the [verified related-work record](related_work.md) for paper sources and the 2026 addendum.

## Complete artifacts

- [Scratch per-cell results and plots](../01_scratch_joint_objectives/results.md), [full seed/item/initial analysis](../01_scratch_joint_objectives/expt_v1/analysis.json).
- [Early-checkpoint per-cell results and plots](../02_early_checkpoint_joint_objectives/results.md), [full seed/item/initial analysis](../02_early_checkpoint_joint_objectives/expt_v1/analysis.json).
- [Frozen program](frozen_program.json), [data manifest](data_manifest.json), [implementation](IMPLEMENTATION.md), [final verification](final_verification.json).
- [Canonical checkpoint](CKPT_unified_training.md), [coordinator checkpoint](../CKPT_MASTER_tmuxnone_cxd_01a10825.md).

The user-designated canonical experiment paths remain in place because the frozen shared harness, manifests and evidence refer to them. Both are marked complete; preserving those dependencies does not mean training remains active. Large checkpoints, raw item evidence and operational receipts are retained in the owned execution storage; private host/account packets are excluded from this public repository.

**TLDR-end:** [unified-training: completed joint-training pilots] experiments/01_scratch_joint_objectives and experiments/02_early_checkpoint_joint_objectives completed all 48 runs and 52 evaluations under budget. The proposed controller improves on sequential training but does not establish superiority over Aioli; early-checkpoint accuracy declines from its untouched initial value.
**Snapshot:**
```text
Training cells: 48/48; failed/interrupted/missing: 0/0/0
Initial evaluations: 4/4
Primary aggregates: 16/16; paired endpoint comparisons: 14/14
Conservative device-hours: 10.68582 / 48
Deterministic implementation tests: 63/63
Artifact audit: 52/52 evaluations; zero errors
Original reviews: FAIL / FAIL; implementer disposition: FIXED
```
