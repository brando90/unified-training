# Unified-training prospective pilot

Updated 10-04-2026 after the independent Opus 5.5 maximum-effort plan review.
The original proposal and pre-review plan remain in `original_proposal.md` and
`PLAN_before_review.md`; they are historical, not the current execution contract.
The original FAIL verdict and every finding remain in `PLAN_REVIEW.md`.
`REVIEW_RECONCILIATION.md` records dispositions. No measured results exist yet.

## Question and scope

Does cost-aware objective selection improve a small-model public-data pilot over
staged training and the same Aioli-style controller without cost adjustment?
The primary exploratory endpoint is WikiText token negative log likelihood for scratch and ARC-Easy length-normalized accuracy for the early checkpoint; GSM8K
exact-answer accuracy, held-out preference ordering, WikiText negative log
likelihood and actual device-seconds are secondary. Proposed-minus-sequential
and proposed-minus-Aioli are the two primary descriptive contrasts. No hypothesis
of universal superiority, formal significance decision, or broad retention claim
is licensed by three seeds and these small finite data pools. No hyperparameter
tuning budget is allocated to any condition; all settings are agent-specified
pilot choices, not tuned strong reproductions of published methods.

## Full matrix and starting states

Each experiment has eight conditions and seeds 0, 1, 2: sequential, parallel_joint,
fixed_joint, smooth_joint, aioli_objective, validation_progress, chord_objective,
rpt_inspired. There are 48 measured cells in the denominator. The prospective
parallel comparison replaces the redundant uniform sampler before freeze; it
preserves the original full denominator. No synthetic benchmark replaces the
public natural-language experiment. CHERRY-RL remains unidentified; CHORD and
RPT are explicitly provisional adaptations, not claims to have identified it.

Experiment 01 constructs the Pythia-70m architecture from a fresh random seed.
Experiment 02 loads Pythia-160m step10000, immutable revision
`bb9bd9573b772c899bda5206af551645138ceed8`. Architecture and initialization within
each experiment are shared across methods. No published test accuracy selects
checkpoint age. All early-checkpoint seeds share weights and vary data/action
randomness; scratch seeds also vary initialization. Full untouched initial
checkpoints are evaluated after the fixed cell queue, not used for selection.

## Data and prospective readiness

Pinned releases, derived split membership, exclusions and file hashes are in
`data_manifest.json`. Training uses WikiText-103, GSM8K and corrected
UltraFeedback-binarized. Both preference responses must fit whole. Finite training
pools are 8,192 corpus blocks, 6,748 math questions and 4,092 preference pairs.
The overlap audit removes exact 13-word overlaps with public benchmark test
content (9 math training, 1 math validation and 3 preference training rows).
A subsequent normalized-question substring audit removes one further math training row and one preference row, including a check against preference-test prompts. These audits use text equality only, never model predictions or scores. Original prepared
files and exclusion receipts remain preserved. Semantic/inherited contamination
is not excluded by this audit.

Controller feedback and development/admission data are disjoint by deterministic
hash partition: GSM8K 357/357, preferences 120/136, WikiText blocks 240/251.
ARC-Easy's 570-item official validation set is development-only. Terminal tests
are all 2,376 ARC-Easy questions and 1,319 GSM8K questions, 564 WikiText blocks
and the predeclared 256 complete held-out preference pairs. The preference result
is a bounded-pool endpoint, not an evaluation of all 2,000 official test pairs.

A single development seed checks initial multiple-choice chance and mathematical
reward before freeze, then 64 and 128 supervised updates from the training pool.
Prefixes 0, .5 and .9 are checked prospectively for a tractable reward signal;
the observed unassisted signal retains prefix 0 for both experiments. The gate
(one mixed-success group in 32 groups of four by 128 updates) is a declared
engineering check, not a literature threshold or evidence of robust reasoning.
This is sparse engineering signal only: the terminal 128-update scratch check is 1/128 correct with 1/32 mixed groups; the early unassisted check is 0/128 and 0/32. At initialization ARC development is 29/128 for scratch (95% Wilson [0.163,0.306]) and 47/128 for early ([0.289,0.453]), against mean chance .2496; p-val=n/a. Scratch reasoning remains floor-limited. All observations, including zeros, parser failures and truncation, remain in
`readiness.json`. No test score or generated gold data enters selection.

## Algorithm and accounting contract

`IMPLEMENTATION.md` gives exact losses, optimizer/masking conventions, moving
reference clock, baseline differences, Aioli equations and cost normalization.
Aioli and the proposed method share initialization, smoothing, loss scales,
probe lengths, floors and update rule; only the proposed matrix divides recovered
objective effects by measured production cost. Neither has a retention guard.
The former proposed controller utility class remains tested historical library
code; the engine's current method mapping is authoritative.

The training budget counts executed padded forward tokens, backward work at
multiplier 2, reference passes, rollout input tokens and all controller/validation
work. Report it in millions of forward-equivalent tokens and also device-seconds.
Terminal evaluation and development/calibration work have separate explicit
ledgers, all within the same 48-device-hour full-program ceiling. Larger batches
are calibrated before freeze, with all original failed and successful calibration
cost retained. `calibration.json` chooses a common per-cell budget and conservative
nested timeouts from validation-only timings. A final admitted update may overshoot
by its predeclared unit bound; no hidden free rollback or result-dependent budget
extension is permitted. Fixed mixture probabilities are **selection probabilities**,
not matched objective compute shares; actual objective work is reported.

## Inference, durability and completion

Every trained model reports conditional item uncertainty; binary endpoints use
Wilson score intervals so zero accuracy does not imply zero uncertainty. Paired
item bootstraps and separate three-seed uncertainty must not be conflated.
Method-level two-stage bootstrap resamples seeds then common items. Missing cells
remain in the full denominator, with complete-case summaries and bounded
best/worst sensitivity for accuracy. No formal hypothesis test is run: p-val=n/a.
WikiText measures in-domain language modeling, not Pile-wide retention. Changes
against untouched initialization and compute/performance plots are descriptive;
no claim to beat a fully tuned Pareto frontier is made.

The required plan report and focused implementation review must both be reconciled
before acceptance. Publish the reviewed setup, then freeze content hashes and
verify the frozen setup landed on main. A separate named deterministic supervisor
runs the full queue, records native process identities and survives coordinator
exit. No healthy detached trainer is duplicated or stopped at a coordinator
boundary. Completed cells require full denominators and independently verified
artifacts, not just exit zero. Initial diagnostics, final analysis and verified
main publication are required for full completion. Missing or failed cells remain
honest unresolved rows; smoke and launch never count as experiment completion.
