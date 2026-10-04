# Unified-training implementation checkpoint

Updated 10-04-2026. Unified-training pilot: random Pythia-70m and immutable
Pythia-160m step10000, eight methods × three seeds each; required reviews repaired,
63 tests pass, reviewed setup and first full-cell results landed, independent
full-set training running, 3/48 cells completed as of 21:09 UTC.

Exclusive source/publication owner: this task on `codex/unified-training-experiments`.
Original base `f9d1f6b92fcebf7098d25abd89186593f13b36ec`; the 19 supplied source
hashes were verified before work and their intentional edits were not reset.
Native runtime is `gpt-6-astra`, effort `ultra`, full access, approval policy `never`,
verified from native launch evidence. Private process/account/input receipts stay
outside Git. The original six-hour bound is unchanged. Two intervening invocations
executed zero tools due to the relocated runtime-companion failure. This is the
one authorized infrastructure recovery, capped at two hours; no further automatic
model continuations or extra model-provider attempts are authorized.

Coordinator heartbeat: `unified-training-finish-reviewed-snap-pilots-and-sync-main`,
every 30 minutes in the existing root chat, stopping only on verified completion
or explicit cancellation. The root source checkout remains read-only until landing.
No collaborator messages were sent. No provider keys, overages or reviewer restart.

## Review and scientific disposition

Both required Claude Code `claude-opus-5-5` maximum-effort reviews are complete.
The plan's final same-session result is FAIL (4 critical, 9 major); the earlier
max-turn transcript is not its final verdict. The focused implementation review
also returned FAIL with concrete fixes. Full original reports are preserved.
`REVIEW_RECONCILIATION.md`, `verification.json` and `acceptance.json` distinguish
implementer FIXED disposition from the unchanged original reviewer verdicts.
The coordinator explicitly authorized deterministic repair/reconciliation without
an additional independent round. All 63 deterministic tests pass, including the
original 26 controller/Aioli tests and new objective, integrity and recovery checks.

All coordinator notes acknowledged: actual runtime/heartbeat, publish-first
milestone, subscription limits, driver/board semantics, validation-only readiness,
retained step10000 revision, critical Aioli counterexample, final plan findings,
six-paper literature addendum, batching/resource findings, sparse-readiness
interpretation, authorized wall-budget fallback, and the preferred method-specific
infrastructure-timeout option. The chosen option keeps equal token-proxy scientific
budgets and the original cost equation; no wall-clock scientific switch was made.

Aioli is repaired: count-mixture response columns are normalized only by initial
validation scales before inversion. The [1,2,0,0] effect / [16,48,16,16] cost
counterexample recovers below 1e-12 error; old cost division reverses preference.
Only the proposed controller divides recovered objective effects by measured
production cost. All probes and validation are charged. Preference loss uses a
matched moving reference and length normalization; RL feedback is actual sampled
reward, not duplicated SFT loss. Parallel independent-optimizer averaging replaces
uniform sampling prospectively, preserving all 48 cells. RPT formatting is fixed.

Scratch primary: WikiText token negative log likelihood. Early-checkpoint primary:
ARC-Easy normalized accuracy. Unassisted GSM8K is a sparse secondary diagnostic.
Initial development ARC: scratch 29/128, early 47/128. Terminal 128-update unassisted
math check: scratch 1/128 correct and 1/32 mixed groups; early 0/128 and 0/32.
These do not establish robust reasoning readiness. Preserve Pythia step10000
`bb9bd9573b772c899bda5206af551645138ceed8`; no test-based choice or synthetic replacement.

## Resource and execution state

Prepared data was preserved, then audited/partitioned without repeating downloads.
All source/data/cache/logs/checkpoints remain on task-owned local scratch.
Train pools after both audits: 8,192 text blocks, 6,748 math rows, 4,092 preferences.
Controller and development splits are disjoint. Official test denominators remain
2,376 ARC and 1,319 GSM items, plus 564 corpus blocks and 256 bounded preferences.

The failed 4.6M-unit forecast and intermediate projections are retained. Final
resource selection gives every cell 19,900,000 charged forward-equivalent tokens,
above the unchanged 15,336,192 minimum for three estimated controller sweeps.
Batch sizes: 16 supervised/preference examples, 16 reinforcement prompts × 4 samples.
Checkpoint interval: 256 updates. Condition-specific infrastructure timeouts use
the declared schedules and identical 2x timing margins; they grant no extra
scientific work. Full 48-cell + four initial-evaluation + preflight + finalization
reservation: 172,296.51 seconds = 47.86 hours, below the original 48-hour ceiling.
This is a forecast with fixed failure limits, not a guaranteed service rate.

Current measured state: 3/48 completed, 0 failed; scratch parallel-joint seed 0 is
training after all three sequential seeds completed. Driver identity is the actual deterministic
supervisor, separate from the model coordinator. Board collector binding is
verified; preparation correctly used a null driver. Liveness and
semantic last-progress timestamps are separate.

Reviewed setup and immutable freeze are verified on main. The named independent
session `unified-training-full-program` owns the complete fixed queue. Its actual
supervisor/trainer native identities are registered privately and verified alive.
First measured update: 10-04-2026 20:18:53 UTC, scratch sequential seed 0, step 1,
39,832 charged tokens including initial validation. The receipt explicitly says
measured=true and smoke=false. This is training progress, not a completed cell.

The first full cell subsequently completed at 20:37 UTC: scratch sequential seed 0,
839 updates, 19,917,904 charged forward-equivalent tokens, 851.33 training seconds
and 233.58 terminal-evaluation seconds. Its 17,904-unit final overshoot is below
the frozen 217,088 bound. The ledger records 419 clipped updates, 65 reinforcement
updates, 31 zero-gradient skips, and zero adaptive-controller iterations for this
nonadaptive baseline. All 564 text blocks, 256 preference pairs, 2,376 multiple-choice
questions and 1,319 math problems were independently checked against unique frozen
item identities; summary and checkpoint hashes match. Recomputed endpoint values
match the published summary. The cell's exit code alone was not used as completion.

| First-cell endpoint | Estimate [95% interval] | Interval / test |
|---|---|---|
| WikiText negative log likelihood, scratch primary | 8.3725 [8.3531, 8.3909] | Corpus-block bootstrap; p-val=n/a |
| ARC-Easy normalized accuracy | 0.2504 [0.2334, 0.2682] | Wilson item interval; p-val=n/a |
| GSM8K exact match | 0.0227 [0.0160, 0.0323] | Wilson item interval; p-val=n/a |
| Raw preference accuracy | 0.5234 [0.4624, 0.5838] | Wilson item interval; p-val=n/a |

These intervals condition on one trained model, not three-seed uncertainty or a
method comparison. All 1,319 math generations hit the fixed 128-token limit;
all extracted answers are the number 8 and none contains the answer delimiter.
The 30 exact matches therefore reflect answer collapse, not demonstrated problem
solving. See the scratch [first-cell diagnostic](../01_scratch_joint_objectives/first_cell_diagnostic.md).
The independent supervisor automatically published the first aggregate/plot
snapshot and advanced to the next seed without a model call or rerun.

The second full cell, scratch sequential seed 1, also passed independent item,
summary and checkpoint verification and automatic main publication. Its WikiText
negative log likelihood is 8.3701 [8.3468, 8.3929], a 95% corpus-block bootstrap
interval conditional on this model; p-val=n/a. Its math answers concentrate on
3 (1,222 items), 2 (96) and 40 (one), with all 1,319 generations truncated.
The third sequential seed subsequently passed the same full verification. The
first baseline's three-seed language-loss mean is 8.5329 [7.8374, 9.2284], a 95%
Student t interval with two degrees of freedom; p-val=n/a. The normal-seed
assumption is weak with only three seeds. Cross-method conclusions and changes
from the untouched initial models remain pending. The third seed's extracted math
answers also concentrate on two numbers: 1 for 1,310 items and 5 for nine.

The supervisor continues all 48 cells, always runs the four initial diagnostics,
verifies full artifacts/denominators, and publishes deterministic incremental/final
reports. Source/data/manifest remain frozen. Preserve healthy children at every
coordinator boundary; do not rerun or alter measured trajectories. Completion
requires full experiments, analysis and verified final publication. No further
automatic model continuation is authorized after this recovery invocation.

The parent's documentation-only main advances were inspected and fast-forwarded
before automatic result publication. All 37 frozen input hashes still match.
The early matrix is admitted and queued, despite the known all-pending generated
results-template sentence saying its gates are pending; its README explicitly
corrects that presentation defect. No frozen code or scientific setting was changed.

## Durable continuation at the recovery bound

The original cumulative six-hour bound and this recovery's two-hour cap are
unchanged. The named independent supervisor owns all remaining deterministic
training, evaluation, analysis and incremental publication; no new model-provider
attempt or reviewer session was launched in this recovery. Its current child is
registered and healthy. Private progress, watchdog, board verification and native
process-identity receipts remain beside the checkout; coordinator status is
`training_running`, never `complete`.

Remaining: 21 scratch cells (including the active parallel-joint seed 0), all 24
early-checkpoint cells, four untouched-initial-model evaluations, complete method
comparisons and final verified publication. Preserve the fixed manifests, all
healthy children and the original queue clock. Failed or missing cells stay in
the denominator. The heartbeat automation named above continues until verified
full completion or explicit cancellation. A completed sequential baseline is not
completion of either experiment.

LANDED 4adbb013e0f7cb233520751240e3785a1ecb8298 https://github.com/brando90/unified-training/commit/4adbb013e0f7cb233520751240e3785a1ecb8298 2026-10-04T20:17:19.422635+00:00

Reviewed setup is verified on main. Immutable freeze and measured admission followed;
full experiment completion remains pending.

LANDED 0bb4ea8c586927c6fc9c5201f00fcffadc044df4 https://github.com/brando90/unified-training/commit/0bb4ea8c586927c6fc9c5201f00fcffadc044df4 2026-10-04T20:19:33.625044+00:00

LANDED 3b845821f26baafd27ec45c6599b8fd5d68db522 https://github.com/brando90/unified-training/commit/3b845821f26baafd27ec45c6599b8fd5d68db522 2026-10-04T20:37:27.745424+00:00

LANDED 2f771a3f4857ae7a326505b4364ff02e9c88cde5 https://github.com/brando90/unified-training/commit/2f771a3f4857ae7a326505b4364ff02e9c88cde5 2026-10-04T20:56:24.584195+00:00

LANDED c7526b926563da4fc40ab4e6cf813a83e635ee26 https://github.com/brando90/unified-training/commit/c7526b926563da4fc40ab4e6cf813a83e635ee26 2026-10-04T21:09:12.797573+00:00
