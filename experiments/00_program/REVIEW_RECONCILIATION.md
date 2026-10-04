# Required reviews and current-source reconciliation

Updated 10-04-2026. Both requested independent reviews used Claude Code
`claude-opus-5-5` with maximum effort through the verified subscription route.
Both completed native processes exited zero and emitted successful result events.
**Both original verdicts are FAIL.** Their complete reports are preserved in
`PLAN_REVIEW.md` (four critical, nine major) and `IMPLEMENTATION_REVIEW.md`.
Machine receipts bind report hashes and actual model/launch evidence. Neither
report is rewritten as a reviewer PASS. The implementer's acceptance decision,
when all deterministic repairs and resource gates pass, is separately recorded
in `acceptance.json`, as expressly authorized by the coordinator's instruction to
reconcile these two reviews without launching another review round.

The plan's same-session continuation is its final result; the earlier max-turn
failure is retained in the receipt but is not the final outcome. The implementation
review received an earlier source snapshot and 41-test log. Later repairs below
are deterministically checked, not falsely attributed to reviewer inspection.
No extra model-provider attempt, key-based request, overage or reviewer restart
was used. Source and evidence remain within the original program allowance.

## Plan findings

| Finding | Disposition and evidence |
|---|---|
| C1: endpoint floors and undersized compute | Narrowed, not declared scientifically solved. Development ARC is scratch 29/128, 0.227 [0.163,0.306], and early 47/128, 0.367 [0.289,0.453], 95% Wilson, p-val=n/a; mean chance .2496. Scratch primary is WikiText token loss; early primary is normalized ARC accuracy. GSM8K is sparse secondary. Preserve step10000. Larger batch calibration and a minimum of three complete controller opportunities precede freeze. Reject the inference that similar historical test scores establish model capacity, and do not use them to choose age. |
| C2: frozen random preference reference and response-length confounding | Fixed prospectively: token-mean preference objective, beta=1, explicitly a DPO adaptation; common hard reference clock .1 through .9 plus .75, which captures staged post-SFT weights. Controller preference feedback is reference-free length-normalized logistic loss. Tests verify uniform per-token language gains cancel despite unequal lengths and reference alignment at .75. Residual semantic length bias is not claimed absent. |
| C3: controller units, clipping, rollback and unspecified probes | Current engine uses in-trajectory 9/1/1/1 probes, four shuffled columns, uniform start, .02 floor, 64 production opportunities between sweeps, eight fixed feedback rows/component. No rollback. Relative loss scales, normalized matrix updates, iteration counts, probabilities and clipping are explicit. Original boundary/clipping tests remain. |
| C4: duplicated SFT/RL proxy and zero-gradient momentum | Fixed: RL feedback is sampled mathematical error with fixed common random numbers and smoothing, not teacher-forced solution loss. Independent optimizer states and complete zero-advantage optimizer skip prevent inherited momentum/decay credit. Tests cover both mechanisms. Feedback remains noisy and cannot credit very long delayed transfer. |
| M1: selection probabilities are not compute shares | Corrected all current descriptions; retain distinct selection-probability baselines and report actual objective work. Do not claim equal per-objective compute allocation or interpret them as a tuned compute frontier. |
| M2: incomplete objective/optimizer specification and weak sequential baseline | Exact masking, beta, aggregation, rollout/group/temperature/cap, optimizer states/learning rate/decay/clipping, zero-gradient rule and reference clock are in IMPLEMENTATION.md. No offload/accumulation; unequal-length full-batch gradient equivalence is tested. Tuning budget is zero for every method; all are agent-specified pilot choices, not tuned published reproductions. |
| M3: Aioli fidelity and bundled proposed changes | Fixed common probe/update setup; proposed differs only by dividing recovered effect columns by measured action cost. Aioli does not divide probe responses by compute. Both omit retention guard. Relative-loss/floor/objective adaptations remain explicit. CHORD is a joint hybrid loss with disclosed differences; RPT remains a diagnostic adaptation mandated by execute.md, with zero rewards retained. |
| M4: missing literature/parallel baseline/unresolved CHERRY | Six-paper verified addendum incorporated. Parallel independent-optimizer updates replace uniform sampling before freeze, preserving eight methods and 48 cells. TRAPO remains an explicit untested competitor. CHERRY identity is still unresolved; no substitute is presented as exact. Reject the suggested reduced 42-row matrix because the authorized full pilot retains 48 rows and named provisional variants. |
| M5: split contamination and retention dataset | Fixed disjoint controller/development/test use; corrected UltraFeedback immutable revision; 13-word and normalized-question substring audit, including GSM training versus preference-test prompts. All exclusions/hashes retained. Preference test remains the original 256 complete bounded pairs, explicitly not the full 2,000-pair release. Broader Pile test expansion is deferred; WikiText is in-domain modeling, not broad retention. |
| M6: ineffective/asymmetric retention guard | Removed from both adaptive methods; no asymmetric guard advantage. Report WikiText and initial-checkpoint capability changes, with no Pile-wide retention claim or scratch "retention" interpretation. |
| M7: sparse reward | Measured, retained as a scientific limitation. At 128 supervised updates, scratch unassisted reward is 1/128 and mixed groups 1/32; early unassisted is 0/128 and 0/32. The earlier nonzero groups only establish engineering signal. Neither is robust reasoning readiness. No silent synthetic-task switch, no invented literature threshold, no test-based setting choice. Final timing uses larger rollout batches for throughput and group coverage. |
| M8: statistics/decision rules | Scratch language loss and early normalized ARC accuracy are prospective primaries; contrasts against sequential and Aioli are descriptive. Conditional Wilson/item intervals, paired item/block comparisons, separate three-seed t intervals, two-stage paired seed/item bootstrap and missing-accuracy identification bounds are implemented. No formal superiority test, Holm claim, retention noninferiority claim or universal-frontier claim. Three seeds and one checkpoint age remain weak evidence. |
| M9: old boundary/resume/clipping/adapters defects | Current boundary/floor/clipping tests predate this handoff and are preserved. Full optimizer/random/stream/probe state is checkpointed. CPU exact recovery now tested through both adaptive sweeps and RL. All adapters execute actual local objectives. GPU state restoration is not claimed bit-for-bit. |

The review's proposed 10% success, 20% mixed groups and correlation cutoffs are
reviewer-suggested engineering thresholds, not validated literature thresholds.
They are not quietly reported as passed. The narrower endpoints and explicit
sparse-reward scope above are the coordinator-authorized prospective disposition.
The initial development gate itself is only a nonzero-signal check.

## Critical coordinator Aioli counterexample

The implementation review called the old cost division a disclosed adaptation;
that does **not** make its matrix recovery valid. The coordinator's algebra takes
precedence: for W diagonal .75/off-diagonal 1/12, true effects [1,2,0,0], and
probe costs [16,48,16,16], exact drops recover the true effects below 1e-12 error.
Dividing response columns by those costs before the same inverse instead gives
approximately [.070747,.034288,.008247,.008247], reversing SFT/PT preference.
The engine removes that division for Aioli. The proposed method unmixes first,
then adjusts each recovered objective column by its production cost; all actual
probe and validation work is still charged. `test_aioli.py` permanently covers
the exact counterexample. This repair was made after the review snapshot and is
not claimed to have been independently re-reviewed. The coordinator explicitly
authorized deterministic reconciliation of the completed reviews without a
further round. No mandatory comparator is silently converted into the proposed
cost-aware rule.

## Focused implementation findings

| Finding | Repair / deterministic evidence |
|---|---|
| P0-1 RPT leading-space mismatch | Symmetrically remove one formatting space from prediction and corpus truth, adjusting token-boundary byte lengths. Tests cover whole token, mid-token, punctuation, absent marker and empty output. |
| P0-2 unbound publication/freeze | Fetch main before freeze; require every source/equation/protocol/calibration artifact to match the verified landed commit byte-for-byte. Runtime repeats content checks and requires the freeze itself and frozen manifests committed. Final calibration records exact producing source hashes and base commit; freeze checks current hashes. All prepared split/model hashes verified. Dirty-source rejection test uses a real temporary Git repository. |
| P0-3 precision | Explicit float32 constructors and parameter assertion, bfloat16 autocast, version/dtype records in calibration/freeze/training. The original real smoke caught a half-precision failure; its logs and cost are retained. |
| P1-1 durability/adoption | --recover holds the same lock and original persisted clock, adopts healthy native identities, marks dead cells interrupted with last event/checkpoint, and continues pending cells. Detached cells have independent sessions. Real process recovery drill kills a fake child, retains interruption, executes the next pending cell and all initial diagnostics. Engine state restoration is a library capability, not automatic regeneration of lost measured work; no lost-work budget reset. |
| P1-2 representative calibration | Final per-objective timing covers PT/SFT/DPO/RL/RPT/hybrid/parallel, actual event/progress writes, validation and checkpoint I/O. Slowest weighted-mean action rate plus measured checkpoint envelope covers all allowed mixtures in the prospective forecast; doubled margin, with every individual rate/max retained. A 256-update checkpoint cadence and equivalent batched choice scoring replace the costly provisional settings. The earlier 4.6M-unit bound failed the unchanged three-sweep minimum and is preserved, not declared admitted. Full disjoint development evaluation is extrapolated per dataset, not by an aggregate ratio. All earlier calibration/readiness work remains charged. Final resource evidence must pass before acceptance. |
| P1-3 conditional/unbounded initials | Always execute the four declared untouched initial evaluations, even with failed cells; each has a status record and the outer group has its reserved fixed timeout. Initial failures remain explicit. |
| P2-1 milestone overhead contaminates adaptive utility | Removed by the scientific redesign: Aioli uses no column-cost division; proposed uses only production action costs after unmixing. Milestone costs remain in total ledger, never in either fitted response denominator. |
| P2-2 failure taxonomy | Record exception type/class/message/stage, explicit cell-timeout events, numerical failure, and native interruption. Preserve every failed/interrupted cell. |
| P2-3 missing reported endpoints | Add raw and implicit preference metrics, WikiText block-paired NLL, all seed/paired/two-stage summaries, initial-checkpoint changes and ledger columns. Raw normalized policy ordering is the preference endpoint; reference-relative ordering is diagnostic. Reports require frozen-source validation and independently verified completed artifacts. |
| P2-4 partial completeness/rerun mixing | Explicit checks for all four denominators; exclusive creation of all item-evidence files prevents mixing attempts. Independent artifact checks additionally verify exact unique item IDs and content hashes. |
| P2-5 embedded benchmark questions | Additional normalized-question substring audit (minimum 40 characters), across preference prompts/responses and neighboring WikiText blocks, plus math training versus preference-test prompts. One further math and one preference row removed; no model scores used. |
| P2-6 no minimum scientific budget | Final calibration requires enough charged work for three complete adaptive sweeps plus intervening production and validation, in addition to fitting the 48-hour ceiling. This guarantees opportunities, not reward robustness or statistical power. |
| P2-7 limited exact-resume claim | Narrowed to CPU bit-for-bit; GPU restores full state without deterministic-kernel claim. Mid-probe and RL continuation tests now exercise actual advance() for both adaptive methods. |
| P2-8 on-policy real-path gap | Tiny real left-padded sampling versus right-padded scoring test passes within 1e-5. Production records mean/max absolute log-probability gaps per RL update; final calibration records accelerator gaps. Cached bfloat16 numerical approximation is disclosed. |

P3 hygiene: freeze checks before changing status; progress locked with separate
liveness/semantic timestamps; atomic replacements sync directories; cells claim
an exclusive file; global-bound rows are explicit; initial diagnostics/statuses
are preserved; item mismatches fail report verification; optimizer restoration
uses PyTorch's device policy; noninitial zero validation loss is accepted; per-cell
objective/zero-gradient/reward/truncation/clipping summaries are recorded; every
new accelerator child performs a fresh availability/memory check; final calibration
holds the owner lock; terminal completion uses a small marker rather than rewriting
the checkpoint. Official question hashes are retained because all prepared test IDs
are verified unique with complete official counts; no duplicate item can collapse
silently. Queue order remains fixed and its ordering is explicit, not selected by
results. No additional model review was requested for these deterministic repairs.

## Pending gates

The final resource gate passes at 19.9 million forward-equivalent tokens/cell, above the unchanged 15,336,192 three-sweep minimum. All 48 fixed-work cells plus four initial evaluations, all preflight and finalization reserve sum to 172,296.51 seconds (47.86 hours), below 172,800 seconds. Infrastructure timeouts are method/model-specific from the same 2x margin; no condition receives extra scientific work. The 63-test suite passes. The resource-calibration receipt, deterministic suite, exact
staged-diff/secret checks, verified setup main commit and immutable freeze are
separate gates. Their status is recorded in acceptance/checkpoint/progress rather
than inferred from this document. Full completion additionally requires 48 full
cells, four initial evaluations, final analysis and verified results publication.
