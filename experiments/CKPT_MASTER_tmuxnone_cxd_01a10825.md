# Unified-training master checkpoint: 48-cell pilot running

**Doc link:** https://github.com/brando90/unified-training/blob/main/experiments/CKPT_MASTER_tmuxnone_cxd_01a10825.md

**Status:** RUNNING — first measured training admitted; no benchmark result yet.
Created: 10-04-2026 13:21 PDT
Last updated: 10-04-2026 13:26 PDT

## 1. Identity and recovery

Coordinator: local Codex desktop, no tmux session; root conversation
`01a10825-fa64-7aa3-b0e0-2f54d5cf6f04`. Working directory:
`/Users/brandomiranda/unified-training`; authenticated Codex profile, execution
model `gpt-6-astra`, reasoning `ultra`, full access, routine approvals disabled.
Resume with `codex resume 01a10825-fa64-7aa3-b0e0-2f54d5cf6f04` under the verified
full-access profile. Private host, process, account and recovery details remain
in private coordinator receipts. The source worker retains exclusive ownership
of program code and scientific settings; this record is coordinator-owned.

## 2. Hazards and scientific boundaries

The question is whether joint pretraining, supervised fine-tuning (SFT), direct
preference optimization (DPO), and reinforcement learning (RL) improve on staged
training and Aioli under the frozen budget. No superiority is established.
Scratch primary: WikiText token negative log likelihood (NLL). Early-checkpoint
primary: normalized accuracy on the AI2 Reasoning Challenge (ARC)-Easy benchmark.
Unassisted mathematical rewards/accuracy are sparse secondary diagnostics.
The exact “CHERRY-RL” identity remains unresolved; named objective adaptations
are not full published-paper reproductions. See [plan](00_program/PLAN.md),
[related work](00_program/related_work.md), and [implementation](00_program/IMPLEMENTATION.md).

Both requested Claude Code `claude-opus-5-5` / `max` reviews completed with original
FAIL verdicts. The implementer's FIXED disposition, 63 passing deterministic tests,
and coordinator acceptance are recorded in [reconciliation](00_program/REVIEW_RECONCILIATION.md),
[acceptance](00_program/acceptance.json), [verification](00_program/verification.json),
[plan review](00_program/PLAN_REVIEW.md), and [implementation review](00_program/IMPLEMENTATION_REVIEW.md).
There is no invented post-fix reviewer PASS. Do not change frozen settings or
reinterpret development diagnostics as held-out results.

Local main was safely synchronized to freeze `0bb4ea8c586927c6fc9c5201f00fcffadc044df4`;
the original 19-file local packet remains preserved in a named stash and archive.
Do not restore that packet over landed source. Coordinator verification compared
37 frozen hashes and 35 landed-source hashes: zero mismatches. Private recovery
receipts retain the precise preservation details.

## 3. Owned workers and execution

The source worker owns `codex/unified-training-experiments`, follows the
[execution runbook](00_program/execute.md), and maintains the authoritative
[source checkpoint](00_program/CKPT_unified_training.md) and experiment ledgers.
Its independently supervised queue continues when the coordinator disconnects.
At 10-04-2026 13:19:47 PDT, scratch sequential seed 0 had 72 optimizer
opportunities and 1,784,728 / 19,900,000 charged forward-equivalent token units;
first measured update: 13:18:53 PDT. Supervisor and detached training child were
independently verified alive; watchdog reported `training_alive`. Snapshot:
1 run active, 0/48 complete; later canonical receipts supersede this observation.

[Experiment 01](01_scratch_joint_objectives/README.md) starts Pythia-70m randomly;
[experiment 02](02_early_checkpoint_joint_objectives/README.md) starts Pythia-160m
at immutable pretraining step 10000. Each has eight methods × three seeds:
48 training cells plus four initial evaluations. Every cell receives 19,900,000
forward-equivalent charged token units; the three-adaptive-sweep minimum remains
15,336,192. [Freeze](00_program/frozen_program.json) forecasts 172,296.5097 seconds
against 172,800 seconds, including 3,109.5097 preflight seconds, 52 evaluations
budgeted at 818 + 60 seconds each, and 1,800 finalization seconds. Method-specific
infrastructure caps retain the same 2× margin. This is a forecast, not a measured
completion time or guaranteed service rate. Preserve full-denominator accounting.

## 4. Watches and recovery ownership

Heartbeat `unified-training-finish-reviewed-snap-pilots-and-sync-main` is ACTIVE,
every 30 minutes in this root chat. It reports meaningful completion, failures,
required action or changed results; it stops only at verified full completion or
explicit cancellation. The local app must be available for that heartbeat; the
remote deterministic queue is independent. The coordinator owns recovery and
publication checks; source-worker continuation/budget limits remain in its runbook.

Watch [scratch results](01_scratch_joint_objectives/results.md),
[early-checkpoint results](02_early_checkpoint_joint_objectives/results.md), both
[cell ledgers](01_scratch_joint_objectives/expt_v1/cell_status.json)
([early](02_early_checkpoint_joint_objectives/expt_v1/cell_status.json)), and the
source checkpoint. Resume the read-only publication watch from an owned checkout:
`gh pr list --repo brando90/unified-training --head codex/unified-training-experiments --state all --json number,state,url`,
then inspect the exact returned pull request and fetched canonical checkpoint.
The native automation tool manages heartbeat recovery; do not create a duplicate.

## 5. Decisions waiting on Brando

None currently gate this frozen pilot. The unresolved “CHERRY-RL” paper identity
remains a literature question for later work and does not justify altering this run.

## 6. Ranked dispatch list

1. Continue the entire admitted frozen queue and preserve failed/interrupted cells;
   source worker remains responsible under the existing execution runbook.
2. On meaningful terminal evidence, reconcile all 48 cells and 52 required evaluations,
   verify artifacts and denominators, and publish the bounded analysis.
3. After confirmed publication, refresh this record and safely synchronize the
   coordinator-owned local checkout. No additional reviewer/model dispatch is planned.

## 7. What landed

Source setup [4adbb013e0f7cb233520751240e3785a1ecb8298](https://github.com/brando90/unified-training/commit/4adbb013e0f7cb233520751240e3785a1ecb8298)
and freeze [0bb4ea8c586927c6fc9c5201f00fcffadc044df4](https://github.com/brando90/unified-training/commit/0bb4ea8c586927c6fc9c5201f00fcffadc044df4)
were pushed directly to main by the source worker. They did not pass through a
pull request. This documentation-only coordinator pull request repairs the missing
durable pull-request record prospectively; it neither rewrites those commits nor
claims they were pull-request merges. Coordinator record pull request #1 is merged;
full experiment status remains RUNNING.

LANDED f721b93259bb1a4cd25fe9a67fdc466096342db2 https://github.com/brando90/unified-training/pull/1 10-04-2026 13:24:37 PDT

The head-matched merge was verified as MERGED. The single required completion
email for pull request #1 was delivered at 10-04-2026 13:26 PDT; its delivery
receipt stays in private coordinator receipts. This follow-up only publishes the
landing record and does not represent another experiment phase or completion.

Rebase safety record: owned backup branch
`codex/backup-coordinator-record-before-main-07465c92`; previous merge base
`0bb4ea8c586927c6fc9c5201f00fcffadc044df4`; fetched main
`07465c9206bb07b66c32170e8b00025e5def9653` (worker's first-training record).
The backup captures this checkpoint before rebasing; compare both patch series
and the coordinator-owned file before merging.

## 8. Next commands

Read the authoritative source checkpoint and live ledgers before inferring progress.
At a landed phase, verify the exact pull-request merge commit and publication
receipt, then let the root coordinator perform the clean main fast-forward.
Never infer experimental completion from an idle agent or an exited coordinator.

**TLDR-end:** [unified-training: master checkpoint] The 48-cell pilot is running;
source and freeze are on main, two original FAIL reviews were reconciled, and no
held-out superiority result exists yet.

**Snapshot:** 10-04-2026 13:19:47 PDT — 1 active run, 0/48 complete;
72 optimizer opportunities, 1,784,728 / 19,900,000 charged units in the first cell.
