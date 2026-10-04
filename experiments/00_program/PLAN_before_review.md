# Unified training: research and execution plan

**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/00_program/PLAN.md>

**TLDR:** Test whether continuously mixing pretraining, instruction learning, preference learning, and reasoning reinforcement learning improves the accuracy–retention tradeoff at a fixed training budget. Run a bounded from-scratch pilot and a separate early-checkpoint pilot on Stanford Network Analysis Project (SNAP) hardware; neither pilot alone establishes superiority over published large-model systems.

Created 10-04-2026. Status: DRAFT FOR REQUESTED CLAUDE CODE REVIEW. This document may change before measurement; freeze each experiment's resolved manifest before its first measured update.

## Scientific question and scope

Does a validation-guided mixture of all four training objectives outperform an ordinary staged schedule and competent joint-training baselines, with the same architecture, initial weights per seed, available data, and inclusive compute budget? The hypothesis is falsifiable: joint objectives can interfere, preference learning can consume scarce capacity, and early reinforcement learning can have no useful reward variance. No guarantee of a better optimum, no-forgetting property, or universal victory is assumed.

The four objectives are next-token cross-entropy pretraining (PT), response-only supervised fine-tuning (SFT), Direct Preference Optimization (DPO) against a frozen initial reference model, and on-policy reinforcement learning (RL) with a deterministic mathematical-answer reward. Existing preference labels are used; no paid model labeling or provider application programming interface calls are authorized. We do not modify public benchmark answers.

Two separate experiments answer different questions:

1. `01_scratch_joint_objectives`: train a small causal language model from random initialization, using the Pythia-70m architecture and tokenizer. General text, instructions, preferences, and mathematical prompts are available from the first update in joint conditions. This is a limited-data feasibility experiment, not web-scale pretraining.
2. `02_early_checkpoint_joint_objectives`: initialize every condition from the same immutable early Pythia-160m checkpoint, initially `step10000`. The Pythia training trajectory has 143,000 steps; verify the checkpoint metadata and training-token count before freezing. Report the inherited pretraining cost separately. This tests continuation, and cannot support a from-scratch claim. Include the untouched starting checkpoint as a diagnostic.

The initial sizes prioritize a full comparison and several seeds on one device. Scaling to a larger model is a separate prospective experiment, based on validation learning and reward diagnostics, not on selecting favorable test results. A negative or floor-limited pilot is a publishable project result, not permission to silently enlarge the budget.

## Literature and baseline identity

Read `related_work.md` and the linked original algorithms before implementing a named method. The original proposal's Aioli title is incorrect: Aioli is a framework for language-model data mixing. Applying its controller to heterogeneous objectives must be labeled **Aioli-objective-adapted**, not an exact published-model reproduction.

The user's “cherry-rl” paper has not yet been identified conclusively. Reinforcement Pre-Training (RPT; arXiv:2506.08007) describes RL as the “cherry on top”; CHORD (arXiv:2508.11408) combines SFT and RL. Both are relevant, but neither name should silently be substituted for the requested reference. Track the unresolved identity explicitly until resolved.

Required first pilot conditions, each with seeds 0, 1, and 2:

| Condition | Definition | Evidence scope |
|---|---|---|
| sequential | PT → SFT → DPO → RL, budget fractions .55/.20/.10/.15 | Generic staged baseline; do not call it a DeepSeek-R1 reproduction |
| uniform_joint | Equal selection probabilities for four objectives | Separates jointness from a tuned scheduler |
| fixed_joint | Constant .55/.20/.10/.15 selection probabilities | Same nominal mix as staged baseline |
| smooth_joint | Linear .70/.20/.05/.05 → .20/.20/.15/.45 | Hand-designed continuous curriculum |
| aioli_objective | Faithful Aioli learning-law probing/controller, with objective-level validation and costs explicitly adapted | Mandatory adapted algorithm baseline |
| validation_progress | Proposed cost-aware validation-progress controller, below | Proposed method |
| chord_objective | CHORD-style decaying SFT assistance to on-policy RL, with declared fixed PT/DPO replay | Related-method adaptation; disclose exact differences |
| rpt_inspired | Reasoning/continuation prediction reward on training text; explicit simplified rollout/reward definition | RPT-inspired adaptation, not 14B distilled RPT reproduction |

This is 24 training cells per experiment, 48 total. Fix all cells before measuring. Retain all failed, missing, and interrupted cells in the denominator. Do not relabel an approximation as a reproduction. If an algorithm cannot be implemented faithfully enough to its stated adaptation, keep its cell pending/blocked and report incomplete baseline coverage.

Korbak et al. (2023), *Pretraining Language Models with Human Preferences*, is direct motivation for incorporating preference information during pretraining: their conditional training learns preference-conditioned behavior earlier and improves preference satisfaction in their studied setting. It does not establish that joint DPO and reasoning RL from random initialization wins. A faithful conditional-pretraining baseline is a follow-up after this narrower objective-scheduling pilot; account for that coverage limitation in every broad comparison.

## Proposed algorithm

Maintain objective-selection probabilities on a simplex with a .02 floor for every objective. Initialize at .55/.20/.10/.15. Each controller window estimates objective-specific utility using held-out **validation** measurements before and after a short training probe. Use relative changes normalized by the fixed initial value of each validation loss, never compare raw PT, DPO, and RL loss magnitudes. Average the standardized loss reductions, divide by measured/proxy charged compute for the probe (expressed in millions of forward-equivalent token operations), and apply an exponentiated-gradient update with a fixed learning rate. Include probe updates in the model trajectory and budget; do not roll back only a proposed method's expensive trials for free.

Use bounded utility clipping, finite-value guards, and fixed exploration. The first window probes every objective in a seeded order. Record validation loss components, chosen actions, probabilities, charged work, zero-variance rewards, and clipping. If general-text validation loss worsens by more than 2% relative to its fixed initial reference, enforce at least .55 PT probability and disclose this agent-proposed retention guard. For random initialization this guard is weak; also report degradation relative to best-so-far as a diagnostic, without hiding a second tuning rule. Compare with the same validation-evaluation allowance for Aioli and the proposed scheduler; charge unused allowance honestly for static baselines only if compute is actually performed.

Define standardized components and their direction in the frozen implementation: general-text negative log likelihood, response negative log likelihood, held-out preference pair negative log likelihood under DPO, and negative log likelihood of correct mathematical responses. Report that the last component is a smooth validation proxy, not benchmark accuracy and not an independently validated predictor of reasoning success. Direct exact-match evaluation remains required.

## Data and evaluation

Select public, pinned dataset revisions, with licenses recorded and raw data excluded from Git. Proposed practical sources: WikiText-103 for natural-language PT; GSM8K (Grade School Math 8K) training examples for SFT and RL; a bounded HuggingFaceH4/ultrafeedback_binarized preference pool for DPO. UltraFeedback preferences are model-generated feedback, not human annotations; this pilot does not establish human alignment. Filter overlength examples using a fixed rule that retains complete chosen/rejected responses rather than truncating away their labeled distinction. Use a small deterministic arithmetic training task only as an explicitly separate diagnostic if natural-language rewards are all zero; never present its accuracy as a public-benchmark result.

Split original training pools into train and validation by stable example hash before any tuning. Exclude all official benchmark test examples from training, replay, mixture estimation, and checkpoint selection; audit normalized prompt overlaps and record removals. Preserve official dataset splits when available. Deduplicate across SFT/RL/validation membership while allowing declared SFT/RL reuse of the same training pool. Pin tokenizer/model/dataset revisions and prepared-data hashes. Record pretraining contamination as unknown for inherited Pythia checkpoints; this experiment cannot prove absence of contamination.

Primary terminal reporting: held-out WikiText test negative log likelihood and perplexity; full official ARC-Easy (AI2 Reasoning Challenge easy split) multiple-choice accuracy; full GSM8K test exact-match accuracy under fixed greedy decoding. Multiple-choice scoring must use continuation likelihood without putting the answer key into the prompt. Report raw and length-normalized likelihood accuracies with one predeclared primary choice. A limited validation subset is permitted for frequent diagnostics; never silently substitute it for full terminal evaluation. Cap response generation at a prespecified token count and report truncation frequency. Small checkpoints may remain at chance or zero; report floors explicitly.

Evaluate at initialization, fixed budget milestones for retention and learning curves, and the final budget. Milestone monitoring uses validation only. Test evaluation occurs only after training choices and checkpoint selection have frozen; use the final checkpoint, not the best test checkpoint. Untouched checkpoint diagnostics must not choose a checkpoint age using test accuracy. Plot accuracy and language loss against inclusive cost, with all methods and seeds present.

## Compute fairness and bounded execution

Use one otherwise available device, set `CUDA_VISIBLE_DEVICES` explicitly, and verify memory/utilization after launch. Initial planning allowance: at most 48 device-hours on one SNAP A100-class device for the two full pilot matrices, preparation, and terminal evaluations. Exact per-cell budgets must be calibrated from a non-measured smoke run before manifest freeze; default target is 25 million forward-equivalent token operations per cell, subject to this complete-matrix preflight. Reducing all cells prospectively before freeze is allowed if the measured throughput cannot fit; shortening a running cell after inspecting its result is not.

Define charged work explicitly: model forward tokens, backward tokens (roughly twice forward cost), reference-model forward tokens, autoregressive generation including repeated prefix processing if caching is disabled, validation/controller probes, and terminal evaluation. Log parameter count and both modeled operations and elapsed device time. Equal optimizer steps are not equal compute. Token-operation accounting is a proxy, not hardware FLOPs; report its limitations and actual device-hours. Primary training budget includes controller overhead; terminal evaluation has a common separately reported reservation. Prefer cached generation, and do not ignore reference or rollout costs.

Training has a fixed token-operation admission rule with bounded final-step overshoot (at most one update); report actual overshoot. Every cell gets a fixed maximum wall-clock infrastructure bound inferred from smoke throughput plus margin. Preflight the sum of full cell limits, setup, evaluation, and finalization. Run sequentially if one device suffices. A hard global safety bound creates explicit interrupted/unattempted rows, never an assertion of completed evaluation. No result-dependent retries or extra seeds; planned crash resume must restore model, optimizer, scheduler, random state, data cursor, ledger, and cumulative cost. Otherwise label a restart as a separate prospective condition.

## Statistical and scientific decisions

The pilot's goal is estimation and falsification, not a powered claim of beating all previous methods. Report each seed and aggregate, 95% uncertainty intervals with the resampling unit, paired differences versus every baseline, and missing counts. Three seeds provide weak training-variance inference: keep seed-level uncertainty separate from paired item-level evaluation uncertainty, and never treat thousands of benchmark items as independent training runs. Use a paired bootstrap over common test items conditional on each trained model; report a seed-level summary separately. For formal multi-baseline superiority, predeclare a joint acceptance test and multiplicity correction in a later powered confirmation; no unsupported universal “wins” here.

Continue toward larger models only if validation measurements show a nondegenerate reward/accuracy signal and an informative retention–reasoning tradeoff. All-zero rewards, chance accuracy, unstable preference loss, or better language loss with worse reasoning are informative failure modes. Log both favorable and unfavorable evidence. Test results cannot change the frozen pilot or retrospectively define its primary metric.

## Review and launch gates

The user explicitly requested quality assurance by Claude Code Opus 5.5 at maximum effort. Resolve and record the actual model and effort from client evidence; try the requested exact model through subscription authentication, never silently substitute a different family/version. A missing required reviewer leaves acceptance pending. Deterministic implementation and data preparation may continue independently, but do not claim the required review passed or publish protected scientific changes as accepted without it.

Review the plan for objective definitions, reference policy choice, compute accounting, data leakage, sparse reward, statistical power, faithful/adapted baseline naming, and durable completion. Address findings, test the implementation, freeze manifests, then execute through a verified full-access `codexd` command-line worker in a named persistent terminal session. Keep the training supervisor independent of model-session lifetime. Verify process identity and first useful output, not just a launch command. Keep private machine/authentication receipts outside this public repository.

Deliverables: this plan and 2026 related work; runnable algorithm/data/evaluation code; tests; pinned configurations and full manifests; both experiment folders with live results and resumable checkpoints; reviewer findings and their resolution; training and evaluation receipts; honest results; reviewed, secret-scanned changes pushed to `main`. A launched worker is an ongoing experiment, not a completed result.
