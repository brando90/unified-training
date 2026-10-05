# Joint reasoning and capability retention: prospective campaign

**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/03_qwen_joint_reasoning_retention/PLAN.md>

**Summary:** Test whether mixing pretraining, supervised fine-tuning, direct preference optimization and reasoning reinforcement learning improves reasoning at a matched resource budget without degrading general capabilities. Separate a capable pretrained model, a controlled exposure comparison, and genuinely random initialization. Positive results are not assumed.

Created: 10-05-2026 10:21 PDT
Status: PREPARATION — numerical budgets and immutable inputs must pass calibration before measurement.

## What the existing evidence does and does not establish

The completed `experiments/01_scratch_joint_objectives` and `experiments/02_early_checkpoint_joint_objectives` contain 48 training runs and four untouched evaluations. Preserve every frozen source, manifest, score and original finding. Their [final report](../00_program/FINAL_REPORT.md) records sparse math reward and declining early-checkpoint science accuracy. Repeating their tiny-budget sweep cannot establish the requested reasoning and retention claim.

The central uncertainty is whether joint optimization works in a regime with useful on-policy reasoning rewards and measurable general capability. A pretrained study addresses that uncertainty; it cannot prove training from random initialization works. The scratch study retains that distinction explicitly.

The requested staged control is pretraining (PT) → supervised fine-tuning (SFT) → direct preference optimization (DPO) → reasoning reinforcement learning (RL). It is not a reproduction of DeepSeek-R1, whose published pipeline starts from a pretrained base and includes cold-start supervision, reasoning RL, rejection-sampling supervision and further RL ([primary source](https://arxiv.org/abs/2501.12948)).

## Experiment matrix and priorities

| Canonical experiment | Setup and question | Prospective matrix | Priority and reason |
|---|---|---|---|
| `experiments/03_qwen_joint_reasoning_retention` | Immutable Qwen2.5-1.5B base; can joint continuation improve reasoning and preserve capabilities? | Six methods × six paired seeds = 36 runs, plus one untouched base | [p 10/10] Resolves the reward-floor and retention gaps directly |
| `experiments/04_qwen_matched_exposure_schedules` | Same base; match objective-specific exposure to isolate scheduling | Sequential versus interleaved replay of one development-frozen joint schedule × three seeds = 6 runs | [p 9/10] Separates interleaving from receiving more useful training data |
| `experiments/05_pythia_scratch_joint_learning` | Random Pythia-160m architecture; substantial corpus learning from initialization | Sequential, fixed joint, smooth joint × three seeds = 9 runs | [p 7/10] Tests the literal initialization claim, with likely reasoning-floor limitations |

Total prospective denominator: **51 training runs**, with every failed, gated, interrupted or unstarted cell retained. This is a design target, not a launched or completed count. Six seeds allow a two-sided exact paired sign-flip test to attain below 0.05; they do not guarantee adequate power. The smaller mechanistic comparisons are descriptive.

Methods for `experiments/03_qwen_joint_reasoning_retention`:

1. **Tuned sequential:** the four stages above; learning rate and stage allocation chosen on development data under a fixed allowance.
2. **Sequential with corpus replay:** same stages plus continuing PT replay during later stages; protects against beating an unnecessarily forgetful baseline.
3. **Fixed joint:** all four components remain active with fixed tuned weights.
4. **Smooth joint:** PT/SFT emphasis gradually shifts toward RL, with positive PT and preference floors.
5. **Aioli objective adaptation:** charged on-trajectory randomized probes recover cross-objective learning effects; use the actual published matrix mechanism, with heterogeneous-objective differences disclosed.
6. **J-R1 retention-constrained adaptive mixture:** the same adaptive foundation plus a gradual PT/SFT-to-RL prior and a validation-only retention constraint. Record exact equations, floor, feedback normalization and constraint response before freeze. No free rollback or retrospective checkpoint choice.

The last row is the primary candidate. Aioli and fixed joint are required controls for any new-controller claim. The previous cost-normalized method is not silently asserted to be superior. This program prioritizes a defensible six-method comparison over collecting many poorly calibrated adaptations.

## Data, readiness and development gates

Use only public, locally trained models and original released datasets; no model-provider inference calls. Resolve model/tokenizer/data revisions and licenses before download; record exact hashes, filtering, source rows and splits. Candidate base is [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B), a pretrained model with 1.54 billion parameters. Its model card does not establish readiness for our protocol.

Reuse the existing WikiText, Grade School Math 8K (GSM8K) and UltraFeedback provenance where suitable, preparing separate tokenizations and split manifests in this experiment. SFT and RL prompt pools must be disjoint. Keep distinct training, tuning, controller, readiness and sealed test sets. Audit exact and normalized text overlap, including benchmark prompts embedded in preference data. Report remaining inherited-pretraining contamination as unknown.

Before selecting a matrix budget, evaluate **128 development prompts × eight sampled completions** with the final-format prompt and answer parser. Begin at 1,024 generated tokens. Admission requires at least 20 mixed-success prompt groups, positive and negative rewards, and at most 10% truncated completions. These are agent-proposed engineering criteria for informative gradients, not guarantees from a paper. Record actual reward, mixed-group and truncation counts.

If truncation fails, calibrate 2,048 tokens on the same development-only procedure, charge both attempts, then freeze the chosen limit. If useful reward still fails, permit one common bounded SFT preparation selected only from development data and label every resulting comparison **joint continuation after common supervision**. Preserve the untouched base, preparation costs and changed scope. If the gate remains unmet, retain a documented failed readiness result; do not launch 36 nominal reasoning runs at a known reward floor.

Give each of the six methods equal development device-time allowance, within six total device-hours. Use a fixed, recorded candidate grid for learning rates and mixture/stage allocations and the same selection score/retention requirements. Expensive controllers do not receive free probes or additional search. Freeze chosen settings before opening test outputs. Readiness/resource work has another four device-hours; no hidden cumulative resets.

## Objectives, optimizer and implementation contract

- PT uses correctly shifted causal cross entropy; SFT masks prompts and padding.
- DPO uses sequence-summed preferred/rejected response log probabilities, with an explicitly frozen initial reference for all schedule comparisons. A post-SFT reference version, if needed, is a separately labeled complete-recipe control, not a hidden baseline handicap.
- RL generates on-policy unassisted responses, applies original exact-answer labels through a deterministic verified parser, masks prompts, and uses group-relative or leave-one-out advantages with the estimator named. Zero-variance groups do not cause spurious optimizer or weight-decay updates. No gold-answer prefixes or rewards for formatting alone.
- Use one shared optimizer consistently for the six main methods; preserve identical optimizer state conventions, clipping, precision and learning-rate clock. This avoids the previous harness's multiple full-size optimizer states becoming a memory confound. Train full weights where calibration fits; do not silently switch to adapters.
- Joint means all four losses contribute throughout the trajectory, through explicitly documented microbatch interleaving or weighted accumulation. Positive scheduler probability is not evidence an objective was actually used: log realized updates, informative RL gradients and per-objective exposure from the start.
- Charge controller probes, reference passes, generated and rescored tokens, discarded samples, validation and optimizer work. Report estimated forward-equivalent token operations and measured device/wall time separately. Do not claim token estimates equal physical floating-point operations.
- Save atomic restartable state: policy, optimizer, scheduler/controller, reference, random generators, stream positions and cumulative budget. Preserve failed attempts; no selection by best seed or extra run after a poor score.

[Aioli](https://arxiv.org/abs/2411.05735) studies data mixing; our heterogeneous-objective use is an adaptation. Keep a differences table and preserve its actual transfer-matrix calculation. References and gold labels are imported unchanged; do not create or relabel benchmark answers.

## Matched resources and admission

New campaign ceiling: **96 device-hours**, separate from the completed pilot. Proposed allocation is 60 hours to `experiments/03_qwen_joint_reasoning_retention`, 10 to `experiments/04_qwen_matched_exposure_schedules`, 14 to `experiments/05_pythia_scratch_joint_learning`, 10 to development/readiness/calibration, and 2 to finalization. Include failed preparation, initialization, evaluation, monitoring occupancy and checkpoints. No new paid services or credits.

Start with one available A100 80 GB device on SNAP. Analytical peak-memory planning range for full 1.5B training with one optimizer, reference and checkpointed activations is approximately 30–60 GB; measure it. Training should be compute-intensive; autoregressive generation and loading may be less utilized. No device reservation while writing code or downloading. Set `CUDA_VISIBLE_DEVICES` explicitly for every launch; never touch another owner's processes.

Time one real update for each objective, one adaptive sweep, checkpoint save/load, batch evaluation, and both model setups. The primary training comparison uses an identical frozen forward-equivalent work target with bounded single-update overshoot, plus actual device-time efficiency curves. Include controller costs in that target. Also report raw examples/tokens and per-objective exposure: efficiency per training sample and efficiency per compute are different claims.

Before admission, prove the sum of every per-cell timeout, full evaluation, retained preparation, publication and cleanup reserve fits 96 hours. Choose one meaningful target before measurement; no outcome-driven early stopping. If the 51-run design cannot fit, prospectively reduce optional breadth first (the separate exposure study, then scratch replication) while retaining all six main methods and six seeds, or explicitly record the unadmitted cells. Do not shrink test denominators or disguise unstarted cells as completed. Freeze the final matrix and report the change before measured updates.

For the scratch study aim for corpus exposure orders of magnitude beyond the old tiny pilot; calculate feasible tokens from throughput, with a target of at least 50 million observed PT tokens per seed when possible. Match total charged work, track differing objective exposure, and report inability to reach that target. Scratch math at a floor is a mechanistic language-learning result, not a demonstration of AIME-level reasoning.

For the exposure study freeze one action/exposure schedule from development seed 9001 before test data are opened. Replay exactly its objective microbatches in interleaved and stage-sorted order, matched by objective examples and tokenization. Sampled RL response lengths and rewards necessarily differ with policy; report these differences and actual compute, and do not call generated trajectories identical. Compare at matched scheduled RL prompts/rollout count, not an impossible exact equality of generated token strings.

## Evaluation, sample efficiency and retention

Primary reasoning endpoint: complete GSM8K test, **1,319 questions**, deterministic prompt/decoding/parser and fixed token cap. Transfer: all **500 MATH-500 questions**, with original labels and a validated mathematical-equivalence parser. American Invitational Mathematics Examination (AIME) is an optional descriptive small-set endpoint if resources permit; it is not a primary endpoint for these model sizes.

Primary general-capability retention: complete Massive Multitask Language Understanding (MMLU) test, with every original subject and item, fixed published multiple-choice likelihood protocol. Record the actual release count; never silently cap it. Secondary retention: full AI2 Reasoning Challenge Easy test (**2,376 questions**) and a frozen language-modeling test corpus. Report held-out preference ordering separately; it is not proof of alignment or safety.

Evaluate the untouched start and terminal checkpoints. At 0%, 25%, 50%, 75% and 100% charged training work, retain checkpoints and evaluate a fixed sealed learning-curve subset (256 GSM8K and 512 subject-stratified MMLU questions) only after settings are frozen. Exclude that subset from controller and development feedback. Save actual work/time coordinates; charge all evaluations. No test feedback into mixtures or checkpoint choice. Stage-boundary diagnostics quantify forgetting relative to both initialization and earlier peak general capability.

Sample efficiency estimands: area under the reasoning-versus-charged-work curve; achieved reasoning at fixed charged budgets; and work/time to a development-prespecified accuracy target (report non-attainment, never extrapolate). Count processed and unique examples by objective and plot token-based curves alongside compute-based curves. Compare general capability at matched reasoning as a descriptive trade-off, never select favorable points after seeing test scores.

Primary candidate-versus-sequential contrast: paired seed difference in terminal GSM8K accuracy. Report all six seeds, mean, standard deviation and a 95% paired Student t interval (weak normality assumption with six seeds), plus exact paired sign-flip p-value under the zero-effect null and exchangeability assumption. Conditional paired item bootstraps are secondary and cannot replace training-seed uncertainty. Other comparisons are descriptive unless a frozen multiplicity correction is specified.

Agent-suggested practical success criteria, fixed prospectively: point gain at least **2 percentage points** in GSM8K and a 95% interval above zero; MMLU and ARC-Easy candidate-minus-initial retention intervals must each exclude losses larger than **1 percentage point**; corpus negative log likelihood must not rise more than **2%** from initialization. Evaluate reasoning improvement and retention jointly. These tolerances are planning choices, not universal accepted thresholds. Failure or wide intervals means negative or inconclusive evidence, not confirmation.

"Avoided alignment tax" requires retained general capability with demonstrated preference/instruction gains; preference likelihood alone warrants only a narrower claim. "Avoided catastrophic forgetting" requires the longitudinal capability measurements. A better final score than sequential alone proves neither.

## Reproducibility, ownership and completion

The worker initializes missing package/dependency instructions and a reproducible environment; reuses tested utilities without altering frozen old artifacts; creates new experiment-owned code, manifests and tests; and records exact source/environment revisions. Deterministic tests must cover masking, preference signs, mathematical parsing, nonzero and zero RL gradients, scheduler behavior, objective-budget accounting, split integrity, resume behavior and completion denominators.

One named SNAP implementation owner publishes source/protocol and then runs a separate deterministic supervisor that survives that agent exiting. A non-model watcher verifies child process identity and fresh progress, preserves healthy training, reports blocked phases, and finalizes only complete artifact sets. Remote state is backed by Git-published live `results.md`; a successful launch is not completion. Bounded recovery never changes measured scientific inputs or resets budgets.

Each experiment keeps a versioned frozen `PROTOCOL.md`, complete cell manifest, per-item results, restart state, `results.md`, and timestamped continuation checkpoint. Raw data and large weights stay in ignored node-local storage, with retrievable manifests. Publish meaningful milestones and final report through verified commits and pull requests. No email or collaborator messages are authorized by this dispatch; use repository receipts and the existing chat.

**TLDR-end:** [unified-training: prospective joint-training campaign] `experiments/03_qwen_joint_reasoning_retention`, `experiments/04_qwen_matched_exposure_schedules` and `experiments/05_pythia_scratch_joint_learning` target 51 matched runs within 96 device-hours. Readiness, full evaluation and resource admission precede measurement; the original hypothesis remains unresolved.

**Snapshot:**
```text
Main comparison: 6 methods × 6 seeds = 36
Exposure control: 2 schedules × 3 seeds = 6
Random initialization: 3 schedules × 3 seeds = 9
Total prospective training cells: 51
Device-time ceiling including preparation/evaluation: 96 hours
```
