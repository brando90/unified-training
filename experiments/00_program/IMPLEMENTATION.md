# Implemented pilot and equations

Updated 10-04-2026. This is the current prospective implementation contract;
acceptance and scientific freeze have separate machine-readable gates. All model
training/scoring is local PyTorch/Transformers work; no model-provider API calls.
Raw data, model weights, logs and private runtime receipts remain outside Git.

## Losses and optimizer state

Pretraining (PT) uses mean token negative log likelihood. Supervised fine-tuning
(SFT) uses response-token mean negative log likelihood, masking prompt and padding,
including end-of-sequence. Both shift labels by one. WikiText is packed into
512-token blocks. Math prompts are `Question: ...\nAnswer:`; responses have a
leading space. Complete preference pairs must fit 512 tokens; neither response
is truncated independently. Tests compare unequal-response-length microbatch
weighted gradients with the full-batch gradient. No offload or accumulation is
used in the actual trainer, avoiding the two bugs identified by Limozin et al.

The preference objective is a **length-normalized DPO adaptation**, not standard
sequence-summed Direct Preference Optimization (DPO). Let `m_pi(y|x)` be the mean
response-token log probability. The loss is
`-mean log sigmoid(beta * [(m_pi(chosen)-m_pi(rejected)) -
(m_ref(chosen)-m_ref(rejected))])`, with beta=1. The stop-gradient reference is
copied from the policy at charged-budget fractions .1,.2,.3,.4,.5,.6,.7,.75,.8,.9,
on the next admitted update after a crossing, identically for every method.
The .75 boundary captures the staged post-SFT policy before preference updates.
Between crossings it is fixed. Hard reference updates are established TR-DPO
prior work, not a novelty claim. A uniform per-token language gain cancels even
when response lengths differ; a deterministic regression checks this. Semantic
or residual length preferences are not thereby ruled out. Controller preference
feedback uses reference-free logistic loss of the same length-normalized policy
margin. Terminal reporting includes both raw policy and reference-relative ordering.

Reinforcement learning (RL) uses sixteen prompts and four independent completions
per prompt, full-vocabulary sampling at temperature 1, cached generation and a
128-token cap. No expert prefix is selected for measured training. Reward is one
iff a fixed last-number extractor matches the answer (`####` takes precedence);
invalid/empty answers receive zero, and truncated responses are scored and counted.
The objective is
`-mean_i [(r_i - mean_{j != i, same prompt} r_j) * sum_t log pi(y_it|x_i,y_i<t)]`.
This leave-one-out baseline is independent of the sampled completion, conditional
on its prompt. No variance division, dense shaping, entropy bonus, old policy,
or Kullback–Leibler penalty is used. Dropout is disabled for generation and scoring.
If all advantages are zero, skip the entire optimizer update, momentum and weight
decay included; still charge generation and forward scoring. Independent AdamW
states exist for PT/SFT/DPO/RL/RPT/hybrid. Each uses learning rate 1e-4, default
betas (.9,.999), epsilon 1e-8, weight decay .01, and gradient norm clipping at 1.
No warmup or stage-specific learning-rate change is hidden. Float32 master weights
and optimizer moments coexist with bfloat16 forward autocast.

## Conditions

| Identifier | Mechanism | Scope relative to published work |
|---|---|---|
| sequential | PT/SFT/DPO/RL charged-work phases ending .55/.75/.85/1 | Generic staged pilot; fixed untuned allocations |
| parallel_joint | .55 PT, .10 DPO, .35 parallel opportunities; from the same parameter snapshot compute SFT and RL Adam updates separately, then average resulting parameters | RL Excursions-inspired, independent moments; both updates charged; replay and tiny model are adaptations |
| fixed_joint | PT/SFT/DPO/RL probabilities .55/.20/.10/.15 | Selection probabilities, not compute shares |
| smooth_joint | Linear probabilities .70/.20/.05/.05 to .20/.20/.15/.45 | Hand-designed curriculum |
| aioli_objective | Count-mixture learning-law probes, uniform initial mixture, full effect-matrix update | Objective adaptation; fixed initial loss normalization and .02 floors |
| validation_progress | Same Aioli configuration; recovered effect columns divided by per-action production cost | One explicit cost-aware difference |
| chord_objective | .55 PT, .10 DPO, .35 hybrid opportunities | CHORD-inspired combination, not a full CHORD-phi reproduction |
| rpt_inspired | Fixed mixture with text-prefix reward replacing the RL slot | Diagnostic transfer of RPT's mechanism to tiny models; natural-language public data retained |

Parallel updates implement `theta_next=theta0+.5*(delta_SFT+delta_RL)` with each
delta computed at theta0 and its own optimizer state. Tests compare the actual
update with two independent reference updates. CHORD uses
`(1-mu)*L_RL + mu*L_SFT_phi`, `mu=.9*(1-budget_fraction)`, and detached token
weight `phi=p*(1-p)` divided by the number of response tokens. It combines
CHORD's weighting ideas with a different estimator, replay and decay schedule; the reinforcement term sums token log probabilities while SFT averages tokens, so mu is not calibrated to the paper's loss scale;
it is not the paper's strongest fixed-mu CHORD-phi recipe. RPT asks for reasoning
then `Prediction:` from 128 corpus context tokens; a nonempty UTF-8 prediction
must exactly prefix the next 32 corpus tokens at a decoded token boundary. One formatting space is removed symmetrically from prediction and continuation, adjusting token boundaries.
No reasoning text independently earns reward. Zero reward is retained, not
quietly replaced by mathematical RL. The exact CHERRY-RL identity is unresolved.

## Aioli recovery and the cost-aware comparison

Let W's row j describe a 12-update probe: nine updates of objective j and one of
each other objective, shuffled. Thus diagonal W=.75 and off-diagonal W=1/12.
Four probes run in a shuffled order on the actual evolving trajectory; no rollback.
For validation component i and probe j,
`D[i,j]=(loss_before[i]-loss_after[i])/initial_loss[i]`.
The model is `D=A W^T`; recover `A=D inverse(W^T)`. **Never divide the response
columns by their unequal probe costs before applying this same count-based W.**
That operation does not describe the design matrix. The counterexample regression
uses effects [1,2,0,0], action costs [1,5,1,1] and probe costs [16,48,16,16]. Correct
recovery has error below 1e-12; the old division reverses SFT/PT preference.

Aioli feeds A into the verified global-minimum-shift/global-sum normalization,
then multiplies current p_j by `exp(sum_i normalized_A[i,j])`, followed by .02
floor projection. Initial p is uniform. The proposed method instead uses
`A_cost[i,j]=A[i,j]/c[j]`, where c[j] is mean measured production cost of action j
in millions of forward-equivalent tokens during that sweep. It unmixes first;
validation overhead never alters W or c, but is fully charged to the global ledger.
The two conditions share all other settings and fixed validation scales. Both
repeat after 64 production updates. Neither has a retention safeguard. Every
controller iteration, normalized matrix, probability, gradient clipping and
partial unfinished probe is logged. In-trajectory drift and small noisy feedback
mean these are local learning-law estimates, not causal-transfer guarantees.

Feedback components are PT token loss, SFT response loss, reference-free normalized
preference logistic loss, and **actual sampled mathematical error**. With k correct
among n=8 fixed controller questions, RL loss is `(n-k+.5)/(n+1)`; common random
numbers are restored after scoring. This avoids duplicating SFT loss as a reward
proxy, but is coarse and noisy; controller and development questions are disjoint.

## Cost, checkpointing and final evaluation

The ledger counts actual padded forward input tokens at 1, backward input tokens
at 2, reference passes at 1, and generation inputs via an executed-forward hook.
Cached decoding includes prefix and every subsequent input, including finished
padded rows that still execute. Initial/milestone/probe validation consumes the
training budget. Terminal evaluation has a separate common reservation. Report
millions of forward-equivalent tokens, estimated parameter operations
`2*parameter_count*charged_tokens`, and synchronized elapsed/device-seconds.
The proxy omits attention, optimizer and memory costs; equal proxy work is not
identical wall time. The full-program calibration includes setup, checkpoint I/O,
all 48 cells, four initial evaluations, all old preflight attempts and finalization.
All methods share the same scientific work cap. Infrastructure timeouts differ prospectively by method/model, using each measured schedule and the same 2x margin, and are fixed before admission. The slowest measured mean per-action rate plus amortized checkpoint cost gets a 2x margin; individual maxima and the earlier rejected forecast remain preserved. Multiple-choice evaluation batches six questions (up to 30 choices), with separate-question score equivalence tested. Checkpoints every 256 updates replace the costly provisional 32-update cadence before freeze.

Atomic checkpoints every 256 updates and stage boundaries retain model/reference,
all optimizer moments, all random states, streams/cursors, scheduler object,
unfinished probes, stage, work and time. Bit-for-bit optimization resume is tested on CPU, including reinforcement updates and mid-probe recovery. Graphics-processor replay restores all state but is not claimed bit-for-bit; deterministic kernels are not forced.
The engine library restores a quiescent checkpoint with exact identity matching; the production --recover path adopts live children and marks dead non-quiescent children interrupted, then continues pending cells. It does not automatically resume a lost trajectory. Lost
uncheckpointed infrastructure work is not regenerated under a reset budget.
The separate supervisor fences native process identity (PID/start tick/boot),
adopts healthy children, and verifies complete artifacts and denominators.

Terminal tests use all official ARC-Easy/GSM8K rows, 564 WikiText blocks and the
predeclared 256 complete preference pairs. ARC primary scoring averages full
choice-text token log probability; raw sums are secondary. GSM8K is greedy,
128-token maximum, fixed numeric parser. No failed/truncated item is dropped.
WikiText is in-domain modeling, not Pile-wide retention; first block tokens are
unscored and the incomplete corpus tail was excluded prospectively. Early Pythia
inherits approximately 20.97 billion Pile tokens; its optimizer is reset for every
condition. Inherited GSM8K-test contamination is historically unlikely because
the Pile predates GSM8K, but provenance alone cannot certify all contamination
absence; WikiText/Pile overlap remains unmeasured.

Wilson intervals describe binary conditional item uncertainty; paired item
bootstrap describes differences conditional on the two models. Three-seed t
intervals and two-stage seed/item bootstraps are exploratory, weak with n=3.
Missing rows remain in all denominators with accuracy sensitivity bounds.
No formal test is run (p-val=n/a), no tuned-frontier or universal-win claim is made.
See `related_work.md` for original algorithm sources and known omitted competitors.

Production reinforcement updates log mean/max absolute sampling-versus-scoring log-probability gaps. A real left-padded tiny-model sampling test agrees with right-padded scoring within 1e-5; bfloat16 and cached-kernel differences on the accelerator remain measured numerical approximations.
