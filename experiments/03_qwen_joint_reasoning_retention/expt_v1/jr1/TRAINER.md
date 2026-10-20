# JR1 trainer (`jr1/train.py`): methods, equations and accounting

**Status:** implementation for development and calibration. Every number in `configs/qwen15b_v0.json` marked
as a development default (learning rate, stage boundaries, mixture weights, batch sizes, work target, controller
window) is replaced by development selection before PROTOCOL v1 is frozen. No measured cell has run.

```bash
cd experiments/03_qwen_joint_reasoning_retention/expt_v1
export JR1_STATE_ROOT=/scratch/users/brando9/ut_jr1          # data under $JR1_STATE_ROOT/data (jr1/data.py)
python -m jr1.train --method jr1 --seed 1101 --config PROTOCOL_v1.json --out $JR1_STATE_ROOT/cells/03/jr1_1101 --resume
python -m jr1.train --method jr1 --seed 9001 --config PROTOCOL_v1.json --out .../dev/jr1_9001 \
    --record-schedule .../dev/jr1_9001/schedule.jsonl                                   # Exp 04 source
python -m jr1.train --replay-schedule .../dev/jr1_9001/schedule.jsonl --order stage_sorted --seed 1101 \
    --config PROTOCOL_v1.json --out $JR1_STATE_ROOT/cells/04/stage_sorted_1101 --resume
python -m unittest jr1.test_train -v                                                     # CPU, tiny Qwen2
```

`--resume` is safe on every launch: with no saved state the cell starts fresh; with state it continues.
Without `--resume` an existing `state/` is refused, so a cell is never silently overwritten.

## Outputs of one cell

| Path | Contents |
|---|---|
| `ckpt_000/` … `ckpt_100/` | HF `save_pretrained` weights + tokenizer + `jr1_ckpt.json` (milestone, work, updates, exposure, wall time). `ckpt_000` is the untouched start and the frozen DPO reference. A checkpoint is written at the first action boundary at or after 25/50/75% of the work target, and at the end. |
| `state/` | `policy.pt` (master weights), `optimizer.pt`, `rng.pt` (Python, NumPy, torch, CUDA, GradScaler), `trainer.json` (Work ledger, counters, exposure, data-stream positions, interleaver credit, controller state, an in-progress probe sweep, log line counts, wall time). Written to a temporary directory and swapped in with renames; `state.old` is used if a crash hit between the renames. |
| `log.jsonl` | One row per action: objective, purpose (train, probe) or validation, example ids, loss, optimizer step taken or not, grad norm, lr, reward statistics, the action's charged work, cumulative work by purpose and the cumulative realized exposure per objective (updates, optimizer steps, examples, tokens, target tokens, RL rollouts, informative and zero-variance groups). On resume the log is cut back to the saved line count, so no update is logged twice. |
| `receipt.json` | Status (running, interrupted, complete, failed, debug_stopped), work total and overshoot, actions, updates, exposure, stream positions (unique examples seen), controller snapshot, wall time across resumes, peak device memory, versions, precision. |

## Objectives (one shared AdamW, full weights)

Each optimizer step uses one microbatch of one objective (microbatch interleaving). With policy `π_θ`,
frozen initial reference `π_0` (`ckpt_000`), prompt `x`, response `y`:

- **PT**: `L = −Σ_t log π_θ(w_t | w_<t) / #targets` over WikiText text (shifted causal cross entropy; padding masked).
- **SFT**: the same over response tokens only; prompt and padding masked; EOS appended to the response.
- **DPO**: `L = −log σ(β[(log π_θ(y_w|x) − log π_θ(y_l|x)) − (log π_0(y_w|x) − log π_0(y_l|x))])` with
  sequence-summed response log probabilities, β = 0.1. The reference is never refreshed.
- **RL**: for each GSM8K prompt draw G unassisted samples (temperature 1, top-k 0, top-p 1, prompt only, no gold
  prefix), reward `r = 1[parsed final answer == released label]` (`common.exact_reward`, the readiness parser),
  leave-one-out advantage `A_i = r_i − mean_{j≠i} r_j`, loss `L = −Σ_i A_i log π_θ(y_i|x_i) / (B·G)`.
  Groups with zero reward variance have `A = 0`; their sequences are not scored. If every group in the batch has
  zero variance there is no backward pass and no optimizer step (so no weight decay and no moment update); the
  rollouts are still charged and logged as a zero-gradient update.

Optimizer: AdamW, gradient clipping, the same lr clock for all methods: `lr(f)` with linear warmup from
`warmup_start_ratio·lr` over `warmup_fraction` of the work target, then cosine to `min_lr_ratio·lr`, where
`f` = charged work / work target. Precision: fp32 master weights and Adam moments with bf16 autocast (pure bf16
AdamW would lose updates of size 1e-6 relative to weights of size 1e-2, below bf16 resolution); `master: bf16` is
available, `compute: fp16` uses a GradScaler (V100), `compute: fp32` is used by the CPU tests. Checkpoints are
saved in bf16 for Qwen (its released weights are bf16, so `ckpt_000` is exact); use `checkpoint: fp32` for a
random fp32 initialization (Exp 05) so the reference stays exact.

## Work accounting

Forward-equivalent tokens, frozen multipliers in `common.MULTIPLIERS`: forward 1, backward 2, reference forward 1,
generation 1 per processed token (prompt prefill + one token per row per decoding step, padding and finished rows
included). Charged per purpose: `train`, `probe` (controller probe updates, which are real training updates) and
`validation` (controller validation passes and the one-time reference pass on controller preference rows).
Optimizer steps are counted separately; device and wall time are reported separately; token counts are estimates,
not measured FLOPs.

The run stops at the first action boundary where total charged work ≥ target. One action is one update, or one
validation pass, or (at the end of a probe mixture) one update plus its validation. Overshoot is therefore at most
the largest single action, and `receipt.json` records `work_overshoot`, `max_action_charge` and
`overshoot_bounded`.

## Methods

Weights `p ∈ Δ⁴` over (PT, SFT, DPO, RL) are shares of optimizer-step actions. A deterministic deficit interleaver
turns them into actions: each action `c_i += p_i` for active objectives (inactive ones hold no credit), take
`argmax c`, subtract one. Realized update counts stay within one of `Σ_t p_i(t)`. Seeds change data order, RL
samples, probe orders (and the initialization for random models), not the interleaver.

| Method | `p(f)` |
|---|---|
| `sequential` | one-hot stage: PT on `[0, b1)`, SFT on `[b1, b2)`, DPO on `[b2, b3)`, RL on `[b3, 1]` (default b = .55/.75/.85, from 00_program; dev-selected) |
| `replay` | same stages; in every stage after PT, `p_PT = ρ` (corpus replay, default 0.2) and the stage objective `1 − ρ` |
| `fixed_joint` | constant weights, all positive (default .55/.20/.10/.15) |
| `smooth_joint` | `Proj_floors((1−f)·p_start + f·p_end)`, PT and DPO floors > 0 (default start .70/.20/.05/.05, end .20/.20/.15/.45, floors .05) |
| `aioli` | Aioli transfer-matrix weights `a_t` (below), initialized uniform, no floor by default |
| `jr1` | `Proj_floors(q)`, `q_i ∝ a_i^(1−λ) · π_i(f)^λ`, prior `π(f) = (1−f)π_start + f π_end`, floors below |

`Proj_floors` is the KL projection onto the simplex with lower bounds, `p_i = max(floor_i, c·w_i)`.

**Aioli sweep.** Before production updates and then every `window` production updates: draw a random order of
the K = 4 probe mixtures; mixture j is `probe_updates` real updates with α of them objective j and
β = (1−α)/(K−1) each of the others (α = .75 with 12 updates gives 9/1/1/1), in a seeded shuffle. Validation
components `v = (PT NLL, SFT NLL, DPO loss, smoothed greedy RL failure rate)` on the controller split are measured
before and after each mixture (the after of one mixture is the before of the next; nothing trains between them).
`Δ[i][j] = (v_i^before − v_i^after) / v_i(0)`. Then `A_i = (Δ_i − β·ΣΔ_i·1)/(α − β)` (solving `W A_i = Δ_i`),
`Â = normalize(A)` (shift a negative global minimum to 0, divide by the global sum), influence `g_j = Σ_i Â_ij`,
`a ← Proj_floor(a ⊙ exp(η g))`, η = 1.

**J-R1** uses the same sweeps and `a_t`, then mixes in the prior geometrically with strength λ (default 0.5,
π_start = .45/.35/.10/.10, π_end = .15/.15/.10/.60). Floors: `ε` = .02 for every objective, PT `ρ_PT` = .05,
DPO `ρ_DPO` = .05. **Retention constraint (validation only):** with `L` the PT NLL on the WikiText controller
split measured in the latest sweep, if `L > (1 + τ) L(0)` (τ = 0.02) the PT floor becomes `m_PT` = .55 until a
later sweep measures `L ≤ (1 + τ) L(0)`. Feedback normalization: each component divided by its initial value.
No rollback, no retrospective checkpoint choice, no test data; all probe and validation work is charged.

### Aioli differences table

| Aioli (Chen et al. 2024, HazyResearch/aioli `learn_params_subroutine`) | This adaptation |
|---|---|
| Mixes data domains of one cross-entropy objective | Mixes four heterogeneous objectives (PT, SFT, DPO, RL) |
| Validation loss per domain, one component per domain | Four validation components with different units (NLL, DPO loss, greedy failure rate), each divided by its initial value so they share a scale |
| Probe mixtures are data proportions inside batches | Probe mixtures are proportions of whole-objective optimizer steps (9/1/1/1 of 12) |
| Uniform update cost per step | Objectives have very different cost (RL generation); weights stay update shares, as in Aioli's W, and costs are charged in the Work ledger, not folded into the matrix |
| Probe and validation compute often reported separately | Every probe update and validation token counts against the same fixed work target as training |
| Same `recover`, global normalize, non-EMA, full-matrix exponentiated update | Kept unchanged (attributed copy of `00_program/aioli.py`; a test checks equality with the frozen module) |
| No probability floor | Floor off by default (`floor: 0`); J-R1 adds explicit floors and the prior |

## Exp 04 schedule replay

`--record-schedule` writes one row per training or probe update of the source run (dev seed 9001): update index,
objective, example ids, purpose, progress and lr; `<path>.meta.json` holds counts per objective, total work and the
file's sha256. Controller validation passes are not training actions and are not replayed.
`--replay-schedule … --order interleaved` replays the rows in recorded order; `--order stage_sorted` stably
sorts them by objective stage (PT, SFT, DPO, RL). Both orders train on exactly the same examples per objective;
RL regenerates fresh rollouts for the same prompts with the replaying policy, so generated tokens and rewards
differ (reported, not equalized). Both orders use the same positional lr clock: action k uses the lr the source
used at its k-th update. Checkpoints in replay are at 25/50/75/100% of scheduled actions, with work recorded.

## Exp 05 (random-initialization Pythia) needs no fork

`model.source: random` builds `from_config` with `init_seed` (defaults to the cell seed), `precision.compute: fp16`
with a GradScaler for V100s, `checkpoint: fp32`, and `objectives.rl.backend` (`hf`; another backend raises).

## Tests (`jr1/test_train.py`, CPU, tiny random Qwen2, offline byte tokenizer)

Masking (prompt, padding, EOS kept, long-prompt truncation), PT loss equals shifted cross entropy, DPO sign and
that DPO steps raise the chosen-minus-rejected margin from exactly 0 at the reference, RL nonzero gradient with a
mixed reward, zero-variance RL leaves weights and optimizer steps bit-identical (with weight decay on) while still
charging generation, leave-one-out values, every method's schedule, interleaver tracking and stage switches,
Aioli weights equal to the frozen `00_program` AioliController, J-R1 prior and retention constraint, per-objective
work charges including generation prefill + decoding, bounded overshoot and charged validation, checkpoints at
every milestone with the reference equal to the initial weights, resume across two interruptions (one inside a
probe sweep, one just past the first controller update) reproducing the uninterrupted log and weights bit-exactly,
identity checks on resume, and replay orders matching the schedule's examples per objective.
