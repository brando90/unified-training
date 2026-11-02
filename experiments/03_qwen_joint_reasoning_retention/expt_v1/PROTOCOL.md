# Pretrained joint reasoning and retention: prospective protocol v1

**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/03_qwen_joint_reasoning_retention/expt_v1/PROTOCOL.md>

**Summary:** Follow the [campaign plan](../PLAN.md). This is the frozen measured protocol v1 for Exp 03: six methods x six paired seeds on the untouched Qwen2.5-1.5B plus one untouched base evaluation.

Status: **FROZEN** 2026-10-31T18:41:16Z. Frozen before any measured update and before any test output was opened; the development selection used only the tune split. Never change this version after its first measured update; later scientific changes need a separate prospective version and do not erase failed evidence.

Planned cells: **36 training cells + 1 untouched base evaluation = 37 array tasks** (denominator never reduced; every failed, interrupted or unstarted cell stays counted). Exp 04 (6 cells) has its own protocol and shares this code and configs.

## Objective and contrast

Can joint continuation (PT, SFT, DPO and unassisted on-policy RL all active, charged to one matched work target) improve GSM8K reasoning of Qwen2.5-1.5B and preserve general capability better than tuned sequential post-training? Primary candidate: J-R1. Required controls: tuned sequential, sequential with corpus replay, fixed joint, smooth joint, Aioli.

## Cell manifest

Array index i of `cells03.sbatch`: seed `1101 + i // 6`, method `(sequential, replay, fixed_joint, smooth_joint, aioli, jr1)[i % 6]`; i = 36 is the untouched base evaluation. Seeds 1101 to 1106 (paired across methods); the dev seed 9001 is never a measured seed.

## Inputs and hashes (binding)

| Item | Value |
|---|---|
| Source repository | `brando90/unified-training`, experiment code under `experiments/03_qwen_joint_reasoning_retention/expt_v1/jr1/` |
| Code that ran on Sherlock | commit `22e2a426a4bd40fa83e825f0b0d498af9a300ba0` (`CODE_COMMIT` of the uploaded bundle; bundle `jr1_sherlock.tgz` sha256 `0be1294e69ce218d9d6249b213294ef5a51871d297bee204fc5968079e9de320`) |
| Code file hashes | train.py `a033f0603bd0ce15dceaf8a4fd3b59df5c37e9320965c6e9ee4332c446351d94`; evaluate.py `8efaeb79e31dc0e4d6537d8de805d306602adecdd89cecf6f72f82b664c66748`; common.py `6360338ee3b0380a65503d08412c64133d89138194844cbe8c403438ded2dc59`; controllers.py `640fc4a98896d52ffb8e1ef8f2fa7d4b881c3aaa3f9af27d7525fbb67151d516`; data.py `c8e8e2fa3a964c93ce448695db7fd3295047be3cf8fb57bfb2161d29d247723d` |
| Model | `Qwen/Qwen2.5-1.5B`, revision `8faed761d45a263340a0528343f099c05c9a4323`, `model.safetensors` sha256 `a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796`; tokenizer is the same snapshot (Sherlock `models/model_manifest.json` hub_check ok on 10 files, prewarm job 48735955) |
| Data manifest | `jr1/data_manifest.json` sha256 `01aa8a12e0a1f6f878dfc9a3fba20352408470d5bbbe132126f4e89a9ae03c46`; 26/26 prepared files matched it on Sherlock (`receipts/data_check.json`, prewarm 48735955) |
| Data sources (revisions in the manifest) | WikiText-103-raw (`Salesforce/wikitext` b08601e0), GSM8K (`openai/gsm8k` 740312ad), UltraFeedback binarized (`HuggingFaceH4/ultrafeedback_binarized` 3949bf5f), MATH-500 (`HuggingFaceH4/MATH-500` 6e4ed1a2), MMLU (`cais/mmlu` c30699e8), ARC-Easy (`allenai/ai2_arc` 210d026f) |
| Python environment | hashed lock `jr1/sherlock/requirements-sherlock.lock` sha256 `087167bb9f1755becb40b53f68c561688062757753b5aa174f2b93048ab6766e`: torch 2.6.0+cu124, transformers 4.57.6; per-job `env.json` receipts record the pip freeze, GPU and driver |
| Dev grid | `jr1/sherlock/dev_grid.json` sha256 `f6676b3c5b7311270057c5ae6b21f9fdb44d5248806e1604ac0853668685b84a` (candidate grid frozen 2026-10-20, work target filled 2026-10-27 before the dev array ran) |
| Dev selection | `dev/selection.json` (committed as `jr1/sherlock/dev_selection.json`) sha256 `b4dc8b69d86ec094e6164fce127097fd365e2790c8977327a4ad1c2ecc7ccadd`, produced 2026-10-31T18:40:19.505031+00:00 by `tools select` |
| GPU SKU for every measured cell | `H100_SXM5` (80 GB, Sherlock `gpu` partition, `-C GPU_SKU:H100_SXM5`); a job on any other GPU refuses to start. Fallback if the queue stalls: `L40S`, recorded and flagged in the cell receipt, never mixed silently |

## Splits, contamination audit, prompts and parsers

- Splits (from the manifest, hashes there): `pt/train` 29,393 articles, `pt/controller` 30, `pt/tune` 30, `pt/validation` 60, `pt/test` 62 (WikiText test NLL); GSM8K train rows split into disjoint `sft/train` 3,161, `rl/train` 3,160, `gsm8k_dev/tune` 512, `gsm8k_dev/controller` 512 and `gsm8k_dev/readiness` 128 (already used for the readiness gate); `sft/controller` 512, `rl/controller` 512; UltraFeedback `pref/train` 59,553, `pref/controller` 512, `pref/tune` 512, `pref/test` 1,985 (held-out preference ordering). Sealed learning-curve subsets `eval/lc_gsm8k` 256 and `eval/lc_mmlu` 512 (57 subject quotas), excluded from every controller and dev signal.
- Contamination audit (exact, normalized and embedded text overlap of every train pool against every eval set; train/dev rows dropped, eval rows never dropped): GSM8K train 0 of 7,473 dropped; preference train 14 of 60,591 dropped (2 embedded MMLU, 8 ARC-Easy, 4 preference-test) plus 505 empty or identical, 36 duplicate-prompt and 3 prompt-in-test pairs filtered; WikiText train 51 of 29,444 articles dropped (overlap with the test articles); validation 0 of 60. Pre-training contamination of Qwen2.5 is unknown (data not public).
- Prompt: `Solve the problem. Show your reasoning, then put the final numeric answer after ####.\n\nProblem: ` + question (MATH-500 uses the `\boxed{}` prefix in `evaluate.py`); generation stops at `\nProblem:` and `\nSolve the problem.`
- GSM8K parser and reward: `common.extract_answer` (last number after the last `####`, else last number in the text, Decimal-normalized), `common.exact_reward` (1 only if equal to the released label); no reward for formatting alone, no gold prefix. MATH-500: math-verify 0.9.0 with latex2sympy2_extended 1.11.0, a different install is refused. MMLU: 5-shot letter log-likelihood (lm-evaluation-harness v0.4 format, dev split shots), all 14,042 test items, left truncation counted. ARC-Easy: 0-shot log-likelihood, all 2,376. All protocol choices are copied into every `summary.json`.
- Readiness gate (predeclared, passed before this freeze; `results.md`): 128 development prompts x 8 unassisted samples at 1,024 new tokens gave 105 mixed-success groups (minimum 20), 268 positive and 756 negative rewards, and 34/1,024 truncations (3.32%, maximum 10%). No common SFT preparation was needed, so the comparison is joint continuation of the untouched base.

## Objectives, optimizer, equations and accounting

Full equations, the Aioli differences table, the interleaver, J-R1 floors and the retention constraint are in `jr1/TRAINER.md` at the code commit above and are frozen with it. In short:

- One shared AdamW on full weights (fp32 master weights and moments, bf16 autocast), betas (0.9, 0.95), eps 1e-8, weight decay 0.01, gradient clip 1.0, cosine lr with 2% linear warmup from 0.1x to a 0.1x floor, one lr clock over charged work; one microbatch of one objective per optimizer step; weights over (PT, SFT, DPO, RL) are shares of optimizer-step actions realized by a deterministic deficit interleaver.
- PT: shifted causal cross entropy, batch 8, max length 1,024. SFT: response-only masked cross entropy with EOS, batch 8. DPO: beta 0.1, sequence-summed response log probabilities against the frozen initial reference (`ckpt_000`, never refreshed), batch 4. RL: 4 GSM8K prompts x 8 unassisted samples at temperature 1, top-k 0, top-p 1, up to 1,024 new tokens, leave-one-out advantage, zero-variance groups give no gradient, no optimizer step and no weight decay; rollouts are charged.
- Aioli and J-R1: K = 4 probe mixtures (alpha 0.75, 12 real updates each), sweep every 200 production updates, controller split of 32 rows per objective, controller RL generation capped at 512 tokens, transfer-matrix recovery as in `00_program/aioli.py`. J-R1 adds the geometric prior, floors (min share 0.02, PT 0.05, DPO 0.05) and the validation-only retention constraint (PT NLL on the controller split more than 2% above its start raises the PT floor to 0.55 until a later sweep is back inside). No rollback, no retrospective checkpoint choice, no test data.
- Work (forward-equivalent tokens): forward 1, backward 2, reference 1, generation 1 per processed token. Probes, validation, reference passes, generation and discarded samples are charged. A cell stops at the first action boundary at or after the target (bounded single-action overshoot, recorded). Token counts are estimates, not FLOPs; wall and device time are reported separately.

## Development selection

Done on the tune split only (512 GSM8K prompts, 30 WikiText articles, 512 preference pairs), dev seed 9001, H100_SXM5, array job 49814693 (25 tasks, all COMPLETED; task 24 is the untouched base: dev GSM8K 0.6992, WikiText NLL 2.2308, preference accuracy 0.4824). Grid: 4 candidates per method (2 learning rates x 2 mixture or stage allocations), equal allowance per method: the same charged work target per candidate (3.5e6 forward-equivalent tokens, 11% of the measured target) and the same wall cap (900 s per candidate, 1.0 device-hour per method, 6.0 in total). Rule written before the run: highest dev GSM8K accuracy among candidates with dev WikiText NLL at most 2% above the base; ties by preference accuracy, then lower NLL, then listing order. All 24 candidates finished inside the allowance and passed retention; no method was flagged `retention_failed` or `no_valid_candidate`.

| Method | Candidate | Train wall (s) | Dev GSM8K acc | NLL ratio to base | Dev pref acc | Choice |
|---|---|---|---|---|---|---|
| sequential | `sequential_lr1e-6_b55` | 387 | 0.7598 | 0.9682 | 0.4805 | **selected** |
| sequential | `sequential_lr1e-6_b40` | 466 | 0.7480 | 0.9709 | 0.4805 |  |
| sequential | `sequential_lr5e-6_b55` | 378 | 0.7344 | 0.9499 | 0.4805 |  |
| sequential | `sequential_lr5e-6_b40` | 431 | 0.7188 | 0.9529 | 0.4805 |  |
| replay | `replay_lr1e-6_pt20` | 402 | 0.7520 | 0.9671 | 0.4805 |  |
| replay | `replay_lr1e-6_pt10` | 386 | 0.7461 | 0.9675 | 0.4844 |  |
| replay | `replay_lr5e-6_pt20` | 418 | 0.7578 | 0.9490 | 0.4805 | **selected** |
| replay | `replay_lr5e-6_pt10` | 390 | 0.7500 | 0.9493 | 0.4824 |  |
| fixed_joint | `fixed_joint_lr1e-6_pt55` | 501 | 0.7754 | 0.9781 | 0.4824 | **selected** |
| fixed_joint | `fixed_joint_lr1e-6_rl30` | 671 | 0.7539 | 0.9881 | 0.4844 |  |
| fixed_joint | `fixed_joint_lr5e-6_pt55` | 403 | 0.7461 | 0.9527 | 0.4844 |  |
| fixed_joint | `fixed_joint_lr5e-6_rl30` | 594 | 0.7578 | 0.9588 | 0.4863 |  |
| smooth_joint | `smooth_joint_lr1e-6_rl45` | 660 | 0.7637 | 0.9803 | 0.4805 |  |
| smooth_joint | `smooth_joint_lr1e-6_rl60` | 676 | 0.7344 | 0.9807 | 0.4805 |  |
| smooth_joint | `smooth_joint_lr5e-6_rl45` | 450 | 0.7559 | 0.9535 | 0.4824 |  |
| smooth_joint | `smooth_joint_lr5e-6_rl60` | 520 | 0.7734 | 0.9536 | 0.4824 | **selected** |
| aioli | `aioli_lr1e-6_eta1` | 785 | 0.7539 | 0.9889 | 0.4844 |  |
| aioli | `aioli_lr1e-6_eta0.3` | 755 | 0.7578 | 0.9890 | 0.4844 | **selected** |
| aioli | `aioli_lr5e-6_eta1` | 575 | 0.7324 | 0.9616 | 0.4844 |  |
| aioli | `aioli_lr5e-6_eta0.3` | 589 | 0.7500 | 0.9624 | 0.4805 |  |
| jr1 | `jr1_lr1e-6_prior0.5` | 760 | 0.7695 | 0.9889 | 0.4844 |  |
| jr1 | `jr1_lr1e-6_prior0.75` | 798 | 0.7656 | 0.9891 | 0.4805 |  |
| jr1 | `jr1_lr5e-6_prior0.5` | 602 | 0.7461 | 0.9619 | 0.4805 |  |
| jr1 | `jr1_lr5e-6_prior0.75` | 630 | 0.7754 | 0.9627 | 0.4824 | **selected** |

Limitation, stated before measurement: dev runs use 11% of the measured work target and one seed, so the selected learning rates (1e-6 versus 5e-6) and allocations are the best of a small grid at short horizon, and the dev differences (about 1 to 3 points on 512 prompts) are within sampling noise. The selection is a fixed procedure, not a claim that these are optimal settings.

## Selected settings per method (frozen configs)

| Method | Candidate | lr | Method settings | Config sha256 |
|---|---|---|---|---|
| sequential | `sequential_lr1e-6_b55` | 1e-06 | `{"boundaries": [0.55, 0.75, 0.85]}` | `de40fef6b8a8128b3e29edcde135f962b266d04f3587e272dac5b2d0e5a6a0b4` |
| replay | `replay_lr5e-6_pt20` | 5e-06 | `{"boundaries": [0.55, 0.75, 0.85], "pt_share": 0.2}` | `f549204deb87b00538776c8937ff7340c21aabf43fbdd40394d275bde46c8078` |
| fixed_joint | `fixed_joint_lr1e-6_pt55` | 1e-06 | `{"weights": {"dpo": 0.1, "pt": 0.55, "rl": 0.15, "sft": 0.2}}` | `882b23181691f5f96b4a292048dec8434ec56a098bda92312d29e79e117ff4ba` |
| smooth_joint | `smooth_joint_lr5e-6_rl60` | 5e-06 | `{"end": {"dpo": 0.1, "pt": 0.15, "rl": 0.6, "sft": 0.15}, "floors": {"dpo": 0.05, "pt": 0.05}, "start": {"dpo": 0.05, "pt": 0.7, "rl": 0.05, "sft": 0.2}}` | `b2579321b04afb1aeba9dfc6a325da07b20095ed73fd1298bd8bc0f20549c6e2` |
| aioli | `aioli_lr1e-6_eta0.3` | 1e-06 | `{"alpha": 0.75, "eta": 0.3, "floor": 0.0, "probe_updates": 12, "window": 200}` | `122de18e455d6a1af39808340047e51f0b4e1ab479c1bf3743d9fd3b5ab2fcbc` |
| jr1 | `jr1_lr5e-6_prior0.75` | 5e-06 | `{"alpha": 0.75, "dpo_floor": 0.05, "eta": 1.0, "floor": 0.0, "min_share": 0.02, "prior_end": {"dpo": 0.1, "pt": 0.15, "rl": 0.6, "sft": 0.15}, "prior_start": {"dpo": 0.1, "pt": 0.45, "rl": 0.1, "sft": 0.35}, "prior_strength": 0.75, "probe_updates": 12, "pt_floor": 0.05, "retention_min_pt": 0.55, "retention_threshold": 0.02, "window": 200}` | `00a61b20bf7dfb5659902f6b8033030c346233aedeeae4136da8549412298b05` |

Configs are `jr1/configs/protocol_v1/<method>.json` (protocol line `jr1-protocol-v1-FROZEN 2026-10-31`; written by `tools freeze-configs` at 2026-10-31T18:41:16Z from the selection above).

## Work target per cell and resource accounting

Per-cell target **W = 3.2e+07 forward-equivalent tokens** for every method and seed (not the calibration placeholder 2e9, which would need 2,584 H100 hours against the 96-hour plan; default taken at the AT-571 handoff). Rates are the AT-571 H100 calibration (`jr1/sherlock/CALIBRATION.md`); sequential and replay rates come from cheap early phases and probably understate their hours.

| Method | H100 work/s (calibrated) | Hours per cell (W/rate + 0.738 h evals) | Hours for 6 seeds |
|---|---|---|---|
| sequential | 49,763 | 0.92 | 5.5 |
| replay | 47,496 | 0.93 | 5.6 |
| fixed_joint | 6,191 | 2.17 | 13.0 |
| smooth_joint | 9,627 | 1.66 | 10.0 |
| aioli | 6,034 | 2.21 | 13.3 |
| jr1 | 6,218 | 2.17 | 13.0 |

Exp 03 estimate: about 60.9 H100 hours including the base evaluation, plus Exp 04 about 10 h, the schedule source run, the dev array (6 h cap) and readiness (4 h): inside the 96-hour plan. Per-cell time limit 2 days with up to 3 self-requeues.

## Estimands, thresholds and denominators

- Primary endpoint: terminal GSM8K accuracy on all 1,319 test questions (greedy, 1,024 new tokens, `common.extract_answer`). Transfer: all 500 MATH-500. Retention: all 14,042 MMLU test items (primary), all 2,376 ARC-Easy items, WikiText test NLL per token, held-out preference ordering accuracy on 1,985 pairs (reported separately, not evidence of alignment).
- Primary contrast: J-R1 minus tuned sequential, paired by seed, terminal GSM8K accuracy; report all six seeds, mean, standard deviation, 95% paired Student t interval and the exact paired sign-flip p-value. Item bootstraps are secondary. Every other comparison is descriptive; no multiplicity correction is claimed.
- Practical success (fixed now): GSM8K gain of at least 2 percentage points with a 95% interval above zero; MMLU and ARC-Easy candidate-minus-initial intervals excluding losses larger than 1 percentage point; corpus NLL rise at most 2% from initialization. These are planning tolerances. Failure or wide intervals is negative or inconclusive evidence.
- Sample efficiency: area under the reasoning-versus-charged-work curve (learning-curve subsets at 0, 25, 50, 75, 100% of work), achieved accuracy at fixed charged work, and work or time to a development-prespecified target (non-attainment reported, never extrapolated).
- Checkpoints at the first action boundary at or after 0, 25, 50, 75 and 100% of the work target; the lc suite (256 GSM8K + 512 MMLU) runs at all five, the full suite at `ckpt_100` and once on the untouched base. Restartable state every 1,000 actions or 1,800 s and at every checkpoint. All evaluation is charged to wall time, not to the training work target.

## Recovery, terminal conditions and storage

- Every job requeues itself near the time limit (at most 3 times) and resumes from its last atomic state; a failed or timed-out task is rerun once with the same command restricted to its index (`JR1_ARRAY=<i> ... submit.sh`), resuming, never restarting. Rerunning by hand is always safe. A cell directory is never deleted or reset; a cell that still fails keeps its place in the denominator as failed.
- A cell refuses to start unless its config is a FROZEN protocol (`tools check-frozen`) and the GPU matches `GPU_SKU`. Recovery never changes a scientific input (config, seed, data, code commit) or resets charged work. No selection by best seed, no extra run after a poor score, no early stopping on outcomes.
- Terminal states per cell: `complete` (training to target, lc suite at five checkpoints, full suite), `failed`, `interrupted`, `unstarted`; all are counted. Slurm END/FAIL mail goes to brandojazz@gmail.com and brando.science@gmail.com for every job.
- Storage: weights, per-item outputs and logs stay on Sherlock `$SCRATCH/ut_jr1` (not committed); compact manifests, receipts, summaries and `results.md` are committed (`jr1/sherlock/collect.sh` packs them with a sha256).

**TLDR-end:** [unified-training Exp 03: protocol v1] FROZEN 2026-10-31T18:41:16Z: 36 cells + 1 base, W = 3.2e+07, H100_SXM5, six dev-selected settings, all inputs hash-bound; no test output opened before the freeze.
