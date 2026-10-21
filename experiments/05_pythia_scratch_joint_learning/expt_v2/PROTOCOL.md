# Exp 05 expt_v2: random-init Pythia-14m and Pythia-70m size ladder (prospective protocol)

**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/05_pythia_scratch_joint_learning/expt_v2/PROTOCOL.md>

**Status: DRAFT.** Every freeze field below is filled except the calibrated charged-work target `W` (field 9).
To freeze: run calibration on the measurement GPU, `python -m p05.freeze --precision <fp16|bf16> --calibration
<calibration.json>`, fill `W` and the cluster's `data/manifest.json` sha256 (field 3b) here, set Status FROZEN, and commit the configs,
`configs/FROZEN.sha256` and this file before the first measured update. `p05/train.py` refuses a DRAFT config or a
frozen config whose sha256 differs from `FROZEN.sha256`. After the first measured update this version never changes;
a later scientific change is a new version (expt_v3, ...), and failed evidence is kept.

Campaign plan: [PLAN.md](../../03_qwen_joint_reasoning_retention/PLAN.md) line 24 (this experiment) and line 73
(scratch exposure target). Companion: [expt_v1](../expt_v1/PROTOCOL.md), the primary Pythia-160m version. This
version is a pre-declared secondary size ladder beyond PLAN.md's 9 scratch cells: identical in every field except
the model sizes, their learning rates and the cell list. Fields that differ from v1 are marked **(v2)**.

## 1. Objective and contrast

Question: from random initialization, at matched total charged work, does training on all four objectives (PT, SFT,
DPO, RL) jointly beat the pretrain-then-post-train pipeline (sequential stages)? The pilot's result was sequential
WikiText NLL 8.53 vs fixed joint 6.02 at a tiny budget; this version tests it with ≥50M observed PT tokens per cell.
**(v2)** Two smaller sizes at the same charged-work target as v1, so the joint-minus-sequential gap can be read
against model size (14m, 70m here; 160m in v1). The size trend is descriptive.

- Methods (jr1 schedules, unchanged): `sequential` (PT → SFT → DPO → RL at work progress 0.55 / 0.75 / 0.85),
  `fixed_joint` (update-count weights PT .55, SFT .20, DPO .10, RL .15 throughout), `smooth_joint` (linear from
  PT .70/SFT .20/DPO .05/RL .05 to PT .20/SFT .20/DPO .15/RL .45, floors PT .05, DPO .05).
- Primary contrast: `fixed_joint − sequential` and `smooth_joint − sequential`, paired by seed, at the terminal
  checkpoint (100% of `W`), separately for each size.
- Primary endpoint: WikiText-103 test NLL per token (nats, lower is better).
- Secondary endpoints: LAMBADA (OpenAI) accuracy and perplexity, ARC-Easy acc_norm, GSM8K accuracy (full 1,319,
  reported even at floor), UltraFeedback test preference accuracy, observed PT tokens, the learning curve at
  0/25/50/75/100% of `W`, and `fixed_joint − smooth_joint`.
- Scope: GSM8K at a floor is a language-learning result, not a reasoning result (PLAN.md line 73).

## 2. Seed and method manifest

**(v2)** 18 cells = 2 sizes × 3 methods × seeds {1, 2, 3}, array indices 0–17, seed-major, so indices 6k..6k+5 are
one complete paired seed for both sizes (`python -m p05.cells --version v2 --list`):

| index | cell | index | cell | index | cell |
|---|---|---|---|---|---|
| 0 | v2/14m/sequential_s1 | 6 | v2/14m/sequential_s2 | 12 | v2/14m/sequential_s3 |
| 1 | v2/14m/fixed_joint_s1 | 7 | v2/14m/fixed_joint_s2 | 13 | v2/14m/fixed_joint_s3 |
| 2 | v2/14m/smooth_joint_s1 | 8 | v2/14m/smooth_joint_s2 | 14 | v2/14m/smooth_joint_s3 |
| 3 | v2/70m/sequential_s1 | 9 | v2/70m/sequential_s2 | 15 | v2/70m/sequential_s3 |
| 4 | v2/70m/fixed_joint_s1 | 10 | v2/70m/fixed_joint_s2 | 16 | v2/70m/fixed_joint_s3 |
| 5 | v2/70m/smooth_joint_s1 | 11 | v2/70m/smooth_joint_s2 | 17 | v2/70m/smooth_joint_s3 |

All 18 stay in the denominator: a failed, interrupted or unstarted cell is reported as such, never dropped or
replaced. No untouched-base cell exists (the base is random), but ckpt_000 of every cell is evaluated.

## 3. Sources, model, tokenizer and data hashes

a. Pinned before measurement:

| item | pin |
|---|---|
| architecture **(v2)** | `EleutherAI/pythia-14m` config.json at `cf967c0a9a04383db6f7b1108d86b2962634b4ac` (GPT-NeoX, 6 layers, d=128, 4 heads) and `EleutherAI/pythia-70m` at `a39f36b100fe8a5377810d56c3f4789b9c53ac42` (6 layers, d=512, 8 heads); both context 2048, rotary, parallel residual, untied embeddings; weights not read |
| tokenizer | GPT-NeoX-20B BPE from the same repos and revisions (identical files) (vocab 50,277 + padding to 50,304; `<\|endoftext|>` = id 0 is BOS/EOS; pad = EOS) |
| PT corpus | WikiText-103 raw v1, `Salesforce/wikitext` at `b08601e04326c79dfdd32d625aee71d232d685c3`, via jr1/data.py |
| jr1 data | every file sha256-checked against the committed `jr1/data_manifest.json`; the ones Exp 05 reads: |
| pt/train | `8aaf0ccf067ab60dc8a15206e486bc35d1273cbe6ffa2c6501c5863d728993ca` (29,393 articles after the audit) |
| pt/test | `f34194ce4f8f06d42bdc866802d22eeb7d1c0f7886fcf61575f331163f7a3218` (62 articles) |
| pt/controller | `4b6658c8ad4db4675a1db46fadafbd671f935a2916c619e94fc343b615f35a37` |
| sft/train | `c3186b484a1507ea9123166e63c770f1179f483506a418e54839cb457172e5df` (3,161 GSM8K train solutions) |
| rl/train | `80276d1a1412acf6d6c753122f75adb4b36dcf590aec90c138c5c211b479c41c` (3,160 GSM8K train questions) |
| pref/train | `da748c8c70f17120bfd4ded6f9fe188723e9a43747c055fbb097147cbb7f684f` (59,553 UltraFeedback pairs) |
| pref/test | `41d81503cbb169467ded793c03cfaa044d38f54db57277d0b88fa3b29ba0f983` (1,985 pairs) |
| eval/gsm8k_test | `153639f96ac7036a5b2f3e89da8b0ba097147774a6ec27f8ed3f2509b779f8c8` (1,319) |
| eval/lc_gsm8k | `30e079611f5ca39b023848a43bc28691648f5d818aec58c2f775ca0f9f7b0f6e` (256) |
| eval/arc_easy_test | `d6174a49ec02b4d571c1352d3c2e6d2c942618c6842ae92222a479c4013c3b40` (2,376) |
| LAMBADA | `EleutherAI/lambada_openai` at `900124bf3b8235c6daf21033af9948b3f07346c4`, `default/test`, 5,153 passages |

The other jr1 controller splits (`sft/controller`, `rl/controller`, `pref/controller`) are read only by jr1's
adaptive methods, which Exp 05 does not run.

b. Exp 05 files, made by `p05/data.py prepare` from the files above (a local prepare on 2026-10-20 reproduced
all 26 jr1 files of `jr1/data_manifest.json` exactly; cluster prep must reproduce these hashes, and `data/manifest.json`
records them with the tokenizer file hashes):

| file | sha256 | rows |
|---|---|---|
| pt_blocks/train | `a44c5c2e8310461d642abbcc861a17ad5d5413dab91cef613c33e605019d8761` | 115,840 blocks |
| eval/lambada | `68ad9311b9c2dc414a50d1ddbec7ca6b698195617a334876289177714c842772` | 5,153 |
| LAMBADA parquet | `99d8a9e54cda761c6340ddd4fa6feda7f846ffb0e716afa52968dea75495eea7` | 5,153 |
| tokenizer.json | `c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624` | |

The 29,393 audited train articles (118,717,293 GPT-NeoX tokens before the audit) pack into 115,840 blocks =
118,620,160 trainer tokens per epoch, so 50M observed PT tokens is 0.42 epoch (a method spending 75% of its work on PT stays within one epoch up to
W ≈ 4.7e8).
`data/manifest.json` itself names the local data root, so its sha256 differs per machine; the cluster's value is
recorded here at freeze: `<at freeze>`.

c. PT blocks: jr1's article-level PT rows are truncated to `max_len` tokens per row, which would waste most of each
article. `p05/data.py` instead concatenates the audited train articles in release order, each followed by
`<|endoftext|>`, and cuts the token stream into exact 1024-token windows (the last partial window is dropped). Each
block is stored as text (character spans of the source articles, ids `p05pt:NNNNNNN`, source article ids listed), so
jr1's unmodified `loss_pt` re-encodes it to 1024 ± 2 tokens (BPE merges at window edges; tested).

## 4. Split and contamination audit

jr1/data.py's audit (exact, casefold/whitespace-normalized and embedded-substring overlap) already removed every
train/development row that overlaps GSM8K test, MATH-500, MMLU test, ARC-Easy test, WikiText test paragraphs or
UltraFeedback test prompts; every hit is in `jr1_data/audit_matches.jsonl` and the counts in its audit manifest. Exp 05 adds LAMBADA: `p05/data.py` checks every PT block against the last 50 characters of
each LAMBADA passage (Aho-Corasick), records the hit count as `lambada_overlap` in the manifest, and keeps the blocks
(LAMBADA comes from BookCorpus, not Wikipedia; a nonzero count is reported next to the LAMBADA numbers, not used to
filter). The 62 WikiText test articles never enter training. Controller and dev splits are not used (no dev
selection, field 7).

## 5. Prompts and parsers

All from jr1, unchanged: SFT/RL/GSM8K prompt `"Solve the problem. Show your reasoning, then put the final numeric
answer after ####.\n\nProblem: {question}\nAnswer:"`; SFT target = the GSM8K train solution + EOS; RL reward = 1 if
`extract_answer` (last number after the last `####`, else the last number; Decimal-normalized) equals the gold answer,
else 0; GSM8K eval greedy, max 1,024 new tokens, same parser. DPO pairs: `encode(prompt) + encode(response) + [eos]`.
ARC-Easy: lm-evaluation-harness v0.4 `arc_easy`, 0-shot, acc_norm. WikiText: rolling log-likelihood, 1,024-token
windows, 256 tokens of context. LAMBADA: lm-evaluation-harness v0.4 `lambada_openai`, context = text up to the last
space, target = ` ` + last word; acc = greedy decoding reproduces the whole target; perplexity over target tokens.
Pref: summed response log-prob, chosen > rejected.

## 6. Optimizer and objective equations

One AdamW over all weights for every objective (jr1 contract, TRAINER.md): peak lr **(v2)** 1e-3 for both sizes, betas (0.9, 0.95), eps 1e-8,
weight decay 0.01, grad-norm clip 1.0, cosine schedule over work progress with 1% linear warmup from 0 and floor 0.1 ×
peak. Fp32 master weights. Per update (batch sizes and lengths):

- PT: shifted causal cross entropy, mean over the batch's target tokens; 32 blocks × 1,024 tokens.
- SFT: cross entropy on response tokens only (prompt and padding masked); 32 rows, max 512 tokens.
- DPO: `−log σ(β[(log π(y_w) − log π_ref(y_w)) − (log π(y_l) − log π_ref(y_l))])`, sequence-summed response
  log-probs, β = 0.1; 8 pairs, max 1,024 tokens.
- RL: on-policy GSM8K, 4 prompts × 8 samples (HF generate, temperature 1.0, max 256 new, prompt ≤ 256 tokens),
  leave-one-out advantages, loss `−Σ_i A_i log π(y_i) / 32`. A batch whose groups all have zero reward variance makes
  no backward and no optimizer step (expected for most or all RL updates from scratch; reported as
  `zero_variance_groups`). Its generation is still charged.

Precision (one GPU type for all 18 cells, decided at calibration): V100 path `fp16` autocast with a dynamic GradScaler
(steps with inf/NaN gradients are skipped and counted, `p05.fp16_skipped_steps`), SDPA attention, no FlashAttention-2,
no vLLM; L40S path `bf16` autocast, SDPA. Checkpoints in fp32. The fp16 and bf16 configs differ only in
`precision.compute` (tested).

## 7. Development selection

None. **(v2)** The learning rate is the Pythia suite's for 70m (Biderman et al. 2023), also used for 14m, which joined
the suite later with the same setting; every other setting is jr1's default
or fixed above before any measurement. No test or learning-curve number feeds back into settings, mixtures or
checkpoint choice.

## 8. Initialization and reference policy

Fresh random init per seed: `AutoModelForCausalLM.from_config` after `torch.manual_seed(seed)`, so the three methods
of one seed start from identical weights and the three seeds differ. Data order: jr1 streams keyed
`jr1:{seed}:{objective}`, identical across methods of one seed. DPO reference = the frozen initial model
(`ckpt_000`, i.e. the random init), jr1's policy unchanged.

## 9. Resource accounting

- Charged work: jr1's ledger in forward-equivalent tokens (forward 1, backward 2, reference forward 1, generation
  prefill + decode 1 per token). Every cell of v1 and v2 runs to the same target `W`; the run stops at the first action
  boundary at or past `W` (overshoot ≤ one update, reported).
- **`W` = `<calibrated>`** (the only open field). Rule, fixed now: `p05/calibrate.py` on the measurement GPU (seed
  9001, never a measured seed) times 10 updates of each objective for each size after 2 warmups, computes per method
  the work needed for 50M observed PT tokens from the schedule weights, and `W` = max over methods and sizes × 1.05,
  rounded up to two significant figures. DRAFT placeholder 3.0e8. Expected order: sequential spends 55% of work on PT
  at 3 work units per PT token, so W ≈ 50M × 3 / 0.55 ≈ 2.7e8.
- If `W` is unreachable within the time and device budget, that is reported (PLAN.md line 73), and the cells run at
  the largest `W` that fits; the shortfall in observed PT tokens is stated with the result.
- Exposure is tracked per objective (updates, rows, tokens, generated tokens) because matched work gives different
  objective exposure by design: observed PT tokens differ across methods and are reported for every cell.
- Device time: every job's wall time (training, evaluation, prep, calibration, failures, requeues) from Slurm records
  and `receipt.json`; evaluation is not charged to `W` but is charged to device hours. The PLAN allocates 14 device
  hours to this experiment (sized for v1's 9 cells); v2's 18 cells are additional, smaller and faster per cell;
calibration gives the real estimate, and any excess is reported before launch.

## 10. Checkpoint and evaluation cadence

Checkpoints (fp32) at 0, 25, 50, 75 and 100% of `W` (`ckpt_000` … `ckpt_100`), plus resumable state every 1,000
updates or 30 minutes. Evaluation after training, per cell: `scratch_lc` (WikiText test, ARC-Easy, the 256-item
GSM8K learning-curve subset, LAMBADA) on all 5 checkpoints; `scratch_full` (WikiText test, ARC-Easy, full GSM8K, pref
test, LAMBADA) on `ckpt_000` and `ckpt_100`. Evaluation runs in fp32 with HF generate.

## 11. Full denominators

GSM8K test 1,319 (learning curve 256); LAMBADA 5,153; ARC-Easy test 2,376; WikiText-103 test 62 articles (every
token scored once; token count in the summary); UltraFeedback test 1,985 pairs. Parse failures and truncations count
as wrong; no item is dropped. jr1/evaluate.py refuses a file whose row count differs.

## 12. Uncertainty and thresholds

Three paired seeds. For each contrast and endpoint: the three per-seed differences, their mean and range, and the
count of seeds favouring joint. With three seeds an exact paired sign-flip test cannot reach p < 0.05 (minimum
two-sided p = 0.25), so the analysis is descriptive. Fixed now: "joint better on the primary endpoint" means all three
paired WikiText NLL differences are negative; "worse" means all three positive; otherwise "mixed". GSM8K is reported
with its 95% Wilson interval per cell; a floor (0 or near 0) is stated, not interpreted as reasoning. Because joint
methods see different PT token counts at matched work, NLL is also shown against observed PT tokens at each checkpoint
(descriptive, no threshold).

## 13. Recovery and terminal conditions

- Slurm signals USR1 15 minutes before the time limit; the job requeues itself (at most 3 times) and resumes from
  the last state. A cell rerun by hand resumes, never restarts; a cell directory is never deleted.
- Infrastructure failures (node, preemption, out of memory from another job) are resumed. A nonfinite loss
  (`FloatingPointError`) or any code error ends the cell as `failed`, kept in the denominator; it is not rerun with
  changed settings in this version.
- Terminal statuses: `complete` (reached `W`), `failed`, `interrupted` (time budget exhausted), `unstarted`.
  `p05/collect.py` lists all 27 Exp 05 cells (v1 + v2) with status, job ids and accounting.

## 14. Storage manifests

Per cell under `$P05_STATE_ROOT/runs/v2/<size>/<method>_s<seed>/`: `config.json` (copy of the frozen config),
`receipt.json` (status, work, exposure, optimizer and fp16-skipped steps, observed PT tokens, config sha256, data
manifest sha256, device), `log.jsonl` (every update), `slurm.json` (every start), `env.json`, `ckpt_*`,
`eval/{lc_*,full_*}/summary.json` + per-item `items_*.jsonl`, `CELL_DONE`. Run-level: `data/manifest.json`,
`receipts/calibration/calibration.json`, `jobs.txt`, `collect/collect.{json,md}`. Code identity: `code/CODE_COMMIT` of
the uploaded bundle (sha256 beside it).

## 15. Pre-declared memory fallback

If calibration shows any objective's peak memory above 90% of the device for any size, every objective's batch size
is halved (PT 16, SFT 16, DPO 4, RL 2 prompts × 8) for all sizes and methods, calibration is rerun, and the change is
recorded here before freeze. No other fallback (gradient checkpointing, shorter sequences) is used in this version.

**TLDR-end:** [unified-training: Exp 05 v2 protocol] DRAFT: random-init Pythia-14m and 70m, sequential vs fixed vs
smooth joint × 3 seeds at the same matched charged-work target as v1 (to be calibrated for ≥50M observed PT tokens);
primary endpoint WikiText-103 test NLL, read against model size.
