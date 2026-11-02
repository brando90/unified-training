# Matched exposure and objective ordering: prospective protocol v1

**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/04_qwen_matched_exposure_schedules/expt_v1/PROTOCOL.md>

**Summary:** Follow the [campaign plan](../../03_qwen_joint_reasoning_retention/PLAN.md). Frozen measured protocol v1 for Exp 04: replay one development-frozen J-R1 action schedule in interleaved and stage-sorted order, three seeds each.

Status: **FROZEN**. Configs frozen 2026-10-31T18:41:16Z; the source schedule was recorded 2026-11-02 by job 50524408 (finished 05:59 PST), before any Exp 04 cell; this file first committed 2026-11-02 in commit cca9104 (the Status line corrected in the commit after it). Frozen before any measured update and before any test output was opened. Never change this version after its first measured update.

Planned cells: **6 training cells (2 orders x 3 seeds)**; denominator never reduced. Exp 04 imports the Exp 03 package (`experiments/03_qwen_joint_reasoning_retention/expt_v1/jr1/`) with no copy, so every input hash, split, prompt, parser, objective equation, accounting rule, evaluation protocol and recovery rule of the Exp 03 PROTOCOL v1 (same commit, same freeze) applies here unchanged and is repeated in short below.

## Objective and contrast

Does interleaving objectives help beyond receiving more useful training data? The same objective microbatches (same examples per objective, same tokenization, same positional lr clock) are replayed in recorded (interleaved) order and in stable stage order (PT, SFT, DPO, RL). RL regenerates fresh rollouts for the same prompts with the replaying policy, so generated tokens and rewards differ and are reported, not equalized; comparisons are at matched scheduled RL prompts and rollout count, not equal generated strings.

## Frozen source schedule

- Source: the frozen J-R1 method of Exp 03 (`configs/protocol_v1/jr1.json`, config sha256 `00a61b20bf7dfb5659902f6b8033030c346233aedeeae4136da8549412298b05`, candidate `jr1_lr5e-6_prior0.75`), dev seed 9001, work target W = 3.2e+07, run on H100_SXM5 as Slurm job 50524408 (`schedule04.sbatch`) with `--record-schedule`. Default taken: the plan's primary candidate (J-R1) is the source method.
- Schedule file: `runs/04/schedule_jr1_9001.jsonl` on Sherlock, one row per training or probe update (update index, objective, example ids, purpose, progress, lr); controller validation passes are not training actions and are not replayed. **sha256 `db3c4c6e448d673851dcd1aef0ec6b34d5480b34edb7b6e9be27210d05357192`.** Metadata (counts per objective, total work): `{"dpo": {"examples": 800, "updates": 200}, "pt": {"examples": 2816, "updates": 352}, "rl": {"examples": 1604, "updates": 401}, "sft": {"examples": 2608, "updates": 326}}`.
- The schedule uses dev seed 9001 and the tune/controller splits only; no test output was opened.

## Cell manifest

Array index i of `cells04.sbatch` (submitted with `--dependency=afterok:` the schedule job): seed `1101 + i // 2`, order `(interleaved, stage_sorted)[i % 2]`. Seeds 1101 to 1103. Each cell runs `jr1.train --replay-schedule <schedule> --order <order> --seed <seed> --config jr1.json`. Checkpoints at 25, 50, 75 and 100% of scheduled actions (work recorded); lc suite at each and the full suite at the end, as in Exp 03.

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

## Splits, prompts, parsers, equations, thresholds, recovery (same as Exp 03 v1)

Identical to the Exp 03 PROTOCOL v1 sections "Splits, contamination audit, prompts and parsers", "Objectives, optimizer, equations and accounting", "Estimands, thresholds and denominators" and "Recovery, terminal conditions and storage". Full denominators per cell: GSM8K 1,319, MATH-500 500, MMLU 14,042, ARC-Easy 2,376, WikiText test 62 articles, preference test 1,985, lc 256 + 512. Exp 04 contrast: per-seed paired difference (interleaved minus stage_sorted) in terminal GSM8K accuracy with all three seeds shown; with three seeds it is descriptive (no p-value claim); retention and corpus NLL reported alongside.

## Development selection, work target and resources

Selected J-R1 settings are those in the Exp 03 table (lr 5e-06, prior strength 0.75); both replay orders use that config's optimizer and lr clock. Work per cell is the schedule's recorded work (about W = 3.2e+07 forward-equivalent tokens, plus the regenerated RL rollouts, which are charged). Estimated about 1.7 h per cell (the AT-571 six-method mean of W/rate plus 0.738 h of evaluations) on H100_SXM5, so six cells are about 10 h, the plan's Exp 04 allocation; the source schedule run (job 50524408) took 0.99 h (00:59:09), less than the 2 h estimated from J-R1 at 2.15 h per cell.

**TLDR-end:** [unified-training Exp 04: protocol v1] FROZEN 2026-10-31T18:41:16Z: 6 cells, schedule sha256 db3c4c6e448d6738, J-R1 seed 9001, H100_SXM5; no test output opened before the freeze.
