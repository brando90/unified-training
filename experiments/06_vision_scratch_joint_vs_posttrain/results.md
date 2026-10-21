# Vision joint from scratch vs pretrain-then-finetune: results

**Doc link:** <https://github.com/brando90/unified-training/blob/ut/exp06-vision/experiments/06_vision_scratch_joint_vs_posttrain/results.md>
**Status:** DONE — 30 of 30 cells complete on Sherlock V100 (fp16); analysis run with the frozen `expt_v1/analyze.py`.
**Last updated:** 10-21-2026 PDT

## Verdict (per the frozen PROTOCOL)

Denominator: 30 prospective cells (2 models x 5 arms x 3 seeds). Completed: 30; failed: 0; interrupted: 0.

- **H1 (joint >= sequential on CIFAR-100): inconclusive** at ViT-Tiny. Best joint arm smooth_joint minus sequential = +1.40 pp,
  97.5% interval [-2.36, +5.16] (n=3 seeds). The interval is too wide for "better" and not above the -1.0 pp non-inferiority margin.
- **H2 (joint retains more Tiny-ImageNet): supported** at ViT-Tiny. +30.10 pp, 95% interval [28.55, 31.66]. Sequential drops to 7.8% on the source task.
- **Scale check (ViT-Small): same verdicts** (H1 inconclusive, +3.72 pp, [-3.03, +10.46]; H2 supported, +21.23 pp, [16.52, 25.95]), so both replicate as the same verdict class.
- Against Brando's claim: nothing counts against it ("sequential better" occurs at neither size). H1 is not shown, only H2. The claim is supported for retention, not for target accuracy, in this setting (small ViTs, 32px, one source/target pair, one recipe, 3 seeds).

## Findings

- Both joint arms are above or near every baseline on CIFAR-100 at ViT-Tiny (fixed_joint 58.64, smooth_joint 58.96 vs sequential 57.56, target_only 56.41), but with 3 seeds only the joint-vs-target_only contrast excludes zero (+2.55 [1.62, 3.48]).
- The stronger baseline, sequential_replay, matches joint on source retention (36.87 vs 37.87 to 38.46) and is 2.68 pp lower on CIFAR-100 (95% [0.05, 5.31], borderline).
- The source helps the target only slightly under this budget: sequential - target_only = +1.15 pp [-0.53, 2.83] at Tiny.
- ViT-Small is worse than ViT-Tiny on CIFAR-100 for every arm (42 to 46 vs 56 to 59) with the same recipe, so the small model is undertrained or poorly tuned for this recipe (60,000 steps, lr 1e-3, 32px). Treat the scale check as "same verdict", not as evidence about scale.
- smooth_joint at ViT-Small has a wide spread across seeds (43.80, 46.44, 48.82) and drives that cell's wide interval. The other four arms are tight (within about 1 pp).
- Sequential's forgetting is visible in the curves: Tiny-ImageNet falls from ~35% to ~8% right after the switch at step 30,000.

Plots (accuracy vs optimizer step, mean over seeds, one line per arm): `analysis/curves_tiny.png`, `analysis/curves_small.png`.

## Per-seed results

| Model | Arm | CIFAR-100 test top-1 per seed (1, 2, 3) | Tiny-ImageNet val top-1 per seed (1, 2, 3) |
|---|---|---|---|
| ViT-Tiny | target_only | 56.68, 56.17, 56.39 | 0.34, 0.39, 0.45 |
| ViT-Tiny | sequential | 58.25, 57.68, 56.76 | 8.34, 8.15, 6.82 |
| ViT-Tiny | sequential_replay | 55.82, 57.06, 55.98 | 37.29, 36.64, 36.67 |
| ViT-Tiny | fixed_joint | 59.21, 57.98, 58.74 | 38.46, 37.94, 38.99 |
| ViT-Tiny | smooth_joint | 59.00, 58.52, 59.37 | 37.72, 38.63, 37.27 |
| ViT-Small | target_only | 42.82, 41.46, 41.67 | 0.63, 0.44, 0.50 |
| ViT-Small | sequential | 42.85, 41.90, 43.16 | 6.99, 6.09, 6.36 |
| ViT-Small | sequential_replay | 43.13, 43.28, 42.71 | 25.70, 26.03, 25.63 |
| ViT-Small | fixed_joint | 43.29, 43.13, 43.25 | 25.32, 26.42, 25.43 |
| ViT-Small | smooth_joint | 46.44, 43.80, 48.82 | 27.69, 25.75, 29.70 |

## Full tables (analyze.py output, `analysis/tables.md`)

Final-step top-1 accuracy (%), mean [95% t interval] over complete seeds. Contrasts are paired by seed;
H1 contrasts use 97.5% intervals (Bonferroni over the two joint arms). p-val=n/a (intervals, no test).

## ViT-Tiny

Cells: complete 15 of 15

| Arm | Complete | CIFAR-100 test top-1 | Tiny-ImageNet val top-1 | img/s | train h |
|---|---:|---|---|---:|---:|
| target_only | 3/3 | 56.41 [55.78, 57.05] (n=3) | 0.39 [0.26, 0.53] (n=3) | 4632 | 0.79 |
| sequential | 3/3 | 57.56 [55.70, 59.43] (n=3) | 7.77 [5.71, 9.83] (n=3) | 5014 | 0.68 |
| sequential_replay | 3/3 | 56.29 [54.61, 57.96] (n=3) | 36.87 [35.96, 37.78] (n=3) | 2870 | 1.13 |
| fixed_joint | 3/3 | 58.64 [57.10, 60.19] (n=3) | 38.46 [37.16, 39.77] (n=3) | 2826 | 1.14 |
| smooth_joint | 3/3 | 58.96 [57.90, 60.02] (n=3) | 37.87 [36.15, 39.59] (n=3) | 2685 | 1.20 |

Best joint arm (higher mean CIFAR-100): **smooth_joint**

| Contrast (a - b, pp) | Level | Seeds | Mean [interval] |
|---|---|---|---|
| H1_best_joint_minus_sequential_cifar (smooth_joint - sequential) | 97.5% (Bonferroni over 2 joint arms) | [1, 2, 3] | 1.40 [-2.36, 5.16] (n=3) |
| H2_best_joint_minus_sequential_tin (smooth_joint - sequential) | 95% | [1, 2, 3] | 30.10 [28.55, 31.66] (n=3) |
| best_joint_minus_sequential_replay_cifar (smooth_joint - sequential_replay) | 95% | [1, 2, 3] | 2.68 [0.05, 5.31] (n=3) |
| best_joint_minus_sequential_replay_tin (smooth_joint - sequential_replay) | 95% | [1, 2, 3] | 1.01 [-1.12, 3.13] (n=3) |
| best_joint_minus_target_only_cifar (smooth_joint - target_only) | 95% | [1, 2, 3] | 2.55 [1.62, 3.48] (n=3) |
| sequential_minus_target_only_cifar (sequential - target_only) | 95% | [1, 2, 3] | 1.15 [-0.53, 2.83] (n=3) |
| fixed_joint_minus_sequential_cifar (fixed_joint - sequential) | 97.5% (Bonferroni over 2 joint arms) | [1, 2, 3] | 1.08 [-1.95, 4.11] (n=3) |
| smooth_joint_minus_sequential_cifar (smooth_joint - sequential) | 97.5% (Bonferroni over 2 joint arms) | [1, 2, 3] | 1.40 [-2.36, 5.16] (n=3) |

**H1 (joint >= sequential on CIFAR-100):** inconclusive  
**H2 (joint retains more Tiny-ImageNet):** supported

## ViT-Small

Cells: complete 15 of 15

| Arm | Complete | CIFAR-100 test top-1 | Tiny-ImageNet val top-1 | img/s | train h |
|---|---:|---|---|---:|---:|
| target_only | 3/3 | 41.98 [40.16, 43.80] (n=3) | 0.52 [0.28, 0.76] (n=3) | 2816 | 1.15 |
| sequential | 3/3 | 42.64 [41.01, 44.27] (n=3) | 6.48 [5.33, 7.63] (n=3) | 3047 | 1.06 |
| sequential_replay | 3/3 | 43.04 [42.31, 43.77] (n=3) | 25.79 [25.26, 26.32] (n=3) | 3211 | 1.00 |
| fixed_joint | 3/3 | 43.22 [43.02, 43.43] (n=3) | 25.72 [24.22, 27.23] (n=3) | 2732 | 1.18 |
| smooth_joint | 3/3 | 46.35 [40.12, 52.59] (n=3) | 27.71 [22.81, 32.62] (n=3) | 2945 | 1.10 |

Best joint arm (higher mean CIFAR-100): **smooth_joint**

| Contrast (a - b, pp) | Level | Seeds | Mean [interval] |
|---|---|---|---|
| H1_best_joint_minus_sequential_cifar (smooth_joint - sequential) | 97.5% (Bonferroni over 2 joint arms) | [1, 2, 3] | 3.72 [-3.03, 10.46] (n=3) |
| H2_best_joint_minus_sequential_tin (smooth_joint - sequential) | 95% | [1, 2, 3] | 21.23 [16.52, 25.95] (n=3) |
| best_joint_minus_sequential_replay_cifar (smooth_joint - sequential_replay) | 95% | [1, 2, 3] | 3.31 [-3.63, 10.26] (n=3) |
| best_joint_minus_sequential_replay_tin (smooth_joint - sequential_replay) | 95% | [1, 2, 3] | 1.93 [-3.48, 7.33] (n=3) |
| best_joint_minus_target_only_cifar (smooth_joint - target_only) | 95% | [1, 2, 3] | 4.37 [-1.82, 10.56] (n=3) |
| sequential_minus_target_only_cifar (sequential - target_only) | 95% | [1, 2, 3] | 0.65 [-1.22, 2.52] (n=3) |
| fixed_joint_minus_sequential_cifar (fixed_joint - sequential) | 97.5% (Bonferroni over 2 joint arms) | [1, 2, 3] | 0.59 [-1.51, 2.68] (n=3) |
| smooth_joint_minus_sequential_cifar (smooth_joint - sequential) | 97.5% (Bonferroni over 2 joint arms) | [1, 2, 3] | 3.72 [-3.03, 10.46] (n=3) |

**H1 (joint >= sequential on CIFAR-100):** inconclusive  
**H2 (joint retains more Tiny-ImageNet):** supported

**Scale check:** same H1 verdict at ViT-Small

## Cells not complete

- none

## Run record

Cluster: Sherlock, `-p gpu -C GPU_GEN:VLT --gres=gpu:1` (V100, AMP fp16), `$SCRATCH/exp06`.
Slurm mail END,FAIL to brandojazz@gmail.com and brando.science@gmail.com on every job.

| Job | Id | Depends on |
|---|---|---|
| Cells all ran on Tesla V100S-PCIE-32GB, torch 2.6.0+cu124, fp16, 60,000 steps, B=192, with 22 to 25 skipped steps per cell (about 0.04%), far below the 1% failure threshold. Raw outputs: `analysis/analysis.json`, `analysis/jobs.txt`; per-cell `final.json`/`curve.jsonl` came from `$SCRATCH/exp06/exp06_results.tgz` (not committed).

Deviations from the submit plan (none touch PROTOCOL):
- FarmShare's `gpu` QOS was full (32/32 submit slots), so the run went to Sherlock, the agreed fallback.
- Raw archives were uploaded from the build machine (sha256 verified against PROTOCOL) instead of downloaded on Sherlock, because the Toronto mirror ran at ~70 KB/s.
- Sherlock's `dev` QOS caps memory at 16G, so the dev smoke ran with `--mem=16G` instead of `submit_exp06.sh`'s 32G; it passed.
- Because the dev GPU is a MIG slice, a V100 run of the same `smoke.sbatch` gated the array (afterok) instead of the dev PASS alone.

**TLDR-end:** [unified-training: Exp 06 vision] 30 of 30 ViT cells done: joint training retains the source task (+30 pp at Tiny, p-interval excludes 0) but H1 on CIFAR-100 is inconclusive (+1.4 pp [-2.4, +5.2]); same at ViT-Small.
