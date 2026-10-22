# Vision joint from scratch vs pretrain-then-finetune: results

**Doc link:** <https://github.com/brando90/unified-training/blob/ut/exp06-vision/experiments/06_vision_scratch_joint_vs_posttrain/results.md>
**Status:** DONE — 50 of 50 cells complete: 30 original, 10 from A1 (Tiny seeds 4-5), 10 from the A1 extension (Small seeds 4-5). Original 30 ran on Sherlock V100 (fp16); analysis run with the frozen `expt_v1/analyze.py`.
**Last updated:** 10-22-2026 PDT

## Amendment A1 extension (n=5 at ViT-Small): H1 joint better at both sizes (exploratory)

**Status of this section: EXPLORATORY.** Seeds 4-5 were added at ViT-Small after the n=3 result was seen (optional stopping).

**Takeaway:** with seeds 4-5 added at ViT-Small, H1 reads "joint better" at ViT-Small too: smooth_joint - sequential = +3.61 pp, 97.5% interval [0.97, 6.26] (n=5; n=3 was +3.72 [-3.03, 10.46], inconclusive). The scale check now gives the same H1 verdict at both sizes (joint better, n=5; Tiny is marginal, lower bound 0.14 pp), and H2 stays supported. Every n=3 interval included 0; the n=5 H1 intervals no longer do. This is exploratory: seeds were added after the n=3 result was seen (optional stopping), the n=3 numbers stay beside it, and `expt_v1/PROTOCOL.md` and cells 0-39 are unchanged. The append-only record is `expt_v1/AMENDMENT_A1_seeds45.md` (extension) and `cells.tsv` tasks 40-49.

**Denominator:** ViT-Small 25 of 25 cells complete (5 arms x 5 seeds), ViT-Tiny 25 of 25; 50 of 50 overall, none failed or interrupted. Sherlock V100 (fp16), jobs 48904130 (seed 4, tasks 40-44), 48904132 (seed 5, tasks 45-49), post 48904137; all COMPLETED exit 0, and each of the 10 per-cell `final.json` files was checked on Sherlock. Seeds 1-3 values are identical to the n=3 run. Outputs: `analysis_n5/` (now n=5 for both sizes).

| ViT-Small contrast (pp) | n=3 [interval] | n=5 [interval] |
|---|---|---|
| H1: smooth_joint - sequential, CIFAR-100 (97.5%) | +3.72 [-3.03, 10.46] | +3.61 [0.97, 6.26] |
| H2: smooth_joint - sequential, Tiny-ImageNet (95%) | +21.23 [16.52, 25.95] | +20.84 [18.73, 22.95] |
| smooth_joint - sequential_replay, CIFAR-100 (95%) | +3.31 [-3.63, 10.26] | +3.30 [0.38, 6.23] |
| smooth_joint - sequential_replay, Tiny-ImageNet (95%) | +1.93 [-3.48, 7.33] | +1.54 [-1.18, 4.27] |
| fixed_joint - sequential, CIFAR-100 (97.5%) | +0.59 [-1.51, 2.68] | +0.72 [-0.13, 1.56] |
| sequential - target_only, CIFAR-100 (95%) | +0.65 [-1.22, 2.52] | +0.73 [-0.30, 1.76] |
| smooth_joint - target_only, CIFAR-100 (95%) | +4.37 [-1.82, 10.56] | +4.34 [1.33, 7.35] |

- **The H1 win rests on smooth_joint alone.** fixed_joint - sequential is +0.72 [-0.13, 1.56], which includes 0. smooth_joint per seed on CIFAR-100: 46.44, 43.80, 48.82, 44.18, 47.62 (sd 2.17 across seeds, versus 0.51 for sequential); its paired difference to sequential is positive in all five seeds (+3.59, +1.90, +5.66, +1.99, +4.92).
- **Replay baseline:** smooth_joint beats sequential_replay on CIFAR-100 by +3.30 [0.38, 6.23], but the retention gap is not distinguishable (+1.54 [-1.18, 4.27]).
- **Caveat carried over:** ViT-Small is worse than ViT-Tiny on CIFAR-100 for every arm (41-46 vs 56-59) under this recipe, so read the scale check as "same verdict", not as evidence about scale. The Tiny interval's lower bound is only 0.14 pp above 0.

**Per-seed results, ViT-Small (seeds 1 to 5)**

| Model | Arm | CIFAR-100 test top-1 per seed | Tiny-ImageNet val top-1 per seed |
|---|---|---|---|
| ViT-Small | target_only | 42.82, 41.46, 41.67, 42.23, 40.96 | 0.63, 0.44, 0.50, 0.44, 0.42 |
| ViT-Small | sequential | 42.85, 41.90, 43.16, 42.19, 42.70 | 6.99, 6.09, 6.36, 6.34, 6.67 |
| ViT-Small | sequential_replay | 43.13, 43.28, 42.71, 42.71, 42.51 | 25.70, 26.03, 25.63, 26.45, 25.12 |
| ViT-Small | fixed_joint | 43.29, 43.13, 43.25, 43.52, 43.19 | 25.32, 26.42, 25.43, 26.12, 24.92 |
| ViT-Small | smooth_joint | 46.44, 43.80, 48.82, 44.18, 47.62 | 27.69, 25.75, 29.70, 25.33, 28.17 |

Full analyze.py output at n=5 for both sizes: [`analysis_n5/tables.md`](analysis_n5/tables.md).

**Next step:** find out why smooth_joint is the only arm with a wide seed spread at ViT-Small (a quick look at its per-seed curves around the step-30,000 switch) before treating "joint better at Small" as more than exploratory.

---

## Amendment A1 (n=5 at ViT-Tiny): H1 now "joint better"

**Status of this section:** seeds 4 and 5 were added at ViT-Tiny only, after the n=3 result was seen to be inconclusive. That is a post-hoc extension
(optional stopping), so read the n=5 interval as exploratory and report the n=3 verdict next to it. `expt_v1/PROTOCOL.md` and cells 0-29 are unchanged; the
append-only record is `expt_v1/AMENDMENT_A1_seeds45.md` and `cells.tsv` tasks 30-39. The n=3 outputs stay in `analysis/` (and Sherlock `analysis_n3`); the n=5 outputs
are in `analysis_n5/`. ViT-Small stays at n=3 in this section (extended to n=5 in the section above).

- **H1 at ViT-Tiny: joint better (n=5).** smooth_joint - sequential = +1.46 pp, 97.5% interval [0.14, 2.78] (n=3 was +1.40 [-2.36, 5.16], inconclusive). The lower bound clears 0 by 0.14 pp, so the verdict is marginal.
- **H2 unchanged: supported.** +29.92 pp [29.07, 30.77] on Tiny-ImageNet (n=3: +30.10 [28.55, 31.66]).
- **Replay baseline:** smooth_joint beats sequential_replay by +2.66 pp on CIFAR-100 [1.73, 3.60] (n=3: [0.05, 5.31]), and by +1.09 pp [0.32, 1.86] on retention.
- **fixed_joint - sequential:** +0.99 pp [0.04, 1.95] (97.5%), also above 0 but only just.
- **Scale check now differs:** ViT-Small (n=3) is still inconclusive (+3.72 [-3.03, 10.46]), so "H1 verdict differs at ViT-Small". Seeds 4-5 at ViT-Small were then run; see the extension section above (H1 joint better, +3.61 [0.97, 6.26]).
- Seeds 1-3 values are identical to the n=3 run; the interval narrowed because the paired differences are consistent (smooth_joint - sequential per seed: +0.75, +0.84, +2.61, +2.11, +1.00).

### Per-seed results, ViT-Tiny (seeds 1 to 5)

| Model | Arm | CIFAR-100 test top-1 per seed | Tiny-ImageNet val top-1 per seed |
|---|---|---|---|
| ViT-Tiny | target_only | 56.68, 56.17, 56.39, 56.78, 55.77 | 0.34, 0.39, 0.45, 0.51, 0.68 |
| ViT-Tiny | sequential | 58.25, 57.68, 56.76, 57.46, 57.57 | 8.34, 8.15, 6.82, 8.04, 8.76 |
| ViT-Tiny | sequential_replay | 55.82, 57.06, 55.98, 56.80, 56.06 | 37.29, 36.64, 36.67, 37.18, 36.47 |
| ViT-Tiny | fixed_joint | 59.21, 57.98, 58.74, 58.25, 58.51 | 38.46, 37.94, 38.99, 37.97, 38.02 |
| ViT-Tiny | smooth_joint | 59.00, 58.52, 59.37, 59.57, 58.57 | 37.72, 38.63, 37.27, 38.34, 37.75 |

### analyze.py output at n=5 (ViT-Tiny; `analysis_n5/tables.md` has ViT-Small too)


Cells: complete 25 of 25

| Arm | Complete | CIFAR-100 test top-1 | Tiny-ImageNet val top-1 | img/s | train h |
|---|---:|---|---|---:|---:|
| target_only | 5/5 | 56.36 [55.85, 56.86] (n=5) | 0.47 [0.31, 0.64] (n=5) | 5102 | 0.70 |
| sequential | 5/5 | 57.54 [56.88, 58.21] (n=5) | 8.02 [7.12, 8.92] (n=5) | 5388 | 0.62 |
| sequential_replay | 5/5 | 56.34 [55.66, 57.03] (n=5) | 36.85 [36.40, 37.30] (n=5) | 4017 | 0.90 |
| fixed_joint | 5/5 | 58.54 [57.95, 59.12] (n=5) | 38.28 [37.72, 38.84] (n=5) | 3966 | 0.91 |
| smooth_joint | 5/5 | 59.01 [58.42, 59.59] (n=5) | 37.94 [37.27, 38.61] (n=5) | 3881 | 0.95 |

Best joint arm (higher mean CIFAR-100): **smooth_joint**

| Contrast (a - b, pp) | Level | Seeds | Mean [interval] |
|---|---|---|---|
| H1_best_joint_minus_sequential_cifar (smooth_joint - sequential) | 97.5% (Bonferroni over 2 joint arms) | [1, 2, 3, 4, 5] | 1.46 [0.14, 2.78] (n=5) |
| H2_best_joint_minus_sequential_tin (smooth_joint - sequential) | 95% | [1, 2, 3, 4, 5] | 29.92 [29.07, 30.77] (n=5) |
| best_joint_minus_sequential_replay_cifar (smooth_joint - sequential_replay) | 95% | [1, 2, 3, 4, 5] | 2.66 [1.73, 3.60] (n=5) |
| best_joint_minus_sequential_replay_tin (smooth_joint - sequential_replay) | 95% | [1, 2, 3, 4, 5] | 1.09 [0.32, 1.86] (n=5) |
| best_joint_minus_target_only_cifar (smooth_joint - target_only) | 95% | [1, 2, 3, 4, 5] | 2.65 [2.28, 3.02] (n=5) |
| sequential_minus_target_only_cifar (sequential - target_only) | 95% | [1, 2, 3, 4, 5] | 1.19 [0.41, 1.96] (n=5) |
| fixed_joint_minus_sequential_cifar (fixed_joint - sequential) | 97.5% (Bonferroni over 2 joint arms) | [1, 2, 3, 4, 5] | 0.99 [0.04, 1.95] (n=5) |
| smooth_joint_minus_sequential_cifar (smooth_joint - sequential) | 97.5% (Bonferroni over 2 joint arms) | [1, 2, 3, 4, 5] | 1.46 [0.14, 2.78] (n=5) |

**H1 (joint >= sequential on CIFAR-100):** joint better  
**H2 (joint retains more Tiny-ImageNet):** supported


---
**The sections below are the original n=3 report (30 cells) and are kept unchanged.**

## Verdict (per the frozen PROTOCOL)

Denominator: 30 prospective cells (2 models x 5 arms x 3 seeds). Completed: 30; failed: 0; interrupted: 0.

- **H1 (joint >= sequential on CIFAR-100): inconclusive** at ViT-Tiny. Best joint arm smooth_joint minus sequential = +1.40 pp,
  97.5% interval [-2.36, +5.16] (n=3 seeds). The interval is too wide for "better" and not above the -1.0 pp non-inferiority margin.
- **H2 (joint retains more Tiny-ImageNet): supported** at ViT-Tiny. +30.10 pp, 95% interval [28.55, 31.66]. Sequential drops to 7.8% on the source task.
- **Scale check (ViT-Small): same verdicts** (H1 inconclusive, +3.72 pp, [-3.03, +10.46]; H2 supported, +21.23 pp, [16.52, 25.95]), so both replicate as the same verdict class.
- Against Brando's claim (from-scratch joint training beats pretrain-then-post-train): the retention advantage is clear; the target-accuracy advantage is not shown at 3 seeds. "Sequential better" occurs at neither size, but H1 stays inconclusive. The stronger baseline, sequential_replay, nearly closes the retention gap. Scope: small ViTs, 32px, one source/target pair, one recipe, 3 seeds.

## Findings

- The stronger baseline, sequential_replay, matches joint on source retention (36.87 vs 37.87 to 38.46) and is 2.68 pp lower on CIFAR-100 (95% [0.05, 5.31], borderline).
- The source helps the target only slightly under this budget: sequential - target_only = +1.15 pp [-0.53, 2.83] at Tiny.
- ViT-Small is worse than ViT-Tiny on CIFAR-100 for every arm (42 to 46 vs 56 to 59) with the same recipe, so the small model is undertrained or poorly tuned for this recipe (60,000 steps, lr 1e-3, 32px). Treat the scale check as "same verdict", not as evidence about scale.
- smooth_joint at ViT-Small has a wide spread across seeds (43.80, 46.44, 48.82) and drives that cell's wide interval. The other four arms are tight (within about 1 pp).
- Sequential's forgetting is visible in the curves: Tiny-ImageNet falls from ~35% to ~8% right after the switch at step 30,000.

Plots (accuracy vs optimizer step, mean over seeds, one line per arm): `analysis/curves_tiny.png`, `analysis/curves_small.png`. Redrawn with `analysis/replot.py` (legend instead of overlapping end labels); no number changed.

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

## Each joint arm vs target_only (CIFAR-100 test top-1, pp)

Paired by seed, 95% Student t interval, n=3 seeds. Computed outside `analyze.py` from the per-seed numbers above with the same formula (`analyze.py` reports only the best joint arm vs target_only); the smooth_joint rows match its output.

| Model | Contrast | Mean [95% interval] |
|---|---|---|
| ViT-Tiny | fixed_joint - target_only | +2.23 [1.30, 3.16] |
| ViT-Tiny | smooth_joint - target_only | +2.55 [1.62, 3.48] |
| ViT-Small | fixed_joint - target_only | +1.24 [-0.42, 2.90] |
| ViT-Small | smooth_joint - target_only | +4.37 [-1.82, 10.56] |

## Full tables (analyze.py output, `analysis/tables.md`)

Final-step top-1 accuracy (%), mean [95% t interval] over complete seeds. Contrasts are paired by seed;
H1 contrasts use 97.5% intervals (Bonferroni over the two joint arms). p-val=n/a (intervals, no test).

### ViT-Tiny

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

### ViT-Small

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

### Cells not complete

- none

## Run record

Cluster: Sherlock, `-p gpu -C GPU_GEN:VLT --gres=gpu:1` (V100, AMP fp16), `$SCRATCH/exp06`.
Slurm mail END,FAIL to brandojazz@gmail.com and brando.science@gmail.com on every job.

| Job | Id | Depends on |
|---|---|---|
| Dev GPU smoke | 48686478 | none |
| V100 smoke (gates the cells) | 48686747 | afterany:48686478 |
| Cells array 0-29%4 | 48686784 | afterok:48686747 |
| Post (analyze + `exp06_results.tgz`) | 48686799 | afterany:48686784 |

Cells all ran on Tesla V100S-PCIE-32GB, torch 2.6.0+cu124, fp16, 60,000 steps, B=192, with 22 to 25 skipped steps per cell (about 0.04%), far below the 1% failure threshold. Raw outputs: `analysis/analysis.json`, `analysis/jobs.txt`; per-cell `final.json`/`curve.jsonl` came from `$SCRATCH/exp06/exp06_results.tgz` (not committed).

Deviations from the submit plan (none touch PROTOCOL):
- FarmShare's `gpu` QOS was full (32/32 submit slots), so the run went to Sherlock, the agreed fallback.
- Raw archives were uploaded from the build machine (sha256 verified against PROTOCOL) instead of downloaded on Sherlock, because the Toronto mirror ran at ~70 KB/s.
- Sherlock's `dev` QOS caps memory at 16G, so the dev smoke ran with `--mem=16G` instead of `submit_exp06.sh`'s 32G; it passed.
- Because the dev GPU is a MIG slice, a V100 run of the same `smoke.sbatch` gated the array (afterok) instead of the dev PASS alone.

**TLDR-end:** [unified-training: Exp 06 vision] 30 of 30 ViT cells done: joint training retains the source task (+30 pp at Tiny, interval excludes 0) but H1 on CIFAR-100 is inconclusive (+1.4 pp [-2.4, +5.2]); same at ViT-Small.
