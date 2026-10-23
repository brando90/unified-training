# Random-initialization joint learning: final results

**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/05_pythia_scratch_joint_learning/results.md>
**Status:** COMPLETE: 27 of 27 prospective cells reached the frozen charged-work target (W = 2.9e8). Failed 0, interrupted 0, unstarted 0.
**Last updated:** 10-23-2026 (cells collected 10-22 23:13 PDT; the 14m collapse diagnosis was added 10-23, and the DPO reference check at 14m is still open)

`experiments/05_pythia_scratch_joint_learning` trained 27 cells from random initialization: Pythia-160m (expt_v1, 9 cells) and
Pythia-14m and Pythia-70m (expt_v2, 18 cells), each = 3 methods x seeds {1, 2, 3}, one matched charged-work target W = 2.9e8
forward-equivalent tokens per cell, one cluster and precision for all (Sherlock `gpu`, V100 32GB, fp16). Every cell has at least
50M observed PT tokens (27 of 27; minimum 53.2M). All numbers below are terminal checkpoint (100% of W) unless stated.

## Headline

1. **Primary endpoint (WikiText-103 test NLL, nats/token), joint beats sequential at every size, for both joint methods, with all
   three paired seeds agreeing (3/3 at each of 3 sizes x 2 contrasts = 18 of 18 paired differences negative).** By the protocol's rule
   this is "joint better".
2. **The size of the gap is not a clean size trend.** Joint-minus-sequential (fixed_joint) is -14.15 (14m), -17.35 (70m), -12.55 (160m):
   largest at 70m, smallest at 160m, ranges overlap across sizes. The size ladder reads as "no monotone trend with 3 sizes and 3 seeds".
3. **The gap is mostly the sequential arm collapsing, not the joint arm being good.** Sequential reaches NLL 3.69 (160m), 3.78 (70m), 4.59 (14m)
   at 50% of W (PT only), then rises to 13.3-24.5 at 100% after SFT, DPO and RL, which is worse than the random initialization
   (10.84, 10.93, 10.99 nats). Joint methods end at 4.08-4.86, i.e. worse than sequential's own pre-post-training checkpoint.
   So "joint avoids destroying the language model that sequential's post-training destroys" is what the primary endpoint measures here.
   14m diagnosis (added 10-23, 3 seeds): the collapse is not an RL artifact and only partly a learning-rate effect. SFT alone already breaks it: NLL 10.76 after the SFT stage at the original lr (13.37 after SFT to the full budget) against 4.54 at the PT checkpoint, before any DPO or RL. Post-training at 0.1x lr roughly halves the damage (6.50 after SFT, 11.85 at the end) but no branch returns to 4.54. Whether the DPO stage is itself buggy is still open (its loss starts at 11-21 instead of 0.69), and this was run at 14m only, not 70m or 160m (section "Collapse diagnosis" below).
4. **Secondary endpoints are at floor or flat.** GSM8K is 0-3% in every cell (joint 2.1-2.6% mean vs sequential 1.0-1.2%; fixed 269 and smooth 289 vs sequential 132
   correct of 9 x 1,319 = 11,871 per arm); this is language learning at floor, not a reasoning result. LAMBADA accuracy
   is 0 in 21 of 27 cells and 0.0002 (1 of 5,153) in 6, so LAMBADA is uninformative; ARC-Easy is 0.252-0.275 (chance 0.25 for 4 choices; paired
   differences straddle 0); UltraFeedback preference accuracy is 0.430-0.441 for every method.
5. fixed_joint vs smooth_joint: no consistent difference (WikiText NLL: fixed lower by 0.02, 0.07, 0.10 at 14m, 70m, 160m; same sign
   3/3 seeds at 70m and 160m only, the sign flips across seeds at 14m).

## Terminal status of all 27 cells (denominator 27)

All 27 are `complete`. The first attempt of every cell (arrays 48765768 and 48765798) crashed on a logging bug: a skipped
fp16 step has a non-finite grad norm, which the log writer rejected. The fix (commit fe2fb96) records `grad_norm: null` plus a flag
for those steps and changes no optimizer, scaler or data behaviour; the cells then resumed and finished (reruns 48796648 for
160m and 48796841 / 49014697 for 14m and 70m; the failed first attempts stay in the job-id column and are not dropped). Per-cell job ids,
charged work, overshoot, observed PT tokens, skipped steps and endpoints:

| cell | job ids (first attempt, resumed run) | work charged | overshoot | observed PT tokens | fp16 skipped steps | train+eval wall h | WikiText NLL | ARC-E acc | GSM8K acc [95% Wilson] (n=1319) | UltraFeedback pref acc (n=1985) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| v1/160m/sequential_s1 | 48765768_0, 48796648_0 | 290,006,224 | 6,224 | 53,182,464 | 6 | 5.04 | 17.328 | 0.266 | 0.0129 [0.0081, 0.0205] | 0.432 |
| v1/160m/fixed_joint_s1 | 48765768_1, 48796648_1 | 290,083,560 | 83,560 | 78,086,144 | 5 | 1.18 | 4.072 | 0.259 | 0.0212 [0.0147, 0.0305] | 0.434 |
| v1/160m/smooth_joint_s1 | 48765768_2, 48796648_2 | 290,011,664 | 11,664 | 71,532,544 | 6 | 1.95 | 4.084 | 0.259 | 0.0243 [0.0172, 0.0340] | 0.440 |
| v1/160m/sequential_s2 | 48765768_3, 48796648_3 | 290,006,688 | 6,688 | 53,182,464 | 7 | 4.63 | 18.756 | 0.265 | 0.0000 [0.0000, 0.0029] | 0.431 |
| v1/160m/fixed_joint_s2 | 48765768_4, 48796648_4 | 290,049,256 | 49,256 | 77,922,304 | 6 | 1.17 | 4.087 | 0.266 | 0.0265 [0.0191, 0.0367] | 0.432 |
| v1/160m/smooth_joint_s2 | 48765768_5, 48796648_5 | 290,003,192 | 3,192 | 71,368,704 | 5 | 1.92 | 4.216 | 0.275 | 0.0220 [0.0154, 0.0314] | 0.438 |
| v1/160m/sequential_s3 | 48765768_6, 48796648_6 | 290,010,408 | 10,408 | 53,182,464 | 7 | 4.00 | 13.813 | 0.262 | 0.0212 [0.0147, 0.0305] | 0.434 |
| v1/160m/fixed_joint_s3 | 48765768_7, 48796648_7 | 290,094,368 | 94,368 | 78,118,912 | 5 | 1.38 | 4.079 | 0.273 | 0.0250 [0.0179, 0.0349] | 0.437 |
| v1/160m/smooth_joint_s3 | 48765768_8, 48796648_8 | 290,004,544 | 4,544 | 71,401,472 | 6 | 1.96 | 4.233 | 0.265 | 0.0303 [0.0223, 0.0410] | 0.438 |
| v2/14m/sequential_s1 | 48765798_0, 48796841_0 | 290,018,728 | 18,728 | 53,182,464 | 6 | 2.80 | 23.567 | 0.255 | 0.0000 [0.0000, 0.0029] | 0.441 |
| v2/14m/fixed_joint_s1 | 48765798_1, 48796841_1 | 290,067,472 | 67,472 | 78,118,912 | 5 | 0.67 | 4.829 | 0.263 | 0.0220 [0.0154, 0.0314] | 0.432 |
| v2/14m/smooth_joint_s1 | 48765798_2, 49014697_2 | 290,000,504 | 504 | 71,532,544 | 6 | 1.05 | 4.857 | 0.257 | 0.0250 [0.0179, 0.0349] | 0.436 |
| v2/14m/sequential_s2 | 48765798_6, 49014697_6 | 290,001,432 | 1,432 | 53,182,464 | 7 | 2.29 | 20.042 | 0.255 | 0.0265 [0.0191, 0.0367] | 0.430 |
| v2/14m/fixed_joint_s2 | 48765798_7, 49014697_7 | 290,029,096 | 29,096 | 78,086,144 | 5 | 0.52 | 4.812 | 0.267 | 0.0273 [0.0198, 0.0376] | 0.434 |
| v2/14m/smooth_joint_s2 | 48765798_8, 49014697_8 | 290,006,848 | 6,848 | 71,401,472 | 5 | 0.84 | 4.848 | 0.273 | 0.0265 [0.0191, 0.0367] | 0.433 |
| v2/14m/sequential_s3 | 48765798_12, 49014697_12 | 290,005,664 | 5,664 | 53,182,464 | 6 | 2.25 | 13.344 | 0.262 | 0.0091 [0.0052, 0.0158] | 0.431 |
| v2/14m/fixed_joint_s3 | 48765798_13, 49014697_13 | 290,094,000 | 94,000 | 78,086,144 | 6 | 0.53 | 4.863 | 0.266 | 0.0197 [0.0135, 0.0287] | 0.437 |
| v2/14m/smooth_joint_s3 | 48765798_14, 49014697_14 | 290,016,464 | 16,464 | 71,565,312 | 6 | 1.03 | 4.861 | 0.270 | 0.0227 [0.0160, 0.0323] | 0.435 |
| v2/70m/sequential_s1 | 48765798_3, 49014697_3 | 290,005,040 | 5,040 | 53,182,464 | 6 | 2.38 | 22.299 | 0.261 | 0.0000 [0.0000, 0.0029] | 0.437 |
| v2/70m/fixed_joint_s1 | 48765798_4, 49014697_4 | 290,050,320 | 50,320 | 78,053,376 | 5 | 0.69 | 4.246 | 0.261 | 0.0167 [0.0110, 0.0251] | 0.439 |
| v2/70m/smooth_joint_s1 | 48765798_5, 49014697_5 | 290,025,432 | 25,432 | 71,434,240 | 5 | 1.19 | 4.309 | 0.253 | 0.0250 [0.0179, 0.0349] | 0.440 |
| v2/70m/sequential_s2 | 48765798_9, 49014697_9 | 290,010,952 | 10,952 | 53,182,464 | 7 | 3.00 | 17.918 | 0.272 | 0.0303 [0.0223, 0.0410] | 0.432 |
| v2/70m/fixed_joint_s2 | 48765798_10, 49014697_10 | 290,055,352 | 55,352 | 77,987,840 | 5 | 0.87 | 4.219 | 0.271 | 0.0243 [0.0172, 0.0340] | 0.435 |
| v2/70m/smooth_joint_s2 | 48765798_11, 49014697_11 | 290,064,016 | 64,016 | 71,335,936 | 5 | 1.00 | 4.309 | 0.267 | 0.0220 [0.0154, 0.0314] | 0.436 |
| v2/70m/sequential_s3 | 48765798_15, 49014697_15 | 290,000,280 | 280 | 53,182,464 | 6 | 2.45 | 24.543 | 0.252 | 0.0000 [0.0000, 0.0029] | 0.436 |
| v2/70m/fixed_joint_s3 | 48765798_16, 49014697_16 | 290,010,992 | 10,992 | 78,053,376 | 6 | 0.86 | 4.242 | 0.266 | 0.0212 [0.0147, 0.0305] | 0.438 |
| v2/70m/smooth_joint_s3 | 48765798_17, 49014697_17 | 290,090,064 | 90,064 | 71,434,240 | 7 | 1.02 | 4.297 | 0.265 | 0.0212 [0.0147, 0.0305] | 0.437 |

Total train+eval wall time 48.7 GPU-h across the 27 cells. Devices: V100-PCIE-32GB, V100S-PCIE-32GB, V100-SXM2-32GB (same fp16 recipe).
Charged work per cell is 290,000,280 to 290,094,368 (target 290,000,000, overshoot at most 0.03%).

## Per-method means by size (3 seeds, terminal checkpoint)

| size | method | WikiText NLL mean (SD) [per-seed] | ARC-E acc | GSM8K acc | pref acc | observed PT tokens (M) |
|---|---|---|---|---|---|---:|
| 14m | sequential | 18.985 (5.193) [23.57, 20.04, 13.34] | 0.257 | 0.0119 | 0.434 | 53.2 |
| 14m | fixed_joint | 4.835 (0.026) [4.83, 4.81, 4.86] | 0.265 | 0.0230 | 0.434 | 78.1 |
| 14m | smooth_joint | 4.855 (0.007) [4.86, 4.85, 4.86] | 0.267 | 0.0248 | 0.435 | 71.5 |
| 70m | sequential | 21.587 (3.370) [22.30, 17.92, 24.54] | 0.262 | 0.0101 | 0.435 | 53.2 |
| 70m | fixed_joint | 4.236 (0.015) [4.25, 4.22, 4.24] | 0.266 | 0.0207 | 0.437 | 78.0 |
| 70m | smooth_joint | 4.305 (0.007) [4.31, 4.31, 4.30] | 0.262 | 0.0227 | 0.438 | 71.4 |
| 160m | sequential | 16.632 (2.544) [17.33, 18.76, 13.81] | 0.264 | 0.0114 | 0.432 | 53.2 |
| 160m | fixed_joint | 4.079 (0.008) [4.07, 4.09, 4.08] | 0.266 | 0.0243 | 0.434 | 78.0 |
| 160m | smooth_joint | 4.178 (0.082) [4.08, 4.22, 4.23] | 0.266 | 0.0255 | 0.439 | 71.4 |

## Paired differences by seed, joint minus sequential (the readout)

The protocol (field 12) fixes a descriptive analysis: with 3 seeds an exact paired sign-flip test cannot go below two-sided p = 0.25.
The 95% t-interval (df = 2) is shown only to size the uncertainty; it is not a test. Seeds are 1, 2, 3, paired.

| size | contrast | metric | seed 1, 2, 3 | mean | range | 95% t-interval of the mean (df=2, descriptive) | seeds favouring joint | protocol verdict |
|---|---|---|---|---:|---|---|---|---|
| 14m | fixed_joint - sequential | WikiText NLL (primary; negative = joint better) | -18.7385, -15.2303, -8.4812 | -14.1500 | [-18.7385, -8.4812] | [-27.1015, -1.1984] | 3/3 | joint better |
| 14m | fixed_joint - sequential | ARC-E acc (positive = joint better) | +0.0080, +0.0114, +0.0038 | +0.0077 | [+0.0038, +0.0114] | [-0.0017, +0.0171] | 3/3 |  |
| 14m | fixed_joint - sequential | GSM8K acc (positive = joint better) | +0.0220, +0.0008, +0.0106 | +0.0111 | [+0.0008, +0.0220] | [-0.0153, +0.0375] | 3/3 |  |
| 14m | fixed_joint - sequential | pref acc (positive = joint better) | -0.0091, +0.0040, +0.0055 | +0.0002 | [-0.0091, +0.0055] | [-0.0198, +0.0201] | 2/3 |  |
| 14m | smooth_joint - sequential | WikiText NLL (primary; negative = joint better) | -18.7108, -15.1943, -8.4831 | -14.1294 | [-18.7108, -8.4831] | [-27.0388, -1.2200] | 3/3 | joint better |
| 14m | smooth_joint - sequential | ARC-E acc (positive = joint better) | +0.0025, +0.0173, +0.0084 | +0.0094 | [+0.0025, +0.0173] | [-0.0090, +0.0278] | 3/3 |  |
| 14m | smooth_joint - sequential | GSM8K acc (positive = joint better) | +0.0250, +0.0000, +0.0136 | +0.0129 | [+0.0000, +0.0250] | [-0.0182, +0.0440] | 2/3 |  |
| 14m | smooth_joint - sequential | pref acc (positive = joint better) | -0.0050, +0.0035, +0.0040 | +0.0008 | [-0.0050, +0.0040] | [-0.0118, +0.0135] | 2/3 |  |
| 70m | fixed_joint - sequential | WikiText NLL (primary; negative = joint better) | -18.0533, -13.6993, -20.3007 | -17.3511 | [-20.3007, -13.6993] | [-25.6892, -9.0130] | 3/3 | joint better |
| 70m | fixed_joint - sequential | ARC-E acc (positive = joint better) | -0.0004, -0.0004, +0.0139 | +0.0043 | [-0.0004, +0.0139] | [-0.0162, +0.0249] | 1/3 |  |
| 70m | fixed_joint - sequential | GSM8K acc (positive = joint better) | +0.0167, -0.0061, +0.0212 | +0.0106 | [-0.0061, +0.0212] | [-0.0257, +0.0469] | 2/3 |  |
| 70m | fixed_joint - sequential | pref acc (positive = joint better) | +0.0015, +0.0035, +0.0025 | +0.0025 | [+0.0015, +0.0035] | [+0.0000, +0.0050] | 3/3 |  |
| 70m | smooth_joint - sequential | WikiText NLL (primary; negative = joint better) | -17.9897, -13.6094, -20.2460 | -17.2817 | [-20.2460, -13.6094] | [-25.6650, -8.8984] | 3/3 | joint better |
| 70m | smooth_joint - sequential | ARC-E acc (positive = joint better) | -0.0076, -0.0051, +0.0126 | -0.0000 | [-0.0076, +0.0126] | [-0.0273, +0.0273] | 1/3 |  |
| 70m | smooth_joint - sequential | GSM8K acc (positive = joint better) | +0.0250, -0.0083, +0.0212 | +0.0126 | [-0.0083, +0.0250] | [-0.0327, +0.0580] | 2/3 |  |
| 70m | smooth_joint - sequential | pref acc (positive = joint better) | +0.0030, +0.0045, +0.0010 | +0.0029 | [+0.0010, +0.0045] | [-0.0015, +0.0073] | 3/3 |  |
| 160m | fixed_joint - sequential | WikiText NLL (primary; negative = joint better) | -13.2561, -14.6688, -9.7344 | -12.5531 | [-14.6688, -9.7344] | [-18.8664, -6.2398] | 3/3 | joint better |
| 160m | fixed_joint - sequential | ARC-E acc (positive = joint better) | -0.0067, +0.0017, +0.0109 | +0.0020 | [-0.0067, +0.0109] | [-0.0200, +0.0239] | 2/3 |  |
| 160m | fixed_joint - sequential | GSM8K acc (positive = joint better) | +0.0083, +0.0265, +0.0038 | +0.0129 | [+0.0038, +0.0265] | [-0.0170, +0.0428] | 3/3 |  |
| 160m | fixed_joint - sequential | pref acc (positive = joint better) | +0.0015, +0.0015, +0.0030 | +0.0020 | [+0.0015, +0.0030] | [-0.0002, +0.0042] | 3/3 |  |
| 160m | smooth_joint - sequential | WikiText NLL (primary; negative = joint better) | -13.2439, -14.5397, -9.5798 | -12.4545 | [-14.5397, -9.5798] | [-18.8454, -6.0636] | 3/3 | joint better |
| 160m | smooth_joint - sequential | ARC-E acc (positive = joint better) | -0.0067, +0.0101, +0.0025 | +0.0020 | [-0.0067, +0.0101] | [-0.0190, +0.0229] | 2/3 |  |
| 160m | smooth_joint - sequential | GSM8K acc (positive = joint better) | +0.0114, +0.0220, +0.0091 | +0.0142 | [+0.0091, +0.0220] | [-0.0029, +0.0312] | 3/3 |  |
| 160m | smooth_joint - sequential | pref acc (positive = joint better) | +0.0076, +0.0071, +0.0040 | +0.0062 | [+0.0040, +0.0076] | [+0.0015, +0.0110] | 3/3 |  |

## Joint-minus-sequential gap against model size (primary endpoint)

| size (params) | fixed_joint - sequential NLL (mean, range) | smooth_joint - sequential NLL (mean, range) | sequential NLL mean | fixed_joint NLL mean | smooth_joint NLL mean |
|---|---|---|---|---|---|
| 14m | -14.15 (-18.74 to -8.48) | -14.13 (-18.71 to -8.48) | 18.98 | 4.835 | 4.855 |
| 70m | -17.35 (-20.30 to -13.70) | -17.28 (-20.25 to -13.61) | 21.59 | 4.236 | 4.305 |
| 160m | -12.55 (-14.67 to -9.73) | -12.45 (-14.54 to -9.58) | 16.63 | 4.079 | 4.178 |

Read: joint is better at every size, the gap is 12.5 to 17.4 nats and varies more across seeds (per-size seed range 4.9 to 10.3 nats) than across sizes,
so no size trend can be claimed. Sequential's NLL is itself unstable across seeds (SD 2.5 to 5.2 nats), the joint arms' SD is 0.007 to 0.08.

## Learning curve: WikiText NLL at each checkpoint (mean of 3 seeds)

| size | method | NLL at 0% | 25% | 50% | 75% | 100% (mean of 3 seeds) |
|---|---|---:|---:|---:|---:|---:|
| 14m | sequential | 10.842 | 5.022 | 4.589 | 10.774 | 18.985 |
| 14m | fixed_joint | 10.842 | 5.581 | 5.117 | 4.913 | 4.835 |
| 14m | smooth_joint | 10.842 | 5.451 | 5.046 | 4.891 | 4.855 |
| 70m | sequential | 10.927 | 4.285 | 3.778 | 10.083 | 21.587 |
| 70m | fixed_joint | 10.927 | 5.085 | 4.631 | 4.363 | 4.236 |
| 70m | smooth_joint | 10.927 | 4.966 | 4.565 | 4.372 | 4.305 |
| 160m | sequential | 10.987 | 4.292 | 3.690 | 8.980 | 16.632 |
| 160m | fixed_joint | 10.987 | 5.061 | 4.546 | 4.223 | 4.079 |
| 160m | smooth_joint | 10.987 | 4.950 | 4.495 | 4.253 | 4.178 |

Sequential's schedule is PT to 55% of W, SFT to 75%, DPO to 85%, RL after. Its NLL is lowest at 50% (PT only), already rises by 75%
(the SFT stage) and is far above random init at 100%. Joint methods improve monotonically but are behind sequential's PT-only checkpoint
at 50%: at matched work they spent a share of W on non-PT objectives, so they saw 71 to 78M PT tokens (vs 53M) and still lag the 50%
sequential point on this endpoint (they are not matched on work against that checkpoint).

## GSM8K pooled counts (full 1,319 per cell, parse failures counted wrong)

| size | method | GSM8K correct / 3 x 1,319 |
|---|---|---|
| 14m | sequential | 47 / 3957 |
| 14m | fixed_joint | 91 / 3957 |
| 14m | smooth_joint | 98 / 3957 |
| 70m | sequential | 40 / 3957 |
| 70m | fixed_joint | 82 / 3957 |
| 70m | smooth_joint | 90 / 3957 |
| 160m | sequential | 45 / 3957 |
| 160m | fixed_joint | 96 / 3957 |
| 160m | smooth_joint | 101 / 3957 |

GSM8K is at the floor for every arm (0 to 3% per cell); joint 82-101 correct per 3,957 (3 seeds x 1,319) vs sequential 40-47, but this is a floor
effect on a 14m to 160m model trained on 118M-token WikiText, not a reasoning or retention claim (PLAN.md line 73).

## RL signal

RL groups with nonzero reward variance (informative) across the 9 cells of each method: sequential 1,930 informative vs 129,354
zero-variance; fixed_joint 2,222 vs 21,158; smooth_joint 5,127 vs 45,721. RL gets a gradient from about 1.5% (sequential), 9.5% (fixed) and 10%
(smooth) of its groups; at these sizes GSM8K reward is almost always 0.

## What this experiment does and does not establish

- Establishes: at matched charged work and at least 50M observed PT tokens, from random init, with these schedules and fp16 on V100s,
  fixed_joint and smooth_joint end with far better WikiText NLL than the sequential pipeline at 14m, 70m and 160m, 3/3 seeds each.
- Does not establish: that joint training yields better reasoning (GSM8K at floor), a size trend (non-monotone, overlapping ranges),
  that the sequential collapse is intrinsic to staged training rather than to this recipe (at 14m it is not an RL artifact and only partly a learning-rate effect, SFT alone breaks it, and the DPO stage is still unchecked; sequential's pre-post-training
  checkpoint is better than any joint endpoint), or anything about LAMBADA (floor).
- Not run: bf16 or any other cluster; more seeds. The lower-post-training-lr and SFT-only diagnostic is now run for 14m (section "Collapse diagnosis" below).

Run artifacts: Sherlock `$SCRATCH/ut_p05` (`collect/collect.{json,md}`, `runs/*/*/*/{receipt,slurm}.json`, `eval/*/summary.json`), results tarball
`p05_results.tgz` sha256 `32e9aa2cb1951843543aa27709183eb3afa76bca5d3b1b0847d6fd8f3cd379dd` (153,035 bytes; summaries only, no checkpoints).

## Collapse diagnosis: stage-by-stage WikiText NLL, 14m, seeds 1, 2, 3 (12 branch runs, all complete)

**Answer: the collapse is not an RL artifact and only partly a learning-rate effect. Whether the DPO stage is itself a recipe or implementation bug is still open (its loss starts at 11-21 instead of 0.69).**

Setup: per seed, the sequential pre-training stage is trained once (progress < 0.55, same frozen 14m fp16 config as the 27-cell run) and four
branches start from that same checkpoint, so after-PT NLL is identical across branches. One A4000 (mercury1 GPU 9). WikiText NLL (nats per
token) is read after every stage. Branches: `orig` (SFT, DPO, RL as before), `lowlr` (same, post-training lr x 0.1: SFT peak 4.86e-5 vs 4.86e-4),
`sftonly` (SFT continued to the full 6,522-update charged-work target, no DPO or RL), `ptonly` (control: keep pre-training to the target).

| branch | after PT | after SFT stage | after DPO | after RL / end | end, seeds 1 / 2 / 3 |
|---|---|---|---|---|---|
| ptonly (control) | 4.54 | - | - | 4.35 | 4.35 / 4.35 / 4.34 |
| sftonly (SFT to full budget) | 4.54 | 13.37 (end of full-budget SFT) | - | 13.37 | 13.42 / 13.25 / 13.44 |
| orig (lr x 1) | 4.54 | 10.76 (shorter SFT stage) | 10.65 | 21.46 | 17.10 / 16.39 / 30.91 |
| lowlr (lr x 0.1) | 4.54 | 6.50 (shorter SFT stage) | 10.37 | 11.85 | 8.71 / 12.17 / 14.69 |

Means of 3 seeds; per-seed values and per-stage training statistics are in `diag_summary.json`. Random-init NLL is about 10.8-11.0. The
`sftonly` row is not the same stage as the `orig` and `lowlr` after-SFT cells: it is SFT run to the full budget, the others stop after the shorter SFT stage.

Readout:
- Not an RL artifact: `orig` is already at 10.76 after its shorter SFT stage, before DPO or RL (3/3 seeds, 10.67-10.85), and SFT to the full budget reaches 13.37
  (3/3 seeds), above random init. Continuing pre-training on the same budget improves NLL to 4.35 (3/3 seeds).
- Only partly an lr effect (compared only between `orig` and `lowlr`): after the SFT stage 6.50 vs 10.76 (3/3 seeds lower), still +2.0 nats over the
  PT checkpoint; after DPO both are about 10.4-10.7 (random-init level); after RL `lowlr` is 11.85 vs 21.46 (lower on 3/3 seeds).
- No branch gets back to the PT-checkpoint NLL of 4.54: 0 of 9 post-training branch endpoints (3 branches x 3 seeds) is below it.

Verdict: forgetting of the pre-training distribution under SFT, DPO and RL post-training with no pre-training replay appears under every post-training
variant tried, and its size depends on the recipe (lr x 0.1 cuts the SFT jump from +6.2 to +2.0 nats and the RL-end mean from 21.46 to 11.85).
The DPO verdict is open: the stage starts at a loss of 11-21 (mean of the first 20 updates), not the ln 2 = 0.69 expected when policy equals
reference, and its gradient norms are about 95-100 (`orig`) and 190-215 (`lowlr`) against about 2 in SFT. A reference-mismatch check on the DPO loss is the next diagnostic.

Caveats:
- 14m only (no 70m or 160m), 3 seeds, one run each.
- Seed spread after RL is large (`orig` 16.4-30.9; `lowlr` seed 3 RL loss unstable, first-20 mean 12.17, last-20 mean -5.50), so the RL-end means are not tight.
- fp16 skipped 6-7 optimizer steps in each `orig` and `lowlr` run and none in `sftonly` (receipts).

Files: `p05/diag.py`, `p05/diag_eval.py`, `diag_summary.json`, `analyze_diag.py`; raw runs on mercury1 `/lfs/mercury1/0/brando9/at1024/runs`.

**TLDR-end (diagnosis):** [unified-training: Exp 05 diagnosis] At 14m, 3 seeds: orig is already at 10.76 NLL after its shorter SFT stage and SFT to the full budget reaches 13.37, against 4.54 at the PT checkpoint, so it is not an RL artifact. Lr x 0.1 roughly halves the damage (RL-end mean 11.85 vs 21.46) but nothing recovers it. The DPO stage may be a recipe bug (loss starts at 11-21, not 0.69) and is still open.
