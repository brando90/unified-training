# Exp 06 analysis tables

Final-step top-1 accuracy (%), mean [95% t interval] over complete seeds. Contrasts are paired by seed;
H1 contrasts use 97.5% intervals (Bonferroni over the two joint arms). p-val=n/a (intervals, no test).

## ViT-Tiny

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

**Scale check:** H1 verdict differs at ViT-Small

## Cells not complete

- none
