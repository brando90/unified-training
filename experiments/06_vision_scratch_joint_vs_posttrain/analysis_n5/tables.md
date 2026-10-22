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

Cells: complete 25 of 25

| Arm | Complete | CIFAR-100 test top-1 | Tiny-ImageNet val top-1 | img/s | train h |
|---|---:|---|---|---:|---:|
| target_only | 5/5 | 41.83 [40.94, 42.72] (n=5) | 0.49 [0.38, 0.59] (n=5) | 2725 | 1.18 |
| sequential | 5/5 | 42.56 [41.93, 43.19] (n=5) | 6.49 [6.06, 6.92] (n=5) | 2896 | 1.12 |
| sequential_replay | 5/5 | 42.87 [42.47, 43.27] (n=5) | 25.79 [25.17, 26.40] (n=5) | 2953 | 1.10 |
| fixed_joint | 5/5 | 43.28 [43.09, 43.46] (n=5) | 25.64 [24.88, 26.40] (n=5) | 2642 | 1.22 |
| smooth_joint | 5/5 | 46.17 [43.48, 48.86] (n=5) | 27.33 [25.09, 29.56] (n=5) | 2771 | 1.17 |

Best joint arm (higher mean CIFAR-100): **smooth_joint**

| Contrast (a - b, pp) | Level | Seeds | Mean [interval] |
|---|---|---|---|
| H1_best_joint_minus_sequential_cifar (smooth_joint - sequential) | 97.5% (Bonferroni over 2 joint arms) | [1, 2, 3, 4, 5] | 3.61 [0.97, 6.26] (n=5) |
| H2_best_joint_minus_sequential_tin (smooth_joint - sequential) | 95% | [1, 2, 3, 4, 5] | 20.84 [18.73, 22.95] (n=5) |
| best_joint_minus_sequential_replay_cifar (smooth_joint - sequential_replay) | 95% | [1, 2, 3, 4, 5] | 3.30 [0.38, 6.23] (n=5) |
| best_joint_minus_sequential_replay_tin (smooth_joint - sequential_replay) | 95% | [1, 2, 3, 4, 5] | 1.54 [-1.18, 4.27] (n=5) |
| best_joint_minus_target_only_cifar (smooth_joint - target_only) | 95% | [1, 2, 3, 4, 5] | 4.34 [1.33, 7.35] (n=5) |
| sequential_minus_target_only_cifar (sequential - target_only) | 95% | [1, 2, 3, 4, 5] | 0.73 [-0.30, 1.76] (n=5) |
| fixed_joint_minus_sequential_cifar (fixed_joint - sequential) | 97.5% (Bonferroni over 2 joint arms) | [1, 2, 3, 4, 5] | 0.72 [-0.13, 1.56] (n=5) |
| smooth_joint_minus_sequential_cifar (smooth_joint - sequential) | 97.5% (Bonferroni over 2 joint arms) | [1, 2, 3, 4, 5] | 3.61 [0.97, 6.26] (n=5) |

**H1 (joint >= sequential on CIFAR-100):** joint better  
**H2 (joint retains more Tiny-ImageNet):** supported

**Scale check:** same H1 verdict at ViT-Small

## Cells not complete

- none
