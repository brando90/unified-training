# Exp 06 analysis tables

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
