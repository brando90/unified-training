# Local CPU smoke output (2026-10-20, cpu_smoke.sh, synthetic data, 'smoke' ViT 32/2/2, S=100, B=24; not a measurement)

prep done in 1.1 s -> <workdir>/data/manifest.json
target_only        seed1 complete src=0 tgt=2400 loss_src first->last=None loss_tgt=(4.138, 2.763) cifar=86.75 tin=0.0
target_only        seed2 complete src=0 tgt=2400 loss_src first->last=None loss_tgt=(4.118, 2.703) cifar=78.25 tin=0.0
target_only        seed3 complete src=0 tgt=2400 loss_src first->last=None loss_tgt=(4.068, 2.708) cifar=49.5 tin=0.0
sequential         seed1 complete src=1200 tgt=1200 loss_src first->last=(4.752, 4.116) loss_tgt=(4.058, 3.493) cifar=46.0 tin=10.25
sequential         seed2 complete src=1200 tgt=1200 loss_src first->last=(4.691, 4.029) loss_tgt=(4.113, 3.553) cifar=53.5 tin=9.75
sequential         seed3 complete src=1200 tgt=1200 loss_src first->last=(4.648, 3.979) loss_tgt=(4.057, 3.5) cifar=47.25 tin=9.75
sequential_replay  seed1 complete src=1200 tgt=1200 loss_src first->last=(4.769, 3.479) loss_tgt=(4.378, 3.317) cifar=65.25 tin=10.25
sequential_replay  seed2 complete src=1200 tgt=1200 loss_src first->last=(4.709, 3.407) loss_tgt=(4.434, 3.35) cifar=53.75 tin=10.5
sequential_replay  seed3 complete src=1200 tgt=1200 loss_src first->last=(4.666, 3.362) loss_tgt=(4.377, 3.312) cifar=40.0 tin=9.75
fixed_joint        seed1 complete src=1200 tgt=1200 loss_src first->last=(4.826, 3.217) loss_tgt=(4.18, 2.919) cifar=46.25 tin=22.75
fixed_joint        seed2 complete src=1200 tgt=1200 loss_src first->last=(4.822, 3.188) loss_tgt=(4.237, 2.89) cifar=47.0 tin=23.0
fixed_joint        seed3 complete src=1200 tgt=1200 loss_src first->last=(4.772, 3.165) loss_tgt=(4.181, 2.92) cifar=29.75 tin=21.25
smooth_joint       seed1 complete src=1200 tgt=1200 loss_src first->last=(4.821, 3.277) loss_tgt=(4.305, 2.952) cifar=28.25 tin=29.25
smooth_joint       seed2 complete src=1200 tgt=1200 loss_src first->last=(4.756, 3.198) loss_tgt=(4.39, 2.951) cifar=29.5 tin=41.5
smooth_joint       seed3 complete src=1200 tgt=1200 loss_src first->last=(4.719, 3.182) loss_tgt=(4.34, 2.952) cifar=34.5 tin=33.25
stop-after rc=3 3 (expect 3 3)
resumed <workdir>/resume at step 40
resumed <workdir>/resume at step 57
RESUME EXACT: curve and final metrics identical to the uninterrupted run
<workdir>/resume: already complete
# Exp 06 analysis tables

Final-step top-1 accuracy (%), mean [95% t interval] over complete seeds. Contrasts are paired by seed;
H1 contrasts use 97.5% intervals (Bonferroni over the two joint arms). p-val=n/a (intervals, no test).

## ViT-Smoke

Cells: complete 15 of 15

| Arm | Complete | CIFAR-100 test top-1 | Tiny-ImageNet val top-1 | img/s | train h |
|---|---:|---|---|---:|---:|
| target_only | 3/3 | 71.50 [23.01, 119.99] (n=3) | 0.00 [0.00, 0.00] (n=3) | 70 | 0.01 |
| sequential | 3/3 | 48.92 [38.93, 58.90] (n=3) | 9.92 [9.20, 10.63] (n=3) | 104 | 0.01 |
| sequential_replay | 3/3 | 53.00 [21.60, 84.40] (n=3) | 10.17 [9.22, 11.12] (n=3) | 83 | 0.01 |
| fixed_joint | 3/3 | 41.00 [16.78, 65.22] (n=3) | 22.33 [19.98, 24.68] (n=3) | 80 | 0.01 |
| smooth_joint | 3/3 | 30.75 [22.53, 38.97] (n=3) | 34.67 [19.15, 50.18] (n=3) | 97 | 0.01 |

Best joint arm (higher mean CIFAR-100): **fixed_joint**

| Contrast (a - b, pp) | Level | Seeds | Mean [interval] |
|---|---|---|---|
| H1_best_joint_minus_sequential_cifar (fixed_joint - sequential) | 97.5% (Bonferroni over 2 joint arms) | [1, 2, 3] | -7.92 [-40.01, 24.18] (n=3) |
| H2_best_joint_minus_sequential_tin (fixed_joint - sequential) | 95% | [1, 2, 3] | 12.42 [10.24, 14.60] (n=3) |
| best_joint_minus_sequential_replay_cifar (fixed_joint - sequential_replay) | 95% | [1, 2, 3] | -12.00 [-27.67, 3.67] (n=3) |
| best_joint_minus_sequential_replay_tin (fixed_joint - sequential_replay) | 95% | [1, 2, 3] | 12.17 [10.73, 13.60] (n=3) |
| best_joint_minus_target_only_cifar (fixed_joint - target_only) | 95% | [1, 2, 3] | -30.50 [-56.32, -4.68] (n=3) |
| sequential_minus_target_only_cifar (sequential - target_only) | 95% | [1, 2, 3] | -22.58 [-70.63, 25.46] (n=3) |
| fixed_joint_minus_sequential_cifar (fixed_joint - sequential) | 97.5% (Bonferroni over 2 joint arms) | [1, 2, 3] | -7.92 [-40.01, 24.18] (n=3) |
| smooth_joint_minus_sequential_cifar (smooth_joint - sequential) | 97.5% (Bonferroni over 2 joint arms) | [1, 2, 3] | -18.17 [-38.36, 2.03] (n=3) |

**H1 (joint >= sequential on CIFAR-100):** inconclusive  
**H2 (joint retains more Tiny-ImageNet):** supported

## Cells not complete

- none
analysis.json
curves_smoke.png
tables.md
CPU SMOKE PASS
