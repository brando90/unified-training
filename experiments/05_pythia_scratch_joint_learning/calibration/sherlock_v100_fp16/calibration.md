# Exp 05 calibration (fp16)

Recommended frozen work target: **2.9e+08** forward-equivalent tokens per cell (max W needed for 50M observed PT tokens x 1.05).

| size | params | objective | s/update | charged work/update | peak GB |
|---|---|---|---|---|---|
| 14m | 14.1M | pt | 0.349 | 98304 | 1.68 |
| 14m | 14.1M | sft | 0.175 | 27523 | 0.73 |
| 14m | 14.1M | dpo | 0.219 | 50778 | 1.45 |
| 14m | 14.1M | rl | 2.297 | 11376 | 0.31 |
| 70m | 70.4M | pt | 0.547 | 98304 | 5.13 |
| 70m | 70.4M | sft | 0.215 | 27523 | 2.39 |
| 70m | 70.4M | dpo | 0.282 | 50778 | 3.67 |
| 70m | 70.4M | rl | 2.882 | 11376 | 1.36 |
| 160m | 162.3M | pt | 0.953 | 98304 | 13.24 |
| 160m | 162.3M | sft | 0.334 | 27523 | 5.88 |
| 160m | 162.3M | dpo | 0.522 | 50778 | 8.71 |
| 160m | 162.3M | rl | 3.737 | 13651 | 3.60 |

| size | method | PT work share | W needed | PT tokens at target | PT epochs | train h/cell |
|---|---|---|---|---|---|---|
| 14m | sequential | 0.550 | 2.73e+08 | 53.2M | 0.45 | 2.73 |
| 14m | fixed_joint | 0.815 | 1.84e+08 | 78.8M | 0.66 | 0.72 |
| 14m | smooth_joint | 0.747 | 2.01e+08 | 72.2M | 0.61 | 1.21 |
| 70m | sequential | 0.550 | 2.73e+08 | 53.2M | 0.45 | 3.48 |
| 70m | fixed_joint | 0.815 | 1.84e+08 | 78.8M | 0.66 | 0.98 |
| 70m | smooth_joint | 0.747 | 2.01e+08 | 72.2M | 0.61 | 1.59 |
| 160m | sequential | 0.550 | 2.73e+08 | 53.2M | 0.45 | 4.02 |
| 160m | fixed_joint | 0.811 | 1.85e+08 | 78.4M | 0.66 | 1.45 |
| 160m | smooth_joint | 0.740 | 2.03e+08 | 71.5M | 0.60 | 2.22 |

Memory fallback needed: 14m=False, 70m=False, 160m=False
