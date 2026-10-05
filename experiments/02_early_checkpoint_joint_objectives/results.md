# Pilot results

**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/02_early_checkpoint_joint_objectives/results.md>

**Status:** COMPLETE — 24/24 training cells and 1/1 untouched initial evaluation; zero failed or interrupted cells.

**Last updated:** 10-04-2026 23:11 PDT.

In `experiments/02_early_checkpoint_joint_objectives/`, the proposed validation-progress controller achieved normalized ARC-Easy (AI2 Reasoning Challenge, Easy subset) accuracy **34.2593% [30.5290%, 37.9896%]**, versus sequential **28.4231% [23.9771%, 32.8691%]** and Aioli **34.2031% [31.1562%, 37.2501%]**. These are three-seed means with 95% Student's t intervals (two degrees of freedom); sample standard deviations are 1.5016, 1.7898, and 1.2266 percentage points, respectively; p-val=n/a (no hypothesis test). The paired proposed-minus-sequential difference is **+5.8361 [+2.6886, +8.9837] percentage points**, and proposed-minus-Aioli is **+0.0561 [−1.8210, +1.9332] percentage points** under the same seed-level interval method. The pilot does not establish an advantage over Aioli or all baselines.

The untouched Pythia-160m step10000 checkpoint scored **37.2475% [35.3256%, 39.2106%]** (95% Wilson item interval; 2,376 items; p-val=n/a), above every trained cell's observed accuracy. Proposed changes from that shared initial checkpoint were **−1.2626 [−3.0724, +0.3788]**, **−3.9983 [−5.8081, −2.2727]**, and **−3.7037 [−5.5135, −1.9360] percentage points** across seeds 0–2. These 95% paired-item bootstrap intervals are conditional on the models and do not represent seed uncertainty; p-val=n/a. The early checkpoint supplies measurable accuracy, but these training conditions did not improve it over the untouched starting point.

See the [final program report](../00_program/FINAL_REPORT.md) for all methods, paired contrasts, initial diagnostics and scope limits. Requested plan and implementation reviews retain their original **FAIL** verdicts; the implementer marked findings **FIXED** in the [review reconciliation](../00_program/REVIEW_RECONCILIATION.md), supported by 63 deterministic tests. No post-fix reviewer PASS is claimed.

| Phase | Status | Verified denominator |
|---|---|---|
| Frozen training matrix | DONE | 24/24 cells; zero failed or interrupted |
| Full terminal evaluations | DONE | 24/24 |
| Untouched initial evaluation | DONE | 1/1 shared checkpoint |
| Primary aggregates and paired initial comparisons | DONE | Eight three-seed aggregates; 24/24 initial comparisons |

| Method | Seed | Status | Training work (million forward-equivalent tokens) | ARC-Easy accuracy | GSM8K exact match | WikiText NLL | Preference raw / implicit accuracy |
|---|---:|---|---:|---:|---:|---:|---|
| sequential | 0 | complete | 19.939 | 0.2976 [0.2795, 0.3163] | 0.0152 [0.0098, 0.0233] | 9.6419 [9.5743, 9.7074] | 0.5312 [0.4701, 0.5915] / 0.3906 [0.3329, 0.4516] |
| sequential | 1 | complete | 19.928 | 0.2639 [0.2466, 0.2820] | 0.0265 [0.0191, 0.0367] | 7.5458 [7.4961, 7.5963] | 0.5781 [0.5169, 0.6370] / 0.5000 [0.4392, 0.5608] |
| sequential | 2 | complete | 19.945 | 0.2912 [0.2733, 0.3098] | 0.0235 [0.0166, 0.0332] | 7.4249 [7.3635, 7.4874] | 0.5078 [0.4469, 0.5685] / 0.4414 [0.3819, 0.5027] |
| parallel_joint | 0 | complete | 19.978 | 0.3481 [0.3292, 0.3674] | 0.0182 [0.0123, 0.0269] | 3.2496 [3.2193, 3.2791] | 0.6484 [0.5881, 0.7043] / 0.6016 [0.5405, 0.6596] |
| parallel_joint | 1 | complete | 19.917 | 0.3451 [0.3263, 0.3645] | 0.0114 [0.0069, 0.0187] | 3.2442 [3.2150, 3.2729] | 0.6016 [0.5405, 0.6596] / 0.5508 [0.4895, 0.6105] |
| parallel_joint | 2 | complete | 19.958 | 0.3497 [0.3308, 0.3692] | 0.0174 [0.0116, 0.0260] | 3.2424 [3.2127, 3.2711] | 0.5859 [0.5248, 0.6446] / 0.4688 [0.4085, 0.5299] |
| fixed_joint | 0 | complete | 19.938 | 0.3464 [0.3275, 0.3657] | 0.0136 [0.0086, 0.0215] | 3.2422 [3.2119, 3.2719] | 0.6055 [0.5445, 0.6634] / 0.5039 [0.4431, 0.5646] |
| fixed_joint | 1 | complete | 19.930 | 0.3396 [0.3209, 0.3589] | 0.0099 [0.0058, 0.0168] | 3.2500 [3.2201, 3.2790] | 0.5898 [0.5287, 0.6483] / 0.5117 [0.4508, 0.5723] |
| fixed_joint | 2 | complete | 19.916 | 0.3615 [0.3425, 0.3811] | 0.0220 [0.0154, 0.0314] | 3.2466 [3.2157, 3.2756] | 0.5977 [0.5366, 0.6559] / 0.5469 [0.4857, 0.6067] |
| smooth_joint | 0 | complete | 19.922 | 0.3523 [0.3333, 0.3717] | 0.0174 [0.0116, 0.0260] | 3.3162 [3.2869, 3.3458] | 0.6094 [0.5484, 0.6671] / 0.5312 [0.4701, 0.5915] |
| smooth_joint | 1 | complete | 19.976 | 0.3401 [0.3213, 0.3594] | 0.0250 [0.0179, 0.0349] | 3.3122 [3.2824, 3.3411] | 0.6406 [0.5802, 0.6969] / 0.5547 [0.4934, 0.6143] |
| smooth_joint | 2 | complete | 19.930 | 0.3548 [0.3358, 0.3743] | 0.0205 [0.0141, 0.0296] | 3.3144 [3.2852, 3.3428] | 0.5703 [0.5091, 0.6295] / 0.5312 [0.4701, 0.5915] |
| aioli_objective | 0 | complete | 19.921 | 0.3535 [0.3346, 0.3730] | 0.0235 [0.0166, 0.0332] | 3.3669 [3.3388, 3.3945] | 0.5820 [0.5208, 0.6408] / 0.5664 [0.5052, 0.6257] |
| aioli_objective | 1 | complete | 19.939 | 0.3291 [0.3105, 0.3483] | 0.0174 [0.0116, 0.0260] | 3.3377 [3.3090, 3.3658] | 0.6250 [0.5643, 0.6820] / 0.4414 [0.3819, 0.5027] |
| aioli_objective | 2 | complete | 19.921 | 0.3434 [0.3246, 0.3628] | 0.0197 [0.0135, 0.0287] | 3.3578 [3.3301, 3.3857] | 0.5898 [0.5287, 0.6483] / 0.5352 [0.4740, 0.5953] |
| validation_progress | 0 | complete | 19.958 | 0.3598 [0.3408, 0.3794] | 0.0190 [0.0129, 0.0278] | 3.3919 [3.3645, 3.4192] | 0.5977 [0.5366, 0.6559] / 0.6094 [0.5484, 0.6671] |
| validation_progress | 1 | complete | 19.938 | 0.3325 [0.3138, 0.3517] | 0.0190 [0.0129, 0.0278] | 3.3887 [3.3604, 3.4169] | 0.5977 [0.5366, 0.6559] / 0.5195 [0.4585, 0.5800] |
| validation_progress | 2 | complete | 19.916 | 0.3354 [0.3167, 0.3547] | 0.0220 [0.0154, 0.0314] | 3.3600 [3.3316, 3.3876] | 0.5781 [0.5169, 0.6370] / 0.5312 [0.4701, 0.5915] |
| chord_objective | 0 | complete | 19.979 | 0.3279 [0.3093, 0.3470] | 0.0212 [0.0147, 0.0305] | 3.2771 [3.2481, 3.3063] | 0.6133 [0.5524, 0.6708] / 0.5938 [0.5326, 0.6521] |
| chord_objective | 1 | complete | 19.934 | 0.3363 [0.3176, 0.3555] | 0.0015 [0.0004, 0.0055] | 3.2573 [3.2277, 3.2856] | 0.6016 [0.5405, 0.6596] / 0.5234 [0.4624, 0.5838] |
| chord_objective | 2 | complete | 19.975 | 0.3266 [0.3080, 0.3457] | 0.0205 [0.0141, 0.0296] | 3.2685 [3.2388, 3.2973] | 0.5977 [0.5366, 0.6559] / 0.6133 [0.5524, 0.6708] |
| rpt_inspired | 0 | complete | 19.929 | 0.3577 [0.3387, 0.3772] | 0.0167 [0.0110, 0.0251] | 3.2360 [3.2053, 3.2659] | 0.6172 [0.5563, 0.6746] / 0.4102 [0.3517, 0.4713] |
| rpt_inspired | 1 | complete | 19.927 | 0.3253 [0.3068, 0.3444] | 0.0197 [0.0135, 0.0287] | 3.2387 [3.2081, 3.2679] | 0.5938 [0.5326, 0.6521] / 0.5156 [0.4546, 0.5762] |
| rpt_inspired | 2 | complete | 19.938 | 0.3510 [0.3321, 0.3704] | 0.0182 [0.0123, 0.0269] | 3.2336 [3.2030, 3.2638] | 0.5898 [0.5287, 0.6483] / 0.4453 [0.3857, 0.5066] |

NLL = negative log likelihood; preference raw ordering is the preference endpoint, implicit ordering is diagnostic. The primary in `experiments/02_early_checkpoint_joint_objectives/` is ARC-Easy normalized accuracy. GSM8K (Grade School Math 8K) unassisted exact match remains a sparse secondary diagnostic.

Per-cell table intervals: 95% Wilson binary-item intervals for accuracies and corpus-block bootstrap intervals for NLL, conditional on each trained model; p-val=n/a (no hypothesis test). Three seeds do not establish universal superiority. Aioli, CHORD (Controllable Harmonization of On- and Off-Policy Reinforcement Learning via Dynamic Weighting), and RPT (Reinforcement Pre-Training) are declared adaptations; exact CHERRY-RL identity remains unresolved.

Complete: 24/24 training cells and 1/1 initial evaluation, with zero failed or interrupted cells. See [frozen protocol](expt_v1/PROTOCOL.md) and [final report](../00_program/FINAL_REPORT.md).

Seed uncertainty, two-stage seed/item intervals, initial-checkpoint changes and all paired comparisons: [analysis.json](expt_v1/analysis.json). All eight methods contain all three declared seeds; no missing cells are omitted from the denominator. p-val=n/a throughout; no formal superiority verdict.

![Terminal tradeoffs](expt_v1/terminal_tradeoffs.png)

**Each point represents one completed seed.** All 24 endpoints are shown against inclusive training work; no frontier superiority is claimed.

![Language learning curves](expt_v1/language_learning_curves.png)

**Controller-set language loss is observed at fixed work milestones.** Curves are descriptive monitoring evidence, separate from development selection and terminal test results.

**TLDR-end:** [unified-training: early-checkpoint results] `experiments/02_early_checkpoint_joint_objectives/` is complete: 24/24 training cells and 1/1 initial evaluation, zero failures. The proposed controller improves accuracy over sequential training, but its Aioli advantage is unresolved and every trained cell's observed accuracy is below the untouched checkpoint.
