# Pilot results

**Status:** Not admitted; review and resource gates remain pending. Missing metrics are unavailable, not zero.

| Method | Seed | Status | Training work (million forward-equivalent tokens) | ARC-Easy accuracy | GSM8K exact match | WikiText NLL | Preference raw / implicit accuracy |
|---|---:|---|---:|---:|---:|---:|---|
| sequential | 0 | pending | unavailable | unavailable | unavailable | unavailable | unavailable |
| sequential | 1 | pending | unavailable | unavailable | unavailable | unavailable | unavailable |
| sequential | 2 | pending | unavailable | unavailable | unavailable | unavailable | unavailable |
| parallel_joint | 0 | pending | unavailable | unavailable | unavailable | unavailable | unavailable |
| parallel_joint | 1 | pending | unavailable | unavailable | unavailable | unavailable | unavailable |
| parallel_joint | 2 | pending | unavailable | unavailable | unavailable | unavailable | unavailable |
| fixed_joint | 0 | pending | unavailable | unavailable | unavailable | unavailable | unavailable |
| fixed_joint | 1 | pending | unavailable | unavailable | unavailable | unavailable | unavailable |
| fixed_joint | 2 | pending | unavailable | unavailable | unavailable | unavailable | unavailable |
| smooth_joint | 0 | pending | unavailable | unavailable | unavailable | unavailable | unavailable |
| smooth_joint | 1 | pending | unavailable | unavailable | unavailable | unavailable | unavailable |
| smooth_joint | 2 | pending | unavailable | unavailable | unavailable | unavailable | unavailable |
| aioli_objective | 0 | pending | unavailable | unavailable | unavailable | unavailable | unavailable |
| aioli_objective | 1 | pending | unavailable | unavailable | unavailable | unavailable | unavailable |
| aioli_objective | 2 | pending | unavailable | unavailable | unavailable | unavailable | unavailable |
| validation_progress | 0 | pending | unavailable | unavailable | unavailable | unavailable | unavailable |
| validation_progress | 1 | pending | unavailable | unavailable | unavailable | unavailable | unavailable |
| validation_progress | 2 | pending | unavailable | unavailable | unavailable | unavailable | unavailable |
| chord_objective | 0 | pending | unavailable | unavailable | unavailable | unavailable | unavailable |
| chord_objective | 1 | pending | unavailable | unavailable | unavailable | unavailable | unavailable |
| chord_objective | 2 | pending | unavailable | unavailable | unavailable | unavailable | unavailable |
| rpt_inspired | 0 | pending | unavailable | unavailable | unavailable | unavailable | unavailable |
| rpt_inspired | 1 | pending | unavailable | unavailable | unavailable | unavailable | unavailable |
| rpt_inspired | 2 | pending | unavailable | unavailable | unavailable | unavailable | unavailable |

NLL = negative log likelihood; preference raw ordering is the preference endpoint, implicit ordering is diagnostic. Scratch primary: WikiText NLL; early-checkpoint primary: ARC-Easy normalized accuracy. Unassisted GSM8K remains a sparse secondary diagnostic.

Intervals: 95% Wilson binary-item score, conditional on each trained model; p-val=n/a (no hypothesis test). Three seeds do not establish universal superiority. Aioli, CHORD (Controllable Harmonization of On- and Off-Policy Reinforcement Learning via Dynamic Weighting), and RPT (Reinforcement Pre-Training) are declared adaptations; exact CHERRY-RL identity remains unresolved.

Complete: 0/24. See `expt_v1/PROTOCOL.md` and the program review reconciliation.

Seed uncertainty, two-stage seed/item intervals, initial-checkpoint changes and all paired comparisons: `expt_v1/analysis.json`. Missing trained seeds remain missing with separate accuracy sensitivity bounds. p-val=n/a throughout; no formal superiority verdict.
