# Unified training

**Doc link:** <https://github.com/brando90/unified-training/blob/main/README.md>

**TLDR:** A research project testing joint objective schedules against staged and modern adapted baselines. The requested Claude Code Opus 5.5 maximum-effort reviews are complete, with findings repaired and deterministically reconciled; resource admission is being finalized; no empirical superiority claim is established.

Test whether mixing pretraining, supervised fine-tuning, preference learning, and reasoning reinforcement learning throughout training improves reasoning while retaining general language capability. The proposed joint method uses validation progress and compute cost to adapt the mixture. Its advantages over staged training remain unproven.

The project has two separate questions: what works from random initialization, and what works when continuing an early pretrained checkpoint. The latter is more affordable and may provide stronger accuracy signals, but cannot establish the former.

- [Current research and execution plan](experiments/00_program/PLAN.md)
- [Related work, refreshed in 2026](experiments/00_program/related_work.md)
- [Experiment index and live status](experiments/README.md)
- [Resumable project checkpoint](experiments/00_program/CKPT_unified_training.md)
- [Original proposal, preserved with limitations](experiments/00_program/original_proposal.md)
- [Original Google Doc proposal](https://docs.google.com/document/d/1j_qSj77AdW0qhZHgQwYfpgLqeumpksd0E67-uj6VL9E/edit)
- [Related zip-mix project](https://github.com/brando90/zip-mix)

The first pilot compares staged training, independent-optimizer parallel averaging and a fixed joint mixture, a smooth curriculum, an Aioli controller adapted to objectives, the proposed validation-progress controller, and explicitly labeled adaptations of CHORD and Reinforcement Pre-Training. The exact paper meant by “cherry-rl” remains unresolved. Named-method adaptations are not reproductions of their published large-model results.

Korbak et al., *Pretraining Language Models with Human Preferences*, motivates the project by demonstrating benefits of incorporating preference information during pretraining in their studied settings. It does not prove that joint reasoning reinforcement learning from random initialization wins, or that mixing objectives prevents forgetting by construction. See the related-work document for primary sources and caveats.

Experiments run on Stanford Network Analysis Project (SNAP) hardware. Code, protocols, and publishable result summaries live with their experiment folders; large models and dataset caches remain in documented, ignored cluster storage. No external model-provider calls, new paid services, or experiment dashboards are required.

The pilot uses WikiText language-modeling loss as the scratch primary and ARC-Easy normalized accuracy as the early-checkpoint primary. Development checks found sparse unassisted math reward; GSM8K is a secondary diagnostic, not evidence of robust reasoning readiness. Both original FAIL reviews and the explicit current-source dispositions are preserved in [review reconciliation](experiments/00_program/REVIEW_RECONCILIATION.md).
