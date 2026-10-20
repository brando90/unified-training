# Vision: joint from scratch vs pretrain-then-finetune (ViT, matched compute)

**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/06_vision_scratch_joint_vs_posttrain/README.md>

**Summary:** Small ViTs (ViT-Tiny primary, ViT-Small scale check) trained from scratch with Tiny-ImageNet-200 as the
generic source ("pretraining") and CIFAR-100 as the target ("post-training"), 32px. Five arms at identical optimizer
steps and batch size, with identical source exposure across the four source-using arms: `target_only`, `sequential`
(pretrain then finetune), `sequential_replay`, `fixed_joint` and `smooth_joint`. Three paired seeds, 30 cells.
Question: does joint training from step 0 match or beat pretrain-then-finetune on the target while forgetting less
of the source? Transformers only.

**Status:** PREPARATION — protocol frozen; 30 cells, none launched.
**Last updated:** 10-20-2026 PDT

## Structure

```text
06_vision_scratch_joint_vs_posttrain/
  README.md
  results.md
  expt_v1/
    PROTOCOL.md        frozen design, hypotheses H1/H2, decision rules, budget, failed-cell rules
    cc.md              execution runbook (FarmShare L40S primary, Sherlock V100 fallback)
    cells.tsv          the 30 cells (array tasks 0-29)
    prep_data.py       CIFAR-100 + Tiny-ImageNet -> 32x32 uint8 .npy + sha256 manifest
    train.py           ViT, the five arms, eval curve, resume-safe checkpoints
    analyze.py         full-denominator status, paired-seed t intervals, H1/H2 verdicts, curves
    env.sh, prep.sbatch, smoke.sbatch, cells.sbatch, post.sbatch, submit_exp06.sh
    cpu_smoke.sh       local CPU smoke on synthetic data (all arms, exact resume, analysis)
```

## Method and decision

1. Prepare data and check raw archive hashes against PROTOCOL.md.
2. GPU smoke: every arm at both sizes, resume check, measured throughput within the frozen budget.
3. Run all 30 cells as one array; keep failures in the denominator.
4. Analyze with the frozen rules; report takeaway, numbers with intervals, next step.

| Phase | Status |
|---|---|
| Design and frozen protocol | Done (expt_v1/PROTOCOL.md) |
| Local CPU smoke | Done (synthetic data) |
| Data prep and GPU smoke | Pending |
| 30-cell execution | Pending |
| Analysis | Pending |
