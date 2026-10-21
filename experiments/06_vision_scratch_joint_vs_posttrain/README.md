# Vision: joint from scratch vs pretrain-then-finetune (ViT, matched compute)

**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/06_vision_scratch_joint_vs_posttrain/README.md>

**Summary:** Small ViTs (ViT-Tiny primary, ViT-Small scale check) trained from scratch with Tiny-ImageNet-200 as the
generic source ("pretraining") and CIFAR-100 as the target ("post-training"), 32px. Five arms at identical optimizer
steps and batch size, with identical source exposure across the four source-using arms: `target_only`, `sequential`
(pretrain then finetune), `sequential_replay`, `fixed_joint` and `smooth_joint`. Three paired seeds, 30 cells.
Question: does joint training from step 0 match or beat pretrain-then-finetune on the target while forgetting less
of the source? Transformers only.

**Status:** DONE — 30/30 cells complete on Sherlock V100 (fp16).
**Verdict:** H1 (joint >= sequential on CIFAR-100) inconclusive at 3 seeds; H2 (joint retains more Tiny-ImageNet) supported, at ViT-Tiny and ViT-Small. See [results.md](results.md).
**Last updated:** 10-21-2026 PDT

## Structure

```text
06_vision_scratch_joint_vs_posttrain/
  README.md
  results.md
  analysis/
    tables.md, per_seed.md, analysis.json, jobs.txt, curves_tiny.png, curves_small.png
    replot.py          redraws the two curve plots with a legend instead of end labels (plot only)
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
| Data prep and GPU smoke | Done (Sherlock V100) |
| 30-cell execution | Done (30/30, 0 failed) |
| Analysis | Done (results.md, analysis/) |
