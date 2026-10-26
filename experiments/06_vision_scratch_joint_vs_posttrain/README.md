# Vision: joint from scratch vs pretrain-then-finetune (ViT, matched compute)

**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/06_vision_scratch_joint_vs_posttrain/README.md>

**Summary:** Small ViTs (ViT-Tiny primary, ViT-Small scale check) trained from scratch with Tiny-ImageNet-200 as the
generic source ("pretraining") and CIFAR-100 as the target ("post-training"), 32px. Five arms at identical optimizer
steps and batch size, with identical source exposure across the four source-using arms: `target_only`, `sequential`
(pretrain then finetune), `sequential_replay`, `fixed_joint` and `smooth_joint`. Three paired seeds (30 cells), extended to five seeds (50 cells) by Amendment A1 and its ViT-Small extension (see `expt_v1/AMENDMENT_A1_seeds45.md`).
Question: does joint training from step 0 match or beat pretrain-then-finetune on the target while forgetting less
of the source? Transformers only.

**Status:** DONE — 50/50 cells complete on Sherlock V100 (fp16): 30 original, 10 from A1 (Tiny seeds 4-5), 10 from the A1 extension (Small seeds 4-5).
**Verdict:** H1 (joint >= sequential on CIFAR-100) inconclusive at 3 seeds, joint better at 5 seeds at both ViT-Tiny (+1.46 pp [0.14, 2.78], marginal) and ViT-Small (+3.61 pp [0.97, 6.26]); the n=5 verdict is exploratory (seeds added after seeing n=3). H2 (joint retains more Tiny-ImageNet) supported, at ViT-Tiny and ViT-Small. See [results.md](results.md).
**Amendment A2 (exploratory, n=1 pair):** one ViT-Tiny seed-1 pair at lr 5e-4 (v1: 1e-3), smooth_joint 61.06 / 38.54 vs sequential 59.46 / 5.95 (CIFAR-100 / Tiny-ImageNet), different hardware from v1; a look, not a test (`expt_v1/AMENDMENT_A2_lowlr.md`, results.md).
**Last updated:** 10-26-2026 PDT

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
