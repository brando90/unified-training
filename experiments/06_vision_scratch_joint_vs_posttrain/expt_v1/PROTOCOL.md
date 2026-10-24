# Vision joint-from-scratch vs pretrain-then-finetune: frozen protocol (expt_v1)

**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/06_vision_scratch_joint_vs_posttrain/expt_v1/PROTOCOL.md>

Status: FROZEN — written and committed before any measured run. Never change this version after the first measured
cell starts; a scientific change needs a new `expt_v2/` with its own protocol, and failed evidence from v1 stays.

## Question

At matched compute, does training one transformer from scratch on a generic source task and the target task together
(joint) do at least as well on the target as the standard recipe of pretraining on the source and then fine-tuning on
the target (sequential), while forgetting less of the source? This is the vision analogue of the claim behind
experiments 01/02/05 (Brando, Oct 20: training from scratch with all objectives basically always beats
pretrain-then-post-train), at a scale that runs on one GPU per cell. Transformers only (Brando, Oct 20 08:19).

## Data

| Role | Dataset | Train | Eval | Classes |
|---|---|---:|---:|---:|
| Source ("pretraining") | Tiny-ImageNet-200, `https://cs231n.stanford.edu/tiny-imagenet-200.zip` | 100,000 | val 10,000 | 200 |
| Target ("post-training") | CIFAR-100 (torchvision; archive `https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz`) | 50,000 | test 10,000 | 100 |

- Tiny-ImageNet 64x64 images are downsampled to 32x32 with an exact 2x2 box filter (mean of each 2x2 block,
  rounded); CIFAR-100 is native 32x32. Both are stored as uint8 NHWC `.npy` by `prep_data.py`, which writes
  `manifest.json` with the sha256 of each raw archive and each array. Expected raw sha256 (computed Oct 20 on the
  build machine): see the "Pinned inputs" table below. A mismatch in a raw archive stops the run before measurement.
- Tiny-ImageNet class index = position of the wnid in sorted `wnids.txt`; CIFAR-100 uses its fine labels.
- No validation split is carved out and nothing is tuned on the eval sets: there is no model selection at all (the
  reported number is the final step), so the test/val sets are only measured, never selected on.
- Overlap note: some Tiny-ImageNet classes are semantically close to CIFAR-100 classes. That is the point of a
  "generic pretraining" source; it is the same for every arm.

## Model

Own ViT in plain PyTorch (`train.py`), 32x32 input, patch 4 (64 patches + CLS), linear patch embedding, learned
position embedding, pre-norm blocks (MHSA + 4x GELU MLP), final LayerNorm on CLS, two linear heads on the shared
trunk: 200-way source head and 100-way target head.

| Name | Width | Depth | Heads | Params (approx.) | Role |
|---|---:|---:|---:|---:|---|
| ViT-Tiny | 192 | 12 | 3 | 5.4M | primary |
| ViT-Small | 384 | 12 | 6 | 21.4M | scale check |

Init: truncated normal std 0.02 for all linear weights, position embedding and CLS; zero biases.

## Arms (matched compute and matched source exposure)

Every arm runs exactly **S = 60,000 optimizer steps at batch B = 192** (11.52M images, so identical FLOPs; both heads
are tiny and evaluated on every image routed to them). Every source-using arm sees exactly **0.5·S·B = 5.76M source
images** (57.6 Tiny-ImageNet epochs); its remaining 5.76M images are CIFAR-100 (115.2 epochs). The per-step source
count follows a cumulative-rounding rule so totals are exact, not approximate.

| Arm | Source fraction of each batch over steps t = 0..S-1 | Source images | Optimizer phases |
|---|---|---:|---|
| `target_only` | 0 (all CIFAR-100; 230.4 epochs) | 0 | one |
| `sequential` | 1 for t < 0.5S, then 0 | 5.76M | two: [0, 0.5S), [0.5S, S) |
| `sequential_replay` | 1 for t < 0.4S, then 1/6 (32 of 192) | 5.76M | two: [0, 0.4S), [0.4S, S) |
| `fixed_joint` | 1/2 (96 of 192) every step from step 0 | 5.76M | one |
| `smooth_joint` | linear 1 -> 0: f(t) = 1 - t/(S-1), mean exactly 1/2 | 5.76M | one |

The names `sequential`, `fixed_joint` and `smooth_joint` mean the same schedules as in `experiments/00_program`
and exps 01/02/05. The fractions above are AT-556's defaults and are not changed here.

Loss: cross-entropy with label smoothing 0.1 on the head of each image's dataset, summed over the batch and divided
by B (so each dataset's weight equals its share of the batch).

## One training recipe for every arm (no per-arm tuning)

- AdamW, peak LR 1e-3, betas (0.9, 0.999), eps 1e-8, weight decay 0.05 (not on biases, LayerNorm, CLS or position
  embedding), gradient-norm clip 1.0, drop path 0.1 (linear over depth), label smoothing 0.1.
- LR: linear warmup over the first 5% of each optimizer phase, then cosine to 0 at the phase end.
- **Sequential arms restart at the phase boundary:** the fine-tuning phase starts with a fresh AdamW state and its own
  5% warmup + cosine to 0, as a separate pretrain -> fine-tune run would. Joint arms and `target_only` have one phase.
- Augmentation on the GPU, identical for both datasets: random crop 32 from reflect-padded 40, horizontal flip p=0.5;
  inputs scaled to [-1, 1]. No other augmentation.
- Precision: FarmShare L40S, bf16 autocast (main path). Sherlock V100 fallback: fp16 autocast + GradScaler. A
  campaign uses one precision for all 30 cells; it is recorded in each `final.json`.
- Seeds 1, 2, 3 set model init, drop path, data order (per-dataset epoch permutations) and augmentation. The same
  seed gives the same model init across arms, so arms are paired by seed.

The recipe is a standard one for small ViTs trained from scratch on 32px data. It was not tuned on these datasets;
that holds equally for every arm.

## Cells and budget

2 models x 5 arms x 3 seeds = **30 cells** (`cells.tsv`, array tasks 0-29, ViT-Tiny first).

Throughput assumption, from training FLOPs ≈ 6 x params x 65 tokens + attention: ViT-Tiny ≈ 2.2 GFLOP and
ViT-Small ≈ 8.6 GFLOP per training image, so a ViT-Small cell is ≈ 1.0e17 FLOP. Assuming at least 20 TFLOP/s
effective (≈ 2,300 img/s ViT-Small; L40S bf16 or V100 fp16), a ViT-Small cell trains in ≈ 1.4 h; even at 5 TFLOP/s
it is ≈ 5.5 h, well under the 2-day limit on one GPU. ViT-Tiny is ≈ 4x cheaper. Evaluation adds 21 passes over
20,000 images per cell (negligible). Expected total ≈ 30-60 GPU-hours, 4 GPUs at a time.

Launch gate (`smoke.sbatch`, before the array): all 5 arms x 2 models run 200 steps with finite loss on the target
GPU, the stop/resume check passes, and the slowest ViT-Small arm's measured img/s implies ≤ 24 h per cell. If not,
nothing is measured under v1; the budget is re-sized in `expt_v2/`.

## Measurements

Every S/20 = 3,000 steps and at step 0 (21 points): CIFAR-100 test top-1 through the target head and Tiny-ImageNet
val top-1 through the source head, plus mean train losses since the previous point, images seen per dataset,
skipped steps and throughput (`curve.jsonl`). Checkpoints at the same cadence (atomic replace) and on SIGUSR1/SIGTERM;
a rerun resumes exactly (sampler positions, RNG states, optimizer and scaler state).

**Endpoints are the final-step (step S) values.** No early stopping, no best-checkpoint selection.

## Hypotheses and decision rules (pre-registered)

All differences are paired by seed, in percentage points (pp) of top-1, with Student's t intervals over the complete
seed pairs (df = n - 1; with 3 seeds df = 2). Only complete cells count; a contrast needs at least 2 complete pairs.

**Best joint arm.** For each model, the joint arm (`fixed_joint` or `smooth_joint`) with the higher mean final
CIFAR-100 top-1. Because this choice uses the same data, H1 intervals are Bonferroni-adjusted over the two joint
arms (97.5% two-sided t intervals), and both individual joint-vs-sequential contrasts are reported too.

**H1 (target): joint ≥ sequential on CIFAR-100.** d1 = best_joint - sequential, final CIFAR-100 test top-1,
ViT-Tiny primary. Verdict, from the 97.5% interval [lo, hi], checked in this order:
1. lo > 0: **joint better**.
2. hi < 0: **sequential better**.
3. lo > -1.0 pp: **joint non-inferior** (margin 1.0 pp).
4. otherwise: **inconclusive**.

H1 is supported if the verdict is "joint better" or "joint non-inferior".

**H2 (retention): joint retains more source.** d2 = best_joint (the same arm chosen for H1) - sequential, final
Tiny-ImageNet val top-1, 95% interval. Supported if lo > 0; not supported if hi < 0; else inconclusive.

**Scale check.** The same rules at ViT-Small. "Replicates" means the same H1 verdict class as ViT-Tiny.

Secondary, reported without verdicts (95% intervals): best_joint - sequential_replay on CIFAR-100 and on
Tiny-ImageNet (the stronger baseline); best_joint - target_only and sequential - target_only on CIFAR-100 (does the
source help at all under this budget); accuracy-vs-step curves for every arm.

What would count against Brando's claim: "sequential better" at ViT-Tiny, or at both sizes. A "joint better" or
"non-inferior" H1 together with a supported H2 supports it for this setting only (small ViTs, 32px, one source/target
pair, one recipe).

## Failed cells

A cell **fails** if (a) training stops with ≥ 50 consecutive non-finite losses, (b) more than 1% of its steps were
skipped (non-finite loss or fp16 overflow), (c) it has no step-S evaluation after its first run plus up to two
resubmissions (each resumes from the checkpoint), or (d) its final.json is missing or unreadable. Failed cells are
never rerun with different settings, stay in the 30-cell denominator, and are listed by task, arm, seed and reason in
results.md. A cell that finishes with low accuracy is data, not a failure. Infrastructure losses (node failure,
preemption, time limit) are resumed, not counted as failures, as long as (c) is met.

## Pinned inputs

| Input | Value |
|---|---|
| Code | the commit named in "Freeze record" below; `cells.tsv` lists all 30 cells |
| CIFAR-100 archive | md5 `eb9058c3a382ffc7106e4002c42a8d85` (torchvision's check), sha256 `85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7` |
| Tiny-ImageNet archive | sha256 `6198c8ae015e2b3e007c7841da39ec069199b9aa3bfa943a462022fe5e43c821` (248,100,043 bytes) |
| Environment (FarmShare) | `/usr/bin/python3.12` venv `~/venvs/exp06`, torch 2.13.0 (PyPI CUDA wheel), its torchvision; lock written to `$EXP06_ROOT/requirements.lock.txt` |
| Environment (Sherlock fallback) | python/3.12.1, venv `$SCRATCH/exp06/venv`, torch 2.6.0+cu124, torchvision 0.21.0, numpy 2.2.6, `--only-binary=:all:` |

The `.npy` hashes depend on the JPEG decoder build and are recorded by `prep_data.py` in `manifest.json` at prep time;
they are copied into results.md and the results tarball.

## Execution

`submit_exp06.sh build | prep | smoke | cells | status` (see `cc.md`). Slurm mail `END,FAIL` to
brandojazz@gmail.com and brando.science@gmail.com. The array runs `%4` (FarmShare: at most 4 GPUs per user). The
post job (afterany) runs `analyze.py` over all 30 rows and writes `$EXP06_ROOT/exp06_results.tgz`.

## Freeze record

Frozen at commit `33ce4dc3a9c61f4a9a45bb9588bc0d3f8270ffb1` (the commit that added this file and all expt_v1 code). The only later change to
this file in v1 is this line recording that hash.
