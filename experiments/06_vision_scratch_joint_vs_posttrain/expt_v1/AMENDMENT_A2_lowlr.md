Amendment A2 (2026-10-24 AT-1211, exploratory, added after the n=5 results were seen): lower-lr look at ViT-Tiny. PROTOCOL.md v1, cells 0-49 and all earlier analyses stay as frozen.

Question: does the marginal joint-over-sequential result and the unexplained seed spread (skipped_steps 21-24 per cell) survive a lower peak LR?

Cells (2): ViT-Tiny, arms `sequential` and `smooth_joint`, seed 1, `--lr 5e-4` (v1: 1e-3); every other setting is the v1 default (60,000 steps, batch 192, wd 0.05, warmup 5%, label smoothing 0.1, drop-path 0.1, clip 1.0).

Launch record (settings at run time):
- Host mercury1.stanford.edu, GPU 9 only (RTX A4000 16 GB), `--amp bf16`, torch 2.8.0+cu128 (CUDA 12.8), Python 3.12, code commit ff52e382711857621d40fc0f1041710af52c3c62 (branch ut/exp06-vision).
- Command: `python train.py --arm <arm> --model tiny --seed 1 --data $R/data --out $R/runs/lr5e-4/<arm>/seed1 --lr 5e-4 --amp bf16`, `$R=/lfs/mercury1/0/brando9/at1211/exp06`, data from `prep_data.py` (CIFAR-100, Tiny-ImageNet-200, fetched 2026-10-24 20:10 PDT).
- Started 2026-10-24 20:11 PDT, sequential first, then smooth_joint. Hardware differs from v1 (V100 fp16 / L40S bf16), so compare only against the same-hardware pair here, not v1 numbers.
- Results: not in yet; will be added to results.md.
