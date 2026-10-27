# JR1 Sherlock calibration (AT-571): L40S versus H100_SXM5

Code commit measured: `d3763c559dfd13c7223e5c60b79798ceee97cc42` (same on both SKUs). Model Qwen2.5-1.5B, snapshot
`8faed761d45a263340a0528343f099c05c9a4323` (model.safetensors sha256 `a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796`).
Data manifest check: 26/26 files match `jr1/data_manifest.json`, no mismatches (receipts/data_check.json from prewarm 48735955).
Raw measurements: `calib/L40S/calibration.json` and `calib/H100_SXM5/calibration.json` (from `$SCRATCH/ut_jr1/calib/<SKU>/`).

## Choice: ONE GPU type for every measured cell = `H100_SXM5`

- H100 does 1.75x to 2.8x the work per second of L40S on every method and 1.5x on full eval (1,880 s versus 2,776 s).
- Both fit: peak training memory is 28.8 to 33.0 GB on both (L40S has 44.5 GB; H100 has 79.1 GB), so memory does not decide.
- With the same work target `W`, H100 costs 52% of the L40S device-hours (2,584 versus 4,930 at W = 2e9).
- Cost to watch: H100_SXM5 is 3 public nodes on Sherlock, so queue waits can be longer than for L40S (8 nodes). Fallback
  if the queue stalls: L40S works with the same code, at about 2x the hours per cell.
- Calibration jobs ran with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; keep it in the cell jobs.

## Measured per method (work = forward-equivalent tokens per second; first 8 updates, 60 for aioli and jr1)

| method | L40S work/s | H100 work/s | speed-up | peak mem L40S (GB) | peak mem H100 (GB) |
|---|---|---|---|---|---|
| fixed_joint | 3,448 | 6,191 | 1.80x | 31.6 | 31.7 |
| smooth_joint | 4,300 | 9,627 | 2.24x | 29.4 | 29.9 |
| sequential | 17,888 | 49,763 | 2.78x | 28.8 | 28.8 |
| replay | 17,531 | 47,496 | 2.71x | 28.8 | 28.8 |
| aioli | 3,452 | 6,034 | 1.75x | 32.4 | 33.0 |
| jr1 | 3,449 | 6,218 | 1.80x | 32.3 | 32.4 |

Per objective, median seconds per update (L40S / H100): SFT 0.577 / 0.255, pre-training 1.399 / 0.497, DPO 1.255 / 0.481,
RL 49.8 / 23.6 (RL dominates cost). Peak eval memory 8.9 GB (full), 5.4 GB (lc).

Checkpoint (3.09 GB): save 7.0 s (L40S) / 7.3 s (H100); load to device 2.2 s / 2.9 s. Restart test: resumed once, 4 actions,
process wall 107 s (L40S) / 105 s (H100), status debug_stopped as intended.

Eval wall time: full suite (GSM8K, MATH-500, MMLU, ARC-Easy, WikiText, preference) 2,775.9 s L40S / 1,879.6 s H100 (HF backend,
status complete); lc suite 155.2 s / 155.4 s. Per-cell overhead (5 lc + 1 full) = 0.987 h L40S, 0.738 h H100.

## Per-cell time estimate (H100_SXM5)

Time per cell = W / work_per_s + 0.738 h of evals.

| method | h per cell at W = 2e9 (the calibration's placeholder) | h per cell at W = 3.17e7 (fits the 96-hour plan) |
|---|---|---|
| fixed_joint | 90.5 | 2.16 |
| smooth_joint | 58.4 | 1.65 |
| sequential | 11.9 | 0.91 |
| replay | 12.4 | 0.92 |
| aioli | 92.8 | 2.20 |
| jr1 | 90.1 | 2.15 |
| six-method set | 356.2 | 10.0 |

Totals at W = 2e9 (calibration.json): Exp 03 36 cells 2,136.9 GPU-h (+0.57 h base eval), Exp 04 six cells 356.2 GPU-h
(its schedule run 90.1 h), all 2,583.7 GPU-h on H100; on L40S 4,929.8 GPU-h. **W = 2e9 is about 27x over the plan:
PLAN.md caps the campaign at 96 device-hours (60 to Exp 03, 10 to Exp 04).** Six replicates of the six-method set for Exp 03
(36 cells) and one set for Exp 04 need 10 h per set: that gives **W of about 3.2e7 on H100** (1.2e7 on L40S), so 03 about 60.1 h
plus 04 about 10.0 h = about 70 GPU-h on H100, plus the dev allowance (6 h) and readiness (4 h) the plan sets aside.
W is frozen with the dev selection (SUBMIT.md step 4), not here; this is the figure that step must use or revise.

Caveats: sequential and replay rates come from 8 early updates that are single-objective (cheap) phases, so their hours
are probably an underestimate once RL phases run; the six-method mean the tool uses for replay cells is in the JSON note.
Rates are from short runs and move with the RL share of each method.

## Jobs (all on Sherlock, partition gpu)

| step | job id | result |
|---|---|---|
| prewarm dev srun (data prepare/verify, check-data, prep-model, pytest 51 passed / 1 skipped) | 48735955 | PREWARM_OK |
| prep (L40S) | 48735996 | COMPLETED, exit 0 |
| calib L40S, first try | 48735998 | FAILED after 1m32s: CUDA OOM (batch logits upcast), fixed in d3763c5 |
| calib H100_SXM5, first try | 48736021 | cancelled while pending (old code) |
| calib L40S, fixed code | 48742783 | COMPLETED, exit 0 (END mail Oct 21 01:36) |
| calib H100_SXM5, fixed code | 48742805 | COMPLETED, exit 0 (END mail Oct 27 08:17; node sh04-01n01) |
