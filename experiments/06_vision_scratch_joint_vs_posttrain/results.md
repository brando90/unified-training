# Vision joint from scratch vs pretrain-then-finetune: results

**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/06_vision_scratch_joint_vs_posttrain/results.md>
**Status:** RUNNING — 30-cell array submitted on Sherlock (V100, fp16) 10-20-2026; no cell has finished yet.
**Last updated:** 10-20-2026 PDT

Denominator: 30 prospective cells (2 models x 5 arms x 3 seeds). Completed: 0; failed: 0; interrupted: 0;
queued: 30. This denominator stays visible whatever happens to individual cells.

| Phase | Status | Evidence |
|---|---|---|
| Design | DONE | [PROTOCOL.md](expt_v1/PROTOCOL.md) |
| Data prep (hashes) | DONE | raw archives match PROTOCOL sha256; 8 `.npy` hashes in `$SCRATCH/exp06/data/manifest.json` match the build machine |
| GPU smoke | DONE (dev) / QUEUED (V100) | dev: `$SCRATCH/exp06/smoke_dev/` (SMOKE PASS); V100: job 48686747 |
| Training (30 cells) | QUEUED | 0/30; array 48686784 (`%4`), starts only if the V100 smoke passes (afterok) |
| Analysis | PENDING | `analysis/tables.md` |

## Run record

Cluster: Sherlock, `-p gpu -C GPU_GEN:VLT --gres=gpu:1` (V100, AMP fp16), `$SCRATCH/exp06`.
Slurm mail END,FAIL to brandojazz@gmail.com and brando.science@gmail.com on every job.

| Job | Id | Depends on |
|---|---|---|
| Dev GPU smoke | 48686478 | — |
| V100 smoke (gates the cells) | 48686747 | afterany:48686478 |
| Cells array 0-29%4 | 48686784 | afterok:48686747 |
| Post (analyze + `exp06_results.tgz`) | 48686799 | afterany:48686784 |

Ids are also in `$SCRATCH/exp06/jobs_exp06.txt`. Expected finish: Wed 10-21-2026, morning to evening PDT, depending on V100 queue.

Dev GPU smoke (Sherlock `dev`, NVIDIA A30 MIG 1g.6gb slice, fp16, 200 steps per arm, B=192): all 10 arm x model runs rc=0;
resume check rc 3,3 then 0 with the same images seen as the uninterrupted run.

| Model | Slowest arm img/s | Peak GiB | Projected full cell (MIG slice) |
|---|---|---|---|
| ViT-Tiny | 2041 | 1.63 | ~1.6 h training + ~21 evals |
| ViT-Small | 1016 | 3.01 | ~3.1 h training + ~21 evals |

These throughputs are from a quarter-A30 slice, so they are a lower bound for V100; the V100 smoke log
(`logs/smoke_v100_48686747.out`) records V100 numbers before any cell starts.

Deviations from the submit plan (none touch PROTOCOL):
- FarmShare's `gpu` QOS was full (32/32 submit slots), so the run went to Sherlock, the agreed fallback.
- Raw archives were uploaded from the build machine (sha256 verified against PROTOCOL) instead of downloaded on Sherlock, because the Toronto mirror ran at ~70 KB/s.
- Sherlock's `dev` QOS caps memory at 16G, so the dev smoke ran with `--mem=16G` instead of `submit_exp06.sh`'s 32G; it passed.
- Because the dev GPU is a MIG slice, a V100 run of the same `smoke.sbatch` gates the array (afterok) instead of the dev PASS alone.

No claim about joint vs sequential training is established by this experiment yet.

**TLDR-end:** [unified-training: Exp 06 vision] 30 ViT cells queued on Sherlock V100 behind a passing GPU smoke; no results yet.
