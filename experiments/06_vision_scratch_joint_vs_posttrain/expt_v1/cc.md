# Execute 06_vision_scratch_joint_vs_posttrain (expt_v1)

**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/06_vision_scratch_joint_vs_posttrain/expt_v1/cc.md>

**Summary:** Run the frozen [PROTOCOL.md](PROTOCOL.md) as written: prep, GPU smoke, one 30-task array, one post job.
Do not change any setting in expt_v1 after the first measured cell starts.

## Where

Primary: FarmShare (L40S, bf16). Fallback: Sherlock V100 (fp16) if FarmShare's GPU queue stalls for more than a day.
`env.sh` picks the cluster (`CLUSTER=farmshare|sherlock`; default sherlock when `$SCRATCH` is set) and sets
`EXP06_ROOT` (FarmShare `~/exp06`, Sherlock `$SCRATCH/exp06`), the venv, precision and partition flags.
Limits: FarmShare QOS gpu allows 4 running GPU jobs and 32 submitted per user (30 array tasks fit; the post job runs on
`normal`); 2-day wall. Sherlock: `-p gpu -C GPU_GEN:VLT`, `dev` refuses sbatch so prep/smoke run via `srun -p dev`.

## Steps (from the unpacked `expt_v1/` folder)

1. `bash submit_exp06.sh build` — venv; check the printed torch/torchvision versions (FarmShare: wait for the job).
2. Sherlock only: `bash submit_exp06.sh download` on the login node.
3. `bash submit_exp06.sh prep` — then check `$EXP06_ROOT/data/manifest.json` raw sha256 against PROTOCOL.md
   "Pinned inputs". A mismatch stops the run.
4. `bash submit_exp06.sh smoke` — must print `SMOKE PASS`, 10 arm lines with rc=0, the resume line (3 then 0) and a
   projected ViT-Small cell time ≤ 24 h. Otherwise stop: nothing is measured under v1.
5. Set results.md Status to `RUNNING (started <date> UTC)`, then `bash submit_exp06.sh cells`. Record the array and
   post job ids (also in `$EXP06_ROOT/jobs.txt`) in results.md.
6. Tasks that end early (time limit, node failure) are resubmitted with `bash submit_exp06.sh resubmit <i> [<j> ...]`;
   each resumes from its checkpoint and a fresh post job follows. At most two resubmissions per cell (PROTOCOL
   "Failed cells").
7. After the post job: fetch `$EXP06_ROOT/exp06_results.tgz`, fill results.md from `analysis/tables.md` (full
   denominator, every failed cell named, H1/H2 verdicts as computed), add the curves, push to the PR branch.

Local check without a GPU: `bash cpu_smoke.sh <python> <workdir>` (synthetic data, all arms, exact resume, analysis).

**TLDR-end:** [unified-training: Exp 06 execution] prep -> smoke -> 30-cell array + post, frozen v1 settings only.
