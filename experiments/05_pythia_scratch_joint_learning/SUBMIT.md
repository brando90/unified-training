# Exp 05 (Pythia from scratch): exact commands, in order

FarmShare L40S (bf16) is the main path; Sherlock V100 32GB (fp16 + GradScaler) is the fallback if the FarmShare
queue stalls for more than a day. Same code, same pinned venv (jr1's hashed lock: torch 2.6.0+cu124, transformers
4.57.6), same data. One cluster, and so one precision, for every measured cell of a version.

Every job mails END,FAIL to brandojazz@gmail.com and brando.science@gmail.com, requeues itself 15 minutes before its
time limit (it resumes), and every submitted id lands in `$P05_STATE_ROOT/jobs.txt`. Arrays run at most 4 GPUs at
once (`%4`); cells have the 2-day maximum. Protocols: [expt_v1](expt_v1/PROTOCOL.md) (160m, 9 cells) and
[expt_v2](expt_v2/PROTOCOL.md) (14m + 70m, 18 cells), both DRAFT until step 4.

## 0. Upload and unpack

FarmShare: from screen 1's OnDemand JupyterLab terminal. Upload `p05_bundle.tgz` (and its `.sha256`) to
`/home/users/brando9/ut_p05/`, then:

```bash
export P05_CLUSTER=farmshare P05_STATE_ROOT=/home/users/brando9/ut_p05      # Sherlock: sherlock, /scratch/users/brando9/ut_p05
mkdir -p $P05_STATE_ROOT && cd $P05_STATE_ROOT
sha256sum -c p05_bundle.tgz.sha256
rm -rf code && tar xzf p05_bundle.tgz && cat code/CODE_COMMIT
S=$P05_STATE_ROOT/code/experiments/05_pythia_scratch_joint_learning/cluster
df -h $P05_STATE_ROOT; quota -s 2>/dev/null     # needs ~80 GB (estimate: 160m cells ~5 GB each = 5 fp32 checkpoints + resume state; 70m ~2.3 GB; 14m ~0.5 GB; venv + data ~10 GB)
```

The venv is built inside the prep job at `$P05_STATE_ROOT/venv` from `/usr/bin/python3.12` (not the Jupyter
python3 or `~/venvs/fs_hf`, whose torch/transformers differ from the lock). Sherlock builds it with
`module load python/3.12.1`.

## 1. Prep (CPU job: venv, data, Pythia config/tokenizer files)

```bash
P05_TEST_ONLY=1 bash $S/submit.sh prep         # validates the script, submits nothing
PREP=$(bash $S/submit.sh prep | tail -1)
# after the END mail:
cat PREP_OK && head -c 600 data/manifest.json   # pt_blocks: ~101,527 blocks, ~104M trainer tokens per epoch
```

Prep downloads the jr1 sources (WikiText-103, GSM8K, UltraFeedback, MATH-500, MMLU, ARC-Easy at pinned
revisions; every file must match jr1/data_manifest.json), packs WikiText-103 into 1024-token GPT-NeoX blocks, writes
LAMBADA, and caches the three Pythia configs and the tokenizer. On Sherlock it reuses `/scratch/users/brando9/ut_jr1/data`
when JR1 prep already made it.

## 2. Tests on a GPU node (optional, ~5 min)

```bash
source $S/env.sh && cd $P05_HOME
srun -p gpu --qos=gpu --gres=gpu:1 -c 4 --mem=16G -t 00:30:00 bash -c "source $S/env.sh && cd \$P05_HOME && \
  \$P05_PY -m pytest -q p05/test_p05.py"
# Sherlock: srun -p gpu -G 1 -C 'GPU_MEM:32GB' ...
```

## 3. Calibration (one GPU job, ~1 h)

```bash
cd $P05_STATE_ROOT
P05_TEST_ONLY=1 bash $S/submit.sh calib         # on Sherlock this checks the V100 constraint syntax
CAL=$(bash $S/submit.sh calib --dependency=afterok:$PREP | tail -1)
# after the END mail:
cat receipts/calibration/calibration.md
```

It times 10 updates of each objective (PT, SFT, DPO, RL) for 14m, 70m and 160m, reports peak memory, and recommends
one charged-work target `W` for every cell: the work at which every method reaches 50M observed PT tokens, × 1.05.
It also prints hours per cell. If any size shows `memory_fallback_needed`, the protocols' field 15 applies (halve
every batch, calibrate again) before freezing. If the hours do not fit, that is reported before launch, not after.

On Sherlock, if `-C` is rejected, pass the constraint another way:
`P05_CONSTRAINT='GPU_SKU:V100S_PCIE|GPU_SKU:V100_PCIE|GPU_SKU:V100_SXM2' bash $S/submit.sh calib` (the 32GB check
is then done by the job itself: `p05.envcheck --require v100_32gb` refuses a 16GB card).

## 4. Freeze (on a branch of the repo, before any measured cell)

```bash
cd <repo>/experiments/05_pythia_scratch_joint_learning
PYTHONPATH=.:../03_qwen_joint_reasoning_retention/expt_v1 python -m p05.freeze --precision bf16 \
  --calibration <calibration.json from step 3>          # Sherlock: --precision fp16
```

This writes the frozen configs (work target = the calibrated `W`) and `expt_v1/configs/FROZEN.sha256`,
`expt_v2/configs/FROZEN.sha256`. Fill `W` and the field-3b hashes (from `data/manifest.json`) in both PROTOCOL.md
files, set them FROZEN, commit, and cut a new bundle (`bash cluster/make_bundle.sh <dir>`); unpack it as in step 0.
Measured jobs refuse a DRAFT config or a frozen config whose sha256 differs from FROZEN.sha256.

## 5. Measured cells

```bash
cd $P05_STATE_ROOT
P05_TEST_ONLY=1 bash $S/submit.sh cells_v1
V1=$(bash $S/submit.sh cells_v1 | tail -1)      # 9 tasks, 4 at a time: 160m x 3 methods x seeds 1-3
V2=$(bash $S/submit.sh cells_v2 --dependency=afterany:$V1 | tail -1)    # 18 tasks: 14m + 70m (keeps the 4-GPU cap)
squeue -u $USER -r -o '%.20i %.12j %.8T %.10M %.20R'
cat jobs.txt
```

Each task trains one cell (`p05.train`, which imports `jr1.train`), then evaluates it: the learning-curve suite on
all 5 checkpoints and the full suite (WikiText-103 test, LAMBADA 5,153, ARC-Easy 2,376, GSM8K 1,319, pref 1,985) on
`ckpt_000` and `ckpt_100`, and writes `CELL_DONE`. Index map: `python -m p05.cells --version v1 --list`.

A failed or timed-out task is rerun with the same command restricted to its indices; it resumes, never restarts:
`P05_ARRAY=4,7 bash $S/submit.sh cells_v1`. Never delete a cell directory: failures stay in the denominator.

## 6. Collect, any time

```bash
source $S/env.sh && cd $P05_STATE_ROOT
$P05_PY -m p05.collect --runs runs --out collect && cat collect/collect.md
tar czf p05_results.tgz collect receipts jobs.txt runs/*/*/*/{receipt.json,slurm.json,eval/*/summary.json}
```

`collect.md` lists all 27 cells with status, job ids, charged work, observed PT tokens (and whether ≥50M), fp16
skipped steps and terminal scores, and the paired gaps by seed (fixed − sequential, smooth − sequential,
fixed − smooth) per size.
