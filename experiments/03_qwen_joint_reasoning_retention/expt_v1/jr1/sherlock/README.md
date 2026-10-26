# JR1 on Sherlock: Exp 03 (36 cells + base) and Exp 04 (6 cells)

Slurm layer for the `jr1` package (data.py, train.py, evaluate.py). Exact commands in order: [SUBMIT.md](SUBMIT.md).

## Environment: the choice and the exact versions

**Chosen: one pinned wheel-only venv on `$SCRATCH`** (`$JR1_STATE_ROOT/venv`, built by `setup_env.sh` from
`requirements-sherlock.lock` with `--only-binary=:all: --require-hashes`), used for every stage: data prep,
training, and evaluation with the evaluator's **HF backend** (`JR1_EVAL_BACKEND=hf`).

| package | version |
|---|---|
| Python | 3.12 (`module load python/3.12.1`) |
| torch | 2.6.0 (PyPI manylinux1 wheel, CUDA 12.4 runtime, cuDNN 9.1.0.70; Sherlock driver 550 = CUDA 12.4) |
| transformers / tokenizers / safetensors | 4.57.6 / 0.22.2 / 0.8.0 |
| huggingface_hub / hf-xet | 0.36.2 / 1.7.0 |
| numpy / pyarrow / pyahocorasick | 2.2.6 / 20.0.0 / 2.3.1 |
| math-verify / latex2sympy2_extended / antlr4 runtime | 0.9.0 / 1.11.0 / 4.13.2 (evaluate.py refuses others) |
| sympy | 1.13.1 (torch 2.6.0's pin) |

The lock (48 packages, every wheel's sha256) was resolved for glibc 2.17 with
`pip install --dry-run --report --platform manylinux2014_x86_64 --python-version 3.12 --implementation cp --abi cp312
--only-binary=:all:`. Every wheel in it is manylinux2014 or older, so it installs on Sherlock's CentOS 7 nodes. The full
test suite and the CPU end-to-end were run in a venv with the same pins (torch 2.6.0+cpu in place of the CUDA build).
Each job also writes `env.json` (pip freeze, torch/CUDA, GPU name and memory, bf16 check) next to its outputs.

**Why not the vLLM Apptainer image** (`$SCRATCH/rl4cv_sherlock/containers/vllm-openai_v0.30.0.sif`, vllm 0.30.0,
torch 2.13+cu130): training cannot use it identically (the trainer's RL rollouts are HF generate by design), so the
image would put training and evaluation on two different torch/transformers stacks, and its Python lacks the pinned
math-verify. HF generation is slower than vLLM; calibration measures the real eval cost. If calibration shows
evaluation dominating, the protocol may switch evaluation to vLLM **before the freeze** (AT-572), for every cell at
once; never mid-campaign.

## Layout on Sherlock (`$JR1_STATE_ROOT` = `/scratch/users/brando9/ut_jr1`)

```
code/            the unpacked bundle (CODE_COMMIT = the git commit it was cut from)
venv/  hf/       the pinned venv; HF cache (HF_HOME); jobs run with HF_HUB_OFFLINE=1 except prep
data/            jr1.data prepare output; receipts/data_check.json compares every file to jr1/data_manifest.json
models/base      -> the pinned Qwen2.5-1.5B snapshot; models/model_manifest.json = sha256 of every file, checked
                    against the Hub's LFS sha256 / git blob ids
calib/<SKU>/     calibration.json (+ per-method short runs, restart, checkpoint and eval timing)
dev/<candidate>/ dev-selection runs; dev/selection.json
runs/03/<method>_s<seed>/, runs/03/base/, runs/04/<order>_s<seed>/, runs/04/schedule_jr1_9001.jsonl
logs/  receipts/  jobs.txt (every submitted job id)  collect/ (result tarballs)
```

## Scripts

| script | what | array |
|---|---|---|
| `setup_env.sh` | venv from the lock (idempotent; login node or inside prep) | |
| `prep.sbatch` | env receipt on the GPU SKU, data prepare + verify + manifest hash check, Qwen download + sha256 | |
| `calib.sbatch` | per objective and per method update time, work/s, peak memory, restart, checkpoint save/load, full + lc eval time | one job per SKU |
| `dev.sbatch` | development selection: frozen grid `dev_grid.json`, dev seed 9001, equal allowance, tune split only | 0-24 (24 = base) |
| `schedule04.sbatch` | Exp 04 source schedule: frozen J-R1, dev seed 9001, `--record-schedule` | |
| `cells03.sbatch` | 6 methods x seeds 1101-1106, index i -> seed 1101+i//6, method i%6; 36 = untouched base | 0-36 |
| `cells04.sbatch` | interleaved / stage_sorted replay x seeds 1101-1103, index i -> seed 1101+i//2, order i%2 | 0-5 |
| `collect.sh` | tarball of every small result (no weights/state) + index.json over the full 37 + 6 denominators | |
| `submit.sh` | adds `-C GPU_SKU:$GPU_SKU`, records ids in jobs.txt; `JR1_TEST_ONLY=1` = `sbatch --test-only` | |

Every sbatch: `-p gpu -G 1 -c 8 --mem=64G`, `--mail-type=END,FAIL --mail-user=brandojazz@gmail.com,brando.science@gmail.com`,
`--requeue --open-mode=append --signal=B:USR1@900`. Near the time limit the job requeues itself (at most
`JR1_MAX_REQUEUE`=3 times) and the task resumes: training with `--resume` from its last atomic state (saved every
30 min and at every checkpoint), evaluation with `--resume` per benchmark. Rerunning any index by hand is always
safe. Each measured cell also refuses to start unless its config is a FROZEN protocol (`tools check-frozen`), and
refuses a GPU other than `GPU_SKU`.

Per cell: train -> lc suite (256 GSM8K + 512 MMLU) at ckpt 000/025/050/075/100 -> full suite at ckpt_100 -> `CELL_DONE`.
ckpt_000 is the untouched model in every cell, so its lc scores must be identical across cells (a free determinism check).

## Development selection (written before any dev run)

`dev_grid.json`: 4 candidates per method (2 learning rates x 2 mixture/stage allocations), all at dev seed 9001,
the same charged work target (filled once by calibration) and the same wall-clock allowance per candidate
(`JR1_DEV_CAND_SECONDS`, 4 x allowance = 1 device-hour per method, 6 in total as PLAN.md fixes). Selection score:
`dev_gsm8k` accuracy on the 512 GSM8K tune prompts; retention rule: `dev_wikitext` NLL per token at most 2% above the
base model's (PLAN.md's corpus-NLL tolerance); ties by `dev_pref` accuracy, then lower NLL, then listing order. No
test or learning-curve file is opened by any dev job (`evaluate.py --suite dev` reads `gsm8k_dev/tune`, `pt/tune`,
`pref/tune`). `tools freeze-configs` then writes `configs/protocol_v1/<method>.json` (FROZEN) for AT-572 to commit.

## CPU end-to-end

`e2e_cpu_sherlock.sh` runs these same .sbatch files with bash on a CPU over the tiny offline Qwen2 (evaluation on the
first 3 real items of every eval/tune file): prep -> calib -> dev (25) -> select -> freeze -> schedule04 -> cells03
(one cell per method + base) -> cells04 (6) -> a rerun of a finished cell (resume path) -> collect -> checks.
