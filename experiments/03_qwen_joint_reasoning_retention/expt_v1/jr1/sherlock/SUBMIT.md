# JR1 on Sherlock: exact commands, in order

All in the Sherlock OnDemand shell as brando9. Every job mails END,FAIL to brandojazz@gmail.com and
brando.science@gmail.com; every submitted id lands in `$SCRATCH/ut_jr1/jobs.txt`. Details: README.md beside this file.

## 0. Upload and unpack (AT-571)

Upload `jr1_sherlock.tgz` to `/scratch/users/brando9/ut_jr1/` (Files > Upload), then:

```bash
mkdir -p $SCRATCH/ut_jr1 && cd $SCRATCH/ut_jr1
sha256sum jr1_sherlock.tgz            # must equal jr1_sherlock.tgz.sha256 from the bundle folder
rm -rf code && tar xzf jr1_sherlock.tgz && cat code/CODE_COMMIT
export JR1_STATE_ROOT=$SCRATCH/ut_jr1
S=$JR1_STATE_ROOT/code/experiments/03_qwen_joint_reasoning_retention/expt_v1/jr1/sherlock
```

## 1. Environment (login node, ~5 min) and tests (dev node, ~5 min)

```bash
bash $S/setup_env.sh                  # venv from the hashed lock; prints torch 2.6.0+cu124, transformers 4.57.6
srun -p dev -c 4 --mem=16G -t 01:00:00 bash -c "source $S/env.sh && cd \$JR1_EXPT && \
  \$JR1_PY -m pytest -q jr1/sherlock/test_sherlock.py jr1/test_train.py jr1/test_evaluate.py jr1/test_data.py"
```

## 2. Prep (one GPU job: data, model, environment receipt on the SKU)

```bash
cd $JR1_STATE_ROOT
PREP=$(GPU_SKU=L40S bash $S/submit.sh prep | tail -1)
# after the END mail:
cat PREP_OK receipts/data_check.json | head -5            # "ok": true = every data file matches jr1/data_manifest.json
grep -A1 '"hub_check"' models/model_manifest.json         # "ok" = every Qwen file matches the Hub's sha256
```

## 3. Calibration on both SKUs (AT-571)

```bash
CAL_L=$(GPU_SKU=L40S      bash $S/submit.sh calib --dependency=afterok:$PREP | tail -1)
CAL_H=$(GPU_SKU=H100_SXM5 bash $S/submit.sh calib --dependency=afterok:$PREP | tail -1)
# after both END mails: fetch calib/L40S/calibration.json and calib/H100_SXM5/calibration.json, e.g.
# https://ondemand.sherlock.stanford.edu/pun/sys/dashboard/files/fs//scratch/users/brando9/ut_jr1/calib/L40S/calibration.json?download=1
```

Choose ONE `GPU_SKU` for every measured cell, the per-cell work target `W` (PROTOCOL v1) and the dev work target
`Wdev` with its per-candidate allowance `T` seconds (4 x T = 1 device-hour per method). Record them in CALIBRATION.md.

## 4. Development selection (AT-572; no test file is opened)

```bash
source $S/env.sh && cd $JR1_STATE_ROOT
$JR1_PY -m jr1.sherlock.tools set-dev-target --work-target <Wdev>     # same edit committed to the PR branch
DEV=$(GPU_SKU=<SKU> JR1_DEV_CAND_SECONDS=<T> bash $S/submit.sh dev | tail -1)      # array 0-24, 24 = base
# after all 25 tasks end:
$JR1_PY -m jr1.sherlock.tools select --dev dev --out dev/selection.json
$JR1_PY -m jr1.sherlock.tools freeze-configs --selection dev/selection.json --work-target <W> --out $JR1_PROTOCOL_DIR
```

Commit `dev/selection.json`, `jr1/configs/protocol_v1/*.json` and the filled PROTOCOL.md (03 and 04, Status FROZEN)
to the PR branch BEFORE step 5. Measured jobs refuse any config that is not FROZEN.

## 5. Measured cells (AT-573)

```bash
source $S/env.sh && cd $JR1_STATE_ROOT
SCHED=$(GPU_SKU=<SKU> bash $S/submit.sh schedule04 | tail -1)          # Exp 04 source schedule, J-R1 seed 9001
C03=$(GPU_SKU=<SKU> bash $S/submit.sh cells03 | tail -1)               # 37 tasks: 36 cells + untouched base (36)
C04=$(GPU_SKU=<SKU> bash $S/submit.sh cells04 --dependency=afterok:$SCHED | tail -1)   # 6 tasks
squeue -u $USER -r -o '%.20i %.12j %.8T %.10M %.20R' > squeue_$(date +%m%d_%H%M).txt
cat jobs.txt
```

`JR1_TEST_ONLY=1` in front of any submit line runs `sbatch --test-only` (validates, estimates the start, submits nothing).
Index map: cells03 i -> seed 1101 + i//6, method (sequential replay fixed_joint smooth_joint aioli jr1)[i%6]; 36 = base.
cells04 i -> seed 1101 + i//2, order (interleaved stage_sorted)[i%2].

## 6. Collect (AT-574), any time

```bash
bash $S/collect.sh            # prints the tarball, its sha256 and the ?download=1 URL; --light leaves out per-item files
```

A failed or timed-out task is rerun once with the same command restricted to its index; it resumes, never restarts:
`JR1_ARRAY=7,13 GPU_SKU=<SKU> bash $S/submit.sh cells03`. Never delete a cell directory: failures stay in the denominator.
