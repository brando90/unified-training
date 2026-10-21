# Random-initialization joint learning

**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/05_pythia_scratch_joint_learning/README.md>

**Summary:** Random-init Pythia; sequential, fixed joint and smooth joint, three paired seeds, all at one matched
charged-work target sized for at least 50M observed PT tokens per cell. expt_v1 is Pythia-160m (9 cells, PLAN.md line
24); expt_v2 is the 14m/70m size ladder (18 cells). The full prospective [campaign plan](../03_qwen_joint_reasoning_retention/PLAN.md)
specifies hypotheses, admission, budgets, endpoints and success criteria. Results remain pending.

**Status:** FROZEN (Sherlock V100-32GB, fp16, W = 2.9e8, calibration job 48759212) — both protocols frozen with config hashes; 27 prospective training cells, none launched.
**Last updated:** 10-21-2026

## Structure

```text
05_pythia_scratch_joint_learning/
  README.md  results.md  CKPT_execution.md
  SUBMIT.md                 exact cluster commands, in order (prep, calibrate, freeze, cells, collect)
  smoke_cpu.sh              CPU smoke: 3 methods x 3 sizes, one cell through the job body with evaluation, calibrate
  expt_v1/PROTOCOL.md       160m protocol (DRAFT; every freeze field but the calibrated work target)
  expt_v1/configs/          pythia160m_{bf16,fp16}.json
  expt_v2/PROTOCOL.md       14m + 70m protocol (DRAFT)
  expt_v2/configs/          pythia{14m,70m}_{bf16,fp16}.json
  p05/                      spec (sizes, pins, cells), data (packed WikiText-103 blocks, LAMBADA, manifest),
                            train (jr1.train.Trainer subclass: fp16 GradScaler counting, V100 guards, receipt),
                            evaluate (jr1.evaluate + LAMBADA, fp32), calibrate, freeze, cells, collect, envcheck, tests
  cluster/                  env.sh, common.sh, jobs.sh, submit.sh, make_bundle.sh, farmshare_l40s/, sherlock_v100/
```

Training is jr1's trainer (`experiments/03_qwen_joint_reasoning_retention/expt_v1/jr1`), imported, not copied:
Exp 05 adds only random-init configs, the packed PT corpus, the fp16 path and LAMBADA.

## Method and decision

1. Verify data provenance, immutable revisions and development-only readiness.
2. Calibrate compute and freeze the full matrix/settings before measurement.
3. Execute every admitted cell, retain failures, evaluate the full denominators.
4. Report reasoning, retention, sample efficiency and resource use with uncertainty.

The decision criteria and scope limitations are in the campaign plan. A positive training loss change or a running process is not proof of the hypothesis.

## Dependencies

The completed pilot under `experiments/00_program` provides frozen utilities and historical evidence. New implementations live in the new canonical experiment homes or a shared source package; old frozen files remain unchanged. Public models/data, the jr1 hashed-lock Python environment and one GPU per cell (FarmShare L40S, or Sherlock V100 32GB) are required. No provider keys or paid inference services are needed.

| Phase | Status |
|---|---|
| Prospective design | Written |
| Code, DRAFT protocols, CPU smoke | Done (10-20-2026) |
| Readiness and resource calibration | Pending |
| Frozen measured protocol | Pending |
| Full execution and evaluation | Pending |
| Verified report and publication | Pending |
