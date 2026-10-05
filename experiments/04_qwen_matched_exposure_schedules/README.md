# Matched exposure and objective ordering

**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/04_qwen_matched_exposure_schedules/README.md>

**Summary:** Sequential and interleaved schedules, three paired seeds, identical development-frozen objective microbatches. The full prospective [campaign plan](../03_qwen_joint_reasoning_retention/PLAN.md) specifies hypotheses, admission, budgets, endpoints and success criteria. Results remain pending.

**Status:** PREPARATION — 6 prospective training cells; none launched.
**Last updated:** 10-05-2026 10:25 PDT

## Structure

```text
04_qwen_matched_exposure_schedules/
  README.md
  results.md
  CKPT_execution.md
  expt_v1/
    PROTOCOL.md
    cc.md
```

## Method and decision

1. Verify data provenance, immutable revisions and development-only readiness.
2. Calibrate compute and freeze the full matrix/settings before measurement.
3. Execute every admitted cell, retain failures, evaluate the full denominators.
4. Report reasoning, retention, sample efficiency and resource use with uncertainty.

The decision criteria and scope limitations are in the campaign plan. A positive training loss change or a running process is not proof of the hypothesis.

## Dependencies

The completed pilot under `experiments/00_program` provides frozen utilities and historical evidence. New implementations live in the new canonical experiment homes or a shared source package; old frozen files remain unchanged. Public models/data, a reproducible Python environment and SNAP graphics hardware are required. No provider keys or paid inference services are needed.

| Phase | Status |
|---|---|
| Prospective design | Written |
| Readiness and resource calibration | Pending |
| Frozen measured protocol | Pending |
| Full execution and evaluation | Pending |
| Verified report and publication | Pending |
