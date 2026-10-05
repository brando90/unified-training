# Unified-training implementation checkpoint

**Status:** COMPLETE — 48/48 frozen cells and 4/4 untouched initial evaluations.
**Last updated:** 10-04-2026 23:20 PDT
**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/00_program/CKPT_unified_training.md>

The bounded joint-training pilot is finished. Read [FINAL_REPORT.md](FINAL_REPORT.md) for results, uncertainty and limits, [final_verification.json](final_verification.json) for the independent artifact audit, and each experiment's results.md for every cell.

## Scientific state

[experiments/01_scratch_joint_objectives](../01_scratch_joint_objectives/README.md): 24/24 complete, three untouched initial evaluations complete. [experiments/02_early_checkpoint_joint_objectives](../02_early_checkpoint_joint_objectives/README.md): 24/24 complete, one untouched initial evaluation complete. Failed/interrupted/missing: 0/0/0. All declared endpoints have full frozen denominators; no outcome-driven reruns or altered settings.

The proposed controller does not establish an advantage over mandatory Aioli on either primary endpoint. Both outperform the staged baseline on observed primary means. All 24 early-checkpoint trained cells have lower observed ARC-Easy accuracy than the untouched checkpoint. Three-seed intervals, all baseline comparisons and initial changes are in the final report; no universal-superiority claim.

Both required Claude Code `claude-opus-5-5` maximum-effort reviews retain original FAIL verdicts. Every finding is recorded in the unchanged REVIEW_RECONCILIATION.md. Acceptance is implementer FIXED with 63 deterministic tests and narrowed scope, not an invented reviewer PASS. Exact CHERRY-RL identity remains unresolved. Math readiness and broad retention remain limitations.

## Operational state

The independent single-device supervisor exited normally at 10-04-2026 23:07:51 PDT after all 48 cells, four initial evaluations, final analysis and verified main publication. Native Codex coordination had already ended under its original cumulative deadline. No automatic model continuation, duplicate queue, new provider attempt or budget extension was used.

Conservative inclusive device accounting: 38,468.953629 seconds (10.68582 hours), including 3,109.509749 preparation seconds and the entire 35,359.443880-second queue lifetime, below the single 48-hour ceiling. Per-cell target 19,900,000 charged units; maximum overshoot 79,320 within 217,088. All healthy children were preserved through natural completion. Private identity/process/storage receipts remain outside Git.

Final read-only audit: all 52 evaluation filesets, 48 checkpoint/summary receipts, 18 prepared splits, 10 cached model files, 37 frozen files, 35 landed source files, and 5 review/calibration evidence hashes verified; zero errors. Token loss, choice labels, parsed math correctness and preference metrics recomputed from the stored evidence. No model evaluation was repeated.

## Publication and ownership

Source setup `4adbb013e0f7cb233520751240e3785a1ecb8298`, freeze `0bb4ea8c586927c6fc9c5201f00fcffadc044df4`, and final generated results `94881e8c8d174793f537b06e942d66e060c401e6` landed directly on main via the source worker. The frozen-program digest remains `3d5cf4a383c0269cd4cd033f656f77d08a47bf391dea569690c9554710bd7c80`.

The source/runtime worker has finished. The coordinator owns only the final documentation/audit publication in an isolated checkout; scientific files and original reviews remain unchanged. Its [master checkpoint](../CKPT_MASTER_tmuxnone_cxd_01a10825.md) records the final pull-request landing. Local main synchronization and heartbeat shutdown are completion actions verified privately after landing, not inferred from a completed model turn.

Final coordinator report/audit pull request [#3](https://github.com/brando90/unified-training/pull/3) merged at `57782fa48c172b1de8210a330d010418273d2778` on 10-04-2026 at 23:18 PDT. The resulting main tree matched the verified branch, and local main safely fast-forwarded to that landing. One completion email was delivered at 23:19 PDT. This record-only follow-up changes no scientific artifact; its final local synchronization and heartbeat shutdown are verified privately.

## Resume contract

Do not restart this completed queue or reuse its budgets. A new experiment needs a separately specified plan, especially any capability-preservation follow-up. Preserve original model/data/checkpoint evidence and all earlier failed calibration attempts. Keep the user-designated canonical paths: frozen harness/manifests depend on them. Never reset the user's checkout or restore the preserved original draft over main.

**TLDR-end:** [unified-training: completed pilot checkpoint] All 48 training cells and 4 initial evaluations are complete and verified; no advantage over Aioli is established. Final report publication and safe local sync signals are tracked by the coordinator.
**Snapshot:**
```text
Scientific state: complete
Training cells: 48/48
Initial evaluations: 4/4
Failures / interrupted / missing: 0 / 0 / 0
Final generated-results commit: 94881e8c8d174793f537b06e942d66e060c401e6
```
