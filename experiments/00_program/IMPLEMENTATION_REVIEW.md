Unified-training pilot implementation review (Pythia-70m from scratch and Pythia-160m step10000; 8 objective-mixing methods × 3 seeds = 48 cells; reviewer Claude Opus 5.5 at maximum effort): **FAIL** pending six must-fix items; 41/41 deterministic tests pass, nothing has been measured, and the plan review is still incomplete.

## Verdict: FAIL

**Basis.** I read the enclosed source and the supplied test log (41/41 OK). I executed nothing. Paths are under `experiments/00_program/` unless noted. A *cell* is one experiment × method × seed unit of training plus evaluation. Priorities run from P0 (highest) to P3.

Most of the objective math is correct (see "Verified by reading" below). The artifact fails for these reasons:
- one reward bug in the Reinforcement Pre-Training (RPT)-inspired method;
- an admission gate that does not tie frozen hashes to the landed, reviewed code;
- unpinned model precision;
- three gaps in durability and denominator completeness.

**PASS condition:** fix every P0 and P1 item, each with the deterministic test named. Land P2 before freeze. P3 is hygiene. The incomplete plan review remains a separate admission blocker whatever this verdict says.

## P0: correctness and admission integrity

**P0-1 · The RPT reward rejects most correct predictions (actual bug).** `objectives.py::prefix_reward`
- **Evidence:** the code strips one leading space from the prediction, but `truth=tokenizer.decode(continuation_ids,...)` keeps its own leading space. With byte-pair-encoding (BPE) tokenizers the next corpus token usually starts with a space.
  - So `Prediction: on the mat` becomes `b"on the mat"`, which is not a prefix of `b" on the mat"`, and the reward is 0.
  - Only an answer with two spaces after the colon scores.
  - Continuations that start with punctuation or a subword piece score normally, so the reward is biased by token type, not just sparse.
- **Fix:** compute `boundaries` on the raw truth, then strip one leading space from the truth as well:
  ```python
  if truth.startswith(b' '):
      truth = truth[1:]
      boundaries = {b - 1 for b in boundaries if b > 1}
  ```
  Add tests with a fake tokenizer that emits leading-space tokens:
  - whole-token prefix → 1
  - mid-token prefix → 0
  - punctuation-initial continuation → 1
  - missing `Prediction:` → 0
  - empty prediction → 0

  Update the matching sentence in IMPLEMENTATION.md.

**P0-2 · Freeze can certify unreviewed code (gate gap).** `configure.py::main`, `supervisor.py::validate_gates`
- **Evidence:**
  - Freeze only runs `git merge-base --is-ancestor <landed> origin/main`, without fetching first, and then hashes whatever is on disk.
  - `validate_gates` only checks that `setup_landed_commit` is non-empty. Uncommitted edits to `*.py` or `PROTOCOL.md` would be frozen and run as "reviewed".
  - `calibration.json` does not record which source produced it.
  - `IMPLEMENTATION.md` holds the equations that PROTOCOL.md delegates to, but it is not hashed. The prepared `.jsonl` splits are not hashed either; only `data_manifest.json` is.
- **Fix:**
  - Run `git fetch origin main` first.
  - For every source and protocol file, require `sha256(git show <landed>:<path>) == digest(path)` and fail closed otherwise. Repeat this check in `validate_gates`.
  - Store source digests and the commit in `calibration.json`, and require them to match at freeze.
  - Add `IMPLEMENTATION.md` to the hash set.
  - Verify each prepared split against `data_manifest.json['splits'][k]['sha256']` at run time.

**P0-3 · Model precision is unpinned (depends on environment; verify before the smoke run).** `training.py::make_models`
- **Evidence:** `from_config(...)` and `from_pretrained(...)` pass no dtype.
  - Pythia's `config.json` declares float16 (fp16) storage.
  - Transformers releases that default to `dtype="auto"` would load the step10000 checkpoint, and possibly the scratch model, in fp16.
  - AdamW (Adam with decoupled weight decay) would then update fp16 master weights with no loss scaling, and the two experiments could silently differ in precision.
  - The tests build an fp32 config, so they cannot detect this.
- **Fix:**
  - Pass float32 (fp32) explicitly to both constructors.
  - After `.to(device)`, assert that every parameter is fp32.
  - Record the torch, transformers and CUDA (NVIDIA graphics processing unit (GPU) runtime) versions, plus the parameter dtype, in `calibration.json`, `training.json` and the freeze record.

## P1: durability and complete denominators

**P1-1 · Recovery, adoption and resume are not implemented.** `supervisor.py::main`, `run_cell`; `training.py::Engine.resume`
- **Evidence:**
  - A restarted supervisor raises "adopt it" or "needs explicit checkpoint reconciliation", but no adopt or reconcile code exists.
  - `run_cell` rejects any cell that already has a receipt, and nothing calls `Engine.resume`.
  - Children are launched with default `Popen` in the supervisor's process group. A terminal hang-up can kill both and leave a permanent `running` receipt.
  - So one supervisor crash stops the rest of a ~48-hour queue.
  - IMPLEMENTATION.md says "exact resume is supported", but that is not true in practice. The milestone iterator lives in `run_cell` local variables (the saved `Engine.next_milestone` is never used), and the training timer restarts from zero.
- **Fix:**
  - **(a) Add an explicit `--recover` mode** that holds the same lock and does three things:
    - waits on alive registered children by identity;
    - marks `running` receipts whose process is dead as `status='interrupted'`, recording the last checkpoint step and last event;
    - continues the pending cells, with the queue budget measured from a persisted `time.time()` start (`monotonic` resets on reboot).
  - **(b)** Launch with `Popen(..., start_new_session=True)` and use `os.killpg` on timeout.
  - **(c) Implement checkpoint resume as a separately named recovery condition, or delete the resume claim.** The recovery version needs:
    - the milestone index and accumulated wall seconds stored in engine state;
    - a `resumed_from_step` event;
    - events after the checkpoint step marked as superseded;
    - evaluation-only recovery from the `stage='evaluation'` checkpoint.
  - **Test:** kill a fake child mid-cell; `--recover` must write `interrupted` and then run the next pending cell.

**P1-2 · Calibration does not represent whole-run wall time.** `calibrate.py::main`
- **Evidence:** throughput comes from 48 updates at progress ≈ 0. That sample misses:
  - the direct preference optimization (DPO) and reinforcement learning (RL) phases of `sequential`;
  - the RL growth in `smooth_joint` (.05 → .45);
  - adaptive drift, which can reach .94 on one objective with .02 floors;
  - per-update `event`/`progress` fsyncs;
  - checkpoint input/output (I/O). Every 32 updates, `run_cell` writes model, reference and two AdamW states. Computed from parameter counts, that is ≈1.1 GB (70m) or ≈2.6 GB (160m) in fp32, and the smoke never exercises it.

  RL updates are generation-bound. By my unmeasured estimate they cost roughly 10–20× more time per charged token than pretraining (PT), so cells that drift toward RL can exceed the frozen `train_timeout_seconds` and become avoidable infrastructure failures.

  The evaluation extrapolation (`max seconds × 4515/64 × 2`) is called "deliberately conservative", but it is not. The full denominator has a larger share of GSM8K (Grade School Math 8K) items (1319/4515 = 29%) than the smoke (16/64 = 25%), and GSM generation dominates time. Only the 2× margin covers the gap.
- **Fix:**
  - Time each update by objective (pt, sft, dpo, rl, rpt, hybrid) and time each validation.
  - Bound each method by integrating its schedule (sequential, smooth) or by the slowest mix its floors allow (adaptive methods).
  - Measure one checkpoint write per architecture and multiply by the expected count. Alternatively, checkpoint by wall time (e.g. every 10 minutes) and drop the frozen reference from periodic checkpoints (rebuild it from seed/revision and verify a stored hash).
  - Time `evaluate` on the full validation splits (491/256/570/715 items) and extrapolate per dataset to 564/256/2376/1319. Keep the 2× margin.
  - If the worst-case bound cannot fit, predeclare a wall-clock stop as a distinct `budget_incomplete` outcome. Such cells are still evaluated and kept in the denominator.

**P1-3 · Initial diagnostics are conditional and have no time limit.** `supervisor.py::main`
- **Evidence:**
  - The four declared untouched-initial evaluations run only if `totals['complete']==48`.
  - Their wait loop has no timeout.
  - One failed cell therefore silently drops four declared evaluations, and a hang blocks finalization, even though the freeze reserved `4 × evaluation_timeout_seconds` for them.
- **Fix:**
  - Always run them after the queue.
  - Enforce `4 × evaluation_timeout_seconds + 60`.
  - Record each evaluation's status, so any missing one is explicit.

## P2: fix before freeze

**P2-1 · Milestone validation cost leaks into probe cost (actual bug, small effect).** `run_cell` + `Engine.advance`
- **Evidence:** the fixed validations at .25/.5/.75/1.0 of budget can land between a probe column's before and after validations.
  - `cost = work.total − start_cost` then charges about one extra validation to that objective.
  - This lowers its `validation_progress` utility or rescales its Aioli column.
- **Fix:** charge milestones under a `milestone_validation` kind and subtract that work from the probe cost, or defer milestones to a column boundary. Add a unit test.

**P2-2 · Failure outcomes are lumped together.** `supervisor.py::main`
- **Evidence:** every incomplete cell becomes `infrastructure_failure`. That includes deterministic exceptions (`FloatingPointError`, invalid validation losses, context-overflow raises) and supervisor-imposed timeouts, for which only the return code is logged.
- **Fix:**
  - Wrap `run_cell` so it writes `failure_class`, the exception type and message, stage, step and charged tokens to the receipt before re-raising.
  - Have the supervisor add `reason='cell_timeout'` when it terminates a child.
  - Classify negative return codes as `infrastructure_interruption`.

**P2-3 · Two of the four official evaluations never reach the report.** `report.py::main`, `update_ledgers`
- **Evidence:**
  - Preference implicit and raw accuracy (256 pairs) are computed but never summarized.
  - WikiText negative log-likelihood (NLL) and preferences have no paired comparisons.
  - The ledger shows only ARC-Easy (AI2 Reasoning Challenge) and GSM8K.
  - Initial diagnostics are never reported.
- **Fix:**
  - Add both preference metrics (declare one primary) and WikiText to the seed summaries, the paired comparisons (block-paired ratio NLL for WikiText) and the ledger.
  - Include initial diagnostics as reference rows.
  - Make `report.py` use the supervisor's completion predicate and refuse to run unless the freeze validates.

**P2-4 · Evaluation completeness checks are partial.** `evaluation.py::evaluate`
- **Evidence:**
  - The only check is an `assert`, which `python -O` strips, and it covers ARC 2376 and GSM 1319 only. WikiText (564) and preference (256) counts are unchecked.
  - Evidence files are opened in append mode, so any rerun mixes attempts.
- **Fix:**
  - Raise explicitly when any of the four counts differs from `data_manifest.json`.
  - Open evidence files exclusively (`'x'`) or use attempt-numbered directories.

**P2-5 · The overlap audit cannot see benchmark questions embedded in templates.** `prepare_data.py::main`
- **Evidence:**
  - Hashing whole prompts misses a GSM8K or ARC test question that sits inside an UltraFeedback instruction template or response.
  - GSM training prompts are never checked against the preference test pool.
- **Fix:**
  - Check normalized substring containment of test questions (≥ ~40 characters) in preference prompts, preference responses and WikiText training lines.
  - Drop GSM-train prompts that appear in the preference test pool.
  - Log the counts.

**P2-6 · The calibrated budget has no scientific floor.** `calibrate.py::main`
- **Evidence:** calibration shrinks `resolved_budget_tokens` to whatever fits and still reports `fits_48_hours=True`. A slow host could therefore admit a budget with only a few controller iterations; one Aioli iteration is 48 probe plus 64 production updates.
- **Fix:** predeclare a minimum budget or a minimum number of adaptive-controller iterations, and fail closed below it.

**P2-7 · "Exact resume" holds only on CPU and is narrowly tested.**
- **Evidence:**
  - Nothing sets `torch.use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG` or a deterministic attention backend, so GPU replay is not bit-for-bit.
  - `test_checkpoint_exact_optimization_resume` covers only PT and supervised fine-tuning (SFT) updates on CPU.
- **Fix:**
  - Either enable deterministic algorithms (measure the cost) or document "full state restored; bit-for-bit only on CPU".
  - Add tests that resume mid-probe through `advance()` for both adaptive methods, and across an RL update.

**P2-8 · Strict on-policy alignment is untested on the real generation path.**
- **Evidence:** the tests patch out `generate`.
- **Fix:**
  - Add a tiny-model test with a left-padded sampled batch (`return_dict_in_generate=True`, `output_logits=True`). The log-softmax at the sampled tokens must match right-padded `token_logps` within fp32 tolerance. This catches position, mask or logits-processor mismatches, such as a default `top_k` that survives.
  - In production, log the mean absolute gap between sampling and scoring log-probabilities for each RL update. bfloat16 (bf16) and key–value (KV) cache arithmetic make it nonzero, so disclose it.

## P3: hygiene

- **Freeze ordering:** `configure.py` sets `status:'frozen'` on manifests before checking the 48-hour bound. Check first.
- **Progress writes:** `common.progress` does an unlocked read-modify-write from two processes (lost updates) and fsyncs every update. Add a lock and throttle to ~30 s.
- **Directory durability:** `atomic_json` and `Engine.checkpoint` should fsync the directory after `os.replace`.
- **Receipt race:** `run_cell` has a time-of-check-to-time-of-use (TOCTOU) window between `receipt.exists()` and writing the running receipt, with Engine construction in between. Use `O_CREAT|O_EXCL` or a per-cell lock.
- **Queue completeness:**
  - When the global bound is hit, mark the remaining cells `not_started_global_bound`.
  - Order the queue seed-first across experiments and methods, so partial completion stays balanced.
- **Ledger status:** the status line in `update_ledgers` always says "in progress".
- **Report robustness:**
  - `report.py`: `diffs.append(paired['estimate'])` raises `KeyError` when `paired_items` returns a mismatch status.
  - Assert that the paired n equals 2376 (ARC) and 1319 (GSM).
- **GSM ids:** GSM ids are hashes of the question text. Use the official positional index and keep the hash as a separate field, so duplicate questions cannot collapse during pairing.
- **Optimizer resume:** the manual `.to(device)` loop over optimizer state in `Engine.resume` moves `step` tensors to the GPU, which `load_state_dict` deliberately keeps on CPU. Delete the loop.
- **Validation loss check:** `Engine.validate` rejects a DPO loss that underflows to exactly 0. Accept `>=0` after the initial validation.
- **RL summary:** add per-cell RL/RPT update counts, zero-gradient skips, mean reward and truncations to `training.json`.
- **GPU recheck:**
  - PROTOCOL requires a GPU availability recheck, but no code performs it.
  - Record `torch.cuda.mem_get_info()` in each receipt and fail fast below a threshold.
  - Have `calibrate.py` take the owner lock.
- **Wording:** the smoke run trains on the training splits; only its scoring is validation-only.
- **Checkpoint stage marker:** the post-evaluation checkpoint rewrites gigabytes just to change `stage`. Write a small marker file instead.

## Disclosed adaptations (not defects)

- **Aioli:**
  - Disclosed: heterogeneous objectives, relative losses, per-column cost normalization, .02 floor, no rollback, probes confounded by order.
  - Algebra checked: for W = (α−β)I + β11ᵀ with α + 3β = 1, A_i = (Δ_i − β·ΣΔ_i)/(α−β) is the exact inverse, and the 9/1/1/1-of-12 probe mixture gives α = .75.
- **`validation_progress`:** the SFT and RL validation components are the same four GSM rows, so that signal is counted twice. There are four rows per component.
- **CHORD (Controllable Harmonization of On- and Off-Policy Reinforcement Learning via Dynamic Weighting):**
  - Disclosed: replay mix, μ = .9(1 − progress), φ = p(1 − p).
  - Add one disclosure: the RL term sums over the sequence while the SFT term averages over tokens, so μ does not act at the magnitude of the paper's mixing weight.
- **DPO reference:** the reference is the initial model. Implicit accuracy is therefore 0 at initialization by construction, and raw preference favors shorter responses.
- **Accounting and scoring choices:**
  - ARC scores normalized by token length.
  - WikiText context resets every 512 tokens.
  - Padded tokens are charged.
  - Selection probability is not compute share.
  - RL updates with all-zero advantages are skipped.
  - AdamW momentum is shared across objectives: the estimator is on-policy, but the optimizer carries state.
- **Tiny-model floors:** GSM and RPT rewards will mostly be zero, so many RL updates will be skipped.

## Verified by reading

**Objectives:**
- The causal shift and position-based masks are correct; padding is never scored.
- SFT trains on response tokens only, including end-of-sequence (EOS).
- The DPO formula is correct, with a no-grad reference.
- Advantages use leave-one-out REINFORCE (RLOO), and the policy gradient is a sequence-sum score function.
- Sampling uses temperature 1, top_k 0 and top_p 1, with truncation at the first EOS.

**Accounting:**
- A forward pre-hook counts generation work.
- Backward is charged at 2× only when it actually runs.

**Evaluation and data:**
- Terminal evaluation has its own ledger, uses full ARC and GSM denominators, and raises on overflow instead of filtering.
- Splits are built with overlap removal.
- Controllers see validation data only.

**Process integrity:**
- Process identity uses the process identifier, start ticks and boot id.
- Completion is never inferred from the exit code alone.
- Each experiment's ledger has all 24 rows.

## Pending operational evidence

1. The plan review is incomplete (no verdict). This is an admission blocker.
2. Real-GPU smoke run and a `calibration.json` with per-objective rates (P1-2) and `fits_48_hours`.
3. Library versions on the run host and the fp32 parameter assertion (P0-3).
4. Landed commit and a freeze record bound to file contents (P0-2).
5. Recovery drill: kill the supervisor and a child mid-cell, and confirm that `--recover` handles it (P1-1).
6. Disk headroom: ≈27 GB (experiment 01) + ≈62 GB (experiment 02) of retained fp32 checkpoints, plus temporary files.
7. GPU availability check at launch, and an external monitor for `watchdog.json` staleness.
8. The sampling-vs-scoring log-probability gap, measured on GPU (P2-8).

**TLDR-end:** [ut-pilot: impl-review] FAIL. Fix the RPT leading-space reward bug, tie frozen hashes to the landed commit, pin fp32 model loading, and add supervisor recovery, always-run time-limited initial diagnostics and per-objective calibration. 41/41 tests pass; the GPU smoke run, calibration, freeze and the incomplete plan review are still pending.

**Snapshot:**
```
objectives.py::prefix_reward  if prediction.startswith(' '):prediction=prediction[1:]        # truth keeps b' ...'
configure.py::main            ['git','merge-base','--is-ancestor',args.landed_commit,'origin/main']  # no content check
training.py::make_models      AutoModelForCausalLM.from_pretrained(spec['id'],revision=...,config=config,...)  # no dtype
supervisor.py::main           raise RuntimeError('interrupted supervisor needs explicit checkpoint reconciliation')
supervisor.py::main           if complete:   # initial diagnostics only at 48/48; wait loop has no timeout
calibrate.py::main            for _ in range(48): d=e.advance()   # progress≈0 only; no checkpoint I/O measured
report.py::main               aggregate keys: arc, gsm, wiki_nll  # preference metrics absent
tests                         41/41 OK (9 Aioli + 17 controller + 15 training/objectives)
```
