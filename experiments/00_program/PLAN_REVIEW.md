Unified-training plan review (Claude Code `claude-opus-5-5`, effort max, read-only): **FAIL**, with 4 critical and 9 major issues. As written, the 25M-unit pilot will hit the floor on its accuracy measures, the preference objective is degenerate from random initialization, and the proposed controller isn't fully defined. No training was run and no inputs were edited.

All six input hashes match `inputs.json`, and `README.md` is byte-identical to `original_proposal.md` (sha256 `26ab60d6…`). This is the one allowed recovery of the same session; it used no further tool actions.

## Critical issues (fix before freezing the manifests)

**C1. Both experiments' accuracy measures will be at floor, and the per-cell budget isn't sized to the device allowance.**
- **Budget size:** 25M forward-equivalent units is about 8.33M trained tokens (3 units per trained token).
- **ARC-Easy (AI2 Reasoning Challenge, easy split), from Pythia's official per-checkpoint evaluations:**
  - Pythia-70m scores 0.272 [0.254, 0.290] at step4 (about 8.4M tokens), identical to random initialization (0.272).
  - It is still 0.268 [0.250, 0.285] at step128 (about 268M tokens) and only reaches 0.298–0.309 at 1–2B tokens.
  - Intervals are Wald 95% intervals assuming 2,376 test items; p-val=n/a, no test run.
- **GSM8K (Grade School Math 8K):** fully pretrained GPT-3 models fine-tuned on GSM8K chain-of-thought reach 3.11% (0.3B), 4.70% (1.3B) and 6.75% (6.7B) (Ho et al., ACL 2023, Table 1; p-val=n/a). At 8M tokens, both of the plan's models should land near 0–3%: a handful of the 1,319 test items, too few to tell methods apart.
- **Experiment 02 (early checkpoint):**
  - Pythia-160m ARC-Easy is 0.436 [0.416, 0.456] at step13000 against 0.435 at the final step143000. It is already at capacity, so it can only measure forgetting.
  - The whole per-cell budget equals 4.0 Pythia optimizer steps (2,097,152 tokens per step), against 20.97B inherited tokens. This is a short fine-tuning continuation in which pretraining (PT) acts as replay.
- **Device time is not the constraint:**
  - Rough estimate, not measured, assuming 10–25% of A100 bf16 peak and excluding generation and evaluation: one 25M-unit cell takes about 0.5–1.2 min of training for 70m and 1.3–3.3 min for 160m.
  - The full matrix is 48 × 25M = 1.2B units. What limits the pilot is endpoint sensitivity, not the 48 device-hours.
  - PLAN line 65 only permits prospective *reductions* of the budget.

**C2. The preference objective is invalid from random initialization, unfair to the staged baseline, and never measured at the end.**
- PLAN line 13 runs Direct Preference Optimization (DPO) against a frozen initial reference. From random initialization that reference is roughly uniform (log V = 10.826 nats).
- Margins start at 0. Once the policy models language at H nats/token, the margin picks up a term β(log V − H)(|y_w| − |y_l|). At β = 0.1 and H = 5 that is 0.58 nats per token of length difference between chosen (y_w) and rejected (y_l) responses:
  - When the chosen response is 10 tokens longer, the pair's gradient weight is 0.0029, so it is effectively ignored.
  - When it is 10 tokens shorter, the weight is 0.997, fully saturated.
- So DPO mostly learns length and language-modelling progress, and the held-out DPO validation loss moves with PT progress (feeds into C4). My check printed the H→0 bound of 1.083 nats/token; the H-dependent form above is the accurate one.
- Staged practice anchors DPO to the model at the start of the DPO stage (the SFT output). Anchoring to the initial model handicaps `sequential` in both experiments.
- PLAN line 59 has no end-of-run preference metric. DPO compute can therefore only lower the reported scores, which favours any controller that starves DPO and leaves "preference learning" untested.

**C3. The proposed controller isn't fully defined.**
- **Learning-rate scale:** the exponentiated-gradient step size η depends on the cost unit.
  - With η = 0.5 and probe costs in forward-equivalent tokens, an update leaves the mix at .550000/.200000/.100000/.150000.
  - With costs in millions of tokens, the RL share only moves from 0.15 to 0.1493.
  - PLAN line 49 promises utility clipping, but `controller.py` lines 227–280 has no clipping or normalization.
  - The result is a "proposed method" that either equals `fixed_joint` plus overhead or swings to extremes.
- **Two different algorithms:** `controller.py` lines 8–13 say each probe starts from the same checkpoint, which is then restored (rollback). PLAN line 47 keeps probe updates in the training trajectory, and `related_work.md` line 56 calls rollback a modification. These have very different overheads.
- **One cost mapping does two jobs:** the same `costs` feeds the utility denominator (line 246) and the budget ledger (lines 271–272). Adding a shared validation overhead reverses the decision: the RL share drops to 0.124 without overhead and rises to 0.157 with +4 shared units.
- **Unspecified:** window length, probe length, number of windows, validation subset size and any smoothing.

**C4. The controller's feedback is predictably biased, mostly against RL.**
- Supervised fine-tuning (SFT) data is GSM8K (PLAN line 55). So the "RL proxy" (negative log-likelihood, NLL, of correct math responses) and the SFT component (response NLL) are the same quantity (line 51).
  - GSM8K NLL therefore gets half the weight.
  - RL is scored by teacher-forced NLL of gold solutions, which it doesn't optimize and can worsen while accuracy improves.
- The DPO component is confounded with length × PT progress (C2).
- **Adam momentum carries over:** a probe with zero gradient (all RL rewards equal, so no advantage) still moves the weights. In a toy run, validation loss went 0.903081 → 0.901382, "crediting" the probe with 1.7e-3 progress. Dividing that shared momentum effect by each objective's own cost gives extra credit to cheap objectives.
- Short probes can't credit PT for enabling later RL reward. That is a limitation to state, not a fix.

## Major issues

- **M1. Sampling probabilities are not compute shares.**
  - The controller samples which objective each update uses, but PLAN line 34 calls `fixed_joint` "the same nominal mix" as `sequential`'s compute fractions.
  - With illustrative per-update costs (PT 1, SFT 1, DPO 2.5, RL 6):
    - `fixed_joint` spends 47.4% of compute on RL, against 15% for `sequential`.
    - `uniform` spends 57.1% on RL.
    - Even the floor mix .94/.02/.02/.02 spends 10.6% on RL.
  - Matching `sequential` would need sampling probabilities .675/.245/.049/.031. The 0.02 floors and the 0.55 retention minimum are also in sampling units.

- **M2. Objectives and the optimizer protocol are under-specified.** Still undefined:
  - Sequence packing and length, the SFT template and masking, and the DPO β and per-token aggregation.
  - The RL algorithm: group size, temperature, maximum new tokens, advantage normalization, KL coefficient (I recommend 0, as CHORD uses), answer-parsing rule, and reward for truncated outputs.
  - Per-objective learning-rate multipliers, global vs per-stage warmup, and shared vs per-objective Adam moments.
  - Optimizer resets between stages, gradient accumulation, and loss aggregation.
  - Why it matters: Limozin et al. (arXiv 2604.23747) traced weak SFT baselines in several published mixed-policy papers to two framework bugs. Once fixed, plain SFT→RL beat every mixed method by +3.8 points (Qwen2.5-Math-7B) and +22.2 points (Llama-3.1-8B). The plan therefore needs equivalence tests for both bug classes and an equal, declared tuning budget for every condition, measured on a development split.

- **M3. Several named baselines are not faithful.**
  - **Aioli:** the published algorithm runs in-trajectory probes on (1−ε)·one-hot+ε·uniform mixtures, uses k shuffled sweeps, and solves A = P⁻¹β. It then normalizes A, updates p_j ∝ p_j·exp(η Σ_i Ā_ij) from a uniform start, and optimizes the unweighted sum of validation losses. The plan must fix all of these; starting from FIXED instead of uniform is a disclosed deviation.
  - **Proposed vs Aioli:** `validation_progress` is roughly Aioli with ε = 0 plus loss normalization, cost division, floors, a retention guard and a FIXED start. That bundle can't be attributed to any single change. Define it as `aioli_objective` plus one declared change (cost normalization).
  - **CHORD:** the plan's "decaying SFT assistance" is CHORD-μ (μ decays 0.9→0.05 over the first 200 steps). The paper reports CHORD-φ as its stronger variant (μ fixed at 0.1, token weight φ = p(1−p), Group Relative Policy Optimization without KL). CHORD weights both losses inside every update, which the `ProbabilityPolicy` sampling interface (lines 113–118) can't express.
  - **RPT (Reinforcement Pre-Training):** RPT starts from DeepSeek-R1-Distill-Qwen-14B. A 70m model from random initialization has no reasoning to reward, so the RPT-inspired cell in experiment 01 is uninformative.

- **M4. Important prior art and comparison methods are missing; cherry-rl is still unresolved.**
  - **TRAPO** (arXiv 2512.17636; ICLR 2026) is cited at README line 115 but absent from `related_work.md`. It mixes SFT on expert prefixes and RL on the model's own completions inside each example.
  - **RL Excursions during Pre-Training** (arXiv 2606.04272, 06-02-2026) closely overlaps experiment 02 and is missing even though `related_work.md` claims a refresh through 10-04-2026.
    - It trains models from scratch and applies RL, SFT and SFT→RL at intermediate checkpoints.
    - It finds RL effective very early.
    - Its "parallel averaging" (separate RL and SFT optimizers with independent moments, θ ← θ̄ + ½(ΔRL + ΔSFT)) beat all other tested methods while preserving general capabilities. That makes it a missing strong joint baseline.
    - Sizes and GSM8K figures (about 2%→18% at 4B tokens) come from an automated page extraction; verify before citing.
  - **Also worth adding:**
    - RL's Razor (arXiv 2509.04259) links forgetting to KL drift from the base model.
    - TR-DPO (arXiv 2404.09656) is the published soft/hard reference-update method behind the C2 fix.
    - Optional, not re-verified: GradNorm, PCGrad and Online Data Mixing.
  - **Cherry-rl:**
    - No matching paper titled or abbreviated CHERRY turned up.
    - RPT's Figure 1 contrasts a "Cherry-on-Top Cake" with a "Cherry Cake (RPT)", so RPT remains the most plausible candidate, but it is unconfirmed.
    - Cherry_LLM (NAACL 2024) is about SFT data selection, not RL.
    - Keep a blocked `cherry_rl` row until the user confirms the paper.

- **M5. Data splits, leakage and contamination.**
  - The controller's feedback, milestone monitoring and tuning/admission all draw on one "validation" split. Use three disjoint splits: controller feedback, development/admission, and test.
  - **UltraFeedback:** pin a revision after the fix. The dataset card says the latest version corrects a few hundred mislabelled completions and removes TruthfulQA-sourced prompts; the original revision `292c1632…` is contaminated. Reserve `test_prefs` (2,000 pairs) for the final preference evaluation. Run a 13-gram check of prompts and responses against the GSM8K and ARC-Easy test sets.
  - **Experiment 02:** the Pile (2020) predates GSM8K (2021), so record inherited GSM8K-test contamination as implausible rather than "unknown". WikiText-103 comes from Wikipedia, which the Pile includes, so its test set likely overlaps inherited training (not measured). Use the Pile test split for retention.

- **M6. Retention is not defined in a usable way.**
  - From scratch the guard never fires: initial loss is about 10.83 nats, so it triggers only above 11.04. Replace "retention" there with a language-modelling tax: PT validation NLL against `sequential`'s PT-only phase at equal PT tokens.
  - In experiment 02, PT on WikiText improves in-domain WikiText loss, so the guard measures adaptation rather than retention. Use Pile test NLL plus SciQ, ARC-Easy and LAMBADA declines against the untouched checkpoint.
  - The guard applies only to the proposed method. Apply it to both adaptive methods or neither, and report how often it triggers.

- **M7. Rewards will be too sparse for RL to learn.**
  - At a 1% pass rate with 8 samples per prompt, only 1 − 0.99⁸ = 7.7% of prompts get any correct sample, so about 92% of rollout compute produces no learning signal. From scratch, reward is exactly zero at first.
  - PLAN line 55's fallback ("switch to an arithmetic diagnostic if rewards are all zero") becomes a result-dependent change unless it is decided before freeze.

- **M8. Statistics and the decision rule.**
  - There is no single primary measure or decision rule.
    - Declare a primary capability endpoint, co-primary retention with a pre-declared noninferiority margin, and two primary contrasts with Holm correction.
    - Proposed vs `sequential`.
    - Proposed vs `aioli_objective`.
  - **Three seeds are too few for significance:** the exact seed-level sign-flip test's smallest possible two-sided p is 0.25 (it would be 0.0625 at n=5 and 0.031 at n=6), and the t-interval multiplier is 4.30. Use a two-stage bootstrap (resample seeds, then items) for method-level intervals; item-level bootstraps cover evaluation noise only.
  - In experiment 02 all seeds share the same initial weights, so seeds capture data-order variance only, and a single checkpoint age can't generalize.
  - **Missing cells:** report k/n, a complete-case analysis and a worst-case-imputation sensitivity check.
  - **Pareto framing:** a "better tradeoff" means beating the frontier of the static mixes, not just moving along it.

- **M9. Implementation and test gaps.**
  - **Failing test:** 13 of 14 tests pass. `test_sequential_boundaries` fails because `math.fsum((.55,.20,.10))` returns `0.8500000000000001`, so progress 0.85 maps to `dpo` instead of `rl`.
  - **No resume:** there is no `state_dict`/`load_state_dict` and the snapshot has no random-number-generator state, so the crash-resume promised in PLAN line 69 is impossible.
  - **Missing code:** no clipping, no exploration, no Aioli/CHORD/RPT adapters, and no loss-weighting interface.
  - **Works correctly:**
    - The floor-respecting KL projection matches a brute-force reference to 3.3e-16 over 20,000 random cases.
    - The input-validation tests hold.

## Concrete fixes (target file and location → minimal change)

| Target | Fix |
|---|---|
| PLAN.md L13, L51 | Replace the frozen initial reference with one rule for every condition: the reference is the policy snapshot taken at each 10% of the budget. For `sequential` this gives the standard SFT-output reference during its DPO stage. Write out exact PT/SFT/DPO/RL definitions (M2). Remove the duplicated SFT/RL component and use direct dev reward, or a proxy validated in Stage 0. Use a length-controlled DPO metric. |
| PLAN.md L28–41 | Express all mixes as compute shares; replace `chord_objective` with a hybrid slot; block `cherry_rl`; defer `rpt_inspired` and `uniform_joint`; replace the cell count with the matrix below. |
| PLAN.md L47–49 | Choose in-trajectory probes (Aioli style). Define the probe/window protocol. Normalize utilities per window and set η in normalized units. Use per-update production cost as the denominator and charge measurement overhead only to the ledger. Apply the retention guard to both adaptive methods. |
| PLAN.md L55–61 | Per-experiment endpoints and data (pilot below); pin the post-fix UltraFeedback revision; three disjoint splits; Pile test for experiment 02; add a preference endpoint; declare the primary multiple-choice scoring rule. |
| PLAN.md L65–69 | Allow prospective budget increases within the unchanged 48 device-hours. Charge padded tokens. Charge cached reference log-probabilities once. Report device-seconds and how rankings change under them. |
| PLAN.md L73 | Estimand, margins, Holm correction, two-stage bootstrap, missing-cell rule, frontier analysis (M8). |
| controller.py L8–13, L121–134 | Make the docstring and PLAN describe the same algorithm. |
| controller.py L186–191; test L32–38 | Use exact phase boundaries (integer basis points, or a comparison on integer charged/budget units). |
| controller.py L181–212, L227–280 | Add compute-share deficit scheduling; separate utility costs from ledger costs; add normalization and clipping. |
| controller.py L282–319 | Add `state_dict`/`load_state_dict`, including `self._rng.getstate()`, plus a kill-and-resume test. |
| controller.py L113–118 | Add a loss-weighting interface for CHORD-φ, parallel averaging and TRAPO. |
| related_work.md L15–30, L42–48 | Add TRAPO, RL Excursions, Limozin et al., RL's Razor and TR-DPO; specify CHORD-φ; keep cherry-rl explicitly unresolved and ask the user. |

## Suggested frozen pilot

**Stage 0 (before freeze; development split only; at most 8 device-hours; never reported as results).**
- Fix the code and tests.
- Measure throughput per objective and generation speed.
- Run a single-seed smoke of `sequential` and SFT-only to set task difficulty and the per-cell budget.
- Set the per-cell budget so the slowest (most RL-heavy) cell fits about 46 minutes. That figure comes from 48 h, minus 8 h for Stage 0, minus about 3 h of evaluation, minus a 25% margin, spread over 36 cells.

**Experiment 01 (Pythia-70m architecture, random initialization per seed).**
- PT on WikiText-103.
- SFT, RL and preferences on a procedurally generated multi-step arithmetic word-problem family. This is a *diagnostic task, not a public benchmark*: equation-only solutions, train/dev/test split by disjoint templates and seeds, and correct-vs-perturbed preference pairs matched in length.
- Primary: diagnostic test exact-match accuracy.
- Co-primary: WikiText-103 test NLL and the language-modelling tax.
- Secondary: preference accuracy and BLiMP (verify it is above chance in Stage 0).
- Reported as expected-floor public checks: GSM8K test and ARC-Easy (random-initialization reference 0.272 acc / 0.252 normalized).

**Experiment 02 (immutable Pythia-160m `step10000`).**
- Fresh optimizer with a declared warmup.
- PT replay on the Pile continuation, or on WikiText with retention measured on the Pile.
- SFT/RL on GSM8K-Aug in equation-only format. Public precedent: GPT-2 124M reaches 42.9 ± 0.2 on GSM8K test (Coconut). Run a 13-gram audit against GSM8K test.
- UltraFeedback DPO with the snapshot reference.
- Primary: GSM8K test exact match.
- Co-primary: Pile test NLL.
- Secondary: ARC-Easy, SciQ and LAMBADA change against the untouched checkpoint; `test_prefs` preference accuracy (length-normalized).

**Conditions in each experiment, seeds 0, 1, 2:**
1. `sequential`
2. `fixed_joint` (compute shares)
3. `smooth_joint` (compute shares)
4. `aioli_objective` (faithful protocol)
5. `validation_progress` (= condition 4 plus cost normalization only)
6. `sft_rl_hybrid` (RL Excursions parallel averaging preferred, CHORD-φ the alternative; choose once before freeze)
7. `cherry_rl`, blocked until the user confirms the paper

That gives 36 measured cells plus 6 blocked rows, all 42 declared in the denominator. `uniform_joint`, TRAPO and `rpt_inspired` (experiment 02 only, and only after confirmation) are optional if Stage 0 shows at least 20% slack.

**Admission gates (fixed before freeze):**
- All tests pass.
- Resume reproduces the same objective sequence and ledger.
- Ledger reconciles to within 0.1%, with at most one update of overshoot.
- Smoke development exact-match is at least 10%, at least 50 of 500 or more items correct, and at least 3× the initial model.
- At the start of RL, at least 20% of prompt groups have mixed rewards.
- The rank correlation between the DPO margin and length difference is at most 0.3 in absolute value.
- The proxy-to-development-accuracy rank correlation is at least 0.5 over at least 8 checkpoints.
- The controller moves the favoured objective's compute share by at least 0.10 within the planned windows, with overhead at most 10%.
- The 13-gram contamination check is zero after removals.
- The sum of all cell caps plus evaluation fits within 0.75 of the remaining allowance.

**Abort rules:**
- A non-finite loss marks the cell failed and keeps it in the denominator.
- A cell exceeding 1.5× its projected time is recorded as interrupted.
- Hitting the global bound leaves the remaining cells as unattempted rows.
- No retries that depend on results, and no test-based changes.
- If admission still fails after one prospective redesign, don't run the matrix; report the pilot as floor-limited.

## Checks run

- Hash verification: 6 of 6 files OK.
- Unit tests: 14 run, 1 failure.
- KL-projection brute-force check: maximum difference 3.3e-16.
- Controller probes for η scale, sampling vs compute shares, utility denominator, Adam carryover, retention reach and resume state, all with the numbers above.
- Sign-flip p-value floors and budget arithmetic.
- Pythia official zero-shot evaluation files for 16 checkpoints.
- Text extraction of Ho et al.'s Table 1.
- Primary-source checks for Aioli, CHORD, TRAPO, RL Excursions, Limozin et al., TR-DPO, RL's Razor, UltraFeedback, the Pythia README and RPT.

## Remaining uncertainty

- Throughput figures are analytic estimates; no GPU runs were allowed.
- Unverified:
  - whether GSM8K-Aug clears 10% within budget at `step10000`;
  - whether BLiMP is above chance in experiment 01;
  - the direction of UltraFeedback's length bias (the C2 argument holds for any length difference);
  - the WikiText–Pile overlap;
  - Pythia's exact learning-rate schedule.
- RL Excursions details are from an automated extraction.
- The cherry-rl identity is unresolved.
- This was a single, time-bounded review.

**Sources:** [Pythia](https://github.com/EleutherAI/pythia) · [Pythia evals](https://github.com/EleutherAI/pythia/tree/main/evals/pythia-v1) · [Aioli](https://arxiv.org/html/2411.05735v2) · [CHORD abs](https://arxiv.org/abs/2508.11408) / [HTML](https://arxiv.org/html/2508.11408v1) · [TRAPO](https://arxiv.org/abs/2512.17636) / [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/54e1381d0c0598127b90af4c940fd3d9-Paper-Conference.pdf) · [RL Excursions](https://arxiv.org/abs/2606.04272) / [HTML](https://arxiv.org/html/2606.04272) · [SFT-then-RL](https://arxiv.org/abs/2604.23747) · [TR-DPO](https://arxiv.org/abs/2404.09656) · [RL's Razor](https://arxiv.org/abs/2509.04259) · [UltraFeedback binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) · [Coconut](https://arxiv.org/html/2412.06769) · [Ho et al. 2023](https://aclanthology.org/2023.acl-long.830.pdf) · [RPT](https://arxiv.org/pdf/2506.08007) · [Cherry_LLM](https://aclanthology.org/2024.naacl-long.421/)

VERDICT: FAIL
CRITICAL_ISSUES: 4
MAJOR_ISSUES: 9
FIXES_APPLIED: 0
STRUCTURAL: SKIP
SUMMARY: The plan is well hedged but not ready to freeze: its 25M-unit pilot hits the floor (ARC-Easy at chance, GSM8K near 0), DPO with a frozen random-initialization reference learns length, and the controller's step size, rollback, cost denominator and RL proxy are undefined or biased. Fix those, add TRAPO, RL Excursions and SFT-then-RL evidence, and run a development-only Stage 0 with the gates above before freezing a 42-row matrix.

**TLDR-end:** [ut: opus55-plan-review] This was the requested single independent quality-assurance review of the unified-training plan, sent to the parent as this report. It fails on 4 critical and 9 major issues: endpoints at floor and an unsized budget, a degenerate DPO reference, an undefined controller, and biased controller feedback. Fix them, run the development-only Stage 0 with its gates, then freeze; this review stops here with no training and no edits.

**Snapshot:**
```
inputs.json sha256: 6/6 OK; README.md == original_proposal.md (26ab60d6…)
unittest: Ran 14 tests — FAILED (failures=1) test_sequential_boundaries: 0.0 != 1
fsum(FIXED[:3])=0.8500000000000001 -> phase at progress .85 -> dpo
eta=0.5, costs in tokens -> p = .550000/.200000/.100000/.150000 (no movement)
fixed .55/.20/.10/.15 sampling -> compute shares pt=.289 sft=.105 dpo=.132 rl=.474
shared overhead 0 -> p_rl=.124 | overhead 4 -> p_rl=.157
Adam zero-gradient probe: loss 0.903081 -> 0.901382 (credited 1.70e-03)
sign-flip min p: n=3 .2500 | n=5 .0625 | n=6 .0312
Pythia-70m ARC-E acc: step0 .2718 | step4 .2723 | step128 .2676 | step1000 .3085
Pythia-160m ARC-E acc: step13000 .4364 | step143000 .4351
Ho et al. GSM8K Fine-tune-CoT: 0.3B 3.11 | 1.3B 4.70 | 6.7B 6.75
```
