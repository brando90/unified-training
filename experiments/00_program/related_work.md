# Unified training: related work and baseline decisions through 10-04-2026

**Doc link:** <https://github.com/brando90/unified-training/blob/main/experiments/00_program/related_work.md>

**TLDR:** Joint supervised and reinforcement learning, reinforcement objectives on pretraining text, and adaptive data mixing already have substantial prior art. The testable project contribution is whether adaptive integration of all four objectives from initialization improves the compute–capability tradeoff over credible matched baselines. Aioli is mandatory; “CHERRY-RL” remains an unresolved paper identity, with Reinforcement Pre-Training a plausible candidate. No experimental superiority has been established here.

Last verified: 10-04-2026. This is a targeted, primary-source refresh, not an exhaustive systematic review. Dates below distinguish first public versions from later conference publication or revision. Paper results are author-reported, not reproduced by this project. Confidence intervals and significance tests are unavailable here unless explicitly reported; no new statistical tests were run on published aggregates (`p-val=n/a`).

## Research question and terminology

The project asks whether one training trajectory that includes **pretraining (PT)**, **supervised fine-tuning (SFT)**, **direct preference optimization (DPO)**, and **reinforcement learning (RL)** can outperform staged training at a matched resource budget. DPO learns from preferred/rejected response pairs; RL learns from sampled responses and rewards. These are distinct objectives. A synthetic correctness preference is a preference-learning test, not evidence of alignment with human preferences.

An adaptive controller changes objective sampling probabilities or loss weights using training/validation information. It must never inspect test labels. “From scratch” means randomly initialized model weights; a public base or reasoning model is not from scratch. “Early checkpoint” means an explicitly identified intermediate pretraining checkpoint whose prior training tokens and optimizer-state availability are recorded.

## Baseline map

| Work and verified date | Relevant mechanism and starting point | Consequence for this project |
|---|---|---|
| [Korbak et al., Pretraining Language Models with Human Preferences](https://proceedings.mlr.press/v202/korbak23a.html), first version 02-16-2023; International Conference on Machine Learning (ICML) 2023 | Preference-conditioned language modeling from initialization; five objectives across three preference tasks | Mandatory conceptual anchor and conditional-training control; it does not establish four-objective reasoning superiority |
| [Aioli: A Unified Optimization Framework for Language Model Data Mixing](https://arxiv.org/abs/2411.05735v2), first version 11-08-2024; revision 04-21-2025; International Conference on Learning Representations (ICLR) 2025 | Online estimation of how training-domain mixtures change validation-domain losses | Mandatory adaptive mixing comparator; extending it from cross-entropy data groups to heterogeneous objectives is an adaptation |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948), first version 01-22-2025; revision 01-04-2026 | Reinforcement post-training of an already pretrained base; R1 includes two SFT and two RL stages | Use a clearly named matched sequential control; do not call PT→SFT→DPO→RL a reproduction of DeepSeek-R1 |
| [UFT: Unifying Supervised and Reinforcement Fine-Tuning](https://arxiv.org/abs/2505.16984v2), first version 05-22-2025; revision 10-19-2025 | Unified Fine-Tuning (UFT) combines exploration with supervised hints; post-training | Comparator for small models whose unassisted success probability is low; the integrated-training idea itself is prior art |
| [Reinforcement Pre-Training](https://arxiv.org/abs/2506.08007), 06-09-2025 | Reinforcement Pre-Training (RPT) rewards reasoning followed by a correct text continuation | Provisional candidate for the user's “cherry” reference; include a clearly labeled adapted control pending identification |
| [SRFT: A Single-Stage Method with Supervised and Reinforcement Fine-Tuning for Reasoning](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a79699db176ed0efc04a9da171e52112-Abstract-Conference.html), preprint 06-24-2025; ICLR 2026 | Supervised Reinforcement Fine-Tuning (SRFT) trains on demonstrations and sampled rollouts with entropy-aware weighting | Closest current comparison for simultaneous SFT and reasoning RL; reserve a real implementation comparison for the benchmark phase |
| [On-Policy RL Meets Off-Policy Experts: Harmonizing Supervised Fine-Tuning and Reinforcement Learning via Dynamic Weighting](https://arxiv.org/abs/2508.11408), first version 08-15-2025 | CHORD, Controllable Harmonization of On- and Off-Policy Reinforcement Learning via Dynamic Weighting, regulates demonstration influence during RL | Strong smooth-transition comparator, particularly from instruction-tuned checkpoints |
| [Reinforcement Learning on Pre-Training Data](https://arxiv.org/abs/2509.19249), first version 09-23-2025 | Reinforcement Learning on Pre-Training data (RLPT) predicts text segments using corpus-derived rewards | Broader text-continuation competitor; include in benchmark-stage scope review alongside RPT |
| [RLP: Reinforcement as a Pretraining Objective](https://arxiv.org/abs/2510.01265v2), first version 09-26-2025; revision 03-01-2026; ICLR 2026 | Reinforcement Learning Pretraining (RLP) rewards sampled reasoning by increased predictive log-likelihood; applied late in pretraining | Directly relevant to continued-training experiments; distinguishes useful latent reasoning from mere reward on the next answer |
| [Olmix: A Framework for Data Mixing Throughout LM Development](https://proceedings.mlr.press/v306/chen26dd.html), first version 02-12-2026; ICML 2026 | Proxy-run mixture optimization and reuse when the domain inventory changes | Current data-mixing comparison; fixed-domain objective mixing does not test its main reuse contribution |
| [Blending Supervised and Reinforcement Fine-Tuning with Prefix Sampling](https://proceedings.mlr.press/v306/huang26g.html), ICML 2026 | Prefix-RFT (prefix-guided reinforcement fine-tuning) uses demonstration prefixes to support exploration | Useful secondary baseline when sparse rewards prevent early-model learning |
| [Understanding Reasoning from Pretraining to Post-Training](https://arxiv.org/abs/2607.16097v2), first version 07-17-2026; revision 08-09-2026 | Controlled chess pipeline with 5-million–1-billion-parameter models, plus 1-billion-parameter math transfer; relates pretraining to later RL | Supports a checkpoint-age sweep; also motivates the competing hypothesis that delaying RL until a stronger foundation can be better |

## What Korbak et al. actually supports

**Preferences can usefully enter the original learning process, rather than only repairing it later.** The [full paper, Sections 3–5](https://proceedings.mlr.press/v202/korbak23a/korbak23a.pdf) trains 124-million-parameter models on 3.32 billion tokens. Its preference tasks concern toxicity, personally identifiable information, and Python style violations. Conditional training attaches a preference condition to text and requests the desirable condition at generation time. This motivates providing target behavior information early.

One reported toxicity-score comparison is 0.0141 for ordinary maximum-likelihood training versus 0.0011 for conditional pretraining; these are scorer outputs, not fractions of people harmed. Evaluation uses 4,096 samples in the principal unconditional-generation procedure. A 95% interval for this comparison is unavailable here; `p-val=n/a`.

**The result is neither an optimization theorem nor a universal absence of capability costs.** Section 4.3 reports a less favorable picture on HumanEval code correctness, where conditional training is not the best preference-training method. Section 5 compares later feedback training from partially pretrained checkpoints. It does not test DPO plus instruction learning plus online reasoning RL from random initialization. Consequently, the paper motivates our hypothesis without proving global optimization, freedom from forgetting, or superiority over current methods.

The authors' [released implementation](https://github.com/tomekkorbak/pretraining-with-human-feedback) provides provenance, but its historical external-model callbacks must not be invoked automatically. This project can use deterministic local scorers and existing labels.

## Resolve “CHERRY-RL” without misattribution

**The exact requested identity is not verified.** Searches for `CHERRY-RL`, `CHERRY RL`, and cherry combined with language-model pretraining and SFT/RL returned no matching joint-training paper. The name also denotes an unrelated reinforcement-learning library. This is a search result, not proof that no such paper exists.

The [RPT paper's Figure 1](https://arxiv.org/html/2506.08007v1) explicitly depicts reinforcement learning as the cherry on a cake, making it a plausible interpretation. However, its experiments start from DeepSeek-R1-Distill-Qwen-14B, use mathematical text, and reward a predicted byte prefix ending at a true token boundary. They do not initialize a general language model randomly. Its limitation section acknowledges reasoning-model initialization.

Use the label **RPT-inspired adaptation, provisional “CHERRY-RL” candidate** until the exact reference is established. A tiny next-token policy-gradient control is not a reproduction of that 14-billion-parameter reasoning experiment. CHORD is another verified paper combining SFT and RL, but similarity in spelling alone does not establish the user's intended reference.

## Implementing and naming Aioli fairly

**Aioli learns cross-domain effects, not just which loss is currently largest.** The [upstream trainer](https://github.com/HazyResearch/aioli/blob/main/trainer/aioli_trainer.py) uses short randomized sweeps through smoothed one-domain mixtures, measures every validation-domain loss change, and solves for a transfer matrix. These probe updates advance the actual training trajectory and consume budget.

For validation domain `i` and sampled training mixture `j`, measure the before-minus-after loss drop `D[i,j]`. Let `W[j,k]` be the fraction of training group `k` in probe mixture `j`. Solve `W A[i,:] = D[i,:]`; average probe sweeps. Without exponential moving-average smoothing, the weight update is proportional to `w[j] exp(eta sum_i A[i,j])`. Upstream shifts a negative global minimum to zero and normalizes the matrix; implementations need numerical guards for degenerate cases.

Using DPO and RL as training groups changes the original setting. Call this **Aioli-objective-adapted**, document validation-loss normalization, and retain a faithful cross-entropy-domain Aioli comparison for the natural-language phase. A diagonal-only estimate is a separate ablation. Cost-aware weighting, loss-scale normalization, lower probability bounds, or rollback probes must be disclosed as modifications.

The [camera-ready paper](https://arxiv.org/abs/2411.05735v2) reports mean test-perplexity improvement of 0.27 points over stratified sampling across 6/6 datasets. A 95% interval for that aggregate is unavailable here; `p-val=n/a`. This motivates retaining stratified/static mixing as a serious baseline. It does not support the blanket claim that all previous mixing methods fail universally.

## Bounded experimental comparisons

**A mechanism pilot can be small while the final scientific target remains broad.** The [execution plan](PLAN.md) proposes eight conditions: sequential, uniform joint, fixed joint, smooth joint, Aioli-objective-adapted, validation-progress control, CHORD-objective-adapted, and RPT-inspired. At two initializations and three seeds this is 48 training cells. Compare the same architecture within each experiment, tokenizer, available data, held-out examples, and inclusive resource accounting. Freeze the manifest before measurement, including failures. This literature review recommends mandatory CHORD coverage in that bounded matrix; it does not add cells or change a frozen manifest.

Preference-conditioned training should be the next fidelity baseline because it isolates the Korbak motivation without assuming DPO behaves identically. It is absent from the initial eight-condition plan and remains required before claiming broad preference-pretraining superiority. Likewise, adapted CHORD/RPT results support only the declared local variants, not a claim of reproducing or defeating the original published systems.

For a joint SFT/RL pilot control, either implement the complete method or name the simplification explicitly. [SRFT's method](https://arxiv.org/html/2506.19767v1) includes demonstration supervision, off-policy reinforcement updates on demonstrations, and positive/negative self-rollout updates; it is not merely any entropy-weighted sum of two losses. [Official SRFT code](https://github.com/fyqqyf/SRFT) is available. [CHORD's method and code link](https://arxiv.org/abs/2508.11408) offer another reproducible integration baseline. The first pilot need not run every paper in the map, but cannot support a claim of beating them all.

**Resource equality must count generation and controller costs.** Match measured accelerator time or estimated floating-point operations, with source tokens, supervised target tokens, generated rollout tokens, reference-model scoring, probe updates, and validation overhead separately reported. Equal optimizer steps alone are inadequate. If a tiny pilot uses step matching for practicality, label it as a functional comparison and withhold compute-efficiency claims.

**The early-checkpoint experiment asks a distinct question.** Use common checkpoints at several recorded pretraining ages, then compare matched continuations. [Pythia's official release](https://github.com/EleutherAI/pythia) exposes initialization and many intermediate steps, including 1,000-step intervals; its final checkpoint is step143000. Intermediate optimizer states are not generally served, so resetting the optimizer must be identical across compared continuations and explicitly recorded. Published historical evaluations also use an older evaluation harness, requiring fresh pinned evaluation rather than direct scoreboard comparison.

A modern fully pretrained small base can help establish a benchmark signal, but must be named a continued-training experiment rather than an early-checkpoint experiment. Use held-out exact-answer tasks with attainable nonzero baseline accuracy, plus general-language loss and preference metrics. Competition mathematics with all-zero accuracy cannot distinguish the methods. A validation-only task-difficulty pilot may select the test regime before the final test set is frozen.

**A positive result must survive the strongest matched control and retention checks.** Report paired differences with 95% intervals, individual seed values, and a named prespecified test against zero difference or a declared noninferiority margin. A loss reduction supports better predictive fit; exact-answer gains support that benchmark's capability; neither establishes general intelligence or universal freedom from forgetting. Define forgetting as a measured decline against an earlier checkpoint on a frozen retention set. Retain negative and interrupted cells in the declared denominator.

## Implementation provenance and remaining benchmark work

| Method | Verified upstream source | Status at this literature refresh |
|---|---|---|
| Aioli | [HazyResearch/aioli](https://github.com/HazyResearch/aioli) | Source and trainer inspected; pin an immutable commit before reproduction |
| Conditional preference pretraining | [tomekkorbak/pretraining-with-human-feedback](https://github.com/tomekkorbak/pretraining-with-human-feedback) | Official source identified; local adaptation requires a differences record |
| SRFT | [fyqqyf/SRFT](https://github.com/fyqqyf/SRFT) | Official source identified; complete objective needed for faithful comparison |
| UFT | [liumy2010/UFT](https://github.com/liumy2010/UFT) | Linked by the original paper |
| CHORD | [Trinity-RFT example](https://github.com/modelscope/Trinity-RFT/tree/main/examples/mix_chord) | Linked by the authors' paper |
| RLP | [NVlabs/RLP](https://github.com/NVlabs/RLP) | Landing repository inspected; visible tree contains documentation/paper and says code release pending; do not report executable training code as available |
| Olmix | [allenai/olmix](https://github.com/allenai/olmix) | Runnable toolkit documented; upstream warns its public migration remains in progress |
| RPT / RLPT | [RPT](https://arxiv.org/abs/2506.08007), [RLPT](https://arxiv.org/abs/2509.19249) | Papers verified; executable official training implementation not verified in this refresh |

Before publication, update the search, settle the “CHERRY-RL” identity, compare the current implemented manifest against this baseline map, pin upstream revisions, and report adaptations in the main result table. The open question is whether joint training provides a better tradeoff under our measured conditions, not whether the desired conclusion can be guaranteed in advance.

## Reviewed addendum and changed pilot design (10-04-2026)

This section supersedes earlier tentative method mappings; the current contract
is `PLAN.md` and `IMPLEMENTATION.md`. All new comparisons are pilot adaptations,
not tuned reproductions. These primary sources were verified in the coordinator's
six-paper literature addendum, then reconciled against the implementation.

- [RL Excursions during Pre-Training](https://arxiv.org/html/2606.04272v1)
  is the closest missing comparison. Its main scale is one billion parameters
  and 50 billion pretraining tokens. Its parallel method computes reinforcement
  learning (RL) and supervised fine-tuning (SFT) updates at the same snapshot
  using separate optimizer moments, then averages the two parameter deltas.
  `parallel_joint` implements that mechanism with matched replay and charged
  work, replacing uniform sampling before freeze. Early-checkpoint figures in
  that paper use different prompt formats before/after training, and some plots
  retain favorable seeds; they do not validate our small-model reward readiness.
  Our full three-seed denominator is never filtered by performance.
- [Trust-Region Adaptive Policy Optimization (TRAPO)](https://arxiv.org/html/2512.17636v1)
  attenuates expert-token gradients by detached `min(p/alpha,1)` and adaptively
  provides expert prefixes to unsuccessful rollout groups. It is an important
  **untested competitor**. A generic SFT/RL mixture or fixed prefix is not TRAPO.
  Our development check retains unassisted prompts, so no TRAPO claim is made.
- [SFT-then-RL Outperforms Mixed-Policy Methods](https://arxiv.org/html/2604.23747v1)
  identifies dropped intermediate gradients with CPU-offloaded optimizers and
  averaging of local loss means instead of global response-token sums/counts.
  Our trainer uses neither offload nor accumulation; an unequal-length gradient
  equivalence test protects the token normalization. Correct implementation does
  not make our untuned staged allocation a strong tuned baseline. Their findings
  also do not establish universal dominance over mixed methods.
- [Trust Region DPO](https://arxiv.org/html/2404.09656v4) establishes hard/soft
  reference updates for Direct Preference Optimization. Our matched hard
  budget-clock reference and length-normalized pair objective are explicit
  adaptations; reference movement is not claimed as novel. The paper's larger
  post-SFT models do not validate a frozen random-initialization reference.
- [RL's Razor](https://arxiv.org/html/2509.04259v1) motivates measuring drift and
  forgetting. It does not prove that arbitrary joint training is optimal.
  WikiText here measures in-domain language modeling; ARC-Easy change from the
  untouched checkpoint is a narrow general-capability diagnostic. Broad
  retention remains unestablished without Pile/SciQ/LAMBADA measurements.
- [Coconut](https://arxiv.org/html/2412.06769v3) provides small-model evidence
  with a fully pretrained GPT-2 and extensive augmented supervised data. Its
  much-cited 42.9±0.2 is the chain-of-thought baseline, not Coconut; the inspected
  source does not establish a 95% interval convention (p-val=n/a). Paper and
  released configuration differ on 50 versus 25 epochs. These facts motivate
  future data-rich experiments, not switching our early initialization or
  declaring its reward signal established by citation. The current public
  natural-language tasks and immutable step10000 are preserved.

No new synthetic gold references, model-provider calls or additional review
rounds were introduced. CHERRY-RL's identity remains unresolved; the provisional
CHORD (Controllable Harmonization of On- and Off-Policy Reinforcement Learning
via Dynamic Weighting) and RPT (Reinforcement Pre-Training) conditions retain
explicit differences from their papers. Stronger CHORD-phi, full TRAPO, broader
retention suites and tuning/sensitivity sweeps remain outside this 48-cell pilot.
