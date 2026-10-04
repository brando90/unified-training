# unified-training

**Joint R1 Training for Large-Scale Reasoning (J-R1) Beats R1**: train pretraining, supervised fine-tuning, preference learning, and reasoning reinforcement learning (RL) *jointly* with a dynamically evolving data/objective mixture, instead of in sequential post-training stages.

- Project proposal (Google Doc): <https://docs.google.com/document/d/1j_qSj77AdW0qhZHgQwYfpgLqeumpksd0E67-uj6VL9E/edit?tab=t.0>
- Affiliation: Stanford Data Science Institute, Marlowe Initiative
- Collaborators: Brando Miranda (STAIR, Stanford Trustworthy AI Research), Sanmi Koyejo (STAIR), AI alignment and optimization researchers

## Objective

Current large language models (LLMs) rely on curriculum-based, sequential training, where different capabilities are learned in distinct phases:

1. **Pretraining (PT):** learning general representations from diverse corpora.
2. **Supervised Fine-Tuning (SFT):** adapting the model to follow human instructions.
3. **Preference Learning (DPO, Direct Preference Optimization / RLHF, RL from Human Feedback):** aligning outputs to human preferences.
4. **Reinforcement Learning for Reasoning (R1 RL):** teaching structured reasoning capabilities.

We hypothesize that this staged paradigm is suboptimal and that training all components simultaneously, with a dynamically evolving mixture, will outperform traditional curriculum learning. Instead of introducing RL late in training, RL, preference learning, and reasoning objectives are gradually integrated from the start using an AIOLI-style dynamic mixture.

**Goal:** replicate and surpass the reasoning capabilities of curriculum-trained models like DeepSeek-R1, but train reasoning, alignment, and knowledge all at once, optimizing the training trajectory from the start.

## Key Hypothesis

- **Traditional staged training forces models into rigid local optima.**
  - Early training heavily biases model convergence toward certain objectives.
  - Transitioning between training phases is difficult due to SGD momentum in parameter space.
  - This creates an "alignment tax" and brittle generalization when RL is applied late in training.
- **Joint training enables synergistic learning.**
  - Preference learning (DPO/RLHF) + RL from early stages can regularize model representations, reducing harmful biases from early PT.
  - Reasoning and alignment are not separate objectives; they should be learned together to improve coherence and safety.
- **Dynamic mixture optimization (AIOLI) prevents catastrophic forgetting and premature convergence.**
  - Instead of hard transitions between training stages, AIOLI dynamically shifts focus over time.
  - Start with high weight on PT, SFT, and DPO, with minimal RL.
  - Gradually increase RL importance as the model gains competence, optimizing reasoning without destabilizing base capabilities.

## Research Questions

1. **Can joint training of PT + SFT + DPO/RLHF + R1 RL outperform sequential curriculum training?**
   - Does mixing objectives early in training lead to more efficient and generalizable reasoning?
   - Can preference learning guide early-stage representations more effectively than late-stage alignment?
2. **Does dynamic mixture training (AIOLI) enable more effective multi-task optimization?**
   - How should mixture proportions evolve over time to prevent catastrophic forgetting?
   - What is the optimal balance between PT, SFT, RLHF, and R1 RL at different training phases?
3. **Does joint training eliminate the need for separate alignment and reasoning phases?**
   - Can RLHF and R1 RL work simultaneously without interfering with each other?
   - Does early exposure to RL prevent the alignment tax that occurs when models are fine-tuned first and RL is applied later?

## Methodology

We compare three training strategies:

1. **Baseline: standard sequential training (DeepSeek-R1 approach)**
   - PT → SFT → RLHF/DPO → R1 RL, each phase trained separately (staged curriculum).
2. **Full joint training (our approach)**
   - All components trained simultaneously: PT + SFT + DPO/RLHF + R1 RL.
   - Dynamic mixture control via AIOLI: starts with PT, SFT, and DPO dominance, then gradually shifts toward RLHF and R1 RL as the model improves.
3. **Ablation studies**
   - No preference learning early on (PT + SFT first, then RLHF/DPO + R1).
   - No RL early on (PT + SFT + DPO first, RL added later).
   - Hard transitions vs. gradual mixture adjustments (testing whether smooth transitions are necessary).

## Data and Compute

- **Datasets:** pretraining corpora (books, code, scientific papers, web data); open-source instruction-tuning datasets; human preference datasets (for RLHF/DPO).
- **Hardware:** A100 80GB GPUs, scaling up to TPUv5 for full training runs.
- **Training duration:** 3–6 months, scaling model sizes from 1B to 70B parameters.

## Evaluation Metrics

1. **Reasoning:** AIME, MATH-500, GPQA, Codeforces.
2. **Generalization and alignment:** MMLU, TruthfulQA, HELM.
3. **Sample efficiency:** compute cost vs. model performance.

## Expected Outcomes

- If joint training outperforms sequential training, it removes the need for separate staged fine-tuning phases in LLM training.
- If AIOLI dynamic mixing proves effective, it becomes a standard for training-pipeline optimization, keeping models out of suboptimal convergence basins.
- If early preference learning helps RL reasoning emerge faster, it could reshape alignment strategies toward more robust, human-aligned AI.

## Why This Might Work

**Avoiding forgetting by not doing post-training.** This method does not overcome forgetting; it avoids it by training all real objectives from the beginning.

- Forgetting is a side effect of post-training, where models are forced to shift away from earlier-learned objectives.
- By optimizing all objectives jointly, the model never needs to unlearn anything, preventing forgetting by design.

**No "most recent training" bias.**

- Current models tend to focus on the most recent data; this is why models trained on reasoning tasks last become good reasoners.
- With joint training there is no single most-recent focus: the model learns reasoning, instruction following, and alignment together, preserving all abilities rather than biasing toward one.
- This encourages balanced generalization rather than overfitting to one training phase.

## Conclusion

This project challenges the necessity of curriculum-based, sequential training by proposing a joint training paradigm that uses dynamic mixture optimization to balance multiple training objectives from the start. By jointly training all objectives, we let optimization dynamics guide the model rather than forcing it into local minima dictated by artificial curriculum constraints. If successful, this could reduce overall training cost, improve reasoning, and make foundation models more sample-efficient, generalizable, and adaptable to new tasks without rigid retraining.

## Related Directions (Marin)

Related goals for work with the [Marin](https://marin.community/) open-source effort:

1. Improve on DoReMi and build information-theoretic pretraining recipes, e.g., automatically optimizing data weights, regularized toward what humans care about.
2. Build a foundation model (ideally 8B+) for formal mathematics that is also general purpose.
3. Try pretraining algorithms that include SFT and RL from scratch and rely less on ad hoc post-training tricks to "fix" base models.

## References

- Tomasz Korbak, Kejian Shi, Angelica Chen, Rasika Bhalerao, Christopher L. Buckley, Jason Phang, Samuel R. Bowman, Ethan Perez. *Pretraining Language Models with Human Preferences.*
- *Trust-Region Adaptive Policy Optimization.*
- AIOLI: dynamic data-mixture optimization (used for the evolving objective mixture).
- DeepSeek-R1 (sequential-training baseline).
