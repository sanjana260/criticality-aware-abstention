# AGENTS.md — Criticality-Aware Abstention Rewards for LLMs

This file is the single source of truth for any agent or collaborator working on this project.
Read it fully before making any design decisions, writing any code, or proposing any changes.
If something here conflicts with another file, this file wins.

---

## 0. Ground Rules — Read This First

These rules apply to every agent working on this project, without exception.

### Never act without human approval

- Do not make any design decision on your own. If a decision needs to be made, stop and ask the human.
- Do not write, run, or push any code until the full plan has been discussed, agreed upon, and explicitly signed off by the human.
- "The human said to do it" is not enough — the specific plan for *how* to do it must also be agreed upon first.
- If a human gives a vague instruction like "build the reward function", do not start building. First present a concrete plan, wait for approval, then build.

### The workflow is always: Plan → Discuss → Agree → Build

1. **Plan:** Propose what you intend to do, step by step, with every design choice made explicit.
2. **Discuss:** The human reviews the plan. Raise any concerns, risks, or alternatives you see.
3. **Agree:** Wait for explicit human sign-off on the plan before touching any code.
4. **Build:** Implement exactly what was agreed. Nothing more, nothing less.

If at any point during building something unexpected comes up — an ambiguity, a problem, a tradeoff you didn't anticipate — **stop immediately and flag it**. Do not resolve it silently. Do not hide it. Do not make a judgment call and move on. Surface it, explain it, and wait for the human to decide.

### No hallucinations

- Do not invent facts, paper results, dataset properties, or API behaviors.
- If you are not sure about something, say so explicitly. "I don't know" is always acceptable. A confident wrong answer is not.
- If you cite a paper or a result, you must have actually read it or have high confidence it exists. Do not fabricate citations.

### No pushes without explicit human permission

- Do not push to any remote repository without the human explicitly saying "push this."
- Do not commit on behalf of the human without being asked.
- Do not open pull requests, create branches, or modify shared state without explicit instruction.

### Flag everything, hide nothing

- If you notice a problem, a risk, a gap, or something that contradicts the plan — say it immediately, even if you think it might slow things down.
- If you are unsure whether something matters, flag it anyway and let the human decide.
- If something you built doesn't work as expected, say so clearly. Do not paper over failures with vague language.

### Deviations from AGENTS.md require human approval

- If you think something in this file should be changed, propose the change and wait for approval. Do not unilaterally deviate from what is written here.
- This applies to the reward function, training approach, evaluation metrics, and all other specified constraints.

---

---

## 1. Project Summary

LLMs are trained to always answer. A wrong answer about a movie plot and a wrong answer about a warfarin dose are treated identically by nearly all current training and evaluation. This project fixes that asymmetry.

We train an LLM with reinforcement learning using a reward function that scales with **question criticality** — how much is at stake if the question is answered wrongly. The model is then tested on **error severity** — how harmful a specific wrong answer is.

The goal: a model that demands more confidence before answering high-stakes questions, while staying helpful on low-stakes questions. This is distinct from prompting the model to be careful (which does not work — see Section 9) and from post-hoc threshold adjustment (which is a baseline, not the contribution).

**Core claim:** A learned, question-intrinsic criticality score c(q), folded into an RL abstention reward, produces question-dependent abstention thresholds as a *learned behavior* — something no existing system achieves.

---

## 2. The Reward Function

The reward function is the core of this project. Do not simplify, modify, or replace it without explicit team agreement.

```
R(q, response) = +1                    if response is correct
               = −λ · κ · δ            if response is wrong
               = −a · (1 − κ)          if response abstains
```

### Variables

| Variable | Name | Type | Range | Description |
|---|---|---|---|---|
| λ | Wrong-answer penalty scale | Hyperparameter (constant) | > 0 | Set before training, tuned experimentally |
| a | Abstention penalty scale | Hyperparameter (constant) | > 0, keep small | Set before training; keep near 0 to avoid collapse |
| κ (kappa) | Criticality | Function of the question | [0, 1] | How much is at stake if this question is answered wrongly. 0 = trivial, 1 = life-threatening. Computed by the criticality judge. |
| δ (delta) | Error severity | Function of the question + answer | [0, 1] | How harmful this specific wrong answer is. 0 = harmless, 1 = catastrophic. Computed by the severity judge. |

### Key Intuitions

- A wrong answer on a trivial question (κ ≈ 0) → penalty ≈ 0. Not a big deal.
- A catastrophic wrong answer on a critical question (κ = 1, δ = 1) → maximum penalty.
- Abstaining on a critical question (κ ≈ 1) → cheap. Better to say "I don't know."
- Abstaining on a trivial question (κ ≈ 0) → costly. The model should not be cowardly on easy questions.

### Derived Abstention Threshold

For a model with correctness probability p on a question, answering beats abstaining if and only if:

```
p > θ(q) = c(q) · (λ − a) / (1 + λ · c(q))
```

θ(q) increases with criticality. This is the testable prediction: at fixed confidence, the model should abstain more on high-κ questions than on low-κ questions.

Setting κ = constant recovers the flat ternary reward (+1 / 0 / −λ) used in prior work. Our reward is a strict generalization of that family.

---

## 3. What Is Novel — Do Not Dilute This

The specific contribution, stated precisely:

> **A learned/estimated question-intrinsic criticality score κ = c(q)** — measuring the consequence of a wrong answer, predicted from question text alone — **used inside an RL abstention reward** to produce question-dependent abstention thresholds as a learned behavior.

This has not been done in any domain as of November 2026.

Three sub-claims that follow from this:

1. **RQ2 core effect:** At matched confidence, the trained model abstains more on high-κ questions than on low-κ questions — and more than a flat-reward baseline trained the same way.
2. **Threshold matching:** The model's empirical abstention boundary tracks the formula θ(q) = c(q)(λ−a)/(1+λc(q)).
3. **Criticality-stratified evaluation:** Reporting evaluation metrics stratified by κ bucket, not just aggregate accuracy, as the standard for this class of system.

Do not propose ideas that reduce the contribution to any of the following (all already done):
- Flat ternary RL rewards with a fixed λ
- Per-question rewards based on model capability or knowledge boundary
- User-supplied risk tolerance scaled into the reward
- Severity weighting on hallucination in image captioning (non-abstention setting)
- Post-hoc threshold adjustment on a pre-trained model (this is a **baseline**, not the contribution)

---

## 4. What Is Already Done — Do Not Claim These

| What | Paper | Why we don't claim it |
|---|---|---|
| Flat ternary RLVR abstention (+1/0/−λ) | TruthRL (ICML 2026), Reinforced Hesitation (arXiv 2511.11500) | Flat λ, not per-question |
| λ-sweep → Pareto frontier of flat thresholds | Reinforced Hesitation | Same flat structure |
| Per-question reward from model's epistemic state | AWA-RL (EMNLP 2026), KARL (arXiv 2604.22779) | Signal = model capability, not question consequence |
| Threshold-shaped abstention via proper scoring rules | Behaviorally Calibrated RL (ICLR 2026) | Risk tolerance t is user-supplied, not learned from question |
| Severity-weighted training on hallucination | HSA-DPO (AAAI 2025) | Image captioning, not QA abstention |
| Decision-theoretic abstention formalism | MedSAFE / npj Digital Medicine 2026 | Framework only — harm never estimated or trained on |
| Proof that prompting doesn't work | RiskEval (EMNLP 2026), I-CALM | Evaluation / motivation, not training |
| Classical cost-sensitive rejection theory | Chow 1957, AISTATS 2021, JMLR 2023 | Per-class costs, not per-question scores from text |

---

## 5. Research Questions

We have five research questions. We acknowledge we cannot answer all five in one semester. Priority order: RQ1 → RQ2 → RQ3 → RQ4 → RQ5.

**RQ1 — Criticality judge validity**
Can an LLM judge score question criticality in agreement with expert-annotated labels, without relying on keyword shortcuts?
- This validates κ before it enters the reward.
- Success criterion: strong correlation with NOHARM / MPIB labels on a held-out sample.

**RQ2 — Core effect (highest priority)**
Does the criticality-aware reward produce better abstention than the flat reward?
- Primary: at fixed confidence, does the model abstain more on high-κ questions?
- Secondary: does the abstention boundary match the predicted θ(q)?
- Baselines to beat: (a) flat-reward RL model, (b) criticality-thresholded abstention without RL training (Shauli's suggested baseline — apply variable confidence threshold post-hoc using the criticality judge, no RL).

**RQ3 — Misleading context**
When convincing but false context is added, does the model still abstain on high-criticality questions?
- Motivated by Wan et al. 2026 ("The Facade of Truth") — LLMs are easily swayed by fabricated plausible evidence.

**RQ4 — Domain fine-tuning**
Does a domain-fine-tuned model (e.g., healthcare-tuned) outperform a general model under the same reward, and is that gain from domain knowledge or the reward?

**RQ5 — Generalization**
Does a policy trained in one domain (e.g., finance) transfer to another (e.g., healthcare)?

---

## 6. Datasets

**To be finalized.** Dataset selection is not yet agreed upon by the team. Do not hardcode dataset choices in any code or make assumptions about dataset availability.

Placeholder roles (will be filled once agreed):
- Severity assessment and criticality judge validation (RQ1)
- Reward training and correctness evaluation (RQ2, RQ4)
- General factual QA for over-refusal testing (RQ2, RQ4, RQ5)
- Misleading context evaluation (RQ3)
- Second domain for transfer (RQ5)

---

## 7. Training Approach and Constraints

These are not suggestions. They are constraints derived from the literature. Deviating from them risks training failure.

### Algorithm
Use **GRPO** (Group Relative Policy Optimization) via the HuggingFace TRL library (`GRPOTrainer`). Do not use PPO unless there is a specific justification — GRPO requires no value network, is lower memory, and is more practical for a semester project.

### Two-Stage Training (required)
**Stage 1:** Train with the flat reward (+1 / 0 / −λ, constant λ, no κ or δ). Let the model learn that abstention is a valid action before it needs to calibrate it.

**Stage 2:** Switch to the full criticality-aware reward. The model now has abstention as an established behavior and can refine when to use it based on κ.

Skipping Stage 1 risks abstention collapse (see Section 9).

### Curriculum Ordering (required)
Order training data from easy to hard within each stage:
- Start with questions where κ is extreme (very low or very high) — the correct behavior is obvious.
- Gradually introduce ambiguous mid-range κ questions.

### Abstention Penalty (required)
Keep `a` small (near 0, e.g., 0.1). Do not make abstaining heavily negative. The model should see abstaining as a safe neutral option, not a punishment.

### Guard Both Sides
Include unanswerable questions in the training mix (~10%) to prevent the model from losing its ability to abstain entirely (the "hallucination tax" from RL training, Hallucination Tax paper arXiv 2505.13988).

---

## 8. Evaluation Metrics

Do not report aggregate accuracy alone. It hides per-criticality regressions.

**Required metrics:**

- **AOC curve (Abstention Operating Characteristic):** appropriate-abstention rate vs. inappropriate-abstention rate, stratified by κ bucket (low / medium / high). This is the headline evaluation for RQ2.
- **Risk-coverage curves (AURC):** accuracy on answered questions vs. coverage (fraction answered), stratified by κ bucket.
- **Criticality-weighted harm:** expected harm computed using κ × δ weighting, not unweighted accuracy.
- **Threshold-matching analysis (RQ2):** plot the model's empirical abstention boundary against the predicted θ(q) formula. Report correlation.

**Baselines required for RQ2:**
1. Flat-reward RL model (same architecture, same training, λ fixed globally)
2. Criticality-thresholded abstention (no RL — apply variable confidence threshold post-hoc using criticality judge)

---

## 9. Known Failure Modes — Avoid These

**Abstention collapse**
Formally proven in arXiv 2608.00301. With discrete abstention actions and large error penalties, the reward gradient and KL anchor both collapse simultaneously. The model either refuses everything or nothing. Our reward concentrates the largest penalties on highest-κ questions — exactly where collapse is most likely. Mitigations: two-stage training, curriculum ordering, small abstention penalty `a`.

**Over-refusal**
Without a penalty for refusing answerable questions, the model becomes cowardly. The `−a·(1−κ)` abstention penalty term exists specifically to prevent this. Abstain-R1 (arXiv 2604.17073) documents this failure mode.

**Hallucination tax from RL**
Standard RL training cuts refusals on unanswerable questions by >80% (arXiv 2505.13988). Guard against this by mixing ~10% unanswerable questions into the training data.

**Prompting doesn't work**
Do not expect that telling the model "this wrong answer costs 10 points" in a prompt will change its abstention behavior. RiskEval (EMNLP 2026) tested this across 11 models and found near-zero effect. Sensitivity must be trained in.

**Verbalized confidence is not decision-faithful**
When measuring the model's confidence for the threshold analysis, do not rely on the model saying "I'm 80% sure." Use self-consistency sampling or token logprobs instead. RiskEval documents that verbalized confidence does not track actual decision behavior.

**Aggregate metrics hiding regressions**
A model that improves on low-κ questions while getting worse on high-κ questions can show flat aggregate accuracy. Always report stratified curves. See the clinical RL caution in Safe RL for Multi-Turn Clinical Agents (arXiv 2611.12728).

---

## 10. Glossary

| Term | Definition |
|---|---|
| **Abstention** | The model outputs "I don't know" / declines to answer instead of producing a response |
| **κ (kappa) / Criticality** | How much is at stake if a question is answered wrongly. Property of the question. Scored [0,1] by the criticality judge |
| **δ (delta) / Error severity** | How harmful a specific wrong answer is. Property of the answer. Scored [0,1] by the severity judge |
| **λ** | Hyperparameter controlling the scale of the wrong-answer penalty. A constant you tune |
| **a** | Hyperparameter controlling the scale of the abstention penalty. Keep small |
| **Flat reward** | Ternary reward +1/0/−λ with the same λ for every question. The standard baseline in prior work |
| **Criticality-aware reward** | Our reward — same ternary structure but penalties scale with κ and δ |
| **Criticality judge** | An LLM prompted with a rubric to score κ for a given question. Runs before training |
| **Severity judge** | An LLM or heuristic that scores δ for a given wrong answer. Runs during training |
| **Coverage** | Fraction of questions the model answers (does not abstain on). Lower coverage = more abstention |
| **Selective prediction** | The formal framework for models that can abstain. Optimizes accuracy at a given coverage level |
| **θ(q)** | The optimal abstention threshold for question q. Equals c(q)(λ−a)/(1+λc(q)). Increases with κ |
| **Abstention collapse** | Training failure where the model refuses everything or nothing, and the gradient dies |
| **Two-stage training** | Train with flat reward first (Stage 1), then switch to criticality-aware reward (Stage 2) |
| **Curriculum** | Ordering training data from easy to hard — extreme κ questions first, ambiguous ones later |
| **AOC curve** | Abstention Operating Characteristic — appropriate vs. inappropriate abstention rate. Main evaluation curve |
| **GRPO** | Group Relative Policy Optimization — the RL algorithm used. Implemented via HuggingFace TRL |
