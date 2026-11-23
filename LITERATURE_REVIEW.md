# Literature Review: Criticality-Aware Abstention Rewards

**Question:** Has anyone trained an LLM to abstain using a reward in which the wrong-answer/abstention penalty is scaled by a per-question *criticality* score — i.e., how catastrophic a wrong answer would be for that question?

**Verdict (short):** **No direct match found.** The two halves of the idea both exist and are individually well-explored — (1) training abstention via error-penalized rewards, and (2) per-question *dynamic* reward shaping — but the per-question signal in every existing system is the **model's own capability/knowledge boundary** or a **user-supplied risk tolerance**, never a **learned estimate of the consequence of being wrong**. The specific combination — a learned question-only criticality score c(q) folded into an RL abstention reward, producing question-dependent abstention thresholds — appears to be an open gap as of November 2026. The gap is acknowledged in print (the npj Digital Medicine review explicitly calls for cost-sensitive abstention policies), so it is likely to be filled soon; several groups are one step away.

**Caveats on sourcing:** Most abstention-reward papers are 2025–2026 preprints (many not yet peer-reviewed). I verified the abstracts of the load-bearing papers directly; details marked *(from prior reading)* come from the earlier literature scan and secondary summaries and should be re-checked against the PDFs before being relied on in a paper.

---

## 1. The idea stated precisely (so "similarity" is checkable)

The proposal is to train with a per-question reward of the form:

```
R(q, response) = +1                          if response correct
               = −λ · c(q)                   if response wrong
               = −a · c(q)                   if response abstains ("I don't know")
```

where `c(q) ∈ [0,1]` is a **criticality score estimated from the question alone** (not from the model's response, closing the manipulation route). For a calibrated model with correctness probability `p`, answering beats abstaining iff `p > θ(q)` with:

```
θ(q) = c(q) · (λ − a) / (1 + λ · c(q))
```

So the optimal abstention threshold is *question-dependent and increasing in criticality* (with `a = 0`: `θ = λc/(1+λc)`). Setting `c(q) ≡ const` recovers the flat ternary reward of TruthRL / Reinforced Hesitation — criticality-awareness is a strict generalization of that family. The two research questions then map to:

- **RQ1:** Does training with the `c(q)`-weighted reward dominate the flat reward on criticality-weighted harm at matched accuracy/coverage?
- **RQ2:** Does the trained policy's abstention boundary track `θ(q)` — i.e., at fixed confidence, abstain more on high-`c` questions, with the crossover at the predicted threshold?

---

## 2. The closest works — the "has this been done?" table

| Paper | Objective / reward | Per-question signal | Is it *question criticality*? | Trained or prompted? | Relation to our idea |
|---|---|---|---|---|---|
| **Reinforced Hesitation** ([arXiv:2511.11500](https://arxiv.org/abs/2511.11500)) | Ternary: +1 / 0 / −λ | None — flat λ per model | ❌ | RLVR (trained) | **Closest methodological cousin.** λ-sweep = Pareto frontier of flat thresholds. Ours = make λ question-dependent via c(q). |
| **TruthRL** ([arXiv:2509.25760](https://arxiv.org/abs/2509.25760), ICML 2026) | Ternary +1/0/−1 via GRPO | None | ❌ | RL (GRPO) | The flat-reward baseline we'd compare against; −1 error penalty fixed for all questions. |
| **Behaviorally Calibrated RL** ([arXiv:2512.19920](https://arxiv.org/abs/2512.19920), ICLR 2026) | Strictly proper scoring rules on verbalized P(correct); stochastic abstention | User-supplied risk tolerance t scaled into the penalty: `t/(1−t)` *(from prior reading)* | ❌ — t is a **prompt-level knob chosen by the user**, not learned from the question | RL (trained) | **Closest conceptual machinery.** Proves proper-scoring-rule rewards give threshold-shaped abstention. Their open limitation: the user must supply t. Our idea: a criticality model supplies t = c(q) automatically. |
| **AWA-RL** ([arXiv:2607.10738](https://arxiv.org/abs/2607.10738), EMNLP 2026) | Abstention reward dynamically shaped by **query-specific prior capability** + on-policy observations | Model's prior success on similar queries | ❌ — capability of the *model*, not consequence for the *user* | RL (trained) | **Closest "per-question dynamic reward" instance.** Signal axis differs: they ask "can I get this right?", we ask "how bad if I'm wrong?" — orthogonal and combinable. |
| **KARL** ([arXiv:2604.22779](https://arxiv.org/abs/2604.22779)) | Knowledge-Boundary-Aware Reward: online boundary estimation from within-group response statistics; two-stage training to escape the "abstention trap" | Model's knowledge boundary on that question | ❌ — again model-relative, not question-intrinsic | RL (trained) | Same distinction as AWA-RL; also the best source for collapse-avoidance training tricks. |
| **HSA-DPO** ([arXiv:2404.14233](https://arxiv.org/abs/2404.14233), AAAI 2025) | DPO preference pairs **weighted by hallucination severity** | Severity of the *wrong content in the response* (LVLM captioning) | Partial — severity of the *answer*, not of the *question*; and not an abstention action | DPO (trained) | **Severity-weighted training objectives exist** — but for object-level caption hallucinations, not question-level abstention in QA. Useful precedent; different problem. |
| **MedSAFE / decision-theoretic abstention** ([npj Digital Medicine 2026](https://www.nature.com/articles/s41746-026-02882-1)) | Expected-utility formalization: answer utility = −(loss + λ_harm·h(x,ŷ)), abstention utility = −c_abs(x); AOC curves | The formalization *allows* per-input harm h(x) and abstention cost c_abs(x) | Framework only — h(x) is a **symbol**, never estimated or trained | Neither — review + evaluation framework | **The theory we'd operationalize.** Explicitly states the threshold "should be context-sensitive" and lists cost-sensitive abstention training as future work. Cites our gap in print. |
| **I-CALM** ([arXiv:2604.03904](https://arxiv.org/abs/2604.03904)) | Prompt announces payoffs for correct/wrong/abstain + elicits verbal confidence | None | ❌ | Prompted only | Shows payoff structures *can* be stated; RiskEval shows models mostly ignore them → motivates training them in. |
| **RiskEval** ([arXiv:2601.07767](https://arxiv.org/abs/2601.07767), EMNLP 2026) | Evaluation: vary error penalty λ ∈ [0,100] in prompt; measure strategic abstention | None (penalty is global per-run) | ❌ | Prompted (eval) | Key negative result: models are **not** cost-aware even at λ=100. The sensitivity must be trained — exactly the premise of our project. |
| **Classical learning-to-reject** (Chow 1957/1970; [cost-sensitive reductions, AISTATS 2021](https://proceedings.mlr.press/v139/charoenphakdee21a/charoenphakdee21a.pdf); [Franc & Průša, JMLR 2023](https://jmlr.org/papers/v24/21-0048.html); [regression with cost-based rejection, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/8ddcba644f602835a52b962d9a119eea-Paper-Conference.pdf)) | Reject option with unequal error costs | Costs are **per-class** (FP vs FN) or scalar | ❌ — no per-example criticality from input text | Loss-based (trained) | Our `θ(q)` derivation is the LLM-generation analogue of Chow's rule; the classical literature's costs are class-level constants, not learned per-question scores. ([Abstaining classification with unequal costs](https://arxiv.org/abs/1806.03445)) |

**Summary of the table:** every trained abstention system either (a) uses a flat penalty (TruthRL, Reinforced Hesitation, KnowRL, Abstain-R1), (b) shapes the penalty per question using the **model's own epistemic state** (AWA-RL, KARL), (c) imports a **user-supplied** risk tolerance into the prompt (behaviorally calibrated RL), or (d) weights training by the **severity of response content** in a non-abstention setting (HSA-DPO). Nobody estimates a question-intrinsic consequence score and puts it in the reward. That is the gap.

---

## 3. Training-time abstention rewards (the flat-penalty family we build on)

These define the experimental baseline and the known trade-off phenomenology:

- **TruthRL: Incentivizing Truthful LLMs via RL** (Wei et al., ICML 2026, [arXiv:2509.25760](https://arxiv.org/abs/2509.25760)). Ternary +1/0/−1 with GRPO; group-relative advantage makes abstention preferable to hallucination; ~28.9–43.5% hallucination reduction vs binary-reward RL across four knowledge benchmarks. *(Numbers differ slightly between sources — check PDF.)*
- **Honesty over Accuracy: Reinforced Hesitation** (Mohamadi, Wang & Li, Nov 2025, [arXiv:2511.11500](https://arxiv.org/abs/2511.11500)). Ternary with **tunable λ**. Verified findings: frontier models almost never abstain even when explicitly warned of severe penalties (GSM8K/MedQA/GPQA); varying λ yields a **Pareto frontier** of models, each optimal for its risk regime; abstention can be exploited at inference as a cascading signal. *(From prior reading: λ=10 → <1% wrong on Knights & Knaves, λ=20 → universal abstention; transient "crisis" spikes mid-training; medical-QA suite with zero prompted abstentions across 11 models.)*
- **The Hallucination Tax of Reinforcement Finetuning** (Song et al., EMNLP 2025 Findings, [arXiv:2505.13988](https://arxiv.org/abs/2505.13988)). RFT cuts refusal on unanswerable questions by >80%; mixing ~10% unanswerable data (SUM dataset) restores it. Motivates guarding both sides of the reward.
- **KnowRL** (Ren et al., ACL 2026, [arXiv:2506.19807](https://arxiv.org/abs/2506.19807)). Factuality reward from knowledge verification into RL; documented reward-hacking dynamics when the refusal reward was made negative (refusals rose then collapsed while wrong answers spiked) *(from prior reading)*.
- **Abstain-R1** (2026, [arXiv:2604.17073](https://arxiv.org/abs/2604.17073)). Clarification-aware RLVR abstention; shows that without a penalty for refusing solvable problems, models become over-conservative and answerable-accuracy drops *(from prior reading)*.
- **RLCR / Beyond Binary Rewards** (Damani et al., [arXiv:2505.17989](https://arxiv.org/abs/2505.17989), code [here](https://github.com/damanimehul/RLCR)). Adds a Brier-term calibration reward to binary correctness; proves joint accuracy+calibration. Notes larger models might hack calibration rewards by confident wrongness with zero confidence (not observed) *(from prior reading)*.
- **FiSCoRe** ([arXiv:2510.24020](https://arxiv.org/abs/2510.24020)). Abstention reward from **fine-grained, sample-specific semantic confidence** rather than a global uncertainty score — again the per-example signal is epistemic, not consequential.
- **Rewarding Efficient Reasoning Improves Abstention** ([arXiv:2609.20846](https://arxiv.org/html/2609.20846v2)). Reward shaping for reasoning quality spills over into abstention behavior.

### Collapse dynamics (directly constrains our reward design)

- **"Abstention as an Action Can Kill Both the Reward Gradient and the KL Anchor: Collapse Law and Repair for Error-Penalized RL"** (Che et al., [arXiv:2608.00301](https://arxiv.org/abs/2608.00301)). Formal theorem: with abstention as a discrete action and error-penalized rewards (+1/0/−λ), both the reward gradient and the KL restoring force are throttled by the same gate-saturation factor → abstention collapse; proposes a repair. **This matters for us:** a criticality-aware reward *concentrates* the largest error penalties on the highest-c questions — exactly where the collapse law predicts failure first. Our reward needs the known mitigations (two-stage/curriculum training, over-refusal penalty scaled by c(q) rather than a constant, KARL-style boundary warm-up).
- KARL's two-stage strategy and AWA-RL's on-policy shaping are the practical state of the art for avoiding the trap.

---

## 4. Consequence/stakes conditioning without training (evidence prompting fails)

- **RiskEval** ([arXiv:2601.07767](https://arxiv.org/abs/2601.07767), EMNLP 2026): sweep error penalties 0→100 in prompts; models neither act strategically on stated penalties nor keep decisions faithful to their verbalized confidence (catalogued as "Strategic Abstention Failure" in the [promptfoo LLM security DB](https://www.promptfoo.dev/lm-security-db/vuln/strategic-abstention-failure-fe576fb6)).
- **I-CALM** ([arXiv:2604.03904](https://arxiv.org/abs/2604.03904)): prompt-level announced payoffs for correctness/falsehood/abstention; improves selective answering of black-box models but does not retrain the policy.
- **Reinforced Hesitation**'s medical suite (11 models × 5 penalty levels): zero abstentions from prompting alone *(from prior reading)*.
- **Cost-Aware Abstention for IT Ticket Classification** ([arXiv:2610.31657](https://arxiv.org/abs/2610.31657)): cost-aware abstention as an *offline, evaluation-time* readiness assessment — costs enter the metric, not the training signal.
- **A Principled Framework for Dynamic Abstention** (ICLR 2026, [OpenReview](https://iclr.cc/virtual/2026/10019330)): threshold-adjusting machinery at inference *(not yet read in detail)*.

Together these establish the field's consensus premise: **penalty sensitivity has to be trained in**, which is the premise of the criticality-aware reward — and nobody has yet trained in *question-dependent* penalty sensitivity.

---

## 5. Where would c(q) come from? (criticality/severity data & estimation)

No dataset exists that labels "how bad is it if a model is wrong about this question" for general QA. The pieces:

**Severity labels on clinical actions/answers**
- **NOHARM** ([arXiv:2512.01241](https://arxiv.org/abs/2512.01241), Stanford): 1,100 eConsult tasks; three board-certified physicians rate appropriateness + harm severity per clinical action on a RAND/UCLA × WHO-harm scale. Key asymmetry finding: **harms of omission dominate severe errors** — so abstention itself must cost something scaled by criticality (the `a·c(q)` term), not be free.
- **NEJM adversarial clinical-context dataset**: WHO harm labels (none→death) per adversarial next step, in the authors' GitHub *(from prior reading)*.
- **HealthBench** (OpenAI): 48,562 physician-written rubric criteria weighted −10…+10 by importance — the most direct "how much does this matter" signal, though per-criterion not per-question.
- **PKU-SafeRLHF**: 265k QA pairs with severity levels (3 tiers, 19 harm categories) — severity of *misuse harm*, would need re-labeling for wrongness-harm.
- **MedHallu** (EMNLP 2025, [arXiv:2502.14302](https://arxiv.org/abs/2502.14302)): 10k PubMedQA-derived medical QA pairs with verifiable answers — a natural correctness substrate to overlay judge-labeled criticality on.
- **Severity-aware evaluation is emerging as a theme**: a 2026 Northeastern thesis ["Rethinking factuality assessment"](https://repository.library.northeastern.edu/files/neu:ms42k2449/fulltext.pdf) argues severity-aware scoring changes model rankings; a [systematic review](https://www.researchsquare.com/article/rs-9741159/v1.pdf) calls for standardized severity-weighted evaluation in healthcare; HALF ([arXiv:2510.12217](https://arxiv.org/abs/2510.12217)) does harm-aware weighting for fairness evaluation. None of these train on severity — they measure it.

**Estimating criticality from question text (c(q) model)**
- Triage/urgency classification of patient messages is an active, mature adjacent task (e.g., [few-shot LLM triage](https://arxiv.org/html/2605.15680v1), ED-triage studies showing over-triage biases) — this *is* question-criticality prediction in clinical form, but it has never been coupled to abstention training.
- Question **difficulty** estimation (how hard for the model) is distinct from criticality (how bad if wrong) and is often conflated in casual discussion; difficulty estimation via hidden-state probes exists ([EMNLP 2025](https://aclanthology.org/anthology-files/pdf/emnlp/2025.emnlp-main.61.pdf)) and is a methods template.
- The rubric-prompted judge → distill-to-small-classifier recipe (validated for harm by HealthBench's grader agreeing with physicians at near inter-physician level) remains the obvious construction; the earlier conversation's design considerations (question-only features to close the manipulation route; clip/bucket heavy-tailed severity — cf. Catastrophic Goodhart) all still stand.

**Risk-averse alignment (adjacent, not abstention)**
- **Risk-Averse Fine-tuning of LLMs** (NeurIPS 2024, [arXiv:2501.06911](https://arxiv.org/html/2501.06911v1)): distributional reward + risk-sensitive (CVaR-style) RLHF, aimed at toxicity — risk-aversion over *reward distributions*, not per-question consequences.
- [How Ethical Should AI Be?](https://www.greta.it/images/credit/credit24/papers/0410/02_Ouyang_KRAHNEN.pdf) — standard alignment makes models drastically risk-averse in advice, an interesting confound for evaluation.
- Conformal/risk-control abstention ([conformal abstention, NeurIPS 2024](https://arxiv.org/html/2405.01563v1); [chance-constrained inference](https://openreview.net/forum?id=cJDhDC69m9)) certifies error **frequency**, not severity — the explicitly recognized limitation our setting targets.

---

## 6. Benchmarks & evaluation infrastructure (for the experiment)

- **AbstentionBench** (NeurIPS 2025 D&B, [arXiv:2506.09038](https://arxiv.org/abs/2506.09038)): 35k+ unanswerable questions, 20 LLMs; abstention unsolved, reasoning models do *worse*; medical subset is among the weakest.
- **Know Your Limits survey** (Wen et al., TACL 2025, [aclanthology.org/2025.tacl-1.26](https://aclanthology.org/2025.tacl-1.26/)): query/model/human-values taxonomy — criticality sits in the under-explored human-values/query-context axis.
- **MedSAFE / AOC curve** (npj Digit. Med. 2026): the **Abstention Operating Characteristic** (appropriate-abstention rate vs inappropriate-abstention rate) is the right headline curve for RQ1 — it exposes over- and under-refusal jointly, and stratifying it by criticality bucket is exactly the analysis our RQ2 needs.
- **Risk-coverage curves / AURC** (standard selective prediction) per criticality bucket.
- **RA-F1** (AWA-RL): capability-reliability F1 — a reusable composite metric.
- Also: MediQ (interactive clinical information-seeking abstention), AbstentionQA (EMNLP 2024, context perturbations), [Chain-of-Self-Questioning](https://arxiv.org/html/2609.17516v1) (inference-time), [Safe RL for Multi-Turn Clinical Agents](https://arxiv.org/html/2611.12728v1) (reports naive RL improving aggregate reward by degrading high-risk cases — a cautionary tale for aggregate-only metrics; *detail unverified*).

---

## 7. Gap analysis — what is and isn't taken

**Taken (do not claim):**
1. Ternary/error-penalized RLVR abstention, including systematic flat-λ sweeps (TruthRL, Reinforced Hesitation).
2. Per-question **dynamic** abstention rewards keyed on the model's epistemic state (AWA-RL, KARL, FiSCoRe).
3. Trained, threshold-shaped calibrated abstention via proper scoring rules (behaviorally calibrated RL), including conditioning on a user-supplied risk tolerance t.
4. Severity-weighted training objectives for hallucination in multimodal captioning (HSA-DPO, AAAI 2025).
5. Decision-theoretic harm-weighted abstention formalisms and evaluation frameworks (MedSAFE); prompted payoff structures (I-CALM) and prompted penalty sweeps (RiskEval) showing prompting fails.
6. Classical cost-sensitive rejection theory (Chow-rule generalizations, per-class costs).
7. The empirical case that penalty sensitivity must be trained, and the collapse law for error-penalized abstention RL.

**Open (the claim we could make):**
1. **A learned/estimated question-intrinsic criticality score c(q)** (harm-of-wrong-answer, predicted from question text alone) **used inside an RL abstention reward** — no paper found doing this in any domain, text or multimodal.
2. **Question-dependent abstention thresholds** demonstrated as a *learned behavior* — i.e., at matched confidence, one policy abstaining more on high-criticality questions (all current work holds the threshold flat per model, or lets the user dial it globally).
3. The **threshold-matching test** (RQ2): comparing the trained policy's empirical abstention boundary against θ(q) = c(q)(λ−a)/(1+λc(q)). This analysis appears nowhere.
4. Criticality-weighted versions of the standard metrics (criticality-weighted harm, stratified AOC/AURC) as the evaluation of record for such training.

**Novelty threats to monitor (groups one step away):**
- The behaviorally-calibrated line (ByteDance Seed-adjacent, ICLR 2026): their t/(1−t) risk-tolerance mechanism needs only an automatic t-supplier — a criticality model is exactly that. They flag the manual-t limitation themselves.
- AWA-RL/KARL line: swapping their capability signal for an external criticality signal is a small patch with the same plumbing.
- The Stanford NOHARM group has the severity labels and clinical distribution and has published the observation that current evaluation (not training) is the bottleneck; the npj review explicitly names cost-sensitive abstention as future work.
- The multimodal severity line (HSA-DPO → 2026 follow-ups) could extend severity weighting from captions to QA abstention.

**Design constraints the literature imposes on our experiment:**
- Guard both sides: over-refusal penalty should also scale with c(q) (omission is the severe error — NOHARM), or the model will learn to abstain precisely where it matters most.
- Expect collapse where penalties are highest (collapse law, [arXiv:2608.00301](https://arxiv.org/abs/2608.00301)): use two-stage/curriculum warm-up (KARL), keep the abstention reward near 0 rather than negative, and consider clipping/bucketing c(q).
- Vary c(q) across training and consider occasionally conditioning on the true c(q) in the prompt (oracle-control arm) — the behavioral-calibration paper's randomization result suggests the conditioning must be trained across values; the oracle arm cleanly separates "learned to be cautious" from "learned to read the label."
- Measure confidence independently of the policy's own verbalizations (self-consistency sampling or logprobs) for the RQ2 threshold analysis — RiskEval showed verbalized confidence is not decision-faithful.
- Aggregate metrics will hide per-criticality regressions; report stratified curves (AOC by criticality bucket), per the clinical-RL caution above.

---

## 8. Reading list for the project, in priority order

1. [Reinforced Hesitation](https://arxiv.org/abs/2511.11500) — the baseline reward family + λ-frontier methodology
2. [Behaviorally Calibrated RL](https://arxiv.org/abs/2512.19920) — proper-scoring-rule machinery + the t-conditioning we'd automate *(read the t/(1−t) section in the PDF; my source is secondary)*
3. [AWA-RL](https://arxiv.org/abs/2607.10738) + [KARL](https://arxiv.org/abs/2604.22779) — the per-question-dynamic-reward plumbing and collapse-avoidance training tricks
4. [Collapse law paper](https://arxiv.org/abs/2608.00301) — the theoretical failure mode our reward must survive
5. [MedSAFE / When silence is safer](https://www.nature.com/articles/s41746-026-02882-1) — the decision-theoretic formalization + AOC evaluation; cite as the framework we operationalize
6. [TruthRL](https://arxiv.org/abs/2509.25760) — the flat ternary baseline and GRPO setup
7. [NOHARM](https://arxiv.org/abs/2512.01241) + [MedHallu](https://arxiv.org/abs/2502.14302) — severity labels + correctness substrate for dataset construction
8. [RiskEval](https://arxiv.org/abs/2601.07767) + [I-CALM](https://arxiv.org/abs/2604.03904) — the "prompting doesn't work" motivation section
9. [Hallucination Tax](https://arxiv.org/abs/2505.13988), [Abstain-R1](https://arxiv.org/abs/2604.17073), [KnowRL](https://arxiv.org/abs/2506.19807), [RLCR](https://arxiv.org/abs/2505.17989) — supporting cast for related work
10. [AbstentionBench](https://arxiv.org/abs/2506.09038) + [Know Your Limits survey](https://aclanthology.org/2025.tacl-1.26/) — evaluation infrastructure and taxonomy positioning

*All arXiv IDs and venues above were located via live search on 2026-11-12; abstracts of starred-primary papers (§2 table, §3) were read directly. Items marked "(from prior reading)" or "*unverified*" rest on the earlier conversation's summaries and should be confirmed against PDFs before citing numbers.*
