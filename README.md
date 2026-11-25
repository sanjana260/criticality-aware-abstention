# Criticality-Aware Abstention

**Research question:** should a model's abstention threshold rise with the *criticality* of a
question — how catastrophic a confident wrong answer would be? And does criticality-aware
thresholding reduce criticality-weighted harm relative to a uniform threshold at matched coverage?

Phase 1 (this repo, no RL): take a base model's **internal confidence** (MCQA letter
probability, isotonic-calibrated), and abstain when it falls below a threshold that depends on a
**judge-assigned criticality level** c(q) ∈ {1..5}. Compare three policies at matched overall
coverage. Phase 2 (later, separate): GRPO with reward `+1 / −1 / 2t(c)−1` per the experiment draft.

## Pipeline

```
python3 -m critab.data_prep          # build data/questions.jsonl (580 MCQA + IDK option)
python3 scripts/judge_gate.py        # CHECKPOINT a: judge sanity pairs + test-retest (needs GLM balance)
python3 -m critab.judge              # score criticality 1-5 with 2 rubric permutations -> data/criticality.jsonl
python3 -m critab.confidence         # letter-prob extraction on Qwen2.5-3B (MPS) -> data/confidence.jsonl + hidden_states.npz
python3 scripts/check_confidence.py  # CHECKPOINT b part 1: degenerate-confidence guard per pool
python3 -m critab.calibrate          # isotonic fit on calib split -> data/calibrated.jsonl
python3 -m critab.evaluate           # P1/P2/P3 policies -> results/evaluation.json
python3 -m critab.plots              # figures -> results/*.png
```

## Pre-registered decisions (fixed before test evaluation)

- **t(c) table:** low {1,2} → 0.60, mid {3} → 0.85, high {4,5} → 0.97 (`critab/config.py`)
- **Threshold matching:** uniform comparator t\* chosen on the calibration split to match the
  criticality policy's overall abstention rate there
- **Split:** 30% calibration / 70% test, stratified by pool, seed 20261123
- **Judge gate:** ≥90% of 20 hand-built sanity pairs ordered correctly; ≥80% test-retest agreement
  (exact + paraphrase) on 30 questions

## Caveats (report says these too)

- The baseline is **oracle-c**: the judge supplies c(q) at test time. Deployable variants
  (criticality in the prompt, distilled predictor) are Phase-2 conditions A/C.
- Labels are the GLM judge's view of harm, not ground truth; quick-look rigor: NOHARM validation
  and a human spot-check are fast-follows.
- Single judge family (GLM); different-family eval judge deferred until another key exists.

## Layout

- `critab/` pipeline modules; `scripts/` gates and runners; `data/` artifacts (gitignored);
  `results/` evaluation outputs, figures, report.
- `LITERATURE_REVIEW.md` — related-work survey and gap analysis (Nov 2026).
- The experiment draft lives at `.context/attachments/waaHhV/RDFA Experiment 1.md` (untracked).
