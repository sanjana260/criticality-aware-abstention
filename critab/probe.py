"""Fast-follow: hidden-state probe as an alternative confidence signal.

Logistic regression on last-layer hidden states -> P(answer correct).
Uses the same calib/test split as everything else; no re-inference needed
(hidden states saved by critab.confidence).

Output: data/probe_confidence.jsonl (per-question probe probability), and an
AUROC comparison vs letter-probability on the test split.
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from .config import DATA_DIR


def main():
    ids = list(np.load(DATA_DIR / "hidden_states.npz")["ids"])
    X = np.load(DATA_DIR / "hidden_states.npz")["X"]
    conf = {r["id"]: r for r in map(json.loads, open(DATA_DIR / "confidence.jsonl"))}
    questions = {r["id"]: r for r in map(json.loads, open(DATA_DIR / "questions.jsonl"))}
    idx_of = {q: i for i, q in enumerate(ids)}

    y = np.array([int(conf[q]["substantive_correct"]) for q in ids])
    split = np.array([questions[q]["split"] for q in ids])
    is_cal = split == "calib"

    scaler = StandardScaler().fit(X[is_cal])
    clf = LogisticRegression(max_iter=2000, C=0.5)
    clf.fit(scaler.transform(X[is_cal]), y[is_cal])

    p = clf.predict_proba(scaler.transform(X))[:, 1]
    test = ~is_cal
    auroc_probe = roc_auc_score(y[test], p[test])
    auroc_letter = roc_auc_score(
        y[test], [conf[q]["confidence_sub_raw"] for q in np.array(ids)[test]]
    )
    print(f"test AUROC  probe={auroc_probe:.3f}  letter-prob={auroc_letter:.3f}")

    with open(DATA_DIR / "probe_confidence.jsonl", "w") as f:
        for i, q in enumerate(ids):
            f.write(json.dumps({"id": q, "probe_prob": float(p[i]), "split": split[i]}) + "\n")
    print(f"wrote data/probe_confidence.jsonl")


if __name__ == "__main__":
    main()
