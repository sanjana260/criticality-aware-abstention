"""Isotonic calibration: raw letter-prob confidence -> P(answer correct).

Fit ONLY on the calibration split. The primary signal is confidence in the
best substantive answer (ignoring the IDK option), because the policy uses
it to decide whether to answer substantively.
"""
import json
import pickle

import numpy as np
from sklearn.isotonic import IsotonicRegression

from .config import DATA_DIR, RESULTS_DIR


def load_merged():
    """Returns dict id -> merged record of questions + confidence + criticality."""
    questions = {r["id"]: r for r in map(json.loads, open(DATA_DIR / "questions.jsonl"))}
    conf = {r["id"]: r for r in map(json.loads, open(DATA_DIR / "confidence.jsonl"))}
    try:
        crit = {r["id"]: r for r in map(json.loads, open(DATA_DIR / "criticality.jsonl"))}
    except FileNotFoundError:
        crit = None
        print("[calibrate] no criticality.jsonl yet; calibration only")

    merged = []
    for qid, q in questions.items():
        c = conf[qid]
        row = {**q, "conf": c["confidence_sub_raw"], "conf_raw_all": c["confidence_raw"],
               "substantive_idx": c["substantive_idx"], "substantive_correct": c["substantive_correct"],
               "chosen_is_idk": c["chosen_is_idk"]}
        if crit is not None:
            row["crit"] = crit[qid]
        merged.append(row)
    return merged


def fit_calibrator(rows):
    cal = [r for r in rows if r["split"] == "calib"]
    X = np.array([r["conf"] for r in cal])
    y = np.array([int(r["substantive_correct"]) for r in cal])
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(X, y)
    with open(RESULTS_DIR / "calibrator.pkl", "wb") as f:
        pickle.dump(iso, f)

    # reliability data (calib split, 10 equal-count bins) for the sanity plot
    order = np.argsort(X)
    Xs, ys = X[order], y[order]
    bins = np.array_split(np.arange(len(Xs)), 10)
    reliability = []
    for b in bins:
        reliability.append({
            "mean_raw": float(Xs[b].mean()),
            "empirical_acc": float(ys[b].mean()),
            "n": int(len(b)),
        })
    with open(RESULTS_DIR / "reliability.json", "w") as f:
        json.dump(reliability, f, indent=2)
    print(f"[calibrate] isotonic fit on {len(cal)} calib rows; reliability.json written")
    return iso


def apply_calibrator(iso, rows):
    for r in rows:
        r["conf_cal"] = float(iso.predict([r["conf"]])[0])
    return rows


def main():
    rows = load_merged()
    iso = fit_calibrator(rows)
    rows = apply_calibrator(iso, rows)
    with open(DATA_DIR / "calibrated.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} calibrated rows -> data/calibrated.jsonl")


if __name__ == "__main__":
    main()
