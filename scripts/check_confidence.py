"""Checkpoint (b) part 1: confidence-distribution sanity check per pool.

Flags degenerate confidence (everything ~1 or ~0) BEFORE any comparisons.
Usage: python3 scripts/check_confidence.py
"""
import json
import sys

sys.path.insert(0, ".")

import numpy as np

from critab.config import DATA_DIR

DEGENERATE_HI = 0.95  # median raw confidence above this => likely saturated pool
DEGENERATE_LO = 0.15  # median below this => model may be guessing everywhere


def describe(values):
    v = np.array(values)
    q = np.percentile(v, [10, 25, 50, 75, 90])
    return f"n={len(v)}  p10={q[0]:.3f} p25={q[1]:.3f} median={q[2]:.3f} p75={q[3]:.3f} p90={q[4]:.3f}  mean={v.mean():.3f}"


def main():
    rows = [json.loads(l) for l in open(DATA_DIR / "confidence.jsonl")]
    questions = {r["id"]: r for r in map(json.loads, open(DATA_DIR / "questions.jsonl"))}
    print("Raw confidence in best substantive answer, by pool:\n")
    issues = []
    by_pool = {}
    for r in rows:
        by_pool.setdefault(questions[r["id"]]["pool"], []).append(r["confidence_sub_raw"])
    for pool, vals in sorted(by_pool.items()):
        line = describe(vals)
        med = float(np.median(vals))
        flag = ""
        if med >= DEGENERATE_HI:
            flag = "  <-- SATURATED? consider harder slice"
            issues.append((pool, "high"))
        elif med <= DEGENERATE_LO:
            flag = "  <-- NEAR-CHANCE? model may guess everywhere"
            issues.append((pool, "low"))
        print(f"{pool:10s} {line}{flag}")

    n_idk = sum(1 for r in rows if r["chosen_is_idk"])
    print(f"\nspontaneous IDK choices (model-initiated): {n_idk}/{len(rows)} ({n_idk/len(rows):.1%})")

    overall = describe([r["confidence_sub_raw"] for r in rows])
    print(f"{'overall':10s} {overall}")
    if issues:
        print("\nACTION NEEDED: pools flagged above - inspect and possibly swap slices (plan checkpoint b).")
    else:
        print("\nNo pool flagged as degenerate. Safe to proceed to calibration once judge labels exist.")


if __name__ == "__main__":
    main()
