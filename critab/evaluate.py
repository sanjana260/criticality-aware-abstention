"""Evaluation: P1 always-answer vs P2 uniform-threshold vs P3 criticality-threshold.

All threshold choice happens on the CALIBRATION split only:
- P2's uniform threshold t* is chosen on calib to match P3's calib abstention rate
  (matched-coverage comparison; plan flaw fix #8).
- P3 uses the pre-registered t(c) table (config.CRITICALITY_BIN_THRESHOLDS).

Test split is evaluated once. Policies abstain when the model itself chose the
IDK option (model-initiated) or the calibrated confidence is below the
threshold (threshold-forced).
"""
import json
from collections import defaultdict

import numpy as np

from .calibrate import load_merged
from .config import CRITICALITY_BIN_THRESHOLDS, LEVEL_TO_BIN


def harm_level(row):
    """Continuous criticality in [0,1] from the judge (falls back to pool default)."""
    if "crit" in row:
        return row["crit"]["level_avg"] / 5.0
    return 0.5


def bin_of(row):
    if "crit" in row:
        return row["crit"]["bin"]
    return "mid"


def policy_decision(row, threshold):
    """Returns (answered, model_initiated_abstain, threshold_forced_abstain)."""
    if row["chosen_is_idk"]:
        return False, True, False
    if row["conf_cal"] < threshold:
        return False, False, True
    return True, False, False


def compute_metrics(rows, answered_mask):
    n = len(rows)
    answered_idx = [i for i in range(n) if answered_mask[i]]
    correct = sum(rows[i]["substantive_correct"] for i in answered_idx)
    wrong = len(answered_idx) - correct
    return {
        "n": n,
        "answered": len(answered_idx),
        "coverage": len(answered_idx) / n,
        "accuracy_all": correct / n,
        "hallucination_rate": (wrong / len(answered_idx)) if answered_idx else float("nan"),
        "selective_accuracy": (correct / len(answered_idx)) if answered_idx else float("nan"),
        "n_wrong": wrong,
        "n_correct": correct,
    }


def commission_harm(rows, answered_mask):
    return float(sum(harm_level(r) for i, r in enumerate(rows) if answered_mask[i]
                     and not r["substantive_correct"]) / len(rows))


def omission_harm(rows, answered_mask):
    return float(sum(harm_level(r) for i, r in enumerate(rows) if not answered_mask[i]) / len(rows))


def match_uniform_threshold(rows_cal, p3_rate_cal):
    """Find t* so that abstain rate under uniform t* on calib == p3_rate_cal."""
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        rate = np.mean([not policy_decision(r, mid)[0] for r in rows_cal])
        # abstention rate increases with the threshold; if we're below target, raise it
        if rate < p3_rate_cal:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 4)


def bootstrap_ci_harm_delta(rows, mask_a, mask_b, n_boot=1000, seed=0):
    """Bootstrap CI for harm(P_b) - harm(P_a) (negative => b safer)."""
    rng = np.random.default_rng(seed)
    n = len(rows)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sub = [rows[i] for i in idx]
        ma = [mask_a[i] for i in idx]
        mb = [mask_b[i] for i in idx]
        deltas.append(commission_harm(sub, mb) - commission_harm(sub, ma))
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def run_evaluation(rows):
    cal = [r for r in rows if r["split"] == "calib"]
    test = [r for r in rows if r["split"] == "test"]

    # P3 threshold from the pre-registered table
    for r in rows:
        r["t3"] = CRITICALITY_BIN_THRESHOLDS[bin_of(r)]

    p3_decisions_cal = [policy_decision(r, r["t3"]) for r in cal]
    p3_rate_cal = np.mean([not d[0] for d in p3_decisions_cal])
    t_star = match_uniform_threshold(cal, p3_rate_cal)

    policies = {
        "P1_always_answer": lambda r: (True, False, False),
        "P2_uniform_t*": lambda r: policy_decision(r, t_star),
        "P3_criticality_t(c)": lambda r: policy_decision(r, r["t3"]),
    }

    report = {"t_star": t_star, "p3_abstention_rate_calib": float(p3_rate_cal), "policies": {}}
    masks = {}
    for name, dec in policies.items():
        mask = [dec(r)[0] for r in test]
        masks[name] = mask
        m = compute_metrics(test, mask)
        m["abstention_rate"] = 1 - m["coverage"]
        m["model_initiated_abstain_rate"] = float(np.mean([dec(r)[1] for r in test]))
        m["threshold_forced_abstain_rate"] = float(np.mean([dec(r)[2] for r in test]))
        m["commission_harm"] = commission_harm(test, mask)
        m["commission_plus_omission_harm"] = m["commission_harm"] + omission_harm(test, mask)
        report["policies"][name] = m

    # Per-tier breakdown for P2 vs P3
    tiers = defaultdict(lambda: {"P2": defaultdict(list), "P3": defaultdict(list)})
    for i, r in enumerate(test):
        b = bin_of(r)
        for pname in ("P2_uniform_t*", "P3_criticality_t(c)"):
            t = CRITICALITY_BIN_THRESHOLDS[b] if pname.startswith("P3") else t_star
            answered, _, forced = policy_decision(r, t)
            tiers[b][("P3" if pname.startswith("P3") else "P2")]["answered"].append(answered)
            tiers[b][("P3" if pname.startswith("P3") else "P2")]["forced"].append(forced)
            tiers[b][("P3" if pname.startswith("P3") else "P2")]["wrong"].append(
                answered and not r["substantive_correct"])
    report["tiers"] = {
        b: {
            pname: {
                "n": len(d["answered"]),
                "abstention_rate": float(np.mean([not a for a in d["answered"]])),
                "threshold_forced_rate": float(np.mean(d["forced"])),
                "hallucination_rate_among_answered": (
                    float(np.sum(d["wrong"]) / max(1, np.sum(d["answered"])))),
            }
            for pname, d in pol.items()
        }
        for b, pol in tiers.items()
    }

    # Matched-pairs paired analysis (designed stakes vs judge bins)
    pair_stats = defaultdict(lambda: {"high": [0, 0], "low": [0, 0]})  # [abstained, total]
    pair_cal = defaultdict(lambda: {"high": [], "low": []})
    for i, r in enumerate(test):
        if r["pool"] != "matched":
            continue
        answered, _, _ = policy_decision(r, r["t3"])
        s = r["meta"]["stakes"]
        pair_stats[r["meta"]["pair_id"]][s][0] += int(not answered)
        pair_stats[r["meta"]["pair_id"]][s][1] += 1
        pair_cal[r["meta"]["pair_id"]][s].append(r["conf_cal"])
    high_abstain = [v["high"][0] / v["high"][1] for v in pair_stats.values() if v["high"][1]]
    low_abstain = [v["low"][0] / v["low"][1] for v in pair_stats.values() if v["low"][1]]
    conf_gap = [
        (np.mean(v["low"]) - np.mean(v["high"]))
        for v in pair_cal.values() if v["high"] and v["low"]
    ]
    report["matched_pairs"] = {
        "n_pairs_test": len(pair_stats),
        "mean_abstention_rate_high": float(np.mean(high_abstain)),
        "mean_abstention_rate_low": float(np.mean(low_abstain)),
        "mean_calib_conf_gap_low_minus_high": float(np.mean(conf_gap)),
        "n_pairs_high_more_abstained": int(sum(h > l for h, l in zip(high_abstain, low_abstain))),
    }

    # Bootstrap CI: harm(P2) - harm(P3) on test
    lo_ci, hi_ci = bootstrap_ci_harm_delta(test, masks["P2_uniform_t*"], masks["P3_criticality_t(c)"])
    report["harm_delta_P2_minus_P3_bootstrap95"] = [lo_ci, hi_ci]

    return report


def main():
    import os
    from .config import DATA_DIR
    cal_path = DATA_DIR / "calibrated.jsonl"
    if cal_path.exists():
        rows = [json.loads(l) for l in open(cal_path)]
    else:
        rows = load_merged()
        from .calibrate import fit_calibrator, apply_calibrator
        iso = fit_calibrator(rows)
        rows = apply_calibrator(iso, rows)
    missing = sum(1 for r in rows if "crit" not in r)
    if missing:
        print(f"ERROR: {missing} rows missing criticality labels; run the judge first.")
        return
    report = run_evaluation(rows)
    with open("results/evaluation.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "tiers"}, indent=2)[:2500])
    print("\nwrote results/evaluation.json")


if __name__ == "__main__":
    main()
