"""Report figures. Palette = dataviz-skill reference instance (validated):
- tiers (ordered) -> ordinal blue ramp: #86b6ef / #3987e5 / #1c5cab (light mode)
- policy pair      -> categorical slots 1-2: #2a78d6 / #eb6834
- ink/grid per skill chrome tokens. Static PNGs (no hover layer).
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import RESULTS_DIR

TIER_COLORS = {"low": "#86b6ef", "mid": "#3987e5", "high": "#1c5cab"}
POLICY_COLORS = {"P2": "#2a78d6", "P3": "#eb6834"}
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["System Font", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASE,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.75,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.titlesize": 11,
    "axes.titlecolor": INK,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
})


def _finish(fig, path):
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] wrote {path}")


def reliability_diagram(reliability):
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    lims = [0, 1]
    ax.plot(lims, lims, color=MUTED, linewidth=1.2, linestyle="--", label="perfect calibration")
    xs = [d["mean_raw"] for d in reliability]
    ys = [d["empirical_acc"] for d in reliability]
    ns = [d["n"] for d in reliability]
    ax.scatter(xs, ys, s=[max(20, n / 2) for n in ns], color="#2a78d6", zorder=3, label="model (calib split)")
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(4, 5),
                    fontsize=7.5, color=INK2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("raw letter-probability confidence (equal-count bins)")
    ax.set_ylabel("observed accuracy")
    ax.set_title("Calibration: raw confidence vs accuracy")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _finish(fig, "reliability.png")


def confidence_by_tier(rows):
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.0), sharey=True)
    bins = np.linspace(0, 1, 21)
    for ax, tier in zip(axes, ["low", "mid", "high"]):
        vals = [r["conf_cal"] for r in rows if r["crit"]["bin"] == tier]
        ax.hist(vals, bins=bins, color=TIER_COLORS[tier], edgecolor=SURFACE, linewidth=0.75)
        med = float(np.median(vals)) if vals else float("nan")
        ax.axvline(med, color=INK2, linewidth=1.0)
        ax.set_title(f"{tier} criticality (n={len(vals)})\nmedian {med:.2f}", fontsize=10)
        ax.set_xlabel("calibrated confidence")
    axes[0].set_ylabel("questions")
    fig.suptitle("Distribution of calibrated confidence by judged criticality tier", y=1.04)
    _finish(fig, "confidence_by_tier.png")


def risk_coverage(rows):
    """Selective error rate among answered, as coverage grows (greedy by confidence), per tier."""
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for tier in ["low", "mid", "high"]:
        sub = sorted(
            (r for r in rows if r["crit"]["bin"] == tier),
            key=lambda r: -r["conf_cal"],
        )
        if len(sub) < 20:
            continue
        n_correct = 0
        xs, ys = [0.0], [float("nan")]
        for i, r in enumerate(sub, 1):
            n_correct += r["substantive_correct"]
            xs.append(i / len(sub))
            ys.append(1 - n_correct / i)
        ax.plot(xs, ys, color=TIER_COLORS[tier], linewidth=2.0, label=tier)
        ax.annotate(tier, (xs[-1], ys[-1]), textcoords="offset points", xytext=(5, 0),
                    fontsize=9, color=INK2)
    ax.set_xlabel("coverage (fraction of tier answered)")
    ax.set_ylabel("error rate among answered")
    ax.set_title("Risk–coverage by judged criticality tier")
    ax.set_xlim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, title="tier", title_fontsize=8)
    _finish(fig, "risk_coverage.png")


def matched_pairs_scatter(rows):
    """Per-pair: mean calibrated confidence + abstention rate under P3, high vs low."""
    by_pair = {}
    for r in rows:
        if r["pool"] != "matched":
            continue
        pid = r["meta"]["pair_id"]
        by_pair.setdefault(pid, {"high": [], "low": []})[r["meta"]["stakes"]].append(r)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0))

    # (a) confidence: high-stakes vs low-stakes twin
    ax1.plot([0, 1], [0, 1], color=MUTED, linewidth=1.0, linestyle="--")
    x, y = [], []
    for pid, d in by_pair.items():
        if d["high"] and d["low"]:
            x.append(np.mean([r["conf_cal"] for r in d["low"]]))
            y.append(np.mean([r["conf_cal"] for r in d["high"]]))
    ax1.scatter(x, y, s=28, color="#3987e5", alpha=0.85, edgecolor=SURFACE, linewidth=0.75)
    ax1.set_xlabel("low-stakes twin: mean calibrated confidence")
    ax1.set_ylabel("high-stakes twin: mean calibrated confidence")
    ax1.set_title("Are twins equally hard? (confidence, high vs low)")
    ax1.set_xlim(0, 1.02)
    ax1.set_ylim(0, 1.02)

    # (b) abstention rate under the criticality policy
    ax2.plot([0, 1], [0, 1], color=MUTED, linewidth=1.0, linestyle="--")
    from critab.evaluate import policy_decision
    from critab.config import CRITICALITY_BIN_THRESHOLDS as THRESH

    x, y = [], []
    for pid, d in by_pair.items():
        hi = [policy_decision(r, THRESH[r["crit"]["bin"]])[0] for r in d["high"]]
        lo = [policy_decision(r, THRESH[r["crit"]["bin"]])[0] for r in d["low"]]
        x.append(1 - np.mean(lo))
        y.append(1 - np.mean(hi))
    ax2.scatter(x, y, s=28, color="#eb6834", alpha=0.85, edgecolor=SURFACE, linewidth=0.75)
    ax2.set_xlabel("low-stakes twin: abstention rate")
    ax2.set_ylabel("high-stakes twin: abstention rate")
    ax2.set_title("Policy abstention on matched twins")
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)
    fig.suptitle("Matched-stakes pairs (each dot = one pair)", y=1.02)
    _finish(fig, "matched_pairs.png")


def main():
    rows = [json.loads(l) for l in open("data/calibrated.jsonl")]
    reliability = json.load(open(RESULTS_DIR / "reliability.json"))
    reliability_diagram(reliability)
    if all("crit" in r for r in rows):
        confidence_by_tier(rows)
        risk_coverage(rows)
        matched_pairs_scatter(rows)
    else:
        print("[plots] criticality labels missing; tier figures skipped")


if __name__ == "__main__":
    main()
