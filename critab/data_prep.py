"""Build the v1 question pool: 4 public datasets + hand-built matched-stakes pairs.

Output: data/questions.jsonl with one record per question:
  {id, pool, domain, question, options, gold_idx, idk_idx, letters, split, meta}

All pools are answerable MCQA; an explicit "cannot answer" option (IDK) is
appended LAST for pools that don't have one. IDK is never gold.
Substantive option order is shuffled (seeded) to remove position bias;
the IDK option always stays last.
"""
import json
import random
import re

from datasets import load_dataset

from .config import (
    CALIB_FRACTION,
    DATA_DIR,
    N_GENERAL,
    N_MATCHED_PAIRS,
    N_MEDICAL,
    SEED,
)

IDK_TEXT = "I cannot answer this question with confidence."
LETTERS = "ABCDEFGHIJKL"

IMAGE_RE = re.compile(r"\b(image|figure|photo|radiograph|x-?ray|endoscop|CT scan|MRI|ECG shown|histopath)\b", re.I)


def rec(qid, pool, domain, question, options, gold_idx, **meta):
    n = len(options)
    return {
        "id": qid,
        "pool": pool,
        "domain": domain,
        "question": question.strip(),
        "options": options,
        "gold_idx": gold_idx,
        "idk_idx": n,  # filled after IDK appended in finalize()
        "meta": meta,
    }


def finalize(r):
    r["options"] = list(r["options"]) + [IDK_TEXT]
    r["idk_idx"] = len(r["options"]) - 1
    r["letters"] = LETTERS[: len(r["options"])]
    return r


def shuffle_substantive(r, rng):
    """Shuffle substantive options (keep IDK slot semantics), remap gold."""
    opts = r["options"]
    idx = list(range(len(opts)))
    rng.shuffle(idx)
    r["options"] = [opts[i] for i in idx]
    # remap gold index through the permutation
    old_gold = r["gold_idx"]
    r["gold_idx"] = idx.index(old_gold)
    return r


def load_arc(rng, n):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    out = []
    for ex in ds:
        labels = list(ex["choices"]["label"])
        texts = list(ex["choices"]["text"])
        key = ex["answerKey"]
        if key not in labels or len(labels) < 3:
            continue
        gold_idx = labels.index(key)
        out.append(
            rec(
                f"arc-{ex['id']}",
                "arc",
                "general",
                ex["question"],
                texts,
                gold_idx,
                source="ARC-Challenge",
            )
        )
    rng.shuffle(out)
    return out[:n]


def load_mmlu_pro(rng, n):
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    out = []
    for ex in ds:
        opts = [o for o in ex["options"] if isinstance(o, str) and o.strip()]
        gold_idx = ex["answer_index"]
        if len(opts) < 4 or gold_idx >= len(opts):
            continue
        out.append(
            rec(
                f"mmlupro-{ex['question_id']}",
                "mmlu_pro",
                "general",
                ex["question"],
                opts,
                gold_idx,
                category=ex["category"],
                source="MMLU-Pro",
            )
        )
    rng.shuffle(out)
    return out[:n]


def load_medmcqa(rng, n):
    ds = load_dataset("openlifescienceai/medmcqa", split="validation")
    out = []
    for ex in ds:
        q = ex["question"]
        try:
            cop = int(ex["cop"])
        except (TypeError, ValueError):
            continue
        if not (0 <= cop <= 3) or len(q) < 20 or IMAGE_RE.search(q):
            continue
        opts = [ex["opa"], ex["opb"], ex["opc"], ex["opd"]]
        if any(not isinstance(o, str) or not o.strip() for o in opts):
            continue
        out.append(
            rec(
                f"medmcqa-{ex['id']}",
                "medmcqa",
                "medical",
                q,
                opts,
                cop,
                subject=ex.get("subject_name"),
                source="MedMCQA",
            )
        )
    rng.shuffle(out)
    return out[:n]


def load_medqa(rng, n):
    """MedQA-USMLE 4-option (swapped in for PubMedQA at checkpoint b: PubMedQA was
    degenerate — model put 100% mass on IDK for 148/150 questions, thresholds never bind)."""
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    out = []
    for i, ex in enumerate(ds):
        opts = [ex["options"][k] for k in "ABCD"]
        gold = "ABCD".index(ex["answer_idx"])
        q = ex["question"]
        if len(q) < 20 or any(not o.strip() for o in opts):
            continue
        out.append(
            rec(
                f"medqa-{i}",
                "medqa",
                "medical",
                q,
                opts,
                gold,
                source="MedQA-USMLE-4-options",
            )
        )
    rng.shuffle(out)
    return out[:n]


def load_matched_pairs(path):
    out = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            out.append(
                rec(
                    row["id"],
                    "matched",
                    row["domain"],
                    row["question"],
                    row["options"],
                    row["gold_idx"],
                    pair_id=row["pair_id"],
                    stakes=row["stakes"],
                    source="hand-built matched pairs",
                )
            )
    return out


def main():
    rng = random.Random(SEED)

    n_arc = N_GENERAL // 2
    n_mmlu = N_GENERAL - n_arc
    n_medmcqa = N_MEDICAL // 2
    n_medqa = N_MEDICAL - n_medmcqa

    pools = {
        "arc": load_arc(rng, n_arc),
        "mmlu_pro": load_mmlu_pro(rng, n_mmlu),
        "medmcqa": load_medmcqa(rng, n_medmcqa),
        "medqa": load_medqa(rng, n_medqa),
        "matched": load_matched_pairs(DATA_DIR / "matched_pairs.jsonl"),
    }

    expected = {"arc": n_arc, "mmlu_pro": n_mmlu, "medmcqa": n_medmcqa,
                "medqa": n_medqa, "matched": 2 * N_MATCHED_PAIRS}
    all_records = []
    for name, rows in pools.items():
        assert len(rows) == expected[name], f"pool {name}: got {len(rows)}, want {expected[name]}"
        for r in rows:
            shuffle_substantive(r, rng)
            finalize(r)
        all_records.extend(rows)

    # Pool-stratified calibration/test split (judge-level balance checked after scoring)
    for name, rows in pools.items():
        rng.shuffle(rows)
        n_cal = round(len(rows) * CALIB_FRACTION)
        for i, r in enumerate(rows):
            r["split"] = "calib" if i < n_cal else "test"

    out_path = DATA_DIR / "questions.jsonl"
    with open(out_path, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    by_pool_split = {}
    for r in all_records:
        by_pool_split.setdefault(r["pool"], {"calib": 0, "test": 0})
        by_pool_split[r["pool"]][r["split"]] += 1
    print(f"wrote {len(all_records)} questions -> {out_path}")
    for p, d in by_pool_split.items():
        print(f"  {p}: {d}")


if __name__ == "__main__":
    main()
