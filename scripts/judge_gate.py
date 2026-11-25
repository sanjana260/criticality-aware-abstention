"""Judge fast gate (checkpoint a): sanity pairs + test-retest before bulk scoring.

Gate criteria (pre-registered in config):
- sanity pairs: >=90% of pairs scored level(high) > level(low), judged blind
- test-retest: >=80% bin agreement on exact repeat AND on GLM-paraphrased questions
Run: python3 scripts/judge_gate.py
"""
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, ".")

from critab import config
from critab.config import DATA_DIR, RETEST_MIN_AGREEMENT, SANITY_PAIRS_MIN_ACCURACY, SEED
from critab.glm import chat_json
from critab.judge import judge_one

PARAPHRASE_SYSTEM = (
    "You rephrase questions. Rewrite the question with different wording but identical meaning, "
    "keeping it a standalone question. Respond ONLY with JSON: {\"paraphrase\": \"...\"}."
)
WORKERS = 4


def run_sanity_pairs() -> float:
    rows = [json.loads(l) for l in open(DATA_DIR / "sanity_pairs.jsonl")]
    by_pair = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        levels = list(ex.map(lambda r: judge_one(r["question"])["level_rounded"], rows))
    for r, lvl in zip(rows, levels):
        by_pair.setdefault(r["pair_id"], {})[r["kind"]] = lvl
    correct = 0
    fails = []
    for pid, d in by_pair.items():
        ordered = d["high"] > d["low"]
        correct += ordered
        if not ordered:
            fails.append((pid, d))
    n = len(by_pair)
    acc = correct / n
    print(f"[gate] sanity pairs: {correct}/{n} ordered correctly ({acc:.0%})")
    for pid, d in fails:
        print(f"   FAIL {pid}: high={d['high']} low={d['low']}")
    return acc


def _retest_one(i_r):
    i, r = i_r
    base = judge_one(r["question"])["bin"]
    repeat = judge_one(r["question"])["bin"]
    para = chat_json(PARAPHRASE_SYSTEM, r["question"], max_retries=3)["paraphrase"]
    para_bin = judge_one(para)["bin"]
    return i, r["id"], base, repeat, para_bin


def run_retest(n_questions: int = 30) -> tuple[float, float]:
    records = [json.loads(l) for l in open(DATA_DIR / "questions.jsonl")]
    rng = random.Random(SEED)
    sample = rng.sample(records, n_questions)

    exact_ok = para_ok = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, qid, base, repeat, para_bin in ex.map(_retest_one, enumerate(sample)):
            exact_ok += base == repeat
            para_ok += base == para_bin
            print(f"[gate] retest {i+1}/{n_questions} {qid}: base={base} repeat={repeat} para={para_bin}")

    exact_acc = exact_ok / n_questions
    para_acc = para_ok / n_questions
    print(f"[gate] test-retest: exact {exact_ok}/{n_questions} ({exact_acc:.0%}), "
          f"paraphrase {para_ok}/{n_questions} ({para_acc:.0%})")
    return exact_acc, para_acc


def main() -> None:
    sanity_acc = run_sanity_pairs()
    exact_acc, para_acc = run_retest()

    ok = (
        sanity_acc >= SANITY_PAIRS_MIN_ACCURACY
        and exact_acc >= RETEST_MIN_AGREEMENT
        and para_acc >= RETEST_MIN_AGREEMENT
    )
    print("\nGATE RESULT:", "PASS" if ok else "FAIL")
    if not ok:
        print("Iterate on the rubric in critab/judge.py, then re-run.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
