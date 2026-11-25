"""Letter-probability confidence extraction for MCQA (local model, MPS).

For each question: one forward pass over the chat-formatted prompt; the
logits at the final position restricted to the option-letter candidate
tokens give per-option probabilities (softmax over candidates).
Also saves the last-layer hidden state at that position (for the
fast-follow probe) and correctness vs gold.
"""
import json
import math
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import BASE_MODEL_ID, DATA_DIR


def build_prompt(tok, question: str, options: list[str], letters: str) -> str:
    lines = [question.strip(), ""]
    for L, o in zip(letters, options):
        lines.append(f"{L}. {o}")
    lines.append("")
    lines.append(
        "Answer with a single letter from the list above. "
        "If you are not confident, it is acceptable to choose the option that says you cannot answer."
    )
    lines.append("Answer:")
    user_content = "\n".join(lines)
    return tok.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def letter_token_ids(tok, letters: str) -> dict[str, list[int]]:
    """Token ids for each letter, with and without a leading space."""
    out = {}
    for L in letters:
        ids = set()
        for variant in (L, " " + L):
            enc = tok.encode(variant, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
        out[L] = sorted(ids)
    return out


def extract(model, tok, records, batch_size: int = 4, hidden_dim: int | None = None,
            resume_ids: set[str] | None = None, mmap_path=None, ids_order=None):
    """Runs forward passes; results are written incrementally (rows -> JSONL via
    row_writer, hidden states -> memory-mapped npy). Resume skips ids already done.

    Batches are processed in order of prompt length (fewer distinct shapes ->
    less MPS kernel compilation; less padding waste). Records order is restored
    by the caller via ids_order.
    """
    # length-sort indices for batching efficiency
    prompts = {r["id"]: build_prompt(tok, r["question"], r["options"], r["letters"]) for r in records}
    cand = {r["id"]: letter_token_ids(tok, r["letters"]) for r in records}
    by_id = {r["id"]: r for r in records}
    order = sorted([r["id"] for r in records if resume_ids is None or r["id"] not in resume_ids],
                   key=lambda q: len(prompts[q]))

    row_writer = open(DATA_DIR / "confidence.jsonl", "a")
    n_done = len(resume_ids or set())
    n_total = len(records)
    t_start = time.time()

    for s in range(0, len(order), batch_size):
        chunk = order[s : s + batch_size]
        enc = tok([prompts[q] for q in chunk], return_tensors="pt", padding=True, padding_side="left")
        input_ids = enc["input_ids"].to(model.device)
        attn = enc["attention_mask"].to(model.device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
        logits = out.logits[:, -1, :].float()
        hidden = out.hidden_states[-1][:, -1, :].float()

        for b, q in enumerate(chunk):
            r = by_id[q]
            row_logits = logits[b]
            letter_logits, letter_probs = {}, {}
            flat = []
            for L, ids in cand[q].items():
                v = max(float(row_logits[i]) for i in ids)
                letter_logits[L] = v
                flat.append(v)
            probs = torch.softmax(torch.tensor(flat), dim=0)
            for i, L in enumerate(cand[q].keys()):
                letter_probs[L] = float(probs[i])

            chosen_L = max(letter_probs, key=letter_probs.get)
            chosen_idx = r["letters"].index(chosen_L)
            sub_letters = [L for i, L in enumerate(r["letters"]) if i != r["idk_idx"]]
            sub_L = max(sub_letters, key=lambda L: letter_probs[L])
            sub_idx = r["letters"].index(sub_L)

            vec = hidden[b].cpu().numpy().astype(np.float32)
            if mmap_path is not None:
                mmap_path[ids_order.index(q)] = vec

            row_writer.write(json.dumps({
                "id": q,
                "letter_probs": letter_probs,
                "letter_logits": letter_logits,
                "chosen_letter": chosen_L,
                "chosen_idx": chosen_idx,
                "chosen_is_idk": chosen_idx == r["idk_idx"],
                "substantive_letter": sub_L,
                "substantive_idx": sub_idx,
                "confidence_raw": letter_probs[chosen_L],
                "confidence_sub_raw": letter_probs[sub_L],
                "correct": chosen_idx == r["gold_idx"],
                "substantive_correct": sub_idx == r["gold_idx"],
            }) + "\n")
        row_writer.flush()
        n_done += len(chunk)
        if (n_done // batch_size) % 10 == 0 or n_done == n_total:
            rate = (n_done - len(resume_ids or set())) / max(1e-9, time.time() - t_start)
            eta = (n_total - n_done) / max(rate, 1e-9) / 60
            print(f"[confidence] {n_done}/{n_total} | {rate:.1f} q/s | ETA {eta:.0f} min")
    row_writer.close()


def main():
    records = [json.loads(l) for l in open(DATA_DIR / "questions.jsonl")]
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # Resume support: skip ids already in confidence.jsonl; reuse mmap array.
    done_ids = set()
    conf_path = DATA_DIR / "confidence.jsonl"
    if conf_path.exists():
        with open(conf_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except json.JSONDecodeError:  # partial line from a killed run
                    break
    ids_order = [r["id"] for r in records]
    hs_path = DATA_DIR / "hidden_states.npy"
    if hs_path.exists():
        mmap = np.lib.format.open_memmap(hs_path, mode="r+")
    else:
        probe = letter_token_ids(tok, records[0]["letters"])
        mmap = None  # created after first forward pass reveals hidden dim

    try:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, dtype=torch.float16, low_cpu_mem_usage=True
        )
    except TypeError:  # older transformers kwarg
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
    model = model.to("mps")
    model.eval()

    # Create the mmap lazily: we need the hidden dim; do one tiny forward pass if needed.
    if mmap is None:
        _r = records[0]
        _enc = tok([build_prompt(tok, _r["question"], _r["options"], _r["letters"])],
                   return_tensors="pt", padding=True, padding_side="left").to(model.device)
        with torch.no_grad():
            _out = model(**_enc, output_hidden_states=True)
        hidden_dim = _out.hidden_states[-1].shape[-1]
        mmap = np.lib.format.open_memmap(hs_path, mode="w+", dtype=np.float32,
                                         shape=(len(records), hidden_dim))
        mmap.flush()
        del _out

    todo = [r for r in records if r["id"] not in done_ids]
    print(f"[confidence] resuming: {len(done_ids)} done, {len(todo)} to go")
    extract(model, tok, records, batch_size=4, resume_ids=done_ids, mmap_path=mmap, ids_order=ids_order)

    ids = np.array(ids_order)
    np.savez(DATA_DIR / "hidden_states.npz", ids=ids, X=np.load(hs_path, mmap_mode="r")[:])
    print(f"wrote {len(done_ids) + len(todo)} rows -> data/confidence.jsonl, data/hidden_states.npz")


if __name__ == "__main__":
    main()
