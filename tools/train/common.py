"""Shared helpers for fine-tuning. Training rows come only from bench/train/*_train.jsonl (train splits,
already filtered against dev/holdout questions by tools/bench/common.filter_train)."""
import json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261118


def load_train(names, cap_per_dataset):
    rows = []
    for n in names:
        rs = [json.loads(l) for l in open(ROOT / "bench" / "train" / f"{n}_train.jsonl", encoding="utf-8")]
        random.Random(SEED).shuffle(rs)
        rows += [{**r, "_ds": n} for r in rs[:cap_per_dataset]]
    random.Random(SEED + 1).shuffle(rows)
    return rows


def ptext(p):   # must match src/retrievers/dense.doc_text
    return (p["title"] + "\n" + p["text"]) if p.get("title") else p["text"]


def hop_ctx(q, p, ctx_chars, sep):   # must match TwoHopRetriever._expand / ChainRerankRetriever
    return q + sep + ((p.get("title") or "") + ": ") + p["text"][:ctx_chars]


def eval_question_set():
    """All dev+holdout question strings (normalised) for a final leakage assertion."""
    qs = set()
    for d in (ROOT / "bench" / "data").glob("*/queries_*.jsonl"):
        for l in open(d, encoding="utf-8"):
            qs.add(" ".join(json.loads(l)["question"].lower().split()))
    return qs


def path_ctx(q, ps, ctx_chars, sep):   # must match ChainRerankRetriever._ctx / MultiHopBeamRetriever._q
    return q + "".join(sep + ((p.get("title") or "") + ": ") + p["text"][:ctx_chars] for p in ps)


def latest_checkpoint(ckpt_dir):
    """Newest checkpoint-<step> under ckpt_dir that finished saving (has trainer_state.json), or None."""
    from pathlib import Path
    d = Path(ckpt_dir)
    if not d.exists():
        return None
    cks = [c for c in d.glob("checkpoint-*") if (c / "trainer_state.json").exists()]
    return str(max(cks, key=lambda c: int(c.name.split("-")[-1]))) if cks else None


CKPT_KW = dict(save_strategy="steps", save_steps=0.1, save_total_limit=1)   # checkpoint every 10% of training steps (HF: <1 = ratio); resume from newest
