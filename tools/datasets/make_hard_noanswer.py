"""Harder no-answer questions: keep the question, remove its gold passage(s) from retrieval.

Usage:
    python -m tools.datasets.make_hard_noanswer

Why
    The original no-answer set asks Collection B questions against Collection A. The topics do
    not overlap, so the retrieved passages are obviously unrelated. Here the question stays in
    its own collection and only the evidence is taken away, so retrieval still returns passages
    from the same article / about the same entities, but none of them contains the answer.

Reads   eval/squad_dev300.jsonl, eval/squad_holdout100.jsonl     (collection squad_v11_dev500)
        eval/hotpot_dev.jsonl,   eval/hotpot_holdout.jsonl        (collection hotpot_dev400)
Writes  eval/hard_noanswer_dev.jsonl       50 SQuAD + 50 HotpotQA
        eval/hard_noanswer_holdout.jsonl   50 SQuAD + 50 HotpotQA
        eval/hard_noanswer_manifest.json

Pool
    dev: only the first 150 questions of each dev file, because 03 ran those end to end with the
    gold passage present, so every question can be paired with its "evidence present" result.
    holdout: all 100 (all were run end to end).

Rules
    SQuAD   remove the single gold passage. Skip if any gold answer string appears (whole tokens,
            case-insensitive) in any other passage of the collection: it might still be answerable.
    HotpotQA bridge: remove the gold passage that contains the answer, keep the other one (the
            model still sees the first hop). Skip if the answer is in both or neither gold
            passage, or anywhere else in the collection.
            comparison, yes/no answer: remove one gold passage at random (seeded). Skip
            "both ... ?" questions answered "no": one side alone can be enough to say no.
            comparison with an entity answer: skipped (the answer is one of the two names in the
            question, so a string check cannot tell whether it leaked).
    The kept gold passage is stored as kept_gold_ids for the partial_oracle mode.
    answers = [] (no-answer); the original answers are kept in orig_answers for analysis.
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.rag_core import _tokens, load_collection, load_jsonl  # noqa: E402

SEED = 41
N_PER_SOURCE = 50
SOURCES = [  # (source, collection, file, pool size, split)
    ("squad", "squad_v11_dev500", "squad_dev300.jsonl", 150, "dev"),
    ("hotpot", "hotpot_dev400", "hotpot_dev.jsonl", 150, "dev"),
    ("squad", "squad_v11_dev500", "squad_holdout100.jsonl", 100, "holdout"),
    ("hotpot", "hotpot_dev400", "hotpot_holdout.jsonl", 100, "holdout"),
]


def tok_str(s: str) -> str:
    return " " + " ".join(_tokens(s)) + " "


def has(text_tok: str, answer: str) -> bool:
    a = " ".join(_tokens(answer))
    return bool(a) and f" {a} " in text_tok


def make_one(q: dict, source: str, doc_tok: dict, rng: random.Random):
    """Return (record, None) or (None, skip_reason)."""
    answers = q["answers"]
    if source == "squad":
        removed, kept = [q["gold_doc_id"]], []
    else:
        g = q["gold_doc_ids"]
        yesno = {a.lower() for a in answers} <= {"yes", "no"}
        if q["type"] == "bridge":
            inside = [d for d in g if any(has(doc_tok[d], a) for a in answers)]
            if len(inside) != 1:
                return None, "bridge: answer in both/neither gold"
            removed, kept = inside, [d for d in g if d not in inside]
        elif yesno:
            if re.search(r"\bboth\b", q["question"], re.I) and answers[0].lower() == "no":
                return None, "comparison: 'both ...? no'"
            r = rng.randrange(2)
            removed, kept = [g[r]], [g[1 - r]]
        else:
            return None, "comparison: entity answer"
    if not (source == "hotpot" and q["type"] == "comparison"):
        leak = [d for d, t in doc_tok.items() if d not in removed and any(has(t, a) for a in answers)]
        if leak:
            return None, "answer string elsewhere in collection"
    rec = {"qid": q["qid"], "question": q["question"], "answers": [], "orig_answers": answers,
           "removed_doc_ids": removed, "kept_gold_ids": kept, "source": source,
           "type": q.get("type") or "single", "orig_split": q.get("split")}
    return rec, None


def main() -> int:
    rng = random.Random(SEED)
    out = {"dev": [], "holdout": []}
    manifest = {"seed": SEED, "n_per_source": N_PER_SOURCE, "sources": []}
    tok_cache = {}
    for source, coll, fname, pool, split in SOURCES:
        if coll not in tok_cache:
            docs = load_collection(ROOT, coll)
            if not docs:
                print(f"[fail] collection {coll} not found", file=sys.stderr)
                return 1
            tok_cache[coll] = {d["doc_id"]: tok_str(d["title"] + " " + d["text"]) for d in docs}
        doc_tok = tok_cache[coll]
        qs = load_jsonl(ROOT / "eval" / fname)[:pool]
        ok, skipped = [], Counter()
        for q in qs:
            rec, why = make_one(q, source, doc_tok, rng)
            if rec:
                rec["collection"], rec["split"], rec["source_file"] = coll, f"hard_{split}", fname
                ok.append(rec)
            else:
                skipped[why] += 1
        picked = rng.sample(ok, min(N_PER_SOURCE, len(ok)))
        order = {q["qid"]: i for i, q in enumerate(qs)}
        picked.sort(key=lambda r: order[r["qid"]])
        out[split] += picked
        manifest["sources"].append({"file": fname, "pool": len(qs), "eligible": len(ok),
                                    "picked": len(picked), "skipped": dict(skipped),
                                    "picked_types": dict(Counter(r["type"] for r in picked))})
        print(f"{fname}: pool {len(qs)}, eligible {len(ok)}, picked {len(picked)}, skipped {dict(skipped)}")
    for split, recs in out.items():
        p = ROOT / "eval" / f"hard_noanswer_{split}.jsonl"
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
        manifest[f"hard_noanswer_{split}.jsonl"] = len(recs)
    (ROOT / "eval" / "hard_noanswer_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
