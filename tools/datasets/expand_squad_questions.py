"""Add more SQuAD questions for Collection A without touching the collection itself.

Usage:
    python -m tools.datasets.expand_squad_questions data/raw/squad/dev-v1.1.json

Reads   eval/squad_dev.jsonl, eval/squad_holdout.jsonl   (the original 50 + 50)
Writes  eval/squad_dev300.jsonl       original 50 dev + 250 new
        eval/squad_holdout100.jsonl   original 50 holdout + 50 new
        eval/squad_expand_manifest.json

Rules
  - New questions come only from Collection A's paragraphs that have not been used
    by any existing question, at most one question per paragraph.
  - Dev and holdout never share a paragraph.
  - The original files are left unchanged, so earlier runs stay comparable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from .prepare_squad import contains_answer, doc_id_for


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("squad_json")
    ap.add_argument("--collection", default="squad_v11_dev500")
    ap.add_argument("--n-dev", type=int, default=300)
    ap.add_argument("--n-holdout", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)

    root = Path(args.root)
    ev = root / "eval"
    load = lambda p: [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    old_dev, old_hold = load(ev / "squad_dev.jsonl"), load(ev / "squad_holdout.jsonl")
    used = {q["gold_doc_id"] for q in old_dev + old_hold}

    coll = root / "collections" / args.collection
    coll_ids = {p.stem for p in coll.glob("*.md")}
    texts = {p.stem: p.read_text(encoding="utf-8").split("\n---\n", 1)[1].strip() for p in coll.glob("*.md")}

    raw = Path(args.squad_json).read_bytes()
    data = json.loads(raw)
    para_qas = {}
    for art in data["data"]:
        for p_idx, p in enumerate(art["paragraphs"]):
            did = doc_id_for(art["title"], p_idx)
            if did in coll_ids:
                para_qas[did] = p["qas"]
    missing = coll_ids - set(para_qas)
    if missing:
        print(f"[fail] {len(missing)} collection paragraphs not found in SQuAD file", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    free = sorted(set(para_qas) - used)
    rng.shuffle(free)
    n_new_dev, n_new_hold = args.n_dev - len(old_dev), args.n_holdout - len(old_hold)
    if n_new_dev + n_new_hold > len(free):
        print(f"[fail] need {n_new_dev + n_new_hold} new paragraphs, only {len(free)} free", file=sys.stderr)
        return 1

    def make(did, split):
        qa = rng.choice(para_qas[did])
        answers = list(dict.fromkeys(a["text"].strip() for a in qa["answers"]))
        return {"qid": qa["id"], "question": qa["question"].strip(), "answers": answers, "gold_doc_id": did,
                "answer_start": qa["answers"][0]["answer_start"], "split": split,
                "answer_also_in_docs": [o for o, t in texts.items() if o != did
                                        and any(contains_answer(t, a) for a in answers)]}

    new_dev = [make(d, "dev") for d in free[:n_new_dev]]
    new_hold = [make(d, "holdout") for d in free[n_new_dev:n_new_dev + n_new_hold]]
    dev, hold = old_dev + new_dev, old_hold + new_hold

    assert not ({q["gold_doc_id"] for q in dev} & {q["gold_doc_id"] for q in hold})
    assert len({q["qid"] for q in dev + hold}) == len(dev) + len(hold)

    for name, rows in (("squad_dev300.jsonl", dev), ("squad_holdout100.jsonl", hold)):
        with (ev / name).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (ev / "squad_expand_manifest.json").write_text(json.dumps({
        "source_sha256": hashlib.sha256(raw).hexdigest(), "seed": args.seed, "collection": args.collection,
        "n_dev": len(dev), "n_holdout": len(hold), "kept_original_dev": len(old_dev),
        "kept_original_holdout": len(old_hold),
    }, indent=2), encoding="utf-8")
    print(f"[ok] eval/squad_dev300.jsonl: {len(dev)} ({len(old_dev)} original + {len(new_dev)} new)")
    print(f"[ok] eval/squad_holdout100.jsonl: {len(hold)} ({len(old_hold)} original + {len(new_hold)} new)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
