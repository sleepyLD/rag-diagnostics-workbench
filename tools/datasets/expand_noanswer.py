"""Expand the no-answer test set (Collection B questions asked against Collection A).

Usage:
    python -m tools.datasets.expand_noanswer data/raw/squad/dev-v1.1.json

Reads   eval/squad_b_questions.jsonl          the original 15 (kept unchanged as the first 15)
Writes  eval/squad_b_noanswer_dev50.jsonl     original 15 + 35 new
        eval/squad_b_noanswer_holdout50.jsonl 50 new, only for the final holdout run
        eval/squad_b_noanswer_manifest.json

Rules (same as prepare_squad_b)
  - Only Collection B paragraphs not used by any existing question; at most one question each.
  - A question is skipped if any of its gold answer strings appears in Collection A
    (whole-word, case-insensitive), because A might then actually answer it.
  - Dev and holdout never share a paragraph.
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
    ap.add_argument("--collection-b", default="squad_v11_b5")
    ap.add_argument("--collection-a", default="squad_v11_dev500")
    ap.add_argument("--n-dev", type=int, default=50)
    ap.add_argument("--n-holdout", type=int, default=50)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)

    root = Path(args.root)
    ev = root / "eval"
    load = lambda p: [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    original = load(ev / "squad_b_questions.jsonl")
    used = {q["gold_doc_id"] for q in original}

    body = lambda p: p.read_text(encoding="utf-8").split("\n---\n", 1)[1]
    a_texts = [body(p) for p in sorted((root / "collections" / args.collection_a).glob("*.md"))]
    b_ids = {p.stem for p in (root / "collections" / args.collection_b).glob("*.md")}
    if not a_texts or not b_ids:
        print("[fail] collection A or B not found", file=sys.stderr)
        return 1

    raw = Path(args.squad_json).read_bytes()
    para_qas = {}
    for art in json.loads(raw)["data"]:
        for i, p in enumerate(art["paragraphs"]):
            did = doc_id_for(art["title"], i)
            if did in b_ids:
                para_qas[did] = p["qas"]

    rng = random.Random(args.seed)
    free = sorted(set(para_qas) - used)
    rng.shuffle(free)
    picked, dropped = [], 0
    need = (args.n_dev - len(original)) + args.n_holdout
    for did in free:
        if len(picked) >= need:
            break
        qas = list(para_qas[did])
        rng.shuffle(qas)
        for qa in qas:
            answers = list(dict.fromkeys(a["text"].strip() for a in qa["answers"]))
            if any(contains_answer(t, a) for t in a_texts for a in answers):
                dropped += 1
                continue
            picked.append({"qid": qa["id"], "question": qa["question"].strip(), "answers": answers,
                           "gold_doc_id": did, "collection": args.collection_b,
                           "answer_start": qa["answers"][0]["answer_start"]})
            break
    if len(picked) < need:
        print(f"[fail] only {len(picked)} usable questions, need {need}", file=sys.stderr)
        return 1

    n_new_dev = args.n_dev - len(original)
    dev = original + [dict(q, split="b_dev") for q in picked[:n_new_dev]]
    hold = [dict(q, split="b_holdout") for q in picked[n_new_dev:]]
    assert not ({q["gold_doc_id"] for q in dev} & {q["gold_doc_id"] for q in hold})

    for name, rows in (("squad_b_noanswer_dev50.jsonl", dev), ("squad_b_noanswer_holdout50.jsonl", hold)):
        with (ev / name).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (ev / "squad_b_noanswer_manifest.json").write_text(json.dumps({
        "source_sha256": hashlib.sha256(raw).hexdigest(), "seed": args.seed,
        "n_dev": len(dev), "n_holdout": len(hold), "kept_original": len(original),
        "dropped_answer_in_A": dropped}, indent=2), encoding="utf-8")
    print(f"[ok] dev {len(dev)} ({len(original)} original + {n_new_dev} new), holdout {len(hold)}; "
          f"{dropped} candidate(s) skipped because the answer also appears in A")
    return 0


if __name__ == "__main__":
    sys.exit(main())
