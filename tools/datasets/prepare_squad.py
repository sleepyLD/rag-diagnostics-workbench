"""Build a small, reproducible RAG subset from SQuAD 1.1 (dev split).

Usage:
    python -m tools.datasets.prepare_squad data/raw/squad/dev-v1.1.json

Output
  collections/squad_v11_dev500/<doc_id>.md   one paragraph = one document = one chunk
                                             (front matter + paragraph text, nothing else)
  eval/squad_dev.jsonl                       questions you may look at while developing
  eval/squad_holdout.jsonl                   questions you only run at the end
  eval/squad_manifest.json                   seed, source hash, chosen articles, counts

Selection
  Whole Wikipedia articles are added (shuffled with --seed) until at least
  --target-paragraphs paragraphs are collected. Keeping articles whole means the
  index contains same-topic distractor paragraphs, which is where retrieval
  actually gets hard. Then at most one question per paragraph is sampled.

Leakage rules
  Questions and answers never go into collections/. The collection files hold only
  the paragraph text plus doc_id/title/source metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

SOURCE_URL = "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json"
LICENSE = "CC BY-SA 4.0 (SQuAD, Rajpurkar et al. 2016)"


def doc_id_for(title: str, p_idx: int) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")
    return f"{safe}__p{p_idx:03d}"


def contains_answer(text: str, answer: str) -> bool:
    # whole-token, case-insensitive match so "1" does not match "1990"
    pat = r"(?<!\w)" + re.escape(answer.strip()) + r"(?!\w)"
    return re.search(pat, text, flags=re.I) is not None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("squad_json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-paragraphs", type=int, default=500)
    ap.add_argument("--n-dev", type=int, default=50)
    ap.add_argument("--n-holdout", type=int, default=50)
    ap.add_argument("--collection", default="squad_v11_dev500")
    ap.add_argument("--root", default=".", help="project root (where collections/ and eval/ live)")
    args = ap.parse_args(argv)

    raw = Path(args.squad_json).read_bytes()
    data = json.loads(raw)
    if data.get("version") != "1.1":
        print(f"[fail] expected SQuAD version 1.1, got {data.get('version')}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    articles = list(enumerate(data["data"]))
    rng.shuffle(articles)

    chosen, paragraphs = [], []
    for a_idx, art in articles:
        if len(paragraphs) >= args.target_paragraphs:
            break
        chosen.append(art["title"])
        for p_idx, p in enumerate(art["paragraphs"]):
            paragraphs.append({"doc_id": doc_id_for(art["title"], p_idx),
                               "title": art["title"].replace("_", " "),
                               "text": p["context"], "qas": p["qas"]})

    # ---- questions: at most one per paragraph -------------------------------
    n_q = args.n_dev + args.n_holdout
    pool = list(paragraphs)
    rng.shuffle(pool)
    if len(pool) < n_q:
        print(f"[fail] only {len(pool)} paragraphs for {n_q} questions", file=sys.stderr)
        return 1
    questions = []
    for p in pool[:n_q]:
        qa = rng.choice(p["qas"])
        answers = list(dict.fromkeys(a["text"].strip() for a in qa["answers"]))
        questions.append({
            "qid": qa["id"],
            "question": qa["question"].strip(),
            "answers": answers,
            "gold_doc_id": p["doc_id"],
            "answer_start": qa["answers"][0]["answer_start"],
        })

    # paragraphs other than the gold one that also contain a gold answer string:
    # a 'miss' on these questions may not be a real retrieval failure
    for q in questions:
        others = [p["doc_id"] for p in paragraphs if p["doc_id"] != q["gold_doc_id"]
                  and any(contains_answer(p["text"], a) for a in q["answers"])]
        q["answer_also_in_docs"] = others

    dev, holdout = questions[:args.n_dev], questions[args.n_dev:]
    for q in dev:
        q["split"] = "dev"
    for q in holdout:
        q["split"] = "holdout"

    # ---- write ---------------------------------------------------------------
    root = Path(args.root)
    coll = root / "collections" / args.collection
    if coll.exists() and any(coll.iterdir()):
        print(f"[stop] {coll} is not empty; delete it first to rebuild", file=sys.stderr)
        return 2
    coll.mkdir(parents=True, exist_ok=True)
    for p in paragraphs:
        fm = (f"---\ndoc_id: {p['doc_id']}\ntitle: {p['title']}\n"
              f"collection: {args.collection}\nsource: SQuAD v1.1 dev\nlicense: {LICENSE}\n---\n\n")
        (coll / f"{p['doc_id']}.md").write_text(fm + p["text"].strip() + "\n", encoding="utf-8")

    ev = root / "eval"
    ev.mkdir(parents=True, exist_ok=True)
    for name, rows in (("squad_dev.jsonl", dev), ("squad_holdout.jsonl", holdout)):
        with (ev / name).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "source_url": SOURCE_URL,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "license": LICENSE,
        "seed": args.seed,
        "collection": args.collection,
        "articles": chosen,
        "n_articles": len(chosen),
        "n_paragraphs": len(paragraphs),
        "n_dev": len(dev),
        "n_holdout": len(holdout),
        "n_questions_with_answer_in_other_docs": sum(bool(q["answer_also_in_docs"]) for q in questions),
    }
    (ev / "squad_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[ok] {len(chosen)} articles -> {len(paragraphs)} paragraphs in {coll}")
    print(f"[ok] {len(dev)} dev + {len(holdout)} holdout questions in {ev}")
    print(f"[info] {manifest['n_questions_with_answer_in_other_docs']} question(s) have their answer string "
          f"in other paragraphs too (see answer_also_in_docs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
