"""Build a second, topic-disjoint SQuAD collection (Collection B) and its questions.

Usage:
    python -m tools.datasets.prepare_squad_b data/raw/squad/dev-v1.1.json

Output
  collections/<collection>/<doc_id>.md     one paragraph per file, same format as Collection A
  eval/squad_b_questions.jsonl              questions whose answers are in B only
  eval/squad_b_manifest.json                articles, seed, counts, excluded questions

Purpose
  1. Collection switching / isolation: A's index must never return B's documents and vice versa.
  2. No-answer test: B's questions asked against A have no answer there, so the right
     behaviour is to abstain.

A question is dropped if any of its gold answer strings also appears (whole-word,
case-insensitive) in Collection A, because then A might actually be able to answer it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from .prepare_squad import LICENSE, contains_answer, doc_id_for

DEFAULT_ARTICLES = ["Super_Bowl_50", "Prime_number", "Genghis_Khan", "Black_Death", "Packet_switching"]


def read_collection_texts(folder: Path) -> list[str]:
    return [p.read_text(encoding="utf-8").split("\n---\n", 1)[1] for p in sorted(folder.glob("*.md"))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("squad_json")
    ap.add_argument("--articles", nargs="+", default=DEFAULT_ARTICLES)
    ap.add_argument("--collection", default="squad_v11_b5")
    ap.add_argument("--exclude-collection", default="squad_v11_dev500",
                    help="questions whose answer appears in this collection are dropped")
    ap.add_argument("--n-questions", type=int, default=15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)

    root = Path(args.root)
    raw = Path(args.squad_json).read_bytes()
    data = json.loads(raw)
    by_title = {a["title"]: a for a in data["data"]}
    missing = [t for t in args.articles if t not in by_title]
    if missing:
        print(f"[fail] not in SQuAD file: {missing}", file=sys.stderr)
        return 1

    other_dir = root / "collections" / args.exclude_collection
    other_texts = read_collection_texts(other_dir) if other_dir.exists() else []
    if not other_texts:
        print(f"[warn] {other_dir} not found; answer-overlap check skipped", file=sys.stderr)
    other_titles = {p.stem.split("__")[0] for p in other_dir.glob("*.md")} if other_texts else set()
    clash = [t for t in args.articles if doc_id_for(t, 0).split("__")[0] in other_titles]
    if clash:
        print(f"[fail] articles already in {args.exclude_collection}: {clash}", file=sys.stderr)
        return 1

    paragraphs = []
    for title in args.articles:
        for p_idx, p in enumerate(by_title[title]["paragraphs"]):
            paragraphs.append({"doc_id": doc_id_for(title, p_idx), "title": title.replace("_", " "),
                               "text": p["context"], "qas": p["qas"]})

    rng = random.Random(args.seed)
    pool = list(paragraphs)
    rng.shuffle(pool)
    questions, dropped = [], []
    for p in pool:
        if len(questions) >= args.n_questions:
            break
        qas = list(p["qas"])
        rng.shuffle(qas)
        for qa in qas:
            answers = list(dict.fromkeys(a["text"].strip() for a in qa["answers"]))
            hit = [a for a in answers if any(contains_answer(t, a) for t in other_texts)]
            if hit:
                dropped.append({"qid": qa["id"], "question": qa["question"], "answer_found_in_other": hit})
                continue
            questions.append({"qid": qa["id"], "question": qa["question"].strip(), "answers": answers,
                              "gold_doc_id": p["doc_id"], "collection": args.collection,
                              "answer_start": qa["answers"][0]["answer_start"], "split": "b_dev"})
            break
    if len(questions) < args.n_questions:
        print(f"[fail] only {len(questions)} usable questions", file=sys.stderr)
        return 1

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
    with (ev / "squad_b_questions.jsonl").open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    manifest = {"source_sha256": hashlib.sha256(raw).hexdigest(), "license": LICENSE, "seed": args.seed,
                "collection": args.collection, "articles": args.articles, "n_paragraphs": len(paragraphs),
                "n_questions": len(questions), "excluded_collection": args.exclude_collection,
                "dropped_because_answer_in_other_collection": dropped}
    (ev / "squad_b_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] {len(args.articles)} articles -> {len(paragraphs)} paragraphs in {coll}")
    print(f"[ok] {len(questions)} questions in eval/squad_b_questions.jsonl "
          f"({len(dropped)} dropped: answer also appears in {args.exclude_collection})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
