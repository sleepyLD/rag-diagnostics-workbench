"""Build a RAG collection and question sets from HotpotQA (dev, distractor setting).

Usage:
    python -m tools.datasets.prepare_hotpotqa data/raw/hotpotqa/hotpot_dev_distractor_v1.json
    python -m tools.datasets.prepare_hotpotqa data/raw/hotpotqa/validation-00000-of-00001.parquet   # Hugging Face copy (needs pyarrow)

Source format (per question, from the official README):
    _id, question, answer, type (bridge / comparison), level,
    supporting_facts: [[title, sent_id], ...]   -> the paragraphs needed to answer
    context: [[title, [sentences...]], ...]     -> 2 gold + 8 distractor paragraphs

Output
  collections/<collection>/<doc_id>.md     one paragraph per file (all context paragraphs of
                                            all sampled questions, de-duplicated by title)
  eval/hotpot_dev.jsonl, eval/hotpot_holdout.jsonl
  eval/hotpot_manifest.json

Why this makes a harder benchmark than SQuAD
  - Every question needs TWO paragraphs (multi-hop or comparison).
  - The 8 distractors were picked by HotpotQA's authors because they look similar
    to the question, so retrieval has realistic near-misses.
  - Questions were not written while reading a single paragraph, so word overlap
    with the gold paragraph is lower.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

LICENSE = "CC BY-SA 4.0 (HotpotQA, Yang et al. 2018)"


def doc_id_for(title: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")[:60] or "doc"
    return f"{slug}__{hashlib.sha1(title.encode('utf-8')).hexdigest()[:6]}"


def load_examples(path: Path) -> list[dict]:
    """Official JSON, or the Hugging Face parquet (hotpotqa/hotpot_qa, distractor/validation-*.parquet).
    The parquet stores supporting_facts / context as {title: [...], sent_id|sentences: [...]};
    convert them back to the official [[title, x], ...] lists."""
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        out = []
        for r in pq.read_table(path).to_pylist():
            sf, ctx = r["supporting_facts"], r["context"]
            out.append({"_id": r["id"], "question": r["question"], "answer": r["answer"],
                        "type": r.get("type"), "level": r.get("level"),
                        "supporting_facts": [list(x) for x in zip(sf["title"], sf["sent_id"])],
                        "context": [list(x) for x in zip(ctx["title"], ctx["sentences"])]})
        return out
    return json.loads(path.read_bytes())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hotpot_json")
    ap.add_argument("--collection", default="hotpot_dev400")
    ap.add_argument("--n-dev", type=int, default=300)
    ap.add_argument("--n-holdout", type=int, default=100)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--root", default=".")
    ap.add_argument("--min-questions", type=int, default=7000, help="sanity check on the input size")
    args = ap.parse_args(argv)

    raw = Path(args.hotpot_json).read_bytes()
    try:
        data = load_examples(Path(args.hotpot_json))
    except Exception as e:                       # e.g. the 40-byte error page from a blocked download
        print(f"[fail] cannot read {args.hotpot_json} ({len(raw)} bytes): {e}", file=sys.stderr)
        return 1
    if len(data) < args.min_questions:
        print(f"[fail] only {len(data)} questions (the real dev set has 7,405) - wrong or partial file?",
              file=sys.stderr)
        return 1
    print(f"[ok] loaded {len(data)} questions from {Path(args.hotpot_json).name}")
    rng = random.Random(args.seed)
    sample = rng.sample(data, args.n_dev + args.n_holdout)

    paragraphs, conflicts = {}, 0
    for ex in sample:
        for title, sents in ex["context"]:
            text = "".join(sents).strip()
            did = doc_id_for(title)
            if did in paragraphs:
                conflicts += paragraphs[did]["text"] != text
                continue
            paragraphs[did] = {"doc_id": did, "title": title, "text": text}

    questions = []
    for i, ex in enumerate(sample):
        titles = list(dict.fromkeys(t for t, _ in ex["supporting_facts"]))
        gold = [doc_id_for(t) for t in titles]
        missing = [g for g in gold if g not in paragraphs]
        if missing:
            print(f"[fail] {ex['_id']}: supporting paragraph not in context: {missing}", file=sys.stderr)
            return 1
        questions.append({"qid": ex["_id"], "question": ex["question"].strip(), "answers": [ex["answer"].strip()],
                          "gold_doc_ids": gold, "gold_doc_id": gold[0], "type": ex.get("type"),
                          "level": ex.get("level"), "split": "dev" if i < args.n_dev else "holdout"})

    root = Path(args.root)
    coll = root / "collections" / args.collection
    if coll.exists() and any(coll.iterdir()):
        print(f"[stop] {coll} is not empty; delete it first to rebuild", file=sys.stderr)
        return 2
    coll.mkdir(parents=True, exist_ok=True)
    for p in paragraphs.values():
        title = p["title"].replace("\n", " ")
        fm = (f"---\ndoc_id: {p['doc_id']}\ntitle: {title}\ncollection: {args.collection}\n"
              f"source: HotpotQA dev distractor\nlicense: {LICENSE}\n---\n\n")
        (coll / f"{p['doc_id']}.md").write_text(fm + p["text"] + "\n", encoding="utf-8")

    ev = root / "eval"
    ev.mkdir(parents=True, exist_ok=True)
    for name, split in (("hotpot_dev.jsonl", "dev"), ("hotpot_holdout.jsonl", "holdout")):
        with (ev / name).open("w", encoding="utf-8") as f:
            for q in questions:
                if q["split"] == split:
                    f.write(json.dumps(q, ensure_ascii=False) + "\n")

    types = Counter(q["type"] for q in questions)
    yes_no = sum(q["answers"][0].lower() in ("yes", "no") for q in questions)
    (ev / "hotpot_manifest.json").write_text(json.dumps({
        "source_sha256": hashlib.sha256(raw).hexdigest(), "license": LICENSE, "seed": args.seed,
        "collection": args.collection, "n_paragraphs": len(paragraphs), "n_dev": args.n_dev,
        "n_holdout": args.n_holdout, "types": types, "levels": Counter(q["level"] for q in questions),
        "yes_no_answers": yes_no, "same_title_different_text": conflicts,
    }, indent=2), encoding="utf-8")
    print(f"[ok] {len(paragraphs)} paragraphs in {coll} ({conflicts} same-title/different-text skipped)")
    print(f"[ok] {args.n_dev} dev + {args.n_holdout} holdout questions; types {dict(types)}; yes/no answers {yes_no}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
