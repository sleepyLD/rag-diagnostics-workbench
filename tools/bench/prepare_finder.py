"""FinDER (Linq-AI-Research/FinDER, CC BY-NC 4.0): expert questions on 10-K filings with evidence passages.
Corpus = every distinct evidence passage of all 5,703 questions (pooled, like the HotpotQA setup); gold = the question's own passages.
    python -m tools.bench.prepare_finder"""
import hashlib
from tools.bench.common import finalize, hf_file, read_parquet, split_questions

REPO = "Linq-AI-Research/FinDER"


def did(t):
    return "finder_" + hashlib.sha1(t.strip().encode("utf-8")).hexdigest()[:14]


def main():
    p, sha = hf_file(REPO, "data/train-00000-of-00001.parquet")
    rows = read_parquet(p)
    corpus, seen, qs = [], set(), []
    for r in rows:
        refs = [x for x in (r.get("references") or []) if x and x.strip()]
        if not refs:
            continue
        g = []
        for t in refs:
            d = did(t)
            if d not in seen:
                seen.add(d)
                corpus.append({"doc_id": d, "title": "", "text": t.strip()})
            g.append(d)
        g = list(dict.fromkeys(g))
        qs.append({"qid": f"finder_{r['_id']}", "question": r["text"], "answers": [r.get("answer") or ""], "gold_doc_ids": g,
                   "type": f"{r.get('category')}|{r.get('type')}", "hops": len(g)})
    dev, holdout, _ = split_questions(qs)
    finalize("finder", corpus, dev, holdout, {"repo": REPO, "revision": sha, "license": "CC BY-NC 4.0 (FinDER)", "domain": "finance"},
             notes="Pooled corpus of all evidence passages (10-K excerpts); queries are short expert-style (abbreviations, jargon); every evidence passage required.")


if __name__ == "__main__":
    main()
