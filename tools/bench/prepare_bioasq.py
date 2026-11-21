"""BioASQ (rag-datasets/rag-mini-bioasq): cross-document biomedical QA. Corpus = 40,221 PubMed passages.
Only questions whose answer needs 2-5 passages (all present in the corpus) are kept; success = every passage found.
    python -m tools.bench.prepare_bioasq"""
import ast
from tools.bench.common import finalize, hf_file, read_parquet, split_questions

REPO = "rag-datasets/rag-mini-bioasq"


def main():
    pp, sha = hf_file(REPO, "data/passages.parquet/part.0.parquet")
    tp, _ = hf_file(REPO, "data/test.parquet/part.0.parquet")
    # v2: the parquet stores 12,220 empty passages as the literal string "nan" and ~30 exact duplicates;
    # v1 kept them, so 346/500 dev questions had an unfindable "nan" gold passage.
    corpus, canon, first, n_nan = [], {}, {}, 0
    for r in read_parquet(pp):
        t = (r["passage"] or "").strip()
        if not t or t.lower() in ("nan", "none", "null"):
            n_nan += 1; continue
        did, key = f"bioasq_{r['id']}", " ".join(t.split())
        if key in first:
            canon[did] = first[key]; continue
        first[key] = canon[did] = did
        corpus.append({"doc_id": did, "title": "", "text": r["passage"]})
    ids = {d["doc_id"] for d in corpus}
    text = {d["doc_id"]: d["text"] for d in corpus}
    qs, dropped = [], {"lt2": 0, "gt5": 0, "missing": 0}
    for r in read_parquet(tp):
        raw = [f"bioasq_{x}" for x in ast.literal_eval(r["relevant_passage_ids"])]
        rel = list(dict.fromkeys(canon[g] for g in raw if g in canon))   # drop "nan" passages, map duplicates
        if not rel:
            dropped["missing"] += 1; continue
        if len(rel) < 2:
            dropped["lt2"] += 1; continue
        if len(rel) > 5:
            dropped["gt5"] += 1; continue
        qs.append({"qid": f"bioasq_{r['id']}", "question": r["question"], "answers": [r["answer"]], "gold_doc_ids": rel,
                   "type": f"xdoc_{len(rel)}", "hops": len(rel)})
    dev, holdout, rest = split_questions(qs)
    train_rows = [{"qid": q["qid"], "question": q["question"], "answers": q["answers"], "type": q["type"],
                   "positives": [{"title": "", "text": text[g]} for g in q["gold_doc_ids"][:3]],
                   "negatives": []} for q in rest[:3000]]
    finalize("bioasq", corpus, dev, holdout, {"repo": REPO, "revision": sha, "license": "CC BY 2.5 (BioASQ via rag-mini-bioasq)", "domain": "medical"},
             train_rows=train_rows,
             notes=f"v2 (nan/duplicate passages removed: {n_nan} nan, {sum(1 for k, v in canon.items() if k != v)} dups). Cross-document: questions with 2-5 relevant PubMed passages, all required. Dropped: {dropped}. Remaining questions after dev/holdout -> train.")


if __name__ == "__main__":
    main()
