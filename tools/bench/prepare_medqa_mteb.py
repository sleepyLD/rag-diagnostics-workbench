"""MedQuAD consumer health QA via mteb/medical_qa (CC0): corpus.jsonl / queries.jsonl / default.jsonl (qrels).
    python -m tools.bench.prepare_medqa_mteb"""
import json
from tools.bench.common import finalize, hf_file, split_questions

REPO = "mteb/medical_qa"


def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    cp, sha = hf_file(REPO, "corpus.jsonl")
    qp, _ = hf_file(REPO, "queries.jsonl")
    rp, _ = hf_file(REPO, "default.jsonl")
    corpus = [{"doc_id": f"mqa_{r['_id']}", "title": r.get("title") or "", "text": r["text"]} for r in rj(cp) if (r.get("text") or "").strip()]
    ids = {d["doc_id"] for d in corpus}
    qtext = {str(r["_id"]): r["text"] for r in rj(qp)}
    rel = {}
    for r in rj(rp):
        q = str(r.get("query-id", r.get("query_id"))); d = f"mqa_{r.get('corpus-id', r.get('corpus_id'))}"
        if float(r.get("score", 1)) > 0 and d in ids:
            rel.setdefault(q, []).append(d)
    qs = [{"qid": f"mqa_{q}", "question": qtext[q], "answers": [], "gold_doc_ids": ds, "gold_sets": [[d] for d in ds],
           "type": "consumer_health", "hops": len(ds)} for q, ds in rel.items() if q in qtext]
    dev, holdout, _ = split_questions(qs)
    finalize("medqa", corpus, dev, holdout, {"repo": REPO, "revision": sha, "license": "CC0 (MedQuAD via MTEB)", "domain": "medical"},
             notes="MedQuAD (NIH consumer health): any relevant answer document counts.")


if __name__ == "__main__":
    main()
