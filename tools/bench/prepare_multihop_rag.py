"""MultiHop-RAG (cross-document news QA). corpus = 609 whole articles; gold = articles (by URL).   python -m tools.bench.prepare_multihop_rag"""
import hashlib, json
from tools.bench.common import finalize, hf_file, split_questions

REPO = "yixuantt/MultiHopRAG"


def did(url):
    return "mhr_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def main():
    cpath, sha = hf_file(REPO, "corpus.json")
    qpath, _ = hf_file(REPO, "MultiHopRAG.json")
    corpus, seen = [], set()
    for a in json.load(open(cpath, encoding="utf-8")):
        d = did(a["url"])
        if d in seen or not (a.get("body") or "").strip():
            continue
        seen.add(d)
        corpus.append({"doc_id": d, "title": a["title"], "text": a["body"], "source_doc_id": d,
                       "source": a.get("source"), "published_at": a.get("published_at"), "category": a.get("category")})
    qs, null_qs, skipped = [], [], 0
    for i, q in enumerate(json.load(open(qpath, encoding="utf-8"))):
        row = {"qid": f"mhr_{i:04d}", "question": q["query"], "answers": [q["answer"]], "type": q["question_type"]}
        if q["question_type"] == "null_query" or not q["evidence_list"]:
            null_qs.append({**row, "gold_doc_ids": [], "hops": 0})
            continue
        gold = list(dict.fromkeys(did(e["url"]) for e in q["evidence_list"]))
        if not all(g in seen for g in gold):
            skipped += 1
            continue
        qs.append({**row, "gold_doc_ids": gold, "gold_titles": list(dict.fromkeys(e["title"] for e in q["evidence_list"])), "hops": len(gold)})
    dev, holdout, _ = split_questions(qs)
    finalize("multihop_rag", corpus, dev, holdout,
             {"repo": REPO, "revision": sha, "files": ["corpus.json", "MultiHopRAG.json"], "license": "ODC-By (Tang & Yang, 2024)"},
             extra_files={"queries_null.jsonl": null_qs},
             notes=f"Whole articles (long; chunking is a later experiment, chunks must carry source_doc_id). Gold = distinct evidence articles. "
                   f"{len(null_qs)} null queries saved separately; {skipped} questions skipped because an evidence URL is not in corpus.json. No train split exists.")


if __name__ == "__main__":
    main()
