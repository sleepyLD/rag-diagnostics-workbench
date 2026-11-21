"""Generic BEIR-format builder (medical / finance IR sets: NFCorpus, FiQA, TREC-COVID ...).

    python -m tools.bench.prepare_beir --name nfcorpus --repo BeIR/nfcorpus --eval-splits dev,test --train-split train --success any
--success any : gold_sets = every relevant doc on its own -> all_gold_hit@k == "at least one relevant doc in top k" (multi-relevance IR sets)
--success all : every relevant doc is required (true multi-evidence questions)
Relevance threshold: --min-score (qrels score >= it). Corpus kept whole (no sub-sampling).
"""
import argparse
from collections import defaultdict
from huggingface_hub import HfApi
from tools.bench.common import finalize, hf_file, iter_parquet, split_questions


def qrels(path, min_score):
    out = defaultdict(dict)
    for line in open(path, encoding="utf-8").read().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        q, d, s = parts[0], parts[1], parts[-1]
        if int(float(s)) >= min_score:
            out[q][d] = int(float(s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--qrels-repo", default=None)
    ap.add_argument("--eval-splits", default="test")
    ap.add_argument("--train-split", default="")
    ap.add_argument("--min-score", type=int, default=1)
    ap.add_argument("--success", choices=["any", "all"], default="any")
    ap.add_argument("--n-dev", type=int, default=500)
    ap.add_argument("--n-holdout", type=int, default=300)
    ap.add_argument("--license", default="see source")
    ap.add_argument("--domain", default="")
    a = ap.parse_args()
    qrepo = a.qrels_repo or a.repo + "-qrels"
    files = [s.rfilename for s in HfApi().dataset_info(a.repo).siblings]
    cfile = next(f for f in files if f.startswith("corpus") and f.endswith(".parquet"))
    qfile = next(f for f in files if f.startswith("queries") and f.endswith(".parquet"))
    cpath, sha = hf_file(a.repo, cfile)
    qpath, _ = hf_file(a.repo, qfile)
    pref = a.name + "_"
    docs = {}
    for r in iter_parquet(cpath):
        if (r.get("text") or "").strip():
            docs[str(r["_id"])] = r
    corpus = [{"doc_id": pref + i, "title": r.get("title") or "", "text": r["text"]} for i, r in docs.items()]
    qtext = {str(r["_id"]): r["text"] for r in iter_parquet(qpath)}
    rel = {}
    qsha = None
    for sp in a.eval_splits.split(","):
        p, qsha = hf_file(qrepo, f"{sp}.tsv")
        for q, ds in qrels(p, a.min_score).items():
            rel.setdefault(q, {}).update(ds)
    qs = []
    for q, ds in rel.items():
        ds = {d: s for d, s in ds.items() if d in docs}
        if not ds or q not in qtext:
            continue
        gold = sorted(ds, key=lambda d: -ds[d])
        row = {"qid": f"{a.name}_q{q}", "question": qtext[q], "answers": [], "gold_doc_ids": [pref + d for d in gold],
               "gold_titles": [docs[d].get("title") or "" for d in gold[:20]], "type": f"ir_{a.success}", "hops": len(gold),
               "n_relevant": len(gold)}
        if a.success == "any":
            row["gold_sets"] = [[pref + d] for d in gold]
        qs.append(row)
    dev, holdout, _ = split_questions(qs, n_dev=a.n_dev, n_holdout=a.n_holdout)
    train_rows = None
    if a.train_split:
        p, _ = hf_file(qrepo, f"{a.train_split}.tsv")
        train_rows = [{"qid": f"{a.name}_t{q}", "question": qtext[q], "answers": [], "type": "ir",
                       "positives": [{"title": docs[d].get("title") or "", "text": docs[d]["text"]} for d in list(ds)[:4] if d in docs],
                       "negatives": []} for q, ds in qrels(p, a.min_score).items() if q in qtext]
        train_rows = [r for r in train_rows if r["positives"]]
    finalize(a.name, corpus, dev, holdout,
             {"repo": f"{a.repo} + {qrepo}", "revision": sha, "qrels_revision": qsha, "license": a.license, "domain": a.domain,
              "eval_splits": a.eval_splits, "min_score": a.min_score, "success": a.success},
             train_rows=train_rows,
             notes=f"BEIR format, whole corpus ({len(corpus)} docs). success={a.success}: "
                   + ("any one relevant doc in top-k counts (multi-relevance IR set)" if a.success == "any" else "every relevant doc required"))


if __name__ == "__main__":
    main()
