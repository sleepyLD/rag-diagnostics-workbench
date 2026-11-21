"""BEIR SciFact (domain shift: scientific claims -> abstracts). Eval = the 300 test queries split dev/holdout; train = train qrels.
    python -m tools.bench.prepare_scifact"""
from collections import defaultdict
from tools.bench.common import finalize, hf_file, read_parquet, split_questions


def qrels(path):
    out = defaultdict(list)
    for line in open(path, encoding="utf-8").read().splitlines()[1:]:
        q, d, s = line.split("\t")
        if int(s) > 0:
            out[q].append(d)
    return out


def main():
    cpath, sha = hf_file("BeIR/scifact", "corpus/corpus-00000-of-00001.parquet")
    qpath, _ = hf_file("BeIR/scifact", "queries/queries-00000-of-00001.parquet")
    test, sha_q = hf_file("BeIR/scifact-qrels", "test.tsv")
    train, _ = hf_file("BeIR/scifact-qrels", "train.tsv")
    docs = {r["_id"]: r for r in read_parquet(cpath)}
    corpus = [{"doc_id": f"scifact_{i}", "title": r["title"], "text": r["text"]} for i, r in docs.items()]
    qtext = {r["_id"]: r["text"] for r in read_parquet(qpath)}
    qs = [{"qid": f"scifact_q{q}", "question": qtext[q], "answers": [], "gold_doc_ids": [f"scifact_{d}" for d in ds],
           "gold_titles": [docs[d]["title"] for d in ds], "type": "claim", "hops": len(ds)} for q, ds in qrels(test).items()]
    dev, holdout, _ = split_questions(qs, n_dev=150, n_holdout=150)
    train_rows = [{"qid": f"scifact_q{q}", "question": qtext[q], "answers": [], "type": "claim",
                   "positives": [{"title": docs[d]["title"], "text": docs[d]["text"]} for d in ds], "negatives": []}
                  for q, ds in qrels(train).items()]
    finalize("scifact", corpus, dev, holdout,
             {"repo": "BeIR/scifact + BeIR/scifact-qrels", "revision": sha, "qrels_revision": sha_q, "license": "CC BY-SA 4.0 (BEIR) / CC BY-NC 2.0 (SciFact, Wadden et al. 2020)"},
             train_rows=train_rows, notes="Original BEIR corpus (5,183 abstracts). Test queries split 150 dev / 150 holdout. No answers (retrieval only).")


if __name__ == "__main__":
    main()
