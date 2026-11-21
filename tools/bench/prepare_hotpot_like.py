"""HotpotQA and 2WikiMultihopQA (same record layout): pooled corpus + dev/holdout + train triples.

    python -m tools.bench.prepare_hotpot_like hotpotqa
    python -m tools.bench.prepare_hotpot_like 2wikimultihopqa
"""
import sys
from tools.bench.common import (build_pooled_corpus, doc_id_for, finalize, hf_file, iter_parquet, read_parquet,
                                split_questions, SEED)
import random

CFG = {
    "hotpotqa": {"repo": "hotpotqa/hotpot_qa", "val": "distractor/validation-00000-of-00001.parquet",
                 "train": ["distractor/train-00000-of-00002.parquet", "distractor/train-00001-of-00002.parquet"],
                 "license": "CC BY-SA 4.0 (Yang et al., EMNLP 2018)"},
    "2wikimultihopqa": {"repo": "framolfese/2WikiMultihopQA", "val": "data/validation-00000-of-00001.parquet",
                        "train": ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"],
                        "license": "Apache-2.0 (Ho et al., COLING 2020; HF mirror framolfese/2WikiMultihopQA)"},
}


def paragraphs(r):
    ctx = r["context"]
    sents = ctx.get("sentences") or ctx.get("content")
    return [(t, " ".join(x.strip() for x in s).strip() if isinstance(s, (list, tuple)) else str(s)) for t, s in zip(ctx["title"], sents)]


def convert(r):
    paras = paragraphs(r)
    by_title = {}
    for t, x in paras:
        by_title.setdefault(t, x)
    gold_titles = [t for t in dict.fromkeys(r["supporting_facts"]["title"]) if t in by_title]
    return {"qid": str(r["id"]), "question": r["question"], "answers": [r["answer"]],
            "gold_doc_ids": [doc_id_for(t) for t in gold_titles], "gold_titles": gold_titles,
            "type": r.get("type"), "hops": len(gold_titles), "level": r.get("level"),
            "_paras": [(doc_id_for(t), t, x) for t, x in by_title.items()]}


def main(name):
    cfg = CFG[name]
    path, sha = hf_file(cfg["repo"], cfg["val"])
    qs = [convert(r) for r in read_parquet(path)]
    qs = [q for q in qs if len(q["gold_doc_ids"]) >= 2]
    dev, holdout, rest = split_questions(qs)
    corpus, n_eval_docs = build_pooled_corpus(dev + holdout, rest, lambda q: q["_paras"])
    strip = lambda q: {k: v for k, v in q.items() if not k.startswith("_")}
    rng = random.Random(SEED)
    train = []
    for f in cfg["train"]:
        p, _ = hf_file(cfg["repo"], f)
        for r in iter_parquet(p):
            if rng.random() > 0.45:      # ~45% sample of train keeps memory low; capped to 30k later
                continue
            q = convert(r)
            if len(q["gold_doc_ids"]) < 2:
                continue
            gold = set(q["gold_titles"])
            train.append({"qid": q["qid"], "question": q["question"], "answers": q["answers"], "type": q["type"],
                          "positives": [{"title": t, "text": x} for _, t, x in q["_paras"] if t in gold],
                          "negatives": [{"title": t, "text": x} for _, t, x in q["_paras"] if t not in gold]})
    finalize(name, corpus, [strip(q) for q in dev], [strip(q) for q in holdout],
             {"repo": cfg["repo"], "revision": sha, "eval_file": cfg["val"], "train_files": cfg["train"], "license": cfg["license"]},
             train_rows=train,
             notes=f"doc = one paragraph, de-duplicated by title. {n_eval_docs} paragraphs come from the eval questions (2 gold + distractors each); "
                   "the rest are paragraphs of other validation questions. Train negatives = the dataset's own distractor paragraphs.")


if __name__ == "__main__":
    main(sys.argv[1])
