"""MuSiQue-Ans (answerable, 2-4 hops): pooled corpus + dev/holdout + train triples.   python -m tools.bench.prepare_musique"""
from tools.bench.common import build_pooled_corpus, doc_id_for, finalize, hf_file, read_jsonl, split_questions

REPO = "dgslibisey/MuSiQue"


def convert(r):
    paras = [(doc_id_for(p["title"], p["paragraph_text"]), p["title"], p["paragraph_text"], bool(p["is_supporting"])) for p in r["paragraphs"]]
    gold = list(dict.fromkeys(d for d, _, _, s in paras if s))
    return {"qid": r["id"], "question": r["question"], "answers": [r["answer"]] + list(r.get("answer_aliases") or []),
            "gold_doc_ids": gold, "gold_titles": list(dict.fromkeys(t for _, t, _, s in paras if s)),
            "type": r["id"].split("__")[0], "hops": int(r["id"][0]), "_paras": paras}


def main():
    dev_path, sha = hf_file(REPO, "musique_ans_v1.0_dev.jsonl")
    qs = [convert(r) for r in read_jsonl(dev_path) if r.get("answerable", True)]
    qs = [q for q in qs if len(q["gold_doc_ids"]) >= 2]
    dev, holdout, rest = split_questions(qs)
    corpus, n_eval_docs = build_pooled_corpus(dev + holdout, rest, lambda q: [(d, t, x) for d, t, x, _ in q["_paras"]])
    strip = lambda q: {k: v for k, v in q.items() if not k.startswith("_")}
    train_path, _ = hf_file(REPO, "musique_ans_v1.0_train.jsonl")
    train = []
    for r in read_jsonl(train_path):
        q = convert(r)
        train.append({"qid": q["qid"], "question": q["question"], "answers": q["answers"], "type": q["type"],
                      "positives": [{"title": t, "text": x} for _, t, x, s in q["_paras"] if s],
                      "negatives": [{"title": t, "text": x} for _, t, x, s in q["_paras"] if not s]})
    finalize("musique", corpus, [strip(q) for q in dev], [strip(q) for q in holdout],
             {"repo": REPO, "revision": sha, "eval_file": "musique_ans_v1.0_dev.jsonl", "train_files": ["musique_ans_v1.0_train.jsonl"],
              "license": "CC BY 4.0 (Trivedi et al., TACL 2022; HF mirror dgslibisey/MuSiQue)"},
             train_rows=train,
             notes=f"doc = one paragraph, id = hash(title+text) because several paragraphs share a title. {n_eval_docs} paragraphs from eval questions.")


if __name__ == "__main__":
    main()
