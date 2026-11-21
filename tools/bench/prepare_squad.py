"""SQuAD 1.1 as the single-hop control: corpus = every validation + train paragraph; eval questions from validation;
train pairs from train (negatives = other paragraphs of the same article).   python -m tools.bench.prepare_squad"""
import random
from collections import defaultdict
from tools.bench.common import SEED, doc_id_for, finalize, hf_file, read_parquet, split_questions

REPO = "rajpurkar/squad"


def main():
    vpath, sha = hf_file(REPO, "plain_text/validation-00000-of-00001.parquet")
    tpath, _ = hf_file(REPO, "plain_text/train-00000-of-00001.parquet")
    val, trn = read_parquet(vpath), read_parquet(tpath)
    corpus, seen = [], set()
    for r in val + trn:
        d = doc_id_for(r["title"], r["context"])
        if d not in seen:
            seen.add(d)
            corpus.append({"doc_id": d, "title": r["title"].replace("_", " "), "text": r["context"]})
    qs = [{"qid": r["id"], "question": r["question"], "answers": list(dict.fromkeys(r["answers"]["text"])),
           "gold_doc_ids": [doc_id_for(r["title"], r["context"])], "gold_titles": [r["title"]], "type": "single", "hops": 1} for r in val]
    dev, holdout, _ = split_questions(qs)
    by_article = defaultdict(list)
    for r in trn:
        by_article[r["title"]].append(r["context"])
    by_article = {t: list(dict.fromkeys(c)) for t, c in by_article.items()}
    rng = random.Random(SEED)
    train = []
    for r in trn:
        others = [c for c in by_article[r["title"]] if c != r["context"]]
        train.append({"qid": r["id"], "question": r["question"], "answers": r["answers"]["text"], "type": "single",
                      "positives": [{"title": r["title"].replace("_", " "), "text": r["context"]}],
                      "negatives": [{"title": r["title"].replace("_", " "), "text": c} for c in rng.sample(others, min(4, len(others)))]})
    finalize("squad", corpus, dev, holdout,
             {"repo": REPO, "revision": sha, "license": "CC BY-SA 4.0 (Rajpurkar et al., EMNLP 2016)"},
             train_rows=train,
             notes="Eval questions come from the validation articles; train articles (disjoint titles) are in the corpus only as distractors. "
                   "Train negatives = other paragraphs of the same article (may contain unlabeled positives).")


if __name__ == "__main__":
    main()
