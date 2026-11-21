"""Rebuild bench/train/bioasq_train.jsonl from EVERY BioASQ question that is not in bioasq dev/holdout
(the eval set keeps only 2-5-passage questions; the 1-passage and >5-passage ones are fine for training).
    python -m tools.bench.bioasq_train_all"""
import ast, json, random
from tools.bench.common import ROOT, SEED, TRAIN, hf_file, norm_q, read_jsonl, read_parquet, write_jsonl

REPO = "rag-datasets/rag-mini-bioasq"


def main():
    pp, _ = hf_file(REPO, "data/passages.parquet/part.0.parquet")
    tp, _ = hf_file(REPO, "data/test.parquet/part.0.parquet")
    text = {f"bioasq_{r['id']}": r["passage"] for r in read_parquet(pp)
            if (r["passage"] or "").strip().lower() not in ("", "nan", "none", "null")}
    d = ROOT / "bench" / "data" / "bioasq"
    evalq = {norm_q(q["question"]) for f in ("queries_dev.jsonl", "queries_holdout.jsonl") for q in read_jsonl(d / f)}
    rows = []
    for r in read_parquet(tp):
        if norm_q(r["question"]) in evalq:
            continue
        rel = [f"bioasq_{x}" for x in ast.literal_eval(r["relevant_passage_ids"]) if f"bioasq_{x}" in text]
        if not rel:
            continue
        rows.append({"qid": f"bioasq_t{r['id']}", "question": r["question"], "answers": [r["answer"]], "type": "train",
                     "positives": [{"title": "", "text": text[g]} for g in rel[:4]], "negatives": []})
    random.Random(SEED).shuffle(rows)
    sha = write_jsonl(TRAIN / "bioasq_train.jsonl", rows)
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    m["train"] = {"n_train": len(rows), "sha256": sha, "note": "all BioASQ questions not in dev/holdout (incl. 1-passage and >5-passage ones)"}
    (d / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(m["train"]))


if __name__ == "__main__":
    main()
