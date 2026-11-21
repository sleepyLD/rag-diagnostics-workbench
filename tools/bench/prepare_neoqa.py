"""NeoQA (fictional news timelines, cross-document). License CC BY-ND 4.0: the decrypted text must NOT be committed
(bench/data/ is git-ignored). Decryption follows the repo's own loader (neoqa.py): base64 + XOR with key % 256.

    python -m tools.bench.prepare_neoqa --key 23456        (key is published in the NeoQA README)
"""
import argparse, base64, json, re
from tools.bench.common import finalize, hf_file, split_questions

REPO = "mglockner/neoqa"
_ID = re.compile(r"\{([^|]+)\|([A-Z]+-\d+,?)+\}")


def strip_ids(text):
    for m in list(_ID.finditer(text)):
        text = text.replace(m.group(0), m.group(1))
    return text


def load(filename, key):
    path, sha = hf_file(REPO, filename)
    k, rows = key % 256, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = bytes(b ^ k for b in base64.b64decode(line))
            try:
                rows.append(json.loads(raw.decode("utf-8")))
            except UnicodeDecodeError:
                rows.append(json.loads("".join(chr(b) for b in raw)))
    return rows, sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", type=int, required=True)
    key = ap.parse_args().key
    corpus, seen, sha = [], set(), None
    for split in ("dev", "test"):
        rows, sha = load(f"data/{split}.news.jsonl", key)
        for a in rows:
            if a["article_id"] in seen:
                continue
            seen.add(a["article_id"])
            corpus.append({"doc_id": a["article_id"], "title": strip_ids(a["headline"]),
                           "text": "\n".join(strip_ids(p) for p in a["passages"]), "source_doc_id": a["article_id"],
                           "timeline_id": a["timeline_id"], "date": a.get("date")})
    qs, n_unans = [], 0
    for split in ("dev", "test"):
        rows, _ = load(f"data/{split}.questions.jsonl", key)
        for q in rows:
            gold = [g for g in (q.get("sufficient_article_ids") or [])]
            if not gold or not all(g in seen for g in gold):
                n_unans += 1
                continue
            sets = [s for s in (q.get("all_sufficient_article_id_combinations") or []) if s and all(g in seen for g in s)]
            qs.append({"qid": q["question_id"], "question": strip_ids(q["question"]), "answers": [strip_ids(q["answer"])],
                       "gold_doc_ids": gold, "gold_sets": sets or [gold], "type": q.get("category"), "hops": len(gold),
                       "timeline_id": q["timeline_id"], "date": q.get("date")})
    dev, holdout, _ = split_questions(qs)
    finalize("neoqa", corpus, dev, holdout,
             {"repo": REPO, "revision": sha, "files": ["data/{dev,test}.news.jsonl", "data/{dev,test}.questions.jsonl"],
              "license": "CC BY-ND 4.0 (Glockner et al., ACL 2025 Findings) - decrypted text must not be redistributed"},
             notes=f"doc = one whole fictional news article. gold_doc_ids = sufficient_article_ids; gold_sets = every sufficient combination "
                   f"(all_gold_hit accepts any of them). {n_unans} questions without a sufficient article set were left out. No train split used.")


if __name__ == "__main__":
    main()
