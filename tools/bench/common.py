"""Shared helpers for the bench dataset builders (tools/bench/prepare_*.py).

Output layout: bench/data/<name>/{corpus.jsonl, queries_dev.jsonl, queries_holdout.jsonl, manifest.json}
               bench/train/<name>_train.jsonl  (question / positives / negatives, from the TRAIN split only)
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import string
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "bench" / "data"
TRAIN = ROOT / "bench" / "train"
SEED = 20261118
N_DEV, N_HOLDOUT = 500, 300
CORPUS_CAP = 40000


def doc_id_for(title: str, text: str = "") -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")[:50] or "doc"
    return f"{slug}__{hashlib.sha1((title + chr(10) + text).encode('utf-8')).hexdigest()[:8]}"


def norm_q(s: str) -> str:
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def hf_file(repo: str, filename: str) -> tuple[Path, str]:
    """Download one file of a HF dataset repo; return (local path, repo revision sha)."""
    from huggingface_hub import HfApi, hf_hub_download
    sha = HfApi().dataset_info(repo).sha
    return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=sha)), sha


def read_parquet(path: Path) -> list[dict]:
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def iter_parquet(path: Path, batch_size: int = 2000):
    import pyarrow.parquet as pq
    for b in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        yield from b.to_pylist()


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            line = json.dumps(r, ensure_ascii=False) + "\n"
            f.write(line)
            h.update(line.encode("utf-8"))
    return h.hexdigest()


def split_questions(questions: list[dict], n_dev: int = N_DEV, n_holdout: int = N_HOLDOUT, seed: int = SEED):
    """De-duplicate by normalised question text, shuffle with a fixed seed, cut dev / holdout / rest."""
    seen, uniq = set(), []
    for q in questions:
        k = norm_q(q["question"])
        if k and k not in seen:
            seen.add(k)
            uniq.append(q)
    rng = random.Random(seed)
    rng.shuffle(uniq)
    n_dev = min(n_dev, max(1, len(uniq) * 5 // 8))
    n_holdout = min(n_holdout, len(uniq) - n_dev)
    return uniq[:n_dev], uniq[n_dev:n_dev + n_holdout], uniq[n_dev + n_holdout:]


def check_dataset(corpus: list[dict], dev: list[dict], holdout: list[dict]) -> dict:
    """Hard checks; raises AssertionError when a dataset is unusable."""
    ids = [d["doc_id"] for d in corpus]
    assert len(ids) == len(set(ids)), "duplicate doc_id in corpus"
    idset = set(ids)
    for name, qs in (("dev", dev), ("holdout", holdout)):
        assert qs, f"{name} is empty"
        assert len({q["qid"] for q in qs}) == len(qs), f"duplicate qid in {name}"
        for q in qs:
            assert q["gold_doc_ids"], f"{name} {q['qid']}: no gold docs"
            missing = [g for g in q["gold_doc_ids"] if g not in idset]
            assert not missing, f"{name} {q['qid']}: gold docs not in corpus: {missing[:3]}"
            for gs in q.get("gold_sets") or []:
                assert all(g in idset for g in gs), f"{name} {q['qid']}: gold_sets doc not in corpus"
    assert not ({q["qid"] for q in dev} & {q["qid"] for q in holdout}), "dev / holdout share qids"
    assert not ({norm_q(q["question"]) for q in dev} & {norm_q(q["question"]) for q in holdout}), "dev / holdout share question text"
    assert all(d["text"].strip() for d in corpus), "empty document text"
    return {"n_corpus": len(corpus), "n_dev": len(dev), "n_holdout": len(holdout), "gold_in_corpus": True,
            "dev_holdout_disjoint": True}


def filter_train(train_rows: list[dict], dev: list[dict], holdout: list[dict], cap: int = 30000, seed: int = SEED):
    """Drop train questions whose text also appears in eval; report gold-title overlap (information only)."""
    eval_q = {norm_q(q["question"]) for q in dev + holdout}
    kept = [r for r in train_rows if norm_q(r["question"]) not in eval_q]
    dropped = len(train_rows) - len(kept)
    rng = random.Random(seed)
    rng.shuffle(kept)
    kept = kept[:cap]
    eval_titles = {t for q in dev + holdout for t in q.get("gold_titles", [])}
    train_titles = {p["title"] for r in kept for p in r["positives"]}
    overlap = len(eval_titles & train_titles) / len(eval_titles) if eval_titles else None
    return kept, {"n_train": len(kept), "dropped_same_question_as_eval": dropped,
                  "eval_gold_titles": len(eval_titles),
                  "eval_gold_title_also_train_positive_ratio": None if overlap is None else round(overlap, 4)}


def finalize(name: str, corpus: list[dict], dev: list[dict], holdout: list[dict], source: dict,
             train_rows: list[dict] | None = None, extra_files: dict[str, list[dict]] | None = None, notes: str = "") -> dict:
    checks = check_dataset(corpus, dev, holdout)
    out = DATA / name
    files = {"corpus.jsonl": write_jsonl(out / "corpus.jsonl", corpus),
             "queries_dev.jsonl": write_jsonl(out / "queries_dev.jsonl", dev),
             "queries_holdout.jsonl": write_jsonl(out / "queries_holdout.jsonl", holdout)}
    for fn, rows in (extra_files or {}).items():
        files[fn] = write_jsonl(out / fn, rows)
    train_info = None
    if train_rows is not None:
        kept, train_info = filter_train(train_rows, dev, holdout)
        train_info["sha256"] = write_jsonl(TRAIN / f"{name}_train.jsonl", kept)
    from collections import Counter
    manifest = {"name": name, "seed": SEED, "source": source, "checks": checks, "sha256": files,
                "dev_types": dict(Counter(str(q.get("type")) for q in dev)),
                "dev_hops": dict(Counter(str(q.get("hops")) for q in dev)),
                "holdout_types": dict(Counter(str(q.get("type")) for q in holdout)),
                "avg_doc_chars": round(sum(len(d["text"]) for d in corpus) / len(corpus), 1),
                "train": train_info, "notes": notes}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("name", "checks", "dev_types", "dev_hops", "avg_doc_chars", "train")}, ensure_ascii=False))
    return manifest


def build_pooled_corpus(eval_qs: list[dict], other_qs: list[dict], paragraphs_of, cap: int = CORPUS_CAP, seed: int = SEED):
    """Corpus = every paragraph of the eval questions, then paragraphs of other questions (fixed-seed order) up to `cap`.
    paragraphs_of(q) -> list of (doc_id, title, text)."""
    corpus, seen = [], set()

    def add(q):
        for did, title, text in paragraphs_of(q):
            if did not in seen and text.strip():
                seen.add(did)
                corpus.append({"doc_id": did, "title": title, "text": text})
    for q in eval_qs:
        add(q)
    n_eval_docs = len(corpus)
    rest = list(other_qs)
    random.Random(seed).shuffle(rest)
    for q in rest:
        if len(corpus) >= cap:
            break
        add(q)
    return corpus, n_eval_docs
