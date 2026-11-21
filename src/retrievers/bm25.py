"""BM25 (Lucene-style idf) on a scipy sparse term x document matrix. Same tokeniser as src/rag_core.bm25_tokens."""
from __future__ import annotations

import re
import time
from collections import Counter

import numpy as np

from .base import Retriever, Timer


def bm25_tokens(s: str) -> list[str]:
    return [w for w in re.findall(r"\w+", s.lower()) if len(w) > 1]


class BM25Retriever(Retriever):
    def __init__(self, name: str = "bm25", k1: float = 1.5, b: float = 0.75, use_title: bool = True):
        super().__init__()
        self.name, self.k1, self.b, self.use_title = name, k1, b, use_title

    def build(self, corpus: list[dict], dataset: str) -> "BM25Retriever":
        from scipy.sparse import csr_matrix
        t0 = time.time()
        self.doc_ids = [d["doc_id"] for d in corpus]
        self.vocab: dict[str, int] = {}
        rows, cols, tfs, dl = [], [], [], np.zeros(len(corpus), dtype=np.float64)
        for j, d in enumerate(corpus):
            toks = bm25_tokens(((d.get("title") or "") + " " if self.use_title else "") + d["text"])
            dl[j] = len(toks)
            for t, c in Counter(toks).items():
                rows.append(self.vocab.setdefault(t, len(self.vocab))); cols.append(j); tfs.append(c)
        rows, cols, tfs = np.asarray(rows), np.asarray(cols), np.asarray(tfs, dtype=np.float64)
        n = len(corpus)
        df = np.bincount(rows, minlength=len(self.vocab))
        idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))
        w = idf[rows] * tfs * (self.k1 + 1.0) / (tfs + self.k1 * (1.0 - self.b + self.b * dl[cols] / max(dl.mean(), 1e-9)))
        self.m = csr_matrix((w.astype(np.float32), (rows, cols)), shape=(len(self.vocab), n))
        self.index_key = f"{dataset}__bm25"
        self.stats = {"build_s": round(time.time() - t0, 2), "cached": False, "n_docs": n,
                      "index_bytes": int(self.m.data.nbytes + self.m.indices.nbytes + self.m.indptr.nbytes)}
        return self

    def scores(self, query: str) -> np.ndarray:
        cnt = Counter(t for t in bm25_tokens(query) if t in self.vocab)
        if not cnt:
            return np.zeros(len(self.doc_ids), dtype=np.float32)
        tids = [self.vocab[t] for t in cnt]
        return np.asarray(self.m[tids].T @ np.asarray([cnt[t] for t in cnt], dtype=np.float32)).ravel()

    def search_batch(self, queries: list[str], k: int) -> list[list[tuple[str, float]]]:
        tm, out = Timer(), []
        k = min(k, len(self.doc_ids))
        for q in queries:
            s = self.scores(q)
            top = np.argpartition(-s, k - 1)[:k]
            top = top[np.argsort(-s[top], kind="stable")]
            out.append([(self.doc_ids[j], float(s[j])) for j in top])
        tm.lap("search_ms")
        self.last_timing = {"encode_ms": 0.0, **tm.out, "post_ms": 0.0}
        return out
