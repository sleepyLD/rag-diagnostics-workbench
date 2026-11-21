"""Rank / score fusion of several retrievers, and chunk -> source-document aggregation."""
from __future__ import annotations

from .base import Retriever, Timer


def rrf(lists: list[list[tuple[str, float]]], k: int = 60, weights: list[float] | None = None, top: int = 100):
    """Reciprocal rank fusion. weights default to 1 for every list."""
    weights = weights or [1.0] * len(lists)
    acc: dict[str, float] = {}
    for w, lst in zip(weights, lists):
        for rank, (doc, _) in enumerate(lst, 1):
            acc[doc] = acc.get(doc, 0.0) + w / (k + rank)
    return sorted(acc.items(), key=lambda x: (-x[1], x[0]))[:top]


def score_fusion(lists: list[list[tuple[str, float]]], weights: list[float] | None = None, top: int = 100):
    """Weighted sum of min-max normalised scores (a document missing from a list contributes 0)."""
    weights = weights or [1.0] * len(lists)
    acc: dict[str, float] = {}
    for w, lst in zip(weights, lists):
        if not lst:
            continue
        hi, lo = max(s for _, s in lst), min(s for _, s in lst)
        span = (hi - lo) or 1.0
        for doc, s in lst:
            acc[doc] = acc.get(doc, 0.0) + w * (s - lo) / span
    return sorted(acc.items(), key=lambda x: (-x[1], x[0]))[:top]


def aggregate_to_source(hits: list[tuple[str, float]], source_of: dict[str, str], top: int = 100):
    """Chunk hits -> source documents, keeping each document's best chunk score (max-pooling)."""
    best: dict[str, float] = {}
    for doc, s in hits:
        src = source_of.get(doc, doc)
        if src not in best or s > best[src]:
            best[src] = s
    return sorted(best.items(), key=lambda x: (-x[1], x[0]))[:top]


class FusionRetriever(Retriever):
    def __init__(self, components: list[Retriever], name: str = "fusion", mode: str = "rrf", weights: list[float] | None = None,
                 rrf_k: int = 60, depth: int = 100, head: int = 5):
        super().__init__()
        assert mode in ("rrf", "score", "head")   # head: first `head` docs of component 0, then component 1 fills the rest
        self.head = head
        self.components, self.name, self.mode, self.weights, self.rrf_k, self.depth = components, name, mode, weights, rrf_k, depth

    def build(self, corpus: list[dict], dataset: str) -> "FusionRetriever":
        for c in self.components:
            if not getattr(c, "doc_ids", None):
                c.build(corpus, dataset)
        self.doc_ids = self.components[0].doc_ids
        self.index_key = "+".join(c.index_key for c in self.components)
        self.stats = {"build_s": sum((c.stats.get("build_s") or 0) for c in self.components),
                      "index_bytes": sum(c.stats.get("index_bytes", 0) for c in self.components),
                      "cached": all(c.stats.get("cached", False) for c in self.components), "n_docs": len(self.doc_ids)}
        return self

    def search_batch(self, queries: list[str], k: int) -> list[list[tuple[str, float]]]:
        timing = {"encode_ms": 0.0, "search_ms": 0.0, "post_ms": 0.0}
        runs = []
        for c in self.components:
            runs.append(c.search_batch(queries, max(k, self.depth)))
            for key in timing:
                timing[key] += c.last_timing.get(key, 0.0)
        tm = Timer()
        if self.mode == "head":
            out = []
            for i in range(len(queries)):
                a, b = [d for d, _ in runs[0][i]], [d for d, _ in runs[1][i]]
                seen, merged = set(), []
                for d in a[:self.head] + b + a[self.head:]:
                    if d not in seen:
                        seen.add(d); merged.append(d)
                out.append([(d, float(len(merged) - r)) for r, d in enumerate(merged[:k])])
        elif self.mode == "rrf":
            out = [rrf([r[i] for r in runs], self.rrf_k, self.weights, k) for i in range(len(queries))]
        else:
            out = [score_fusion([r[i] for r in runs], self.weights, k) for i in range(len(queries))]
        tm.lap("post_ms")
        timing["post_ms"] += tm.out["post_ms"]
        self.last_timing = timing
        return out
