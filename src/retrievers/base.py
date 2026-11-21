"""Common interface for the bench retrievers.

    r.build(corpus, dataset)                    corpus: list of {"doc_id", "title", "text", ...}
    r.search_batch(queries, k) -> list[list[(doc_id, score)]]      best first
    r.last_timing   -> {"encode_ms", "search_ms", "post_ms"} of the most recent search_batch call
    r.stats         -> {"build_s", "index_bytes", "cached", ...}

Wrappers (fusion, rerank, multi-hop, chunk->document aggregation) take other retrievers and expose the same interface.
"""
from __future__ import annotations

import time


class Retriever:
    name = "retriever"

    def __init__(self):
        self.last_timing: dict = {}
        self.stats: dict = {}

    def build(self, corpus: list[dict], dataset: str) -> "Retriever":
        raise NotImplementedError

    def search_batch(self, queries: list[str], k: int) -> list[list[tuple[str, float]]]:
        raise NotImplementedError


class Timer:
    """Wall-clock stage timer that waits for the GPU before reading the clock."""

    def __init__(self, cuda: bool = False):
        self.cuda = cuda
        self.t = self._now()
        self.out: dict = {}

    def _now(self):
        if self.cuda:
            import torch
            torch.cuda.synchronize()
        return time.perf_counter()

    def lap(self, name: str):
        t = self._now()
        self.out[name] = self.out.get(name, 0.0) + (t - self.t) * 1000.0
        self.t = t


def atomic_save_npy(path, arr):
    """np.save to <path>.tmp then os.replace: a crash never leaves a half-written cache file under the real name."""
    import os
    import numpy as np
    tmp = str(path) + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def atomic_write_text(path, text):
    import os
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
