"""Phase B retrievers: cross-encoder rerank, late interaction (ColBERT, exact MaxSim on GPU),
learned sparse (SPLADE-style SparseEncoder) and a training-free two-hop beam retriever.
All share the Retriever interface (build / search_batch / last_timing / stats)."""
from __future__ import annotations

import re
import time

import numpy as np

from .base import Retriever, Timer, atomic_save_npy, atomic_write_text
from .dense import doc_text


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# --------------------------------------------------------------------------- rerank
class RerankRetriever(Retriever):
    """first-stage top `depth` -> cross-encoder scores -> re-sorted. Documents outside the depth keep their order below."""

    def __init__(self, first: Retriever, model: str, name: str = "rerank", depth: int = 100, batch_size: int = 64,
                 max_length: int = 512, fp16: bool = True, use_title: bool = True, model_kwargs: dict | None = None, scorer=None):
        super().__init__()
        self.first, self.model_name, self.name, self.depth = first, model, name, depth
        self.batch_size, self.max_length, self.fp16, self.use_title = batch_size, max_length, fp16, use_title
        self.model_kwargs, self.scorer = model_kwargs or {}, scorer

    def build(self, corpus, dataset):
        if not getattr(self.first, "doc_ids", None):
            self.first.build(corpus, dataset)
        self.text = {d["doc_id"]: doc_text(d, self.use_title) for d in corpus}
        self.doc_ids = self.first.doc_ids
        if self.scorer is None:
            from sentence_transformers import CrossEncoder
            self.ce = CrossEncoder(self.model_name, max_length=self.max_length, device="cuda" if _cuda() else "cpu", **self.model_kwargs)
            if self.fp16 and _cuda():
                self.ce.model.half()
            self.scorer = lambda pairs: np.asarray(self.ce.predict(pairs, batch_size=self.batch_size, show_progress_bar=False), dtype=np.float32)
        self.index_key = f"{self.first.index_key}+rerank:{self.model_name}@{self.depth}"
        self.stats = {**self.first.stats, "rerank_model": self.model_name, "depth": self.depth}
        return self

    def search_batch(self, queries, k):
        first = self.first.search_batch(queries, max(k, self.depth))
        timing = dict(self.first.last_timing)
        tm = Timer(_cuda())
        pairs, spans = [], []
        for q, hits in zip(queries, first):
            top = hits[:self.depth]
            spans.append((len(pairs), len(top)))
            pairs.extend((q, self.text[d]) for d, _ in top)
        scores = self.scorer(pairs) if pairs else np.zeros(0, dtype=np.float32)
        out = []
        for (s0, n), hits in zip(spans, first):
            top = sorted(zip([d for d, _ in hits[:n]], scores[s0:s0 + n].tolist()), key=lambda x: -x[1])
            floor = (top[-1][1] if top else 0.0) - 1.0
            rest = [(d, floor - i * 1e-6) for i, (d, _) in enumerate(hits[n:])]
            out.append((top + rest)[:k])
        tm.lap("rerank_ms")
        timing["rerank_ms"] = tm.out["rerank_ms"]
        timing["post_ms"] = timing.get("post_ms", 0.0) + tm.out["rerank_ms"]   # keep the 3-stage sum == total
        self.last_timing = timing
        return out


# --------------------------------------------------------------------------- ColBERT
class ColBERTRetriever(Retriever):
    """Exact late interaction: all document token vectors in one fp16 GPU matrix; per query
    MaxSim = segment-max over each document's tokens, summed over query tokens. Encoder: PyLate."""

    def __init__(self, model: str, name: str = "colbert", index_root: str = "index/bench", batch_size: int = 64,
                 doc_length: int = 300, query_length: int = 32, chunk_tokens: int = 2_000_000, use_title: bool = True):
        super().__init__()
        self.model_name, self.name, self.index_root = model, name, index_root
        self.batch_size, self.doc_length, self.query_length, self.chunk, self.use_title = batch_size, doc_length, query_length, chunk_tokens, use_title

    def build(self, corpus, dataset):
        import hashlib, json, re, torch
        from pathlib import Path
        import sentence_transformers.sentence_transformer.model_card as _mc
        if not hasattr(_mc, "generate_model_card"):           # pylate 1.0 vs sentence-transformers 6.x (model cards only)
            _mc.generate_model_card = lambda *a, **k: ""
        from pylate import models
        dev = "cuda" if _cuda() else "cpu"
        self.model = models.ColBERT(model_name_or_path=self.model_name, device=dev, document_length=self.doc_length, query_length=self.query_length)
        if dev == "cuda":
            self.model.half()
        self.doc_ids = [d["doc_id"] for d in corpus]
        texts = [doc_text(d, self.use_title) for d in corpus]
        h = hashlib.sha1((self.model_name + str(self.doc_length)).encode())
        for t in texts:
            h.update(t.encode("utf-8")); h.update(b"\0")
        self.index_key = f"{dataset}__colbert-{re.sub(r'[^A-Za-z0-9.]+', '-', self.model_name)}__{h.hexdigest()[:12]}"
        path = Path(self.index_root) / self.index_key
        t0 = time.time()
        if (path / "meta.json").exists() and (path / "tokens.npy").exists():     # meta written last = complete cache
            tok, off, cached = np.load(path / "tokens.npy"), np.load(path / "offsets.npy"), True
            build_s = json.loads((path / "meta.json").read_text())["build_s"]
        else:
            embs = self.model.encode(texts, batch_size=self.batch_size, is_query=False, show_progress_bar=True)
            lens = np.asarray([len(e) for e in embs])
            off = np.concatenate([[0], np.cumsum(lens)])
            tok = np.concatenate([np.asarray(e, dtype=np.float16) for e in embs])
            build_s, cached = round(time.time() - t0, 2), False
            path.mkdir(parents=True, exist_ok=True)
            atomic_save_npy(path / "tokens.npy", tok); atomic_save_npy(path / "offsets.npy", off)
            atomic_write_text(path / "meta.json", json.dumps({"build_s": build_s, "n_tokens": int(len(tok)), "dim": int(tok.shape[1])}))
        self.tok = torch.from_numpy(tok).to(dev)
        self.seg = torch.repeat_interleave(torch.arange(len(off) - 1, device=dev), torch.from_numpy(np.diff(off)).to(dev))
        self.cuda = dev == "cuda"
        self.stats = {"build_s": build_s, "cached": cached, "index_bytes": int(tok.nbytes), "n_tokens": int(len(tok)), "n_docs": len(corpus)}
        return self

    def search_batch(self, queries, k):
        import torch
        tm, out = Timer(self.cuda), []
        qs = self.model.encode(queries, batch_size=self.batch_size, is_query=True, show_progress_bar=False)
        tm.lap("encode_ms")
        n = len(self.doc_ids)
        for qe in qs:
            q = torch.as_tensor(np.asarray(qe), dtype=self.tok.dtype, device=self.tok.device)
            best = torch.full((n, q.shape[0]), -1e4, dtype=torch.float32, device=self.tok.device)
            for s in range(0, len(self.tok), self.chunk):          # chunked so the tokens x qlen matrix stays small
                sim = (self.tok[s:s + self.chunk] @ q.T).float()
                seg = self.seg[s:s + self.chunk]
                best.scatter_reduce_(0, seg[:, None].expand_as(sim), sim, reduce="amax", include_self=True)
            score = best.sum(1)
            v, i = torch.topk(score, min(k, n))
            out.append([(self.doc_ids[j], float(x)) for j, x in zip(i.tolist(), v.tolist())])
        tm.lap("search_ms")
        self.last_timing = {**tm.out, "post_ms": 0.0}
        return out


# --------------------------------------------------------------------------- SPLADE
class SparseRetriever(Retriever):
    """Learned sparse retrieval with sentence-transformers SparseEncoder (SPLADE family); scipy CSR dot products."""

    def __init__(self, model: str, name: str = "splade", index_root: str = "index/bench", batch_size: int = 32,
                 max_seq_length: int = 512, use_title: bool = True):
        super().__init__()
        self.model_name, self.name, self.index_root = model, name, index_root
        self.batch_size, self.max_seq_length, self.use_title = batch_size, max_seq_length, use_title

    @staticmethod
    def _to_csr(t, dim=None):
        from scipy.sparse import csr_matrix
        t = t.coalesce().cpu()
        idx, val = t.indices().numpy(), t.values().float().numpy()
        return csr_matrix((val, (idx[0], idx[1])), shape=(t.shape[0], dim or t.shape[1]))

    def build(self, corpus, dataset):
        import hashlib, re
        from pathlib import Path
        from scipy.sparse import load_npz, save_npz, vstack
        from sentence_transformers import SparseEncoder
        self.model = SparseEncoder(self.model_name, device="cuda" if _cuda() else "cpu")
        self.model.max_seq_length = min(self.max_seq_length, self.model.max_seq_length or self.max_seq_length)
        self.doc_ids = [d["doc_id"] for d in corpus]
        texts = [doc_text(d, self.use_title) for d in corpus]
        h = hashlib.sha1(self.model_name.encode())
        for t in texts:
            h.update(t.encode("utf-8")); h.update(b"\0")
        self.index_key = f"{dataset}__sparse-{re.sub(r'[^A-Za-z0-9.]+', '-', self.model_name)}__{h.hexdigest()[:12]}"
        path = Path(self.index_root) / self.index_key
        t0 = time.time()
        if (path / "docs.npz").exists() and (path / "ok").exists():          # "ok" marker written after the npz
            self.m, cached = load_npz(path / "docs.npz").tocsr(), True
        else:
            dim = None          # vocabulary size, taken from the encoded tensor
            parts = []
            for i in range(0, len(texts), 2048):
                e = self.model.encode_document(texts[i:i + 2048], batch_size=self.batch_size, convert_to_sparse_tensor=True, show_progress_bar=True)
                parts.append(self._to_csr(e, dim))
            self.m, cached = vstack(parts).tocsr(), False
            path.mkdir(parents=True, exist_ok=True)
            import os
            save_npz(path / "docs.tmp.npz", self.m)
            os.replace(path / "docs.tmp.npz", path / "docs.npz")
            atomic_write_text(path / "ok", "1")
        self.mT = self.m.T.tocsr()
        self.dim = self.m.shape[1]
        self.stats = {"build_s": round(time.time() - t0, 2), "cached": cached, "n_docs": len(corpus),
                      "index_bytes": int(self.m.data.nbytes + self.m.indices.nbytes + self.m.indptr.nbytes),
                      "avg_doc_nnz": round(self.m.nnz / max(1, self.m.shape[0]), 1)}
        return self

    def search_batch(self, queries, k):
        tm = Timer(_cuda())
        q = self._to_csr(self.model.encode_query(queries, convert_to_sparse_tensor=True, show_progress_bar=False), self.dim)
        tm.lap("encode_ms")
        s = (q @ self.mT).toarray()
        kk = min(k, len(self.doc_ids))
        out = []
        for row in s:
            top = np.argpartition(-row, kk - 1)[:kk]
            top = top[np.argsort(-row[top], kind="stable")]
            out.append([(self.doc_ids[j], float(row[j])) for j in top])
        tm.lap("search_ms")
        self.last_timing = {**tm.out, "post_ms": 0.0}
        return out


# --------------------------------------------------------------------------- two-hop beam
class TwoHopRetriever(Retriever):
    """Training-free MDR-style beam search. hop 1: top `beam` docs for q. hop 2: for each beam doc d,
    search with  q + [SEP] + title(d) + text(d)[:ctx_chars]  and score the pair (d, c) = s1(d) + w2 * s2(c|d).
    Output order: documents in order of their best pair; documents never paired follow by hop-1 order."""

    def __init__(self, base: Retriever, name: str = "twohop", beam: int = 5, per_beam: int = 20, ctx_chars: int = 400, w2: float = 1.0,
                 sep: str = " ; ", use_title: bool = True):
        super().__init__()
        self.base, self.name, self.beam, self.per_beam, self.ctx, self.w2, self.sep, self.use_title = base, name, beam, per_beam, ctx_chars, w2, sep, use_title

    def build(self, corpus, dataset):
        if not getattr(self.base, "doc_ids", None):
            self.base.build(corpus, dataset)
        self.doc = {d["doc_id"]: d for d in corpus}
        self.doc_ids = self.base.doc_ids
        self.index_key = f"{self.base.index_key}+twohop(b{self.beam},p{self.per_beam},c{self.ctx},w{self.w2})"
        self.stats = {**self.base.stats}
        return self

    def _expand(self, q, d):
        doc = self.doc[d]
        return q + self.sep + ((doc.get("title") or "") + ": " if self.use_title else "") + doc["text"][:self.ctx]

    def search_batch(self, queries, k):
        timing = {"encode_ms": 0.0, "search_ms": 0.0, "post_ms": 0.0}
        hop1 = self.base.search_batch(queries, max(k, 100))
        for key in timing:
            timing[key] += self.base.last_timing.get(key, 0.0)
        exp, owner = [], []
        for qi, (q, hits) in enumerate(zip(queries, hop1)):
            for d, s in hits[:self.beam]:
                exp.append(self._expand(q, d)); owner.append((qi, d, s))
        hop2 = self.base.search_batch(exp, self.per_beam + 1) if exp else []
        for key in timing:
            timing[key] += self.base.last_timing.get(key, 0.0)
        tm = Timer()
        pairs = [[] for _ in queries]
        for (qi, d, s1), cands in zip(owner, hop2):
            for c, s2 in cands:
                if c != d:
                    pairs[qi].append((s1 + self.w2 * s2, d, c))
        out = []
        for qi, hits in enumerate(hop1):
            seen, ranked = set(), []
            n = len(pairs[qi]) + len(hits) + 2
            for sc, d, c in sorted(pairs[qi], key=lambda x: -x[0]):
                for x in (d, c):
                    if x not in seen:
                        seen.add(x); ranked.append(x)
            for d, _ in hits:
                if d not in seen:
                    seen.add(d); ranked.append(d)
            out.append([(d, float(n - i)) for i, d in enumerate(ranked[:k])])
        tm.lap("post_ms")
        timing["post_ms"] += tm.out["post_ms"]
        self.last_timing = timing
        return out


# --------------------------------------------------------------------------- chain rerank (Beam Retrieval style)
class ChainRerankRetriever(Retriever):
    """Beam-Retrieval-inspired (Zhang et al., NAACL 2024) two-hop reranking over a first-stage candidate pool.

    hop 1:  s1(c)   = CE(q, c)                     for c in the top `pool` first-stage candidates
    hop 2:  s2(c|b) = CE(q + sep + b, c)           for the `beam` best hop-1 docs b and every other candidate c
    chain:  S(b, c) = log σ(s1(b)) + log σ(s2(c|b))
    Output: documents in order of the best chain they belong to; then the rest by hop-1 score.
    Differences from the paper: one shared cross-encoder for both hops (context or no context), candidates come
    from an open-corpus first stage instead of the fixed distractor set, two hops at inference."""

    def __init__(self, first: Retriever, model: str, name: str = "chain", pool: int = 30, beam: int = 4, ctx_chars: int = 600,
                 sep: str = " [SEP] ", batch_size: int = 128, max_length: int = 384, fp16: bool = True, scorer=None,
                 expand: Retriever | None = None, expand_k: int = 0, hops: int = 2, beam3: int = 3):
        """expand / expand_k: for every hop-1 beam doc b, also search the corpus with  q ; b  and add the top expand_k
        results to b's hop-2 candidates (open-corpus beam retrieval). hops=3 adds a third hop on the beam3 best chains."""
        super().__init__()
        self.first, self.model_name, self.name, self.pool, self.beam = first, model, name, pool, beam
        self.expand, self.expand_k, self.hops, self.beam3 = expand, expand_k, hops, beam3
        self.ctx, self.sep, self.batch_size, self.max_length, self.fp16, self.scorer = ctx_chars, sep, batch_size, max_length, fp16, scorer

    def build(self, corpus, dataset):
        if not getattr(self.first, "doc_ids", None):
            self.first.build(corpus, dataset)
        if self.expand is not None and not getattr(self.expand, "doc_ids", None):
            self.expand.build(corpus, dataset)
        self.text = {d["doc_id"]: doc_text(d) for d in corpus}
        self.doc = {d["doc_id"]: d for d in corpus}
        self.doc_ids = self.first.doc_ids
        if self.scorer is None:
            from sentence_transformers import CrossEncoder
            ce = CrossEncoder(self.model_name, max_length=self.max_length, device="cuda" if _cuda() else "cpu")
            if self.fp16 and _cuda():
                ce.model.half()
            import torch
            self.scorer = lambda pairs: np.asarray(ce.predict(pairs, batch_size=self.batch_size, show_progress_bar=False,
                                                              activation_fn=torch.nn.Identity()), dtype=np.float32)   # raw logits
        self.index_key = (f"{self.first.index_key}+chain:{self.model_name}(p{self.pool},b{self.beam},h{self.hops},x{self.expand_k})"
                          + (f"+exp:{self.expand.index_key}" if self.expand is not None else ""))
        self.stats = {**self.first.stats, "chain_model": self.model_name, "pool": self.pool, "beam": self.beam}
        return self

    @staticmethod
    def _logsig(x):
        return -np.logaddexp(0.0, -x)

    def _ctx(self, q, path):
        return q + "".join(self.sep + (self.doc[d].get("title") or "") + ": " + self.doc[d]["text"][:self.ctx] for d in path)

    def search_batch(self, queries, k):
        first = self.first.search_batch(queries, max(k, self.pool))
        timing = dict(self.first.last_timing)
        tm = Timer(_cuda())
        cands = [[d for d, _ in h[:self.pool]] for h in first]
        # hop 1
        p1 = [(q, self.text[c]) for q, cs in zip(queries, cands) for c in cs]
        s1_all = self.scorer(p1) if p1 else np.zeros(0, dtype=np.float32)
        s1, pos = [], 0
        for cs in cands:
            s1.append(dict(zip(cs, (self._logsig(x) for x in s1_all[pos:pos + len(cs)])))); pos += len(cs)
        beams = [[(d,) for d in sorted(s1[qi], key=lambda d: -s1[qi][d])[:self.beam]] for qi in range(len(queries))]
        chains = [[((d,), s1[qi][d]) for d in s1[qi]] for qi in range(len(queries))]
        prev = [{(d,): s1[qi][d] for (d,) in beams[qi]} for qi in range(len(queries))]
        for hop in range(2, self.hops + 1):
            extra = {}
            if self.expand is not None and self.expand_k:
                eq, eo = [], []
                for qi, (q, bs) in enumerate(zip(queries, beams)):
                    for path in bs:
                        eq.append(self._ctx(q, path)); eo.append((qi, path))
                if eq:
                    res = self.expand.search_batch(eq, self.expand_k + len(eo[0][1]))
                    for key in ("encode_ms", "search_ms"):
                        timing[key] = timing.get(key, 0.0) + self.expand.last_timing.get(key, 0.0)
                    for (qi, path), r in zip(eo, res):
                        extra[(qi, path)] = [d for d, _ in r if d not in path][:self.expand_k]
            pairs, owner = [], []
            for qi, (q, bs) in enumerate(zip(queries, beams)):
                for path in bs:
                    ctx = self._ctx(q, path)
                    pool = list(dict.fromkeys(cands[qi] + extra.get((qi, path), [])))
                    for c in pool:
                        if c not in path:
                            pairs.append((ctx, self.text[c])); owner.append((qi, path, c))
            sc = self.scorer(pairs) if pairs else np.zeros(0, dtype=np.float32)
            new = [dict() for _ in queries]
            for (qi, path, c), x in zip(owner, sc):
                key = path + (c,)
                val = prev[qi][path] + float(self._logsig(x))
                if frozenset(key) not in {frozenset(p) for p in new[qi]} or val > new[qi].get(key, -1e9):
                    new[qi][key] = val
            for qi in range(len(queries)):
                chains[qi] = sorted(new[qi].items(), key=lambda x: -x[1]) + chains[qi]
                width = self.beam3 if hop >= 2 else self.beam
                beams[qi] = [p for p, _ in sorted(new[qi].items(), key=lambda x: -x[1])[:width]]
                prev[qi] = dict(new[qi])
        out = []
        for qi, hits in enumerate(first):
            seen, ranked = set(), []
            # deepest chains first (they were prepended), each block sorted by score
            for path, _ in chains[qi]:
                for d in path:
                    if d not in seen:
                        seen.add(d); ranked.append(d)
            ranked += [d for d, _ in hits if d not in seen]
            n = len(ranked)
            out.append([(d, float(n - i)) for i, d in enumerate(ranked[:k])])
        tm.lap("rerank_ms")
        timing["rerank_ms"] = tm.out["rerank_ms"]
        timing["post_ms"] = timing.get("post_ms", 0.0) + tm.out["rerank_ms"]
        self.last_timing = timing
        return out


# --------------------------------------------------------------------------- n-hop beam (MDR generalised)
class MultiHopBeamRetriever(Retriever):
    """MDR-style beam search for `hops` hops (Xiong et al., 2021), training-free or with a fine-tuned encoder.
    A path is a list of docs; its query is  q + sep + doc_1 + sep + doc_2 ...  (each doc = title: text[:ctx_chars]).
    path score = sum of the per-hop retrieval scores; `beam` best paths survive each hop.
    Output: docs in order of the best path they appear in (deeper paths first), then hop-1 order."""

    def __init__(self, base: Retriever, name: str = "mhbeam", hops: int = 3, beam: int = 5, per_beam: int = 10, ctx_chars: int = 300,
                 sep: str = " ; ", path_rerank: str | None = None, path_ctx: int = 700, path_batch: int = 32):
        super().__init__()
        self.base, self.name, self.hops, self.beam, self.per_beam, self.ctx, self.sep = base, name, hops, beam, per_beam, ctx_chars, sep
        # path_rerank: a cross-encoder scores (question, doc_1 + doc_2 + ...) for every beam path, so a 2nd-hop doc that
        # shares no words with the question is judged together with the bridge doc (Beam Retrieval idea, zero-shot CE)
        self.path_rerank, self.path_ctx, self.path_batch, self.ce = path_rerank, path_ctx, path_batch, None

    def build(self, corpus, dataset):
        if not getattr(self.base, "doc_ids", None):
            self.base.build(corpus, dataset)
        self.doc = {d["doc_id"]: d for d in corpus}
        self.doc_ids = self.base.doc_ids
        self.index_key = f"{self.base.index_key}+mhbeam(h{self.hops},b{self.beam},p{self.per_beam},c{self.ctx})" + (
            f"+pathrr:{self.path_rerank}@{self.path_ctx}" if self.path_rerank else "")
        self.stats = {**self.base.stats}
        if self.path_rerank and self.ce is None:
            import torch
            from sentence_transformers import CrossEncoder
            self.ce = CrossEncoder(self.path_rerank, max_length=512, device="cuda" if _cuda() else "cpu", activation_fn=torch.nn.Identity())
            if _cuda():
                self.ce.model.half()
        return self

    def _q(self, q, path):
        return q + "".join(self.sep + (self.doc[d].get("title") or "") + ": " + self.doc[d]["text"][:self.ctx] for d in path)

    def search_batch(self, queries, k):
        timing = {"encode_ms": 0.0, "search_ms": 0.0, "post_ms": 0.0}

        def run(qs, kk):
            res = self.base.search_batch(qs, kk)
            for key in timing:
                timing[key] += self.base.last_timing.get(key, 0.0)
            return res
        hop1 = run(queries, max(k, 100))
        beams = [[((d,), s) for d, s in h[:self.beam]] for h in hop1]
        finished = [list(b) for b in beams]
        for _ in range(self.hops - 1):
            exp, owner = [], []
            for qi, (q, bs) in enumerate(zip(queries, beams)):
                for path, sc in bs:
                    exp.append(self._q(q, path)); owner.append((qi, path, sc))
            if not exp:
                break
            res = run(exp, self.per_beam + len(owner[0][1]))
            nxt = [[] for _ in queries]
            for (qi, path, sc), cands in zip(owner, res):
                for c, s in cands:
                    if c not in path:
                        nxt[qi].append((path + (c,), sc + s))
            beams = []
            for qi in range(len(queries)):
                seen, keep = set(), []
                for p, s in sorted(nxt[qi], key=lambda x: -x[1]):
                    key = frozenset(p)
                    if key not in seen:
                        seen.add(key); keep.append((p, s))
                    if len(keep) == self.beam:
                        break
                beams.append(keep)
                finished[qi] = keep + finished[qi]          # deeper paths first
        tm = Timer(_cuda())
        if self.ce is not None:
            pairs, own = [], []
            for qi, q in enumerate(queries):
                for pi, (path, _) in enumerate(finished[qi]):
                    txt = " || ".join((self.doc[d].get("title") or "") + ": " + self.doc[d]["text"][:self.path_ctx] for d in path)
                    pairs.append((q, txt)); own.append((qi, pi))
            sc = self.ce.predict(pairs, batch_size=self.path_batch, show_progress_bar=False) if pairs else []
            rescored = [[] for _ in queries]
            for (qi, pi), v in zip(own, sc):
                rescored[qi].append((finished[qi][pi][0], float(v)))
            finished = [sorted(r, key=lambda x: -x[1]) for r in rescored]
        out = []
        for qi, h in enumerate(hop1):
            seen, ranked = set(), []
            for path, _ in finished[qi]:
                for d in path:
                    if d not in seen:
                        seen.add(d); ranked.append(d)
            for d, _ in h:
                if d not in seen:
                    seen.add(d); ranked.append(d)
            n = len(ranked)
            out.append([(d, float(n - i)) for i, d in enumerate(ranked[:k])])
        tm.lap("post_ms")
        timing["post_ms"] += tm.out["post_ms"]
        self.last_timing = timing
        return out


# --------------------------------------------------------------------------- Qwen3-Reranker scorer
class Qwen3RerankScorer:
    """Qwen3-Reranker (Qwen team, 2025; arXiv:2506.05176) as in its model card: a causal LM judges "yes"/"no";
    score = log P(yes) - log P(no) at the last position. Plugs into RerankRetriever(scorer=...)."""

    def __init__(self, model: str, instruction: str, max_length: int = 1024, batch_size: int = 8):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16).to("cuda").eval()
        self.yes, self.no = self.tok.convert_tokens_to_ids("yes"), self.tok.convert_tokens_to_ids("no")
        self.prefix = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
                       "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.pre_ids = self.tok.encode(self.prefix, add_special_tokens=False)
        self.suf_ids = self.tok.encode(self.suffix, add_special_tokens=False)
        self.instruction, self.max_length, self.batch_size = instruction, max_length, batch_size

    def __call__(self, pairs):
        import torch
        out = []
        body_max = self.max_length - len(self.pre_ids) - len(self.suf_ids)
        for i in range(0, len(pairs), self.batch_size):
            texts = [f"<Instruct>: {self.instruction}\n<Query>: {q}\n<Document>: {d}" for q, d in pairs[i:i + self.batch_size]]
            enc = self.tok(texts, padding=False, truncation=True, max_length=body_max, add_special_tokens=False)
            ids = [self.pre_ids + x + self.suf_ids for x in enc["input_ids"]]
            batch = self.tok.pad({"input_ids": ids}, padding=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                logits = self.model(**batch).logits[:, -1, :]
            lp = torch.log_softmax(torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1).float(), dim=1)
            out.extend((lp[:, 1] - lp[:, 0]).tolist())
        return np.asarray(out, dtype=np.float32)


# --------------------------------------------------------------------------- LLM decomposition (Self-Ask / IRCoT style)
DECOMP_SYS = """You break a complex question into simple single-hop sub-questions, one at a time.
You see the question and the facts found so far. Write the NEXT sub-question needed, written so it can be searched on its own:
replace references like "the director" or "that city" with the names already found in the facts.
If the facts already answer the question, write DONE.
Reply with exactly one line:  Sub-question: <text>   or   DONE"""
ANSWER_SYS = """Answer the sub-question in a few words using only the passages. If they do not contain the answer, reply UNKNOWN.
Reply with exactly one line:  Answer: <short answer>"""


class DecomposeRetriever(Retriever):
    """Self-Ask (Press et al., 2023) / IRCoT (Trivedi et al., ACL 2023) style iterative retrieval with a local LLM (Ollama).
    step t: LLM writes sub-question s_t from (question, facts so far) -> single-hop retriever -> LLM extracts a short answer a_t.
    Output: the per-step top docs interleaved (rank 1 of every step first, then rank 2 ...), then the `fallback` list."""

    def __init__(self, base: Retriever, llm: str = "qwen3:4b-instruct", name: str = "decomp", max_steps: int = 4, per_step: int = 3,
                 answer_passages: int = 3, fallback: Retriever | None = None, url: str = "http://localhost:11434", think=None):
        super().__init__()
        self.base, self.llm, self.name, self.max_steps, self.per_step, self.answer_passages = base, llm, name, max_steps, per_step, answer_passages
        self.fallback, self.url, self.think = fallback, url, think

    def build(self, corpus, dataset):
        from src.rag_core import OllamaGenerator
        for r in (self.base, self.fallback):
            if r is not None and not getattr(r, "doc_ids", None):
                r.build(corpus, dataset)
        self.doc = {d["doc_id"]: d for d in corpus}
        self.doc_ids = self.base.doc_ids
        self.gen = OllamaGenerator(self.llm, url=self.url, num_ctx=4096, think=self.think)
        self.gen.check()
        self.index_key = f"{self.base.index_key}+decomp:{self.llm}(s{self.max_steps},p{self.per_step})" + (
            f"+fb:{self.fallback.index_key}" if self.fallback is not None else "")
        self.stats = {**self.base.stats, "llm": self.llm}
        self.trace = []
        return self

    @staticmethod
    def _line(text, key):
        import re
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        m = re.search(rf"{key}\s*:\s*(.+)", text, flags=re.I)
        return m.group(1).strip() if m else None

    def _one(self, q, timing):
        facts, steps = [], []
        llm_ms = 0.0
        for _ in range(self.max_steps):
            fact_txt = "\n".join(f"- {s} -> {a}" for s, a in facts) or "(none yet)"
            r = self.gen.chat(DECOMP_SYS, f"Question: {q}\nFacts found so far:\n{fact_txt}")
            llm_ms += r["latency_s"] * 1000
            if "DONE" in r["text"].upper().split("SUB-QUESTION")[0] and not self._line(r["text"], "Sub-question"):
                break
            sub = self._line(r["text"], "Sub-question") or (q if not steps else None)
            if not sub or any(sub.lower() == s.lower() for s, _ in facts):
                break
            hits = self.base.search_batch([sub], max(self.per_step, self.answer_passages))[0]
            for key in ("encode_ms", "search_ms"):
                timing[key] += self.base.last_timing.get(key, 0.0)
            steps.append([d for d, _ in hits[:self.per_step]])
            ctx = "\n\n".join(f"[{d}] {(self.doc[d].get('title') or '')}: {self.doc[d]['text'][:1200]}" for d, _ in hits[:self.answer_passages])
            a = self.gen.chat(ANSWER_SYS, f"Passages:\n{ctx}\n\nSub-question: {sub}")
            llm_ms += a["latency_s"] * 1000
            facts.append((sub, self._line(a["text"], "Answer") or "UNKNOWN"))
        timing["post_ms"] += llm_ms
        return steps, facts

    def search_batch(self, queries, k):
        timing = {"encode_ms": 0.0, "search_ms": 0.0, "post_ms": 0.0}
        fb = self.fallback.search_batch(queries, max(k, 100)) if self.fallback is not None else [[] for _ in queries]
        if self.fallback is not None:
            for key in timing:
                timing[key] += self.fallback.last_timing.get(key, 0.0)
        out = []
        for q, fbl in zip(queries, fb):
            steps, facts = self._one(q, timing)
            self.trace.append({"q": q, "style": "selfask", "steps": steps, "facts": facts})
            seen, ranked = set(), []
            for rank in range(self.per_step):
                for st in steps:
                    if rank < len(st) and st[rank] not in seen:
                        seen.add(st[rank]); ranked.append(st[rank])
            ranked += [d for d, _ in fbl if d not in seen]
            n = len(ranked)
            out.append([(d, float(n - i)) for i, d in enumerate(ranked[:k])])
        self.last_timing = timing
        return out


# --------------------------------------------------------------------------- iterative evidence retrieval: IRCoT / Iter-RetGen
IRCOT_SYS = """You answer a complex question step by step, using the retrieved passages.
Write ONLY the next single sentence of your reasoning (one new fact, naming the entities involved).
If you can already answer, write a final sentence that starts with "So the answer is"."""
ITER_SYS = """Answer the question using the passages. First reason briefly (2-4 short sentences naming the entities involved),
then end with a line "Answer: <short answer>"."""


class IterativeRetriever(Retriever):
    """Three evidence-by-evidence retrieval strategies under one roof (same single-hop `base`, same LLM, same step budget):
      ircot      (Trivedi et al., ACL 2023):  retrieve with q; then each new CoT sentence is the next query.
      iterretgen (Shao et al., Findings EMNLP 2023): retrieve with q; generate a full answer draft; next query = q + draft.
      selfask    (Press et al., 2023): see DecomposeRetriever (sub-question + intermediate answer); here for completeness.
    Output: step-0 docs (query = q) and every later step's docs interleaved by rank, then `fallback` (if any)."""

    def __init__(self, base: Retriever, style: str = "ircot", llm: str = "qwen3:4b-instruct", name: str = "iter", max_steps: int = 4,
                 per_step: int = 3, ctx_docs: int = 8, fallback: Retriever | None = None, url: str = "http://localhost:11434"):
        super().__init__()
        assert style in ("ircot", "iterretgen")
        self.base, self.style, self.llm, self.name, self.max_steps, self.per_step = base, style, llm, name, max_steps, per_step
        self.ctx_docs, self.fallback, self.url = ctx_docs, fallback, url

    def build(self, corpus, dataset):
        from src.rag_core import OllamaGenerator
        for r in (self.base, self.fallback):
            if r is not None and not getattr(r, "doc_ids", None):
                r.build(corpus, dataset)
        self.doc = {d["doc_id"]: d for d in corpus}
        self.doc_ids = self.base.doc_ids
        self.gen = OllamaGenerator(self.llm, url=self.url, num_ctx=8192)
        self.gen.check()
        self.index_key = f"{self.base.index_key}+{self.style}:{self.llm}(s{self.max_steps},p{self.per_step})"
        self.stats = {**self.base.stats, "llm": self.llm}
        self.trace = []
        return self

    def _ret(self, query, timing, seen):
        hits = self.base.search_batch([query], self.per_step + len(seen))[0]
        for key in ("encode_ms", "search_ms"):
            timing[key] += self.base.last_timing.get(key, 0.0)
        return [d for d, _ in hits if d not in seen][:self.per_step]

    def _ctx(self, docs):
        return "\n\n".join(f"[{i + 1}] {(self.doc[d].get('title') or '')}: {self.doc[d]['text'][:900]}" for i, d in enumerate(docs[:self.ctx_docs]))

    def _one(self, q, timing):
        import re
        seen, steps, log, llm_ms = set(), [], [], 0.0
        first = self._ret(q, timing, seen); steps.append(first); seen.update(first)
        cot = []
        for _ in range(self.max_steps - 1):
            docs = [d for st in steps for d in st]
            if self.style == "ircot":
                r = self.gen.chat(IRCOT_SYS, f"Passages:\n{self._ctx(docs)}\n\nQuestion: {q}\nReasoning so far: {' '.join(cot) or '(none)'}\nNext sentence:")
                sent = re.sub(r"<think>.*?</think>", "", r["text"], flags=re.S).strip().split("\n")[0][:300]
                llm_ms += r["latency_s"] * 1000
                if not sent:
                    break
                cot.append(sent); log.append(sent)
                if sent.lower().startswith("so the answer is"):
                    break
                query = sent
            else:
                r = self.gen.chat(ITER_SYS, f"Passages:\n{self._ctx(docs)}\n\nQuestion: {q}")
                draft = re.sub(r"<think>.*?</think>", "", r["text"], flags=re.S).strip()[:600]
                llm_ms += r["latency_s"] * 1000
                log.append(draft)
                query = q + " " + draft.replace("\n", " ")
            new = self._ret(query, timing, seen)
            if not new:
                break
            steps.append(new); seen.update(new)
        timing["post_ms"] += llm_ms
        return steps, log

    def search_batch(self, queries, k):
        timing = {"encode_ms": 0.0, "search_ms": 0.0, "post_ms": 0.0}
        fb = self.fallback.search_batch(queries, max(k, 100)) if self.fallback is not None else [[] for _ in queries]
        if self.fallback is not None:
            for key in timing:
                timing[key] += self.fallback.last_timing.get(key, 0.0)
        out = []
        for q, fbl in zip(queries, fb):
            steps, log = self._one(q, timing)
            self.trace.append({"q": q, "style": self.style, "steps": steps, "log": log})
            seen, ranked = set(), []
            for rank in range(self.per_step):
                for st in steps:
                    if rank < len(st) and st[rank] not in seen:
                        seen.add(st[rank]); ranked.append(st[rank])
            ranked += [d for d, _ in fbl if d not in seen]
            n = len(ranked)
            out.append([(d, float(n - i)) for i, d in enumerate(ranked[:k])])
        self.last_timing = timing
        return out


# --------------------------------------------------------------------------- entity split (cross-company)
class EntitySplitRetriever(Retriever):
    """Cross-entity questions ("compare CADIZ INC and HOUSTON AMERICAN ENERGY ...") retrieve one entity's evidence and
    crowd out the other. Detect the corpus entities named in the question (corpus field `entity_field`, e.g. company),
    run one sub-query per entity ("<entity>: <question>") through the base retriever, and interleave the per-entity
    rankings at source-document level so every entity gets slots in the top k. <2 entities found -> base ranking.
    Rule-based analogue of query decomposition (Self-Ask / Multi-Meta-RAG metadata routing) with no LLM call."""

    SUFFIX = {"INC", "CORP", "CORPORATION", "CO", "LTD", "PLC", "LP", "LLC", "INCORPORATED", "COMPANY", "THE"}

    def __init__(self, base: Retriever, name: str = "entsplit", entity_field: str = "company", per_entity_k: int = 100):
        super().__init__()
        self.base, self.name, self.field, self.per_k = base, name, entity_field, per_entity_k

    @staticmethod
    def _norm(s):
        return " " + " ".join(re.sub(r"[^A-Z0-9]+", " ", s.upper().replace("_", " ")).split()) + " "

    def build(self, corpus, dataset):
        self.base.build(corpus, dataset)
        self.doc_ids = self.base.doc_ids
        self.src = {d["doc_id"]: d.get("source_doc_id", d["doc_id"]) for d in corpus}
        ents = {}
        hdr = re.compile(rf"^{re.escape(self.field)}: ([^|\n]+)")
        for d in corpus:
            v = d.get(self.field)
            if not v:   # chunked corpora keep metadata only in the contextual header line ("company: X | form: ...")
                m = hdr.match(d.get("text", ""))
                v = m.group(1).strip() if m else None
            if not v or v in ents:
                continue
            nm = re.sub(r"\s*\([^)]*\)\s*$", "", v)          # drop "(TICKER)"
            w = self._norm(nm).split()
            while len(w) > 1 and w[-1] in self.SUFFIX:
                w.pop()
            if w and len(" ".join(w)) >= 4:
                ents[v] = (" " + " ".join(w) + " ", nm)
        self.ents = ents
        self.index_key = f"{self.base.index_key}+entsplit:{self.field}"
        self.stats = {**self.base.stats, "n_entities": len(ents)}
        return self

    def search_batch(self, queries, k):
        tm = Timer(_cuda())
        found = []
        for q in queries:
            nq = self._norm(q)
            hits = sorted({(nq.find(key), nm) for key, nm in self.ents.values() if key in nq})
            found.append([nm for _, nm in hits])
        base = self.base.search_batch(queries, max(k, self.per_k))
        timing = dict(self.base.last_timing)
        subs, owner = [], []
        for i, (q, f) in enumerate(zip(queries, found)):
            if len(f) >= 2:
                for nm in f:
                    subs.append(f"{nm}: {q}"); owner.append(i)
        self.stats["n_split_queries"] = self.stats.get("n_split_queries", 0) + len(set(owner))
        sub_res = self.base.search_batch(subs, max(k, self.per_k)) if subs else []
        per = {}
        for i, r in zip(owner, sub_res):
            per.setdefault(i, []).append(r)
        out = []
        for i, b in enumerate(base):
            if i not in per:
                out.append(b[:k]); continue
            lists, seen, merged, pos = per[i] + [b], set(), [], [0] * (len(per[i]) + 1)
            while len(merged) < k and any(p < len(l) for p, l in zip(pos, lists)):
                for li, l in enumerate(lists):
                    while pos[li] < len(l):
                        d = l[pos[li]][0]; pos[li] += 1
                        if self.src.get(d, d) not in seen:
                            seen.add(self.src.get(d, d)); merged.append(d); break
            out.append([(d, float(k - r)) for r, d in enumerate(merged[:k])])
        tm.lap("entsplit_ms")
        timing["post_ms"] = timing.get("post_ms", 0.0) + tm.out["entsplit_ms"]
        self.last_timing = timing
        return out


# --------------------------------------------------------------------------- set selection by requirement coverage (D1)
REQ_SYS = ("You plan evidence retrieval. List the distinct pieces of information that must be found to answer the question "
           "(2 to 4 items; 1 item if the question needs only one fact). Write each item as a short standalone search query "
           "that names the specific entity, source, company or date it concerns. Output only lines that start with '- '.")


class CoverageSelectRetriever(Retriever):
    """Set selection instead of top-k ranking (SetR, Lee et al., ACL 2025; coverage-matrix + greedy max-marginal coverage).
    1) a local LLM lists the question's information requirements r_1..r_m (cached per question);
    2) the base reranker's cross-encoder scores every (r_j, candidate) pair for the top `pool` candidates -> coverage matrix C;
    3) greedy: pick the candidate with the largest  sum_j max(0, C_ij - covered_j) + lam * relevance_i , update covered_j;
       one pick per source document (chunked corpora); the rest of the base ranking follows.
    Questions with <2 requirements keep the base ranking. No training; the LLM is called once per question."""

    def __init__(self, base: Retriever, name: str = "covsel", llm: str = "qwen3:4b-instruct", pool: int = 30, select: int = 10,
                 lam: float = 0.3, url: str = "http://localhost:11434", cache: str | None = None):
        super().__init__()
        self.base, self.name, self.llm, self.pool, self.select, self.lam, self.url, self.cache_path = base, name, llm, pool, select, lam, url, cache

    def build(self, corpus, dataset):
        import json
        from pathlib import Path
        from src.rag_core import OllamaGenerator
        self.base.build(corpus, dataset)
        self.doc_ids = self.base.doc_ids
        self.text = {d["doc_id"]: doc_text(d, True) for d in corpus}
        self.src = {d["doc_id"]: d.get("source_doc_id", d["doc_id"]) for d in corpus}
        self.gen = OllamaGenerator(self.llm, url=self.url, num_ctx=2048)
        self.gen.check()
        self.cache_file = Path(self.cache_path or f"results/bench/_cache/reqs_{dataset}_{self.llm.replace(':', '_')}.json")
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.reqs = json.loads(self.cache_file.read_text(encoding="utf-8")) if self.cache_file.exists() else {}
        self.index_key = f"{self.base.index_key}+covsel:{self.llm}(p{self.pool},s{self.select},l{self.lam})"
        self.stats = {**self.base.stats, "llm": self.llm, "n_set_selected": 0}
        return self

    def _requirements(self, q, timing):
        import json
        if q not in self.reqs:
            r = self.gen.chat(REQ_SYS, f"Question: {q}")
            timing["post_ms"] += r["latency_s"] * 1000
            items = [l[2:].strip() for l in re.sub(r"<think>.*?</think>", "", r["text"], flags=re.S).splitlines() if l.strip().startswith("- ")]
            self.reqs[q] = [x for x in items if x][:4]
            tmp = self.cache_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.reqs, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.cache_file)
        return self.reqs[q]

    def search_batch(self, queries, k):
        base = self.base.search_batch(queries, max(k, 100))
        timing = dict(self.base.last_timing)
        timing.setdefault("post_ms", 0.0)
        tm = Timer(_cuda())
        out = []
        for q, hits in zip(queries, base):
            reqs = self._requirements(q, timing)
            if len(reqs) < 2:
                out.append(hits[:k]); continue
            cand = [d for d, _ in hits[:self.pool]]
            pairs = [(r, self.text[d]) for d in cand for r in reqs]
            C = np.asarray(self.base.scorer(pairs), dtype=np.float64).reshape(len(cand), len(reqs))
            C = 1 / (1 + np.exp(-C)) if (C.min() < 0 or C.max() > 1) else C          # logits -> [0,1]
            rel = np.asarray([s for _, s in hits[:self.pool]], dtype=np.float64)
            rel = 1 / (1 + np.exp(-rel)) if (rel.min() < 0 or rel.max() > 1) else rel
            covered, picked, used_src = np.zeros(len(reqs)), [], set()
            while len(picked) < min(self.select, len(cand)):
                best, bi = -1e9, None
                for i, d in enumerate(cand):
                    if i in picked or self.src.get(d, d) in used_src:
                        continue
                    g = np.clip(C[i] - covered, 0, None).sum() + self.lam * rel[i]
                    if g > best:
                        best, bi = g, i
                if bi is None:
                    break
                picked.append(bi); used_src.add(self.src.get(cand[bi], cand[bi]))
                covered = np.maximum(covered, C[bi])
            sel = [cand[i] for i in picked]
            rest = [d for d, _ in hits if d not in set(sel)]
            ranked = sel + rest
            out.append([(d, float(len(ranked) - r)) for r, d in enumerate(ranked[:k])])
            self.stats["n_set_selected"] += 1
        tm.lap("covsel_ms")
        timing["post_ms"] += tm.out["covsel_ms"]
        self.last_timing = timing
        return out
