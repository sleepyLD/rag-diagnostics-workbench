"""Bi-encoder retrieval: embeddings cached under index/bench/, exact inner product on the GPU (or CPU)."""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

from .base import Retriever, Timer, atomic_save_npy, atomic_write_text


# one-line task descriptions; public BEIR sets use the instructions from the Qwen3-Embedding / MTEB evaluation prompts
TASK_INSTRUCTIONS = {
    "scifact": "Given a scientific claim, retrieve documents that support or refute the claim",
    "nfcorpus": "Given a question, retrieve relevant documents that best answer the question",
    "fiqa": "Given a financial question, retrieve user replies that best answer the question",
    "hotpotqa": "Given a multi-hop question, retrieve documents that can help answer the question",
    "2wikimultihopqa": "Given a multi-hop question, retrieve documents that can help answer the question",
    "musique": "Given a multi-hop question, retrieve documents that can help answer the question",
    "multihop_rag": "Given a question about recent news, retrieve news article passages that help answer the question",
    "neoqa": "Given a question about fictional news events, retrieve news article passages that help answer the question",
    "medqa": "Given a medical exam question, retrieve textbook passages that help answer the question",
    "bioasq": "Given a biomedical question, retrieve PubMed abstract passages that answer the question",
    "squad": "Given a question, retrieve Wikipedia passages that answer the question",
    "finder": "Given a financial analyst question, retrieve 10-K filing passages that answer the question",
    "finrate": "Given a question about company filings, retrieve SEC filing passages that answer the question",
}


def task_instruction(dataset: str) -> str:
    for k in sorted(TASK_INSTRUCTIONS, key=len, reverse=True):
        if dataset.startswith(k):
            return TASK_INSTRUCTIONS[k]
    return "Given a web search query, retrieve relevant passages that answer the query"


class STEncoder:
    """sentence-transformers encoder. encode(texts, is_query) -> L2-normalised float32 array."""

    def __init__(self, model: str, query_prefix: str = "", doc_prefix: str = "", max_seq_length: int = 512,
                 batch_size: int = 64, fp16: bool = True, trust_remote_code: bool = False, device: str | None = None,
                 use_model_prompts: bool = False):
        import torch
        from sentence_transformers import SentenceTransformer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model, device=self.device, trust_remote_code=trust_remote_code)
        self.model.max_seq_length = min(max_seq_length, self.model.max_seq_length or max_seq_length)
        if fp16 and self.device == "cuda":
            self.model.half()
        self.query_prefix, self.doc_prefix, self.batch_size = query_prefix, doc_prefix, batch_size
        # use_model_prompts: take the query/document prompt the model ships in its sentence-transformers config
        # (names differ between models), instead of a hand-written prefix
        self.q_prompt = self.d_prompt = None
        if use_model_prompts:
            names = set((getattr(self.model, "prompts", None) or {}).keys())
            self.q_prompt = next((n for n in ("query", "web_search_query", "retrieval.query", "search_query", "Retrieval-query") if n in names), None)
            self.d_prompt = next((n for n in ("document", "passage", "retrieval.passage", "search_document", "Retrieval-document") if n in names), None)
            print(f"model prompts {sorted(names)} -> query={self.q_prompt} doc={self.d_prompt}", flush=True)

    def encode(self, texts: list[str], is_query: bool, show_progress: bool = False) -> np.ndarray:
        prefix = self.query_prefix if is_query else self.doc_prefix
        pn = self.q_prompt if is_query else self.d_prompt
        emb = self.model.encode([prefix + t for t in texts], batch_size=self.batch_size, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=show_progress, **({"prompt_name": pn} if pn else {}))
        return np.asarray(emb, dtype=np.float32)


def doc_text(d: dict, use_title: bool = True) -> str:
    return (d["title"] + "\n" + d["text"]) if (use_title and d.get("title")) else d["text"]


class DenseRetriever(Retriever):
    def __init__(self, model: str, name: str | None = None, use_title: bool = True, index_root: str | Path = "index/bench",
                 encoder=None, search_device: str | None = None, **encoder_kwargs):
        super().__init__()
        self.model_name, self.name, self.use_title = model, name or model, use_title
        self.index_root, self.encoder, self.encoder_kwargs = Path(index_root), encoder, encoder_kwargs
        self.search_device = search_device
        self.task_instruct = bool(self.encoder_kwargs.pop("task_instruct", False))

    def _key(self, dataset: str, texts: list[str]) -> str:
        h = hashlib.sha1()
        h.update(json.dumps({"model": self.model_name, "use_title": self.use_title,
                             **{k: v for k, v in self.encoder_kwargs.items() if k in ("doc_prefix", "max_seq_length", "fp16", "use_model_prompts")}}, sort_keys=True).encode())
        for t in texts:
            h.update(t.encode("utf-8")); h.update(b"\x00")
        return f"{dataset}__{re.sub(r'[^A-Za-z0-9.]+', '-', self.model_name)}__{h.hexdigest()[:12]}"

    def build(self, corpus: list[dict], dataset: str) -> "DenseRetriever":
        import torch
        if self.encoder is None:
            self.encoder = STEncoder(self.model_name, **self.encoder_kwargs)
        if self.task_instruct:   # per-dataset one-line task instruction (MTEB-style "Instruct: {task}\nQuery:"); documents unchanged
            task = task_instruction(dataset)
            self.encoder.query_prefix, self.encoder.q_prompt = f"Instruct: {task}\nQuery:", None
        self.doc_ids = [d["doc_id"] for d in corpus]
        texts = [doc_text(d, self.use_title) for d in corpus]
        self.index_key = self._key(dataset, texts)
        path = self.index_root / self.index_key
        t0 = time.time()
        # meta.json is written LAST, so its presence means the cache is complete (crash-safe)
        if (path / "meta.json").exists() and (path / "embeddings.npy").exists() and \
                json.loads((path / "doc_ids.json").read_text(encoding="utf-8")) == self.doc_ids:
            emb, cached = np.load(path / "embeddings.npy"), True
            meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
            build_s = meta.get("build_s")
        else:
            emb, cached = self.encoder.encode(texts, is_query=False, show_progress=True), False
            build_s = round(time.time() - t0, 2)
            path.mkdir(parents=True, exist_ok=True)
            atomic_save_npy(path / "embeddings.npy", emb)
            atomic_write_text(path / "doc_ids.json", json.dumps(self.doc_ids))
            atomic_write_text(path / "meta.json", json.dumps({"dataset": dataset, "model": self.model_name, "n_docs": len(corpus),
                                                              "dim": int(emb.shape[1]), "build_s": build_s}, indent=1))
        dev = self.search_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.emb = torch.from_numpy(emb).to(dev)
        self.cuda = dev == "cuda"
        self.stats = {"build_s": build_s, "cached": cached, "index_bytes": int(emb.nbytes), "dim": int(emb.shape[1]), "n_docs": len(corpus)}
        return self

    def search_batch(self, queries: list[str], k: int) -> list[list[tuple[str, float]]]:
        import torch
        tm = Timer(self.cuda)
        q = torch.from_numpy(self.encoder.encode(queries, is_query=True)).to(self.emb.device)
        tm.lap("encode_ms")
        if getattr(self, "prf_k", 0):   # Rocchio pseudo-relevance feedback in embedding space (dense PRF, e.g. ANCE-PRF / Rocchio)
            _, top = torch.topk(q @ self.emb.T, self.prf_k, dim=1)
            q = q + self.prf_beta * self.emb[top].mean(dim=1)
            q = q / q.norm(dim=1, keepdim=True)
        scores, idx = torch.topk(q @ self.emb.T, min(k, len(self.doc_ids)), dim=1)
        tm.lap("search_ms")
        scores, idx = scores.cpu().tolist(), idx.cpu().tolist()
        out = [[(self.doc_ids[j], float(s)) for j, s in zip(ii, ss)] for ii, ss in zip(idx, scores)]
        tm.lap("post_ms")
        self.last_timing = tm.out
        return out
