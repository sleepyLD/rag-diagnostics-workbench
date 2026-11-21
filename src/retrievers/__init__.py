"""Bench retrievers. build_retriever(config) turns a JSON-able config into a Retriever (not yet built)."""
from .base import Retriever
from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .fusion import FusionRetriever, aggregate_to_source, rrf, score_fusion
from .advanced import ChainRerankRetriever, ColBERTRetriever, DecomposeRetriever, IterativeRetriever, CoverageSelectRetriever, EntitySplitRetriever, MultiHopBeamRetriever, RerankRetriever, SparseRetriever, TwoHopRetriever

__all__ = ["Retriever", "BM25Retriever", "DenseRetriever", "FusionRetriever", "aggregate_to_source", "rrf", "score_fusion", "build_retriever"]


def build_retriever(cfg: dict, _cache: dict | None = None) -> Retriever:
    """cfg["type"]: dense | bm25 | fusion. Components of a fusion are configs themselves."""
    cfg = dict(cfg)
    kind, name = cfg.pop("type"), cfg.pop("name", None)
    if kind == "dense":
        prf_k, prf_beta = cfg.pop("prf_k", 0), cfg.pop("prf_beta", 0.5)
        r = DenseRetriever(cfg.pop("model"), name=name, **cfg)
        r.prf_k, r.prf_beta = prf_k, prf_beta
        return r
    if kind == "bm25":
        return BM25Retriever(name=name or "bm25", **cfg)
    if kind == "fusion":
        comps = [build_retriever(c) for c in cfg.pop("components")]
        return FusionRetriever(comps, name=name or "fusion", **cfg)
    if kind == "rerank":
        first, model = build_retriever(cfg.pop("first")), cfg.pop("model")
        if cfg.pop("scorer_type", "ce") == "qwen3":
            from .advanced import Qwen3RerankScorer
            sc = Qwen3RerankScorer(model, cfg.pop("instruction"), max_length=cfg.pop("max_length", 1024), batch_size=cfg.pop("batch_size", 8))
            return RerankRetriever(first, model, name=name or "rerank", scorer=sc, **cfg)
        return RerankRetriever(first, model, name=name or "rerank", **cfg)
    if kind == "chain":
        exp = cfg.pop("expand", None)
        return ChainRerankRetriever(build_retriever(cfg.pop("first")), cfg.pop("model"), name=name or "chain",
                                    expand=build_retriever(exp) if exp else None, **cfg)
    if kind == "decomp":
        fb = cfg.pop("fallback", None)
        return DecomposeRetriever(build_retriever(cfg.pop("base")), name=name or "decomp", fallback=build_retriever(fb) if fb else None, **cfg)
    if kind == "iter":
        fb = cfg.pop("fallback", None)
        return IterativeRetriever(build_retriever(cfg.pop("base")), name=name or "iter", fallback=build_retriever(fb) if fb else None, **cfg)
    if kind == "mhbeam":
        return MultiHopBeamRetriever(build_retriever(cfg.pop("base")), name=name or "mhbeam", **cfg)
    if kind == "colbert":
        return ColBERTRetriever(cfg.pop("model"), name=name or "colbert", **cfg)
    if kind == "sparse":
        return SparseRetriever(cfg.pop("model"), name=name or "splade", **cfg)
    if kind == "twohop":
        return TwoHopRetriever(build_retriever(cfg.pop("base")), name=name or "twohop", **cfg)
    if kind == "covsel":
        return CoverageSelectRetriever(build_retriever(cfg.pop("base")), name=name or "covsel", **cfg)
    if kind == "entsplit":
        return EntitySplitRetriever(build_retriever(cfg.pop("base")), name=name or "entsplit", **cfg)
    raise ValueError(f"unknown retriever type: {kind}")
