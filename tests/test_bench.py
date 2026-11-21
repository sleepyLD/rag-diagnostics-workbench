"""Bench skeleton: metrics, fusion, BM25, dense (fake encoder), dataset checks. CPU only, no downloads."""
import numpy as np
import pytest

from src.retrievers import BM25Retriever, DenseRetriever, FusionRetriever, aggregate_to_source, build_retriever, rrf, score_fusion
from tools.bench import metrics as M
from tools.bench.common import check_dataset, filter_train, split_questions

CORPUS = [{"doc_id": "a", "title": "Paris", "text": "Paris is the capital of France."},
          {"doc_id": "b", "title": "Berlin", "text": "Berlin is the capital of Germany."},
          {"doc_id": "c", "title": "Seine", "text": "The Seine river flows through Paris."},
          {"doc_id": "d", "title": "Cats", "text": "Cats sleep most of the day."}]


def test_query_metrics():
    m = M.query_metrics(["x", "a", "y", "b"], ["a", "b"])
    assert m["recall@1"] == 0 and m["recall@3"] == 0.5 and m["recall@5"] == 1
    assert m["all_gold_hit@3"] == 0 and m["all_gold_hit@5"] == 1
    assert m["mrr"] == 0.5
    assert 0 < m["ndcg@10"] < 1
    assert M.query_metrics(["a", "b"], ["a", "b"])["ndcg@10"] == pytest.approx(1.0)
    assert M.query_metrics(["x"], ["a"])["mrr"] == 0


def test_gold_sets_alternatives():
    m = M.query_metrics(["c", "d", "x"], ["a", "b"], gold_sets=[["a", "b"], ["c", "d"]])
    assert m["all_gold_hit@3"] == 1 and m["recall@3"] == 0 and m["mrr"] == 1


def test_compute_groups_and_bootstrap():
    rows_a = [{"qid": str(i), "gold_doc_ids": ["a"], "type": "t" + str(i % 2), "hops": 1, "retrieved": [["x", 1.0], ["a", 0.5]]} for i in range(40)]
    rows_b = [{**r, "retrieved": [["a", 1.0]]} for r in rows_a]
    c = M.compute(rows_a)
    assert c["n"] == 40 and c["overall"]["recall@1"] == 0 and c["by_type"]["t0"]["n"] == 20
    b = M.paired_bootstrap(rows_a, rows_b, "recall@1")
    assert b["diff_b_minus_a"] == 1 and b["significant"]
    assert not M.paired_bootstrap(rows_a, rows_a, "recall@1")["significant"]


def test_rrf_and_score_fusion():
    l1, l2 = [("a", 9), ("b", 8), ("c", 1)], [("b", 5), ("d", 4)]
    assert [d for d, _ in rrf([l1, l2])][0] == "b"
    assert [d for d, _ in rrf([l1, l2], weights=[1, 0])] == ["a", "b", "c", "d"][:3] + ["d"]
    s = dict(score_fusion([l1, l2]))
    assert s["a"] == pytest.approx(1.0) and s["b"] == pytest.approx(0.875 + 1.0) and s["d"] == pytest.approx(0.0)
    assert aggregate_to_source([("c1", 3), ("c2", 5), ("c3", 4)], {"c1": "A", "c2": "B", "c3": "A"}) == [("B", 5), ("A", 4)]


def test_bm25():
    r = BM25Retriever().build(CORPUS, "toy")
    hits = r.search_batch(["capital of Germany", "zzz unknown"], 2)
    assert hits[0][0][0] == "b" and len(hits[1]) == 2
    assert r.stats["index_bytes"] > 0 and "search_ms" in r.last_timing


class FakeEncoder:
    """Bag-of-letters embedding: deterministic, no model download."""
    def encode(self, texts, is_query, show_progress=False):
        v = np.zeros((len(texts), 26), dtype=np.float32)
        for i, t in enumerate(texts):
            for ch in t.lower():
                if "a" <= ch <= "z":
                    v[i, ord(ch) - 97] += 1
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def test_dense_cache_and_fusion(tmp_path):
    pytest.importorskip("torch")
    d = DenseRetriever("fake/model", index_root=tmp_path, encoder=FakeEncoder(), search_device="cpu").build(CORPUS, "toy")
    assert not d.stats["cached"]
    assert d.search_batch(["Paris is the capital of France."], 1)[0][0][0] == "a"
    d2 = DenseRetriever("fake/model", index_root=tmp_path, encoder=FakeEncoder(), search_device="cpu").build(CORPUS, "toy")
    assert d2.stats["cached"]
    f = FusionRetriever([d, BM25Retriever().build(CORPUS, "toy")]).build(CORPUS, "toy")
    out = f.search_batch(["capital of Germany"], 3)
    assert "b" in [d for d, _ in out[0][:2]] and set(f.last_timing) == {"encode_ms", "search_ms", "post_ms"}


def test_build_retriever_config():
    assert isinstance(build_retriever({"type": "bm25", "name": "x"}), BM25Retriever)
    f = build_retriever({"type": "fusion", "mode": "rrf", "components": [{"type": "bm25"}, {"type": "bm25", "k1": 1.2}]})
    assert len(f.components) == 2


def test_dataset_checks():
    dev = [{"qid": "1", "question": "Q one?", "gold_doc_ids": ["a"]}]
    hold = [{"qid": "2", "question": "Q two?", "gold_doc_ids": ["b"]}]
    assert check_dataset(CORPUS, dev, hold)["gold_in_corpus"]
    with pytest.raises(AssertionError):
        check_dataset(CORPUS, dev, [{"qid": "2", "question": "Q two?", "gold_doc_ids": ["zz"]}])
    with pytest.raises(AssertionError):
        check_dataset(CORPUS, dev, [{"qid": "3", "question": "q ONE", "gold_doc_ids": ["b"]}])
    qs = [{"qid": str(i), "question": f"question {i}"} for i in range(50)] + [{"qid": "dup", "question": "Question 3!"}]
    d, h, rest = split_questions(qs, 20, 10)
    assert len(d) == 20 and len(h) == 10 and len(rest) == 20
    kept, info = filter_train([{"question": "Q one", "positives": [{"title": "Paris"}]}, {"question": "new", "positives": [{"title": "X"}]}], dev, hold)
    assert len(kept) == 1 and info["dropped_same_question_as_eval"] == 1


def test_rerank_and_twohop_toy(tmp_path):
    from src.retrievers import RerankRetriever, TwoHopRetriever
    bm = BM25Retriever().build(CORPUS, "toy")
    # scorer prefers documents mentioning "Seine"
    rr = RerankRetriever(bm, "fake", depth=2, scorer=lambda pairs: np.asarray([float("Seine" in t) for _, t in pairs], dtype=np.float32)).build(CORPUS, "toy")
    out = rr.search_batch(["Paris capital river Seine"], 3)[0]
    assert out[0][0] == "c" and len(out) == 3 and out[0][1] >= out[1][1] >= out[2][1]
    th = TwoHopRetriever(BM25Retriever().build(CORPUS, "toy"), beam=1, per_beam=2).build(CORPUS, "toy")
    hops = th.search_batch(["capital of France"], 3)[0]
    assert hops[0][0] == "a" and {d for d, _ in hops} == {"a", "b", "c"}
    assert set(th.last_timing) == {"encode_ms", "search_ms", "post_ms"}


def test_chain_rerank_toy():
    from src.retrievers import ChainRerankRetriever
    corpus = CORPUS + [{"doc_id": "d", "title": "Seine", "text": "The Seine ends in the English Channel."}]
    def scorer(pairs):   # hop 1 likes "Paris"; hop 2 (context contains Paris) likes "Channel"
        return np.asarray([3.0 if ("[SEP]" in q and "Channel" in t) else (2.0 if "Paris" in t and "[SEP]" not in q else -2.0)
                           for q, t in pairs], dtype=np.float32)
    ch = ChainRerankRetriever(BM25Retriever().build(corpus, "toy"), "fake", pool=4, beam=1, scorer=scorer).build(corpus, "toy")
    top = [d for d, _ in ch.search_batch(["Where does the river of Paris end?"], 4)[0]]
    assert top[:2] in (["a", "d"], ["c", "d"]) and len(top) == 4


def test_mhbeam_and_chain_expand_toy():
    from src.retrievers import ChainRerankRetriever, MultiHopBeamRetriever
    corpus = CORPUS + [{"doc_id": "d", "title": "English Channel", "text": "The English Channel separates England and France."}]
    mh = MultiHopBeamRetriever(BM25Retriever().build(corpus, "toy"), hops=3, beam=2, per_beam=2).build(corpus, "toy")
    out = mh.search_batch(["river of Paris"], 4)[0]
    assert len(out) == 4 and len({d for d, _ in out}) == 4
    def scorer(pairs):
        return np.asarray([3.0 if ("[SEP]" in q and "Channel" in t) else (2.0 if "Paris" in t and "[SEP]" not in q else -2.0)
                           for q, t in pairs], dtype=np.float32)
    ch = ChainRerankRetriever(BM25Retriever().build(corpus, "toy"), "fake", pool=1, beam=1, scorer=scorer, hops=3, beam3=1,
                              expand=BM25Retriever().build(corpus, "toy"), expand_k=3).build(corpus, "toy")
    top = [d for d, _ in ch.search_batch(["Where does the river of Paris end?"], 4)[0]]
    assert "d" in top[:3]          # only reachable through the expansion search (pool=1)


def test_decompose_toy(monkeypatch):
    from src.retrievers import DecomposeRetriever
    corpus = CORPUS + [{"doc_id": "d", "title": "English Channel", "text": "The Seine ends in the English Channel."}]
    class FakeGen:
        def __init__(self): self.n = 0
        def check(self): pass
        def chat(self, system, user):
            if system.startswith("You break"):
                if "(none yet)" in user: return {"text": "Sub-question: Which river flows through Paris?", "latency_s": 0.0}
                if "Channel" in user: return {"text": "DONE", "latency_s": 0.0}
                return {"text": "Sub-question: Where does the Seine end?", "latency_s": 0.0}
            return {"text": "Answer: Seine" if "Sub-question: Which river" in user else "Answer: English Channel", "latency_s": 0.0}
    import src.rag_core as rc
    monkeypatch.setattr(rc, "OllamaGenerator", lambda *a, **k: FakeGen())
    r = DecomposeRetriever(BM25Retriever().build(corpus, "toy"), per_step=2, answer_passages=1).build(corpus, "toy")
    top = [d for d, _ in r.search_batch(["Where does the river of Paris end?"], 3)[0]]
    assert "c" in top and "d" in top and len(r.trace[0]["facts"]) == 2


def test_iterative_toy(monkeypatch):
    from src.retrievers import IterativeRetriever
    corpus = CORPUS + [{"doc_id": "d", "title": "English Channel", "text": "The Seine ends in the English Channel."}]
    class FakeGen:
        def check(self): pass
        def chat(self, system, user):
            if "next single sentence" in system:
                return {"text": "So the answer is the English Channel." if "Seine ends" in user and "Reasoning so far: The" in user
                        else "The Seine ends in the English Channel.", "latency_s": 0.0}
            return {"text": "The Seine ends in the Channel.\nAnswer: English Channel", "latency_s": 0.0}
    import src.rag_core as rc
    monkeypatch.setattr(rc, "OllamaGenerator", lambda *a, **k: FakeGen())
    for style in ("ircot", "iterretgen"):
        r = IterativeRetriever(BM25Retriever().build(corpus, "toy"), style=style, per_step=1, max_steps=3).build(corpus, "toy")
        top = [d for d, _ in r.search_batch(["What flows through Paris?"], 3)[0]]
        assert "d" in top and len(r.trace[0]["steps"]) >= 2, (style, top, r.trace)
