"""Unit tests for the parts of src/rag_core.py that do not need a model."""
import numpy as np

from src import rag_core as rc


def test_parse_output_variants():
    assert rc.parse_output("Answer: Harrods\nSources: [A__p1], [A__p2]") == ("Harrods", ["A__p1", "A__p2"], True)
    assert rc.parse_output("Harrods\nSources: [A__p1]") == ("Harrods", ["A__p1"], True)       # no 'Answer:'
    assert rc.parse_output("Sullivan Bay on Port Phillip") == ("Sullivan Bay on Port Phillip", [], False)
    assert rc.parse_output("<think>hmm</think>\nAnswer: X\nSources: []") == ("X", [], True)


def test_abstain_detection():
    assert rc.is_abstain("INSUFFICIENT_EVIDENCE") and rc.is_abstain("UNKNOWN") and rc.is_abstain("insufficient_evidence.")
    assert rc.is_abstain("<unknown>")                                     # 1.7B copied the template
    assert not rc.is_abstain("Harrods")


def test_angle_bracket_citations_are_cleaned():
    assert rc.parse_output("Answer: Anglican\nSources: [<Private_school__p015>]")[1] == ["Private_school__p015"]
    assert rc.parse_output("Answer: A\nSources: [<doc_id> ABC__p050]")[1] == ["ABC__p050"]
    assert rc.parse_output("Answer: 2n\nSources: [<doc_id>]")[1] == []


def test_multi_gold_question():
    q = {"qid": "x", "question": "q", "answers": ["a"], "gold_doc_ids": ["d1", "d2"], "gold_doc_id": "d1"}
    assert rc.gold_ids(q) == ["d1", "d2"]
    assert rc.gold_ids({"gold_doc_id": "d1"}) == ["d1"]


def test_scores():
    q = "The Talbot Hughes collection was a gift from which company?"
    s = rc.score("The Talbot Hughes collection was a gift from Harrods.", ["Harrods"], q)
    assert s["f1"] < 0.5 and s["f1_qstrip"] == 1.0 and s["contains_gold"]
    s = rc.score("High-frequency electricity", ["atmospheric", "atmospheric electricity"],
                 "What kind of electricity was Tesla investigating?")
    assert s["f1"] == 0.5 and s["f1_qstrip"] == 0.0 and not s["contains_gold"]   # old F1 false positive
    s = rc.score("apicomplexan-related diseases", ["apicomplexan-related diseases"],
                 "What kind of diseases do drugs target apicoplasts for?")
    assert s["f1_qstrip"] == 1.0
    assert not rc.contains_gold("1990", "1")                                       # whole tokens only
    assert rc.score("x", [], "q")["em"] is None                                   # no-answer question


def test_index_key_depends_on_collection_model_and_content():
    docs = [{"doc_id": "a", "text": "hello"}]
    k1 = rc.index_key("A", "BAAI/bge-small-en-v1.5", docs)
    assert k1 != rc.index_key("B", "BAAI/bge-small-en-v1.5", docs)
    assert k1 != rc.index_key("A", "other-model", docs)
    assert k1 != rc.index_key("A", "BAAI/bge-small-en-v1.5", [{"doc_id": "a", "text": "hello!"}])


class _FakeDense(rc.DenseRetriever):
    def __init__(self, docs, scores):
        self.ids = [d["doc_id"] for d in docs]
        self.doc_by_id = {d["doc_id"]: d for d in docs}
        self.index_key = "c__dense__x"
        self._s = np.array(scores, dtype=float)

    def scores(self, query):
        return self._s


def test_bm25_and_rrf_fusion():
    docs = [{"doc_id": "a", "title": "Tampa", "text": "WFTS-TV and WWSB are ABC affiliates"},
            {"doc_id": "b", "title": "ABC", "text": "the network headquarters"},
            {"doc_id": "c", "title": "Other", "text": "nothing relevant"}]
    bm = rc.BM25Retriever(docs, "c")
    assert bm.search("Tampa affiliates", 1)[0]["doc_id"] == "a"
    dense = _FakeDense(docs, [0.1, 0.9, 0.5])          # dense likes b, then c, then a
    hy = rc.HybridRetriever(dense, bm)
    s = hy.scores("Tampa affiliates")
    # a: dense rank 3, bm25 rank 1 ; b: dense 1, bm25 >=2 ; fused scores follow 1/(60+r)
    assert np.isclose(s[0], 1 / 63 + 1 / 61)
    assert hy.ranks_of("Tampa affiliates", ["a", "zzz"]) == [hy.ranks_of("Tampa affiliates", ["a"])[0], None]
    assert rc.rank_positions(np.array([0.2, 0.9, 0.2])).tolist() == [2, 1, 3]   # ties keep document order


def test_context_budget_drops_whole_passages():
    hits = [{"doc_id": f"d{i}", "title": "t", "text": "x" * 100} for i in range(5)]
    ctx, used, dropped = rc.build_context(hits, char_budget=250)
    assert used == ["d0", "d1"] and dropped == ["d2", "d3", "d4"]


def test_search_can_exclude_passages():
    docs = [{"doc_id": d, "title": d, "text": d} for d in "abc"]
    dense = _FakeDense(docs, [0.9, 0.5, 0.1])
    assert [h["doc_id"] for h in dense.search("q", 2)] == ["a", "b"]
    assert [h["doc_id"] for h in dense.search("q", 2, exclude=["a", "zzz"])] == ["b", "c"]
    assert [h["doc_id"] for h in dense.search("q", 5, exclude=["b"])] == ["a", "c"]   # excluded never returned
