"""Checks on the SQuAD subset built by tools.datasets.prepare_squad."""
import json
from pathlib import Path

import pytest

from tools.datasets import prepare_squad

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "squad" / "dev-v1.1.json"
COLL = ROOT / "collections" / "squad_v11_dev500"

pytestmark = pytest.mark.skipif(not RAW.exists(), reason="SQuAD dev-v1.1.json not downloaded")


def load(name):
    return [json.loads(l) for l in (ROOT / "eval" / name).read_text(encoding="utf-8").splitlines()]


def body(path):
    text = path.read_text(encoding="utf-8")
    return text.split("\n---\n", 1)[1].strip()


def test_counts():
    m = json.loads((ROOT / "eval" / "squad_manifest.json").read_text(encoding="utf-8"))
    assert len(list(COLL.glob("*.md"))) == m["n_paragraphs"] >= 500
    assert len(load("squad_dev.jsonl")) == 50 and len(load("squad_holdout.jsonl")) == 50


def test_dev_and_holdout_disjoint():
    dev, hold = load("squad_dev.jsonl"), load("squad_holdout.jsonl")
    assert not {q["qid"] for q in dev} & {q["qid"] for q in hold}
    assert not {q["gold_doc_id"] for q in dev} & {q["gold_doc_id"] for q in hold}


def test_gold_doc_exists_and_answer_offset_is_correct():
    for q in load("squad_dev.jsonl") + load("squad_holdout.jsonl"):
        text = body(COLL / f"{q['gold_doc_id']}.md")
        a = q["answers"][0]
        # the paragraph text is stored stripped; SQuAD contexts have no leading spaces
        assert text[q["answer_start"]:q["answer_start"] + len(a)] == a, q["qid"]


def test_no_questions_leak_into_the_collection():
    corpus = "\n".join(p.read_text(encoding="utf-8") for p in COLL.glob("*.md"))
    for q in load("squad_dev.jsonl") + load("squad_holdout.jsonl"):
        assert q["question"] not in corpus
        assert q["qid"] not in corpus
    allowed = {"doc_id", "title", "collection", "source", "license"}
    for p in COLL.glob("*.md"):
        fm = p.read_text(encoding="utf-8").split("\n---\n", 1)[0].removeprefix("---\n")
        keys = {line.split(":", 1)[0] for line in fm.splitlines()}
        assert keys == allowed, (p.name, keys)


def test_build_is_reproducible(tmp_path):
    assert prepare_squad.main([str(RAW), "--root", str(tmp_path)]) == 0
    for name in ("squad_dev.jsonl", "squad_holdout.jsonl"):
        assert (tmp_path / "eval" / name).read_text(encoding="utf-8") == \
               (ROOT / "eval" / name).read_text(encoding="utf-8")
    new = sorted(p.name for p in (tmp_path / "collections" / "squad_v11_dev500").glob("*.md"))
    assert new == sorted(p.name for p in COLL.glob("*.md"))
