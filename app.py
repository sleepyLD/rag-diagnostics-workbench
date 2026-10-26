"""RAG 工作台：展示用介面。

啟動（專案根目錄）：
    .\\.venv\\Scripts\\python.exe -m streamlit run app.py

左側選知識庫、檢索方法、生成模型、Top-k、RAG 開關；右側輸入問題或挑一題範例，
會顯示答案、引用、取回的段落與分數，以及實際送給模型的 Prompt。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from src import rag_core as rc

ROOT = Path(__file__).resolve().parent
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
OLLAMA_URL = "http://localhost:11434"
CONTEXT_CHAR_BUDGET = 8000
SAMPLE_QUESTIONS = {          # 每個知識庫可挑的範例題（有標準答案，可以順便看答對沒有）
    "squad_v11_dev500": "squad_dev300.jsonl",
    "hotpot_dev400": "hotpot_dev.jsonl",
    "squad_v11_b5": "squad_b_noanswer_dev50.jsonl",
}

st.set_page_config(page_title="RAG 工作台", layout="wide")


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="載入 Embedding 模型…")
def get_embedder():
    return rc.Embedder(EMBED_MODEL)


@st.cache_resource(show_spinner="載入知識庫…")
def get_docs(collection: str):
    docs = rc.load_collection(ROOT, collection)
    return docs, {d["doc_id"]: d for d in docs}


@st.cache_resource(show_spinner="建立或讀取索引…")
def get_retriever(collection: str, method: str):
    docs, _ = get_docs(collection)
    if method == "bm25":
        return rc.BM25Retriever(docs, collection)
    dense = rc.DenseRetriever(rc.build_or_load_index(ROOT, docs, collection, get_embedder()), docs, get_embedder())
    if method == "dense":
        return dense
    return rc.HybridRetriever(dense, get_retriever(collection, "bm25"))


@st.cache_data(ttl=30)
def installed_models() -> list[str]:
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).json()
        return sorted(m["name"] for m in tags.get("models", []))
    except requests.RequestException:
        return []


def list_collections() -> list[str]:
    return sorted(p.name for p in (ROOT / "collections").iterdir() if p.is_dir() and any(p.glob("*.md")))


@st.cache_data
def sample_questions(collection: str) -> list[dict]:
    f = SAMPLE_QUESTIONS.get(collection)
    if not f or not (ROOT / "eval" / f).exists():
        return []
    return rc.load_jsonl(ROOT / "eval" / f)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("設定")
    collections = list_collections()
    collection = st.selectbox("知識庫（Collection）", collections,
                              index=collections.index("squad_v11_dev500") if "squad_v11_dev500" in collections else 0)
    method = st.radio("檢索方法", ["dense", "bm25", "hybrid"], horizontal=True,
                      help="dense：向量；bm25：關鍵字；hybrid：兩者名次用 RRF 融合")
    top_k = st.slider("Top-k（交給模型幾段）", 1, 10, 3)
    use_rag = st.toggle("使用 RAG", value=True, help="關掉 = 模型只看到問題（No-RAG）")

    models = installed_models()
    if models:
        default = models.index("qwen3:4b-instruct") if "qwen3:4b-instruct" in models else 0
        gen_model = st.selectbox("生成模型", models, index=default)
    else:
        st.error("連不到 Ollama，請先啟動 Ollama")
        gen_model = st.text_input("生成模型", "qwen3:4b-instruct")
    no_think = st.checkbox("關閉「思考」", value="instruct" not in gen_model,
                           help="qwen3:1.7b、qwen3.5 等會先思考再回答，關掉比較快")

    st.caption(f"Embedding：`{EMBED_MODEL}`  \nContext 上限：{CONTEXT_CHAR_BUDGET} 字元")


# ---------------------------------------------------------------------------
# Question
# ---------------------------------------------------------------------------

st.title("RAG 工作台")
samples = sample_questions(collection)
picked = None
if samples:
    labels = ["（自己輸入）"] + [f"{i + 1}. {q['question']}" for i, q in enumerate(samples)]
    choice = st.selectbox(f"範例題（{SAMPLE_QUESTIONS[collection]}）", labels)
    if choice != labels[0]:
        picked = samples[labels.index(choice) - 1]
question = st.text_area("問題", value=picked["question"] if picked else "", height=80)
go = st.button("送出", type="primary", disabled=not question.strip())

if picked:
    golds = rc.gold_ids(picked)
    in_this = [g for g in golds if g in get_docs(collection)[1]]
    st.caption(f"標準答案：{' ／ '.join(picked['answers'])}　｜　正確段落：{', '.join(golds)}"
               + ("" if in_this else "　（不在這個知識庫裡 → 應該拒答）"))

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if go:
    q = dict(picked) if picked and picked["question"] == question else {"qid": "manual", "question": question}
    retriever = get_retriever(collection, method) if use_rag else None
    gen = rc.OllamaGenerator(gen_model, OLLAMA_URL, think=False if no_think else None)
    with st.spinner("檢索與生成中…"):
        try:
            rec = rc.ask(q, "rag" if use_rag else "no_rag", gen, retriever, k=top_k, char_budget=CONTEXT_CHAR_BUDGET)
        except RuntimeError as e:
            st.error(str(e))
            st.stop()
    st.session_state["last"] = {"rec": rec, "collection": collection, "method": method, "use_rag": use_rag}

last = st.session_state.get("last")
if last:
    rec, docs_by_id = last["rec"], get_docs(last["collection"])[1]
    left, right = st.columns([3, 2])
    with left:
        st.subheader("答案")
        if rec["abstained"]:
            st.warning(f"模型拒答：{rec['answer']}")
        else:
            st.success(rec["answer"] or "（空白）")
        if rec["gold_answers"]:
            ok = bool(rec["contains_gold"]) or (rec["f1_qstrip"] or 0) >= 0.5
            st.write(f"自動評分：{'✅ 答對' if ok else '❌ 答錯'}（contains_gold={rec['contains_gold']}，"
                     f"f1_qstrip={rec['f1_qstrip']}）")
    with right:
        st.subheader("紀錄")
        st.write(f"模式：`{rec['mode']}`　模型：`{rec['gen_model']}`")
        if last["use_rag"]:
            st.write(f"索引：`{rec['index_key']}`")
            st.write(f"引用：{', '.join(rec['cited_ids']) or '（沒有引用）'}")
            if rec["invalid_citations"]:
                st.error(f"引用了不在 Context 裡的 ID：{rec['invalid_citations']}")
            if rec.get("gold_doc_ids"):
                st.write(f"正確段落找到 {rec['gold_found']}/{len(rec['gold_doc_ids'])}")
        st.write(f"耗時 {rec['latency_s']}s，prompt {rec['prompt_tokens']} tokens")

    if last["use_rag"]:
        st.subheader(f"取回的段落（{last['method']}，Top-{len(rec['retrieved'])}）")
        for i, h in enumerate(rec["retrieved"], 1):
            d = docs_by_id[h["doc_id"]]
            tags = []
            if h["doc_id"] in rec.get("gold_doc_ids", []):
                tags.append("⭐ 正確段落")
            if h["doc_id"] in rec["cited_ids"]:
                tags.append("📎 被引用")
            if h["doc_id"] in rec["dropped_ids"]:
                tags.append("✂ 超過字數上限，沒放進 Context")
            with st.expander(f"#{i}　{h['doc_id']}　score {h['score']}　{'　'.join(tags)}", expanded=i == 1):
                st.caption(d["title"])
                st.write(d["text"])

    with st.expander("實際送給模型的 Prompt"):
        if last["use_rag"]:
            hits = [dict(docs_by_id[h["doc_id"]], score=h["score"]) for h in rec["retrieved"]]
            context, _, _ = rc.build_context(hits, CONTEXT_CHAR_BUDGET)
            st.code(rc.SYSTEM_RAG, language=None)
            st.code(f"Context:\n{context}\n\nQuestion: {rec['question']}", language=None)
        else:
            st.code(rc.SYSTEM_NO_RAG, language=None)
            st.code(f"Question: {rec['question']}", language=None)
    with st.expander("模型原始輸出"):
        st.code(rec["raw_output"], language=None)
