"""End-to-end QA on top of a finished retrieval run (reuses src/rag_core: prompt, Ollama generator, parser, scorer).

    python -m tools.bench.run_qa results/bench/<run_id> [--k 5] [--mode rag|oracle] [--model qwen3:4b-instruct] [--limit 0]
Writes results/bench/<run_id>/qa_<mode>_k<k>_<model>.jsonl and qa_<...>_summary.json.
- Chunked corpora: the k best *chunks* of the retrieved documents are used as passages.
- "Unanswerable" gold (NeoQA false-premise / uncertain): counted correct iff the model abstains.
- oracle: passages = gold documents (one gold set), to separate retrieval from generation failures.
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import rag_core as rc  # noqa: E402
from tools.bench.common import read_jsonl  # noqa: E402

UNANSWERABLE = {"unanswerable", "insufficient information", "insufficient_evidence"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--mode", default="rag", choices=["rag", "oracle"])
    ap.add_argument("--model", default="qwen3:4b-instruct")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--char-budget", type=int, default=12000)
    a = ap.parse_args()
    run = Path(a.run) if Path(a.run).is_absolute() else ROOT / a.run
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    ddir = ROOT / "bench" / "data" / cfg["dataset"]
    corpus = {d["doc_id"]: d for d in read_jsonl(ddir / "corpus.jsonl")}
    by_src = {}
    for d in corpus.values():
        by_src.setdefault(d.get("source_doc_id", d["doc_id"]), []).append(d)
    qs = {q["qid"]: q for q in read_jsonl(ddir / f"queries_{cfg['split']}.jsonl")}
    rows = [json.loads(l) for l in open(run / "retrieval.jsonl", encoding="utf-8")]
    if a.limit:
        rows = rows[:a.limit]
    gen = rc.OllamaGenerator(a.model, num_ctx=8192)
    gen.check()
    tag = f"qa_{a.mode}_k{a.k}_{a.model.replace(':', '-')}"
    out, n_ok, t0 = [], 0, time.time()

    def passages(doc_ids):
        hits = []
        for i, did in enumerate(doc_ids):
            d = corpus.get(did) or (by_src.get(did) or [None])[0]
            if d is None:
                continue
            text = d["text"] if did in corpus else " ".join(c["text"] for c in by_src[did])[:3000]
            hits.append({"rank": i + 1, "doc_id": did, "score": None, "title": d.get("title", ""), "text": text})
        return hits

    for r in rows:
        q = qs[r["qid"]]
        ids = ([s for s in (q.get("gold_sets") or [q["gold_doc_ids"]])][0] if a.mode == "oracle" else [d for d, _ in r["retrieved"][:a.k]])
        ctx, used, _ = rc.build_context(passages(ids), a.char_budget)
        res = gen.chat(rc.SYSTEM_RAG, f"Context:\n{ctx}\n\nQuestion: {q['question']}")
        ans, cited, _ = rc.parse_output(res["text"])
        golds = q.get("answers") or []
        if golds and golds[0].strip().lower() in UNANSWERABLE:
            correct = rc.is_abstain(ans)
            sc = {"correct": bool(correct)}
        else:
            sc = rc.score(ans, golds, q["question"])     # project rule (README): contains gold, or F1 after removing question words >= 0.5
            correct = (not rc.is_abstain(ans)) and bool(sc["contains_gold"] or (sc["f1_qstrip"] or 0) >= 0.5)
        n_ok += bool(correct)
        out.append({"qid": r["qid"], "type": q.get("type"), "answer": ans, "gold": golds, "correct": bool(correct),
                    "gold_in_context": any(set(s) <= set(used) for s in (q.get("gold_sets") or [q["gold_doc_ids"]])),
                    "latency_s": res["latency_s"], "raw": res["text"][:400]})
    with open(run / f"{tag}.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    summ = {"run": run.name, "mode": a.mode, "k": a.k, "model": a.model, "n": len(out), "accuracy": round(n_ok / max(1, len(out)), 4),
            "acc_when_gold_in_context": round(sum(o["correct"] for o in out if o["gold_in_context"]) / max(1, sum(o["gold_in_context"] for o in out)), 4),
            "minutes": round((time.time() - t0) / 60, 1)}
    (run / f"{tag}_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
