"""GPL / InPars-style synthetic training questions from a target corpus (no dev/holdout questions are read except for a
near-duplicate guard). Each sample needs TWO related documents (cross-document), mirroring the bottleneck found in
MultiHop-RAG / NeoQA: the second document is only relevant in combination with the first.

    python -m tools.train.gen_synth_queries --dataset neoqa --n 1500 [--model qwen3:4b-instruct]
Pairs: same `--group-field` value (e.g. timeline_id) if given, else a BM25 neighbour of the anchor document.
Output: bench/train/<dataset>_synth.jsonl  {question, positives:[{doc_id,title,text}x2], negatives:[...8 BM25 neighbours]}
"""
import argparse, json, random, re, sys, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.rag_core import OllamaGenerator  # noqa: E402
from src.retrievers.bm25 import BM25Retriever  # noqa: E402
from tools.train.common import SEED  # noqa: E402
from tools.progress import report, start_parent_watch  # noqa: E402

SYS = """You write exam questions about news articles.
Write ONE question that can only be answered by combining a fact from Article A with a fact from Article B.
- Mention specific names, organisations, places or outlets so the question is searchable; do not copy whole sentences.
- The answer must be short (a few words) and appear in the articles.
Reply exactly:
Question: <question>
Answer: <short answer>"""


def words(s):
    return set(re.findall(r"\w+", s.lower()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--model", default="qwen3:4b-instruct")
    ap.add_argument("--group-field", default="")
    ap.add_argument("--meta", default="", help="fields shown to the LLM, e.g. source,published_at")
    ap.add_argument("--max-chars", type=int, default=1800)
    a = ap.parse_args()
    start_parent_watch()
    ddir = ROOT / "bench" / "data" / a.dataset
    corpus = [json.loads(l) for l in open(ddir / "corpus.jsonl", encoding="utf-8")]
    byid = {d["doc_id"]: d for d in corpus}
    # near-duplicate guard against every eval question of this dataset family
    evalq = []
    for p in (ROOT / "bench" / "data").glob(f"{a.dataset.split('_c')[0]}*/queries_*.jsonl"):
        evalq += [words(json.loads(l)["question"]) for l in open(p, encoding="utf-8")]
    bm = BM25Retriever().build(corpus, a.dataset)
    groups = defaultdict(list)
    if a.group_field:
        for d in corpus:
            groups[d.get(a.group_field)].append(d["doc_id"])
    rnd = random.Random(SEED)
    gen = OllamaGenerator(a.model, num_ctx=8192)
    gen.check()
    out_path = ROOT / "bench" / "train" / f"{a.dataset}_synth.jsonl"
    done, rejected, t0 = 0, {"format": 0, "near_dup": 0}, time.time()
    anchors = [d["doc_id"] for d in corpus]
    rnd.shuffle(anchors)

    def show(d):
        head = " | ".join(f"{f}: {d.get(f)}" for f in a.meta.split(",") if f and d.get(f))
        return (head + "\n" if head else "") + f"Title: {d.get('title', '')}\n{d['text'][:a.max_chars]}"

    start = 0
    if out_path.exists():                     # crash-safe: continue after the last anchor already written
        prev = []
        for line in open(out_path, encoding="utf-8"):
            try:
                prev.append(json.loads(line))
            except json.JSONDecodeError:
                break
        done = len(prev)
        start = max((r.get("anchor_i", -1) for r in prev), default=-1) + 1
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:      # drop a possibly cut last line
            for r in prev:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"resuming: {done} questions already generated, next anchor {start}", flush=True)
    with open(out_path, "a", encoding="utf-8", newline="\n") as f:
        for i in range(start, len(anchors) * 3):
            if done >= a.n:
                break
            A = byid[anchors[i % len(anchors)]]
            neigh = [d for d, _ in bm.search_batch([A["title"] + " " + A["text"][:600]], 30)[0] if d != A["doc_id"]]
            if a.group_field:
                pool = [x for x in groups[A.get(a.group_field)] if x != A["doc_id"]]
                cand = [x for x in neigh if x in set(pool)][:10] or pool
            else:
                cand = neigh[:5]
            if not cand:
                continue
            B = byid[rnd.choice(cand[:10])]
            r = gen.chat(SYS, f"Article A:\n{show(A)}\n\nArticle B:\n{show(B)}")
            m = re.search(r"Question:\s*(.+)", r["text"])
            if not m or len(m.group(1)) < 20:
                rejected["format"] += 1; continue
            q = m.group(1).strip()
            qw = words(q)
            if any(len(qw & e) / max(1, len(qw | e)) > 0.6 for e in evalq):
                rejected["near_dup"] += 1; continue
            negs = [byid[x] for x in neigh if x not in (A["doc_id"], B["doc_id"])][:8]
            keep = lambda d: {"doc_id": d["doc_id"], "title": d.get("title", ""), "text": d["text"]}
            f.write(json.dumps({"qid": f"synth_{a.dataset}_{done}", "question": q, "positives": [keep(A), keep(B)],
                                "negatives": [keep(d) for d in negs], "type": "synthetic_2doc", "anchor_i": i}, ensure_ascii=False) + "\n")
            f.flush()
            done += 1
            report(done, a.n, stage=f"synthetic questions {a.dataset}")
            if done % 100 == 0:
                print(f"{done} questions, {time.time() - t0:.0f}s, rejected {rejected}", flush=True)
    info = {"dataset": a.dataset, "n": done, "rejected": rejected, "model": a.model, "group_field": a.group_field,
            "minutes": round((time.time() - t0) / 60, 1), "eval_questions_checked": len(evalq), "near_dup_threshold": 0.6}
    (ROOT / "bench" / "train" / f"{a.dataset}_synth.info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
