"""Run one retrieval method on one bench dataset.

    python -m tools.bench.run_retrieval --dataset hotpotqa --method bge-small --split dev
    (--method: a name in tools/bench/methods.py or a path to a JSON config; holdout needs --allow-holdout)

Writes results/bench/<run_id>/{config.json, retrieval.jsonl, metrics.json}.
Latency: after a warm-up, LAT_N questions are searched one at a time (batch = 1); p50 / p95 per stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.retrievers import aggregate_to_source, build_retriever  # noqa: E402
from tools.bench import metrics as M  # noqa: E402
from tools.bench.common import read_jsonl  # noqa: E402
from tools.bench.methods import METHODS  # noqa: E402
from tools.progress import report, start_parent_watch, ten_percent_batch  # noqa: E402

TOP_K, WARMUP, BATCH = 100, 20, 64
LAT_N = int(os.environ.get("BENCH_LAT_N", 200))     # heavy rerankers: fewer latency samples


def pct(xs, p):
    return round(float(np.percentile(xs, p)), 3)


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "holdout", "all", "null"])  # all: full-benchmark sets (finrate_all)
    ap.add_argument("--allow-holdout", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/bench")
    args = ap.parse_args(argv)
    start_parent_watch()
    if args.split == "holdout" and not args.allow_holdout:
        raise SystemExit("holdout is reserved for the final run; pass --allow-holdout on purpose")
    cfg = METHODS[args.method] if args.method in METHODS else json.loads(Path(args.method).read_text(encoding="utf-8"))
    if not any(t in json.dumps(cfg) for t in ('"decomp"', '"iter"')):   # no LLM needed -> free Ollama's VRAM first
        try:
            import urllib.request
            ps = json.loads(urllib.request.urlopen("http://localhost:11434/api/ps", timeout=3).read())
            for m in ps.get("models", []):
                req = urllib.request.Request("http://localhost:11434/api/generate", method="POST",
                                             data=json.dumps({"model": m["name"], "keep_alive": 0}).encode(),
                                             headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=30).read()
                print(f"unloaded Ollama model {m['name']} to free VRAM", file=sys.stderr, flush=True)
        except Exception:
            pass
    skip = ROOT / "results" / "bench" / "_skip.json"          # {"method": ["dataset", ...]} — lets the main reviewer trim a running matrix
    if skip.exists() and args.dataset in json.loads(skip.read_text(encoding="utf-8")).get(cfg.get("name", args.method), []):
        print(f"SKIPPED by results/bench/_skip.json: {args.dataset} / {args.method}", file=sys.stderr)
        return 3
    ddir = ROOT / "bench" / "data" / args.dataset
    corpus = read_jsonl(ddir / "corpus.jsonl")
    queries = read_jsonl(ddir / f"queries_{args.split}.jsonl")
    if args.limit:
        queries = queries[:args.limit]

    try:
        import torch
        cuda = torch.cuda.is_available()
    except Exception:
        torch, cuda = None, False
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    r = build_retriever(cfg).build(corpus, args.dataset)
    build_wall = round(time.time() - t0, 2)
    vram_build = round(torch.cuda.max_memory_allocated() / 2**20, 1) if cuda else None
    source_of = {d["doc_id"]: d.get("source_doc_id", d["doc_id"]) for d in corpus}
    if not (cfg.get("aggregate") or any(v != k for k, v in source_of.items())):   # chunked corpus -> always max-pool to documents
        source_of = None

    def search(qs, k):
        res = r.search_batch(qs, k * 5 if source_of else k)
        return [aggregate_to_source(h, source_of, k) for h in res] if source_of else res

    if cuda:
        torch.cuda.reset_peak_memory_stats()
    qtext = [q["question"] for q in queries]
    # ---- crash-safe main pass: every finished batch is appended (fsync) to results/bench/_partial/<key>.jsonl;
    #      a re-run of the same dataset/method/split/limit skips the questions already in that file.
    part_dir = ROOT / args.out / "_partial"
    part_dir.mkdir(parents=True, exist_ok=True)
    ppath = part_dir / f"{args.dataset}__{config_hash(cfg)}__{args.split}__{args.limit}.jsonl"
    done = {}
    if ppath.exists():
        for line in open(ppath, encoding="utf-8"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:      # last line cut by a crash
                break
            done[row["qid"]] = row
        print(f"resuming: {len(done)} / {len(queries)} questions already done ({ppath.name})", file=sys.stderr, flush=True)
    todo = [i for i, q in enumerate(queries) if q["qid"] not in done]
    BATCH_ = ten_percent_batch(len(todo), int(os.environ.get("BENCH_BATCH", BATCH)))   # checkpoint every <= ~10% of the questions
    stage = f"{args.dataset}/{cfg.get('name', args.method)}"
    report(len(done), len(queries), stage=stage + " (main pass)")
    crash_after = int(os.environ.get("BENCH_CRASH_AFTER_BATCHES", 0))      # test hook only
    t0 = time.time()
    with open(ppath, "a", encoding="utf-8", newline="\n") as pf:
        for nb, b in enumerate(range(0, len(todo), BATCH_), 1):
            idx = todo[b:b + BATCH_]
            n_tr = len(getattr(r, "trace", None) or [])
            res = search([qtext[i] for i in idx], TOP_K)
            new_tr = (getattr(r, "trace", None) or [])[n_tr:]
            for j, (i, h) in enumerate(zip(idx, res)):
                row = {"qid": queries[i]["qid"], "hits": [[d, float(sc)] for d, sc in h]}
                if j < len(new_tr):
                    row["trace"] = new_tr[j]
                done[row["qid"]] = row
                pf.write(json.dumps(row, ensure_ascii=False) + "\n")
            pf.flush()
            os.fsync(pf.fileno())
            report(len(done), len(queries), stage=stage + " (main pass)")
            if crash_after and nb >= crash_after:
                os._exit(9)
    batch_s = time.time() - t0
    n_new = len(todo)
    hits = [done[q["qid"]]["hits"] for q in queries]
    traces = [done[q["qid"]].get("trace") for q in queries]

    for q in qtext[:min(WARMUP, LAT_N)]:
        search([q], TOP_K)
    lat, stages = [], {"encode_ms": [], "search_ms": [], "post_ms": []}
    for li, q in enumerate(qtext[:LAT_N]):
        if li % max(1, LAT_N // 10) == 0:
            report(li, min(LAT_N, len(qtext)), stage=stage + " (latency)")
        t = time.perf_counter()
        search([q], TOP_K)
        if cuda:
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t) * 1000.0)
        for s in stages:
            stages[s].append(r.last_timing.get(s, 0.0))
    vram_search = round(torch.cuda.max_memory_allocated() / 2**20, 1) if cuda else None

    run_id = f"{datetime.now():%Y%m%d_%H%M%S}_{args.dataset}_{cfg.get('name', args.method)}_{args.split}"
    out = ROOT / args.out / run_id
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (q, h) in enumerate(zip(queries, hits)):
        rows.append({"qid": q["qid"], "gold_doc_ids": q["gold_doc_ids"], **({"gold_sets": q["gold_sets"]} if q.get("gold_sets") else {}),
                     "type": q.get("type"), "hops": q.get("hops"), "latency_ms": round(lat[i], 3) if i < len(lat) else None,
                     "retrieved": [[d, round(s, 5)] for d, s in h]})
    with open(out / "retrieval.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = json.loads((ddir / "manifest.json").read_text(encoding="utf-8"))
    config = {"run_id": run_id, "dataset": args.dataset, "split": args.split, "method": cfg, "method_hash": config_hash(cfg),
              "top_k": TOP_K, "n_queries": len(queries), "n_corpus": len(corpus), "corpus_sha256": manifest["sha256"]["corpus.jsonl"],
              "queries_sha256": manifest["sha256"][f"queries_{args.split}.jsonl"], "index_key": getattr(r, "index_key", None),
              "torch": getattr(torch, "__version__", None), "gpu": torch.cuda.get_device_name(0) if cuda else None,
              "latency_protocol": f"warmup {WARMUP}, then first {min(LAT_N, len(qtext))} questions at batch=1, top_k={TOP_K}"}
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")
    if any(t is not None for t in traces):   # LLM decomposition: sub-questions and intermediate answers, for error analysis
        with open(out / "trace.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for q, t in zip(queries, traces):
                f.write(json.dumps({"qid": q["qid"], **(t or {})}, ensure_ascii=False) + "\n")
    m = M.compute(rows)
    m["latency"] = {"n": len(lat), "device": "cuda" if cuda else "cpu", "total_ms_p50": pct(lat, 50), "total_ms_p95": pct(lat, 95),
                    "total_ms_mean": round(float(np.mean(lat)), 3),
                    **{f"{s[:-3]}_ms_p50": pct(v, 50) for s, v in stages.items()}, **{f"{s[:-3]}_ms_p95": pct(v, 95) for s, v in stages.items()},
                    "batch_qps": round(n_new / batch_s, 1) if n_new and batch_s > 0 else None,
                    "resumed_questions": len(queries) - n_new}
    m["index"] = {**r.stats, "build_wall_s": build_wall, "index_mb": round(r.stats.get("index_bytes", 0) / 2**20, 2),
                  "vram_build_peak_mb": vram_build, "vram_search_peak_mb": vram_search}
    (out / "metrics.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    try:
        ppath.unlink()                        # run finished: the partial file is no longer needed
    except OSError:
        pass
    print(json.dumps({"run_id": run_id, **{k: m["overall"][k] for k in ("recall@10", "all_gold_hit@10", "ndcg@10")},
                      "p50_ms": m["latency"]["total_ms_p50"], "p95_ms": m["latency"]["total_ms_p95"], "index_mb": m["index"]["index_mb"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
