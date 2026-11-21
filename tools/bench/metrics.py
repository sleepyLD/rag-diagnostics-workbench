"""Retrieval metrics, recomputed from results/bench/<run_id>/retrieval.jsonl only.

    python -m tools.bench.metrics results/bench/<run_id>                      # print + rewrite metrics.json (keeps latency/index blocks)
    python -m tools.bench.metrics results/bench/<run_a> --compare results/bench/<run_b> [--metric all_gold_hit@10]

Row format: {"qid", "gold_doc_ids", "gold_sets"?, "type", "hops", "retrieved": [[doc_id, score], ...], "latency_ms"?}
- recall@k        share of gold_doc_ids inside the top k
- all_gold_hit@k  1 if EVERY gold doc is inside the top k (any one of gold_sets when a question has alternatives)
- mrr             1 / rank of the first gold doc (0 if none in the list)
- ndcg@10         binary relevance
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

RECALL_KS = (1, 3, 5, 10, 20, 100)
ALLHIT_KS = (3, 5, 10, 20)


def query_metrics(ranked: list[str], gold: list[str], gold_sets: list[list[str]] | None = None) -> dict:
    gold_set = set(gold)
    if not gold_set:      # unanswerable / null queries: no gold evidence, retrieval metrics undefined -> 0
        return {**{f"recall@{k}": 0.0 for k in RECALL_KS}, **{f"all_gold_hit@{k}": 0.0 for k in ALLHIT_KS}, "mrr": 0.0, "ndcg@10": 0.0}
    sets = [set(s) for s in (gold_sets or [gold]) if s]
    relevant = gold_set.union(*sets) if sets else gold_set
    m = {}
    for k in RECALL_KS:
        top = set(ranked[:k])
        m[f"recall@{k}"] = len(gold_set & top) / len(gold_set)
    for k in ALLHIT_KS:
        top = set(ranked[:k])
        m[f"all_gold_hit@{k}"] = float(any(s <= top for s in sets))
    first = next((i for i, d in enumerate(ranked, 1) if d in relevant), None)
    m["mrr"] = 1.0 / first if first else 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:10], 1) if d in gold_set)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold_set), 10) + 1))
    m["ndcg@10"] = dcg / idcg
    return m


def load_rows(run_dir: str | Path) -> list[dict]:
    with open(Path(run_dir) / "retrieval.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def per_query(rows: list[dict]) -> dict[str, dict]:
    return {r["qid"]: query_metrics([d for d, _ in r["retrieved"]], r["gold_doc_ids"], r.get("gold_sets")) for r in rows}


def mean_of(ms: list[dict]) -> dict:
    return {k: round(float(np.mean([m[k] for m in ms])), 4) for k in ms[0]} if ms else {}


def compute(rows: list[dict]) -> dict:
    pq = per_query(rows)
    out = {"n": len(rows), "overall": mean_of(list(pq.values())), "by_type": {}, "by_hops": {}}
    for field, key in (("type", "by_type"), ("hops", "by_hops")):
        groups = defaultdict(list)
        for r in rows:
            groups[str(r.get(field))].append(pq[r["qid"]])
        out[key] = {g: {"n": len(ms), **mean_of(ms)} for g, ms in sorted(groups.items())}
    return out


def paired_bootstrap(rows_a: list[dict], rows_b: list[dict], metric: str = "all_gold_hit@10", n_boot: int = 5000, seed: int = 0) -> dict:
    """Mean difference (b - a) on the shared questions with a 95% percentile interval."""
    a, b = per_query(rows_a), per_query(rows_b)
    qids = sorted(set(a) & set(b))
    diff = np.asarray([b[q][metric] - a[q][metric] for q in qids])
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"metric": metric, "n": len(qids), "mean_a": round(float(np.mean([a[q][metric] for q in qids])), 4),
            "mean_b": round(float(np.mean([b[q][metric] for q in qids])), 4), "diff_b_minus_a": round(float(diff.mean()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)], "significant": bool(lo > 0 or hi < 0)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--compare")
    ap.add_argument("--metric", default="all_gold_hit@10")
    args = ap.parse_args(argv)
    rows = load_rows(args.run_dir)
    if args.compare:
        print(json.dumps(paired_bootstrap(rows, load_rows(args.compare), args.metric), indent=1))
        return 0
    m = compute(rows)
    p = Path(args.run_dir) / "metrics.json"
    old = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    p.write_text(json.dumps({**old, **m}, indent=1), encoding="utf-8")
    print(json.dumps(m["overall"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
