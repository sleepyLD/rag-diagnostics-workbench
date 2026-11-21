"""Audit + summary: recompute metrics from every results/bench/*/retrieval.jsonl (never trusts printed numbers).

    python -m tools.bench.summarize [--split dev] [--metric all_gold_hit@10] [--out docs/retriever_study/summary_dev.md]
Checks per run: n_queries matches the dataset file, queries sha256 matches the manifest, recomputed metric == metrics.json.
Latest ok run per (dataset, method_hash) is used.
"""
import argparse, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.bench import metrics as M  # noqa: E402

SHOW = ["all_gold_hit@5", "all_gold_hit@10", "recall@10", "ndcg@10", "mrr"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--metric", default="all_gold_hit@10")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    runs, problems = {}, []
    for d in sorted((ROOT / "results" / "bench").glob(f"*_{a.split}")):
        if not (d / "metrics.json").exists():
            continue
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        if cfg.get("n_queries") is None:
            continue
        ddir = ROOT / "bench" / "data" / cfg["dataset"]
        man = json.loads((ddir / "manifest.json").read_text(encoding="utf-8"))
        rows = M.load_rows(d)
        n_file = sum(1 for _ in open(ddir / f"queries_{a.split}.jsonl", encoding="utf-8"))
        if len(rows) != n_file:          # --limit runs (smoke tests) are excluded from the table
            continue
        if cfg["queries_sha256"] != man["sha256"][f"queries_{a.split}.jsonl"]:
            problems.append(f"{d.name}: queries sha mismatch"); continue
        stored = json.loads((d / "metrics.json").read_text())
        re = M.compute(rows)
        if abs(re["overall"][a.metric] - stored["overall"][a.metric]) > 1e-4:
            problems.append(f"{d.name}: recomputed {a.metric} {re['overall'][a.metric]} != stored {stored['overall'][a.metric]}")
        runs[(cfg["dataset"], cfg["method"].get("name"), cfg["method_hash"])] = {
            "run": d.name, **{k: re["overall"][k] for k in SHOW}, "by_hops": re.get("by_hops", {}),
            "p50": stored["latency"]["total_ms_p50"], "p95": stored["latency"]["total_ms_p95"],
            "index_mb": stored["index"]["index_mb"], "build_s": stored["index"].get("build_s")}
    datasets = sorted({k[0] for k in runs})
    methods = list(dict.fromkeys(k[1] for k in sorted(runs, key=lambda k: k[1])))
    lines = [f"# 檢索結果（{a.split}，自動重算自 retrieval.jsonl）\n", f"主指標：**{a.metric}**（全部正確段落都在前 10）。latency = p50 ms（GPU、batch=1、暖機後）。\n"]
    lines.append("| 方法 | " + " | ".join(datasets) + " | 平均 | p50 ms |")
    lines.append("|---|" + "---|" * (len(datasets) + 2))
    table = {}
    for m in methods:
        cells, vals, lat = [], [], []
        for ds in datasets:
            hit = [v for k, v in runs.items() if k[0] == ds and k[1] == m]
            if hit:
                v = hit[-1]; cells.append(f"{v[a.metric]:.3f}"); vals.append(v[a.metric]); lat.append(v["p50"])
            else:
                cells.append("—")
        avg = sum(vals) / len(vals) if len(vals) == len(datasets) else None
        table[m] = {"avg": avg, "lat": sorted(lat)[len(lat) // 2] if lat else None}
        lines.append(f"| {m} | " + " | ".join(cells) + f" | {avg:.3f} |" if avg is not None else f"| {m} | " + " | ".join(cells) + " | — |")
        lines[-1] += f" {table[m]['lat']:.1f} |" if table[m]["lat"] is not None else " — |"
    lines.append("\n稽核：" + ("全部通過" if not problems else "；".join(problems)))
    txt = "\n".join(lines)
    print(txt)
    if a.out:
        Path(ROOT / a.out).write_text(txt + "\n", encoding="utf-8")
        (ROOT / a.out).with_suffix(".json").write_text(json.dumps({"|".join(k): v for k, v in runs.items()}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
