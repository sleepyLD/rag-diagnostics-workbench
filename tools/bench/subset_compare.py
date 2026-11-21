"""Compare methods on the same question subset (the qids of a --limit run), recomputing from retrieval.jsonl.
    python -m tools.bench.subset_compare --dataset musique --ref-run <run_id with the subset> --methods a,b,c"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.bench import metrics as M  # noqa: E402


def latest_runs():
    out = {}
    for l in open(ROOT / "results/bench/_index.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["status"] == "ok":
            out[(r["dataset"], r["method"], r.get("limit", 0))] = r["run_id"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--methods", required=True)
    a = ap.parse_args()
    runs = latest_runs()
    rows = {}
    for m in a.methods.split(","):
        rid = runs.get((a.dataset, m, a.limit)) or runs.get((a.dataset, m, 0))
        if rid:
            rows[m] = {r["qid"]: r for r in M.load_rows(ROOT / "results/bench" / rid)}
    common = set.intersection(*[set(v) for v in rows.values()]) if rows else set()
    print(f"{a.dataset}: {len(common)} common questions")
    for m, rr in rows.items():
        sub = [rr[q] for q in sorted(common)]
        mm = M.compute(sub)["overall"]
        lat = json.loads((ROOT / "results/bench" / (runs.get((a.dataset, m, a.limit)) or runs.get((a.dataset, m, 0))) / "metrics.json").read_text())["latency"]
        print(f"  {m:22s} all@5 {mm['all_gold_hit@5']:.3f}  all@10 {mm['all_gold_hit@10']:.3f}  recall@10 {mm['recall@10']:.3f}  p50 {lat['total_ms_p50']:.0f} ms")


if __name__ == "__main__":
    main()
