"""All finished full-dev runs -> one long table (recomputed from retrieval.jsonl).
    python -m tools.bench.export_tables --out docs/retriever_study/tables_dev.json"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.bench import metrics as M  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    latest = {}
    for l in open(ROOT / "results/bench/_index.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["status"] == "ok" and r.get("limit", 0) == 0 and r["split"] == a.split:
            latest[(r["dataset"], r["method"])] = r["run_id"]
    rows = []
    for (ds, m), rid in sorted(latest.items()):
        d = ROOT / "results/bench" / rid
        if not (d / "metrics.json").exists():
            continue
        mm = M.compute(M.load_rows(d))["overall"]
        lat = json.loads((d / "metrics.json").read_text())["latency"]
        rows.append({"dataset": ds, "method": m, "n": M.compute(M.load_rows(d))["n"], "all@5": mm["all_gold_hit@5"], "all@10": mm["all_gold_hit@10"],
                     "recall@5": mm["recall@5"], "recall@10": mm["recall@10"], "p50_ms": lat["total_ms_p50"], "run_id": rid})
    Path(ROOT / a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=0), encoding="utf-8")
    print(len(rows), "rows;", len({r['dataset'] for r in rows}), "datasets;", len({r['method'] for r in rows}), "methods")


if __name__ == "__main__":
    main()
