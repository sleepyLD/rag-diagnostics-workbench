"""Run run_qa on the latest ok retrieval run of (dataset, method, split).
    python -m tools.bench.run_qa_latest --dataset multihop_rag_c256m --method rr-bgev2m3_wrrf --split null [--limit 0] [--k 5]"""
import argparse, json, subprocess, sys
from tools.bench.common import ROOT

ap = argparse.ArgumentParser()
for a in ("--dataset", "--method", "--split"):
    ap.add_argument(a, required=True)
ap.add_argument("--limit", default="0"); ap.add_argument("--k", default="5")
a = ap.parse_args()
rid = None
for l in open(ROOT / "results" / "bench" / "_index.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r["status"] == "ok" and (r["dataset"], r["method"], r.get("split", "dev"), r.get("limit", 0)) == (a.dataset, a.method, a.split, 0):
        rid = r["run_id"]
assert rid, f"no ok run for {a.dataset}/{a.method}/{a.split}"
print("run", rid, flush=True)
sys.exit(subprocess.call([sys.executable, "-m", "tools.bench.run_qa", f"results/bench/{rid}", "--k", a.k, "--limit", a.limit], cwd=str(ROOT)))
