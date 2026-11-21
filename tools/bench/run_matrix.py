"""Run a datasets x methods grid, one subprocess per cell (frees the GPU between models). Resumable.

    python -m tools.bench.run_matrix --datasets hotpotqa,musique --methods bge-small,bm25 [--split dev] [--limit N]
State: results/bench/_index.jsonl (one line per attempt). A cell with an "ok" line for the same method hash is skipped;
a cell that already failed twice is skipped too.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.bench import methods as _m
from tools.bench.methods import DATASETS, METHODS, PHASE_A  # noqa: E402
from tools.bench.run_retrieval import config_hash  # noqa: E402
from tools.progress import start_parent_watch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--methods", default=",".join(PHASE_A), help="comma list, or a list name in methods.py e.g. PHASE_B1")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--allow-holdout", action="store_true", help="passed through to run_retrieval (final confirmation only)")
    ap.add_argument("--out", default="results/bench")
    a = ap.parse_args()
    start_parent_watch()
    if hasattr(_m, a.methods):
        a.methods = ",".join(getattr(_m, a.methods))
    idx = ROOT / a.out / "_index.jsonl"
    idx.parent.mkdir(parents=True, exist_ok=True)
    past = [json.loads(l) for l in idx.read_text(encoding="utf-8").splitlines()] if idx.exists() else []
    def data_sha(ds):
        m = json.loads((ROOT / "bench" / "data" / ds / "manifest.json").read_text(encoding="utf-8"))
        return m["sha256"]["corpus.jsonl"][:12] + ":" + m["sha256"][f"queries_{a.split}.jsonl"][:12]

    def rec_sha(p):   # old index lines have no data_sha -> read it from the run's config.json
        if "data_sha" not in p and p.get("run_id"):
            try:
                c = json.loads((ROOT / a.out / p["run_id"] / "config.json").read_text(encoding="utf-8"))
                p["data_sha"] = c["corpus_sha256"][:12] + ":" + c["queries_sha256"][:12]
            except Exception:
                p["data_sha"] = None
        return p.get("data_sha")

    for method in a.methods.split(","):          # method outer loop: model weights stay in the OS file cache
        for ds in a.datasets.split(","):
            h = config_hash(METHODS[method])
            if not (ROOT / "bench" / "data" / ds / "manifest.json").exists():
                print(f"no data for {ds}", flush=True)
                continue
            dsha = data_sha(ds)   # a rebuilt dataset (new corpus / queries) invalidates old results
            same = [p for p in past if (p["dataset"], p["method"], p["split"], p["method_hash"], p.get("limit", 0)) == (ds, method, a.split, h, a.limit)
                    and rec_sha(p) in (dsha, None if p["status"] != "ok" else "x")]
            if any(p["status"] == "ok" for p in same) or sum(p["status"] != "ok" for p in same) >= 2:
                print(f"skip {ds} / {method}", flush=True)
                continue
            if not (ROOT / "bench" / "data" / ds / "manifest.json").exists():
                print(f"no data for {ds}", flush=True)
                continue
            t0 = time.time()
            cmd = [sys.executable, "-m", "tools.bench.run_retrieval", "--dataset", ds, "--method", method, "--split", a.split, "--out", a.out]
            if a.limit:
                cmd += ["--limit", str(a.limit)]
            if a.allow_holdout:
                cmd += ["--allow-holdout"]
            p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
            last = (p.stdout.strip().splitlines() or [""])[-1]
            rec = {"dataset": ds, "method": method, "split": a.split, "method_hash": h, "limit": a.limit, "data_sha": dsha,
                   "status": "ok" if p.returncode == 0 else "failed", "seconds": round(time.time() - t0, 1)}
            if p.returncode == 0:
                rec.update(json.loads(last))
            else:
                rec["error"] = p.stderr.strip()[-1500:]
            past.append(rec)
            with open(idx, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(json.dumps(rec, ensure_ascii=False)[:600], flush=True)


if __name__ == "__main__":
    main()
