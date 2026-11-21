"""Diagnose a reranker that ranks worse than its first stage: re-score saved first-stage candidates with
{base, fine-tuned} x {fp32, fp16} x {sigmoid, raw logit} and report all_gold_hit@10, score saturation and ties.
    python -m tools.train.diag_reranker --model models/rr-m3-dom --datasets medqa,bioasq,finrate_dr --n 100"""
import argparse, json
import numpy as np
from tools.progress import report, start_parent_watch
from tools.train.common import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--first", default="wrrf_basedom_bm25")
    ap.add_argument("--datasets", default="medqa,bioasq,finrate_dr")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--depth", type=int, default=30)
    a = ap.parse_args()
    start_parent_watch()
    import torch
    from sentence_transformers import CrossEncoder
    R = ROOT / "results" / "bench"
    runs = {}
    for l in open(R / "_index.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["status"] == "ok" and r.get("limit", 0) == 0 and r.get("split", "dev") == "dev":
            runs[(r["dataset"], r["method"])] = r["run_id"]
    out = {}
    combos = [(m, prec) for m in (a.base, a.model) for prec in ("fp32", "fp16")]
    for ci, (mname, prec) in enumerate(combos):
        ce = CrossEncoder(mname, max_length=512, device="cuda", activation_fn=torch.nn.Identity())
        if prec == "fp16":
            ce.model.half()
        for ds in a.datasets.split(","):
            text = {}
            for l in open(ROOT / "bench" / "data" / ds / "corpus.jsonl", encoding="utf-8"):
                d = json.loads(l); text[d["doc_id"]] = ((d.get("title") or "") + "\n" + d["text"]).strip()
            Q = {json.loads(l)["qid"]: json.loads(l)["question"] for l in open(ROOT / "bench" / "data" / ds / "queries_dev.jsonl", encoding="utf-8")}
            rows = [json.loads(l) for l in open(R / runs[(ds, a.first)] / "retrieval.jsonl", encoding="utf-8")][:a.n]
            hit_l, hit_s, first_hit, logits = 0, 0, 0, []
            for j in rows:
                cand = [d for d, _ in j["retrieved"][:a.depth]]
                s = np.asarray(ce.predict([(Q[j["qid"]], text[d]) for d in cand], batch_size=32, show_progress_bar=False), dtype=np.float64)
                logits += s.tolist()
                sets = [set(x) for x in (j.get("gold_sets") or [j["gold_doc_ids"]])]
                ok = lambda order: any(x <= set(order[:10]) for x in sets)
                hit_l += ok([cand[i] for i in np.argsort(-s, kind="stable")])
                sig = np.float16(1 / (1 + np.exp(-s))) if prec == "fp16" else 1 / (1 + np.exp(-s))
                hit_s += ok([cand[i] for i in np.argsort(-np.asarray(sig, dtype=np.float64), kind="stable")])
                first_hit += ok(cand)
            L = np.asarray(logits)
            out[f"{mname}|{prec}|{ds}"] = {"first_stage@10": first_hit / len(rows), "rerank_logit@10": hit_l / len(rows), "rerank_sigmoid@10": hit_s / len(rows),
                                           "logit_mean": round(float(L.mean()), 2), "logit_p5": round(float(np.percentile(L, 5)), 2),
                                           "logit_p95": round(float(np.percentile(L, 95)), 2), "frac_logit_gt_5": round(float((L > 5).mean()), 3)}
            print(json.dumps({f"{mname}|{prec}|{ds}": out[f"{mname}|{prec}|{ds}"]}), flush=True)
        report(ci + 1, len(combos), stage=f"{mname} {prec}")
        del ce; torch.cuda.empty_cache()
    p = ROOT / "results" / "diag_reranker.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
