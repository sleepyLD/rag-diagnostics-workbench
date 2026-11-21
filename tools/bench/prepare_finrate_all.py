"""Fin-RATE full evaluation set: EVERY DR-QA / EC-QA / LT-QA question whose gold chunks are in the corpus (paper: 2,500 each),
written as queries_all.jsonl (split "all"); dev/holdout files are only a formal partition required by finalize().
EC/LT questions whose gold spans >=2 filings get gold_sets = >=1 gold chunk per filing (same rule as finrate_xdoc v2);
recall@k always uses every gold chunk (= the paper's Recall@K).
    python -m tools.bench.prepare_finrate_all            -> bench/data/finrate_all
    python -m tools.bench.prepare_finrate_all --variant finrate_xdoc_named2_c256m   -> bench/data/finrate_all_named2_c256m"""
import argparse, hashlib, itertools, json, shutil, zipfile
from tools.bench.common import DATA, finalize, hf_file, norm_q

REPO = "GGLabYale/Fin-RATE"


def build():
    zp, sha = hf_file(REPO, "corpus/corpus.zip")
    corpus = []
    with zipfile.ZipFile(zp) as zf, zf.open("corpus.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            if (r.get("text") or "").strip():
                corpus.append({"doc_id": "fr_" + r["_id"], "title": r.get("title") or "", "text": r["text"]})
    ids = {d["doc_id"] for d in corpus}
    title = {d["doc_id"]: d["title"] for d in corpus}
    filing = lambda g: "_".join(title[g].split("_")[:3])
    allq, miss = [], {}
    for fn, tag in (("DR-QA.json", "dr"), ("EC-QA.json", "ec"), ("LT-QA.json", "lt")):
        p, _ = hf_file(REPO, f"qa/{fn}")
        miss[tag] = 0
        for r in json.load(open(p, encoding="utf-8")):
            gold = [r["doc_id"]] if r.get("doc_id") else list(dict.fromkeys(r.get("doc_ids") or []))
            gold = ["fr_" + g for g in gold]
            if not gold or any(g not in ids for g in gold):
                miss[tag] += 1; continue
            q = {"qid": f"fr_{r['q_id']}", "question": r["question"], "answers": [r["answer"]], "gold_doc_ids": gold,
                 "type": tag, "hops": len(gold)}
            groups = list(dict.fromkeys(filing(g) for g in gold))
            if tag != "dr" and len(groups) >= 2:
                per = [[g for g in gold if filing(g) == f] for f in groups]
                n = 1
                for x in per: n *= len(x)
                if n <= 512:
                    q["gold_sets"] = [list(t) for t in itertools.product(*per)]
                q["type"] = tag + "_multi"
            allq.append(q)
    seen, uniq = set(), []
    for q in allq:
        k = norm_q(q["question"])
        if q["qid"] not in seen and k not in seen:
            seen.add(q["qid"]); seen.add(k); uniq.append(q)
    half = len(uniq) // 2
    finalize("finrate_all", corpus, uniq[:half], uniq[half:],
             {"repo": REPO, "revision": sha, "license": "MIT (Fin-RATE, Jiang et al., arXiv:2602.07294)", "domain": "finance"},
             extra_files={"queries_all.jsonl": uniq},
             notes=f"Full Fin-RATE question set (split 'all'); missing-gold dropped: {miss}. dev/holdout files are a formal split only.")


def variant(src):
    s, a = DATA / src, DATA / "finrate_all"
    dst = DATA / f"finrate_all_{src.split('finrate_xdoc_', 1)[1]}"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(s / "corpus.jsonl", dst / "corpus.jsonl")
    for f in a.glob("queries_*.jsonl"):
        shutil.copyfile(f, dst / f.name)
    m = json.loads((s / "manifest.json").read_text(encoding="utf-8"))
    m.update({"name": dst.name, "derived_from": [src, "finrate_all"],
              "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in dst.glob("*.jsonl")}})
    (dst / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    print(dst.name, "ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", default="")
    a = ap.parse_args()
    variant(a.variant) if a.variant else build()
