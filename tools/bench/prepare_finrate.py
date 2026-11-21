"""Fin-RATE (GGLabYale/Fin-RATE, MIT; Jiang et al., 2026) over chunked SEC filings.
  finrate_xdoc : EC-QA (cross-company comparison) + LT-QA (cross-year tracking) -> cross-document, every gold chunk required
  finrate_dr   : DR-QA (detail & reasoning inside one filing section) -> single-document
    python -m tools.bench.prepare_finrate"""
import itertools, json, zipfile
from tools.bench.common import finalize, hf_file, split_questions

REPO = "GGLabYale/Fin-RATE"


def main():
    zp, sha = hf_file(REPO, "corpus/corpus.zip")
    corpus = []
    with zipfile.ZipFile(zp) as zf:
        with zf.open("corpus.jsonl") as fh:
            for line in fh:
                r = json.loads(line)
                if (r.get("text") or "").strip():
                    corpus.append({"doc_id": "fr_" + r["_id"], "title": r.get("title") or "", "text": r["text"]})
    ids = {d["doc_id"] for d in corpus}

    def load(fn, tag):
        p, _ = hf_file(REPO, f"qa/{fn}")
        out, miss = [], 0
        for r in json.load(open(p, encoding="utf-8")):
            gold = [r["doc_id"]] if r.get("doc_id") else list(dict.fromkeys(r.get("doc_ids") or []))
            gold = ["fr_" + g for g in gold]
            if not gold or any(g not in ids for g in gold):
                miss += 1; continue
            out.append({"qid": f"fr_{r['q_id']}", "question": r["question"], "answers": [r["answer"]], "gold_doc_ids": gold,
                        "type": tag, "hops": len(gold), "key_points": r.get("key_points")})
        return out, miss
    ec, m_ec = load("EC-QA.json", "cross_company")
    lt, m_lt = load("LT-QA.json", "cross_year")
    dr, m_dr = load("DR-QA.json", "single_doc")
    title = {d["doc_id"]: d["title"] for d in corpus}
    filing = lambda g: "_".join(title[g].split("_")[:3])      # ticker_date_form = one SEC filing
    xdoc, capped = [], 0
    for q in ec + lt:
        groups = list(dict.fromkeys(filing(g) for g in q["gold_doc_ids"]))
        if len(groups) < 2:
            continue
        per = [[g for g in q["gold_doc_ids"] if filing(g) == f] for f in groups]
        n = 1
        for x in per: n *= len(x)
        if n <= 512:   # success = >=1 gold chunk from EVERY gold filing (cross-document requirement)
            q["gold_sets"] = [list(t) for t in itertools.product(*per)]
        else:
            capped += 1  # too many combinations -> keep strict "every chunk" rule
        q["hops"] = len(groups)
        xdoc.append(q)
    src = {"repo": REPO, "revision": sha, "license": "MIT (Fin-RATE, Jiang et al., arXiv:2602.07294)", "domain": "finance"}
    d1, h1, _ = split_questions(xdoc)
    finalize("finrate_xdoc", corpus, d1, h1, src,
             notes=f"v2: EC-QA + LT-QA whose gold chunks span >=2 SEC filings (cross-company / cross-year). Success = >=1 gold chunk from every gold filing (gold_sets); strict rule kept for {capped} questions with >512 combinations. v1 required every chunk (up to 15) and was unreachable at k=10. missing-gold dropped: EC {m_ec}, LT {m_lt}; "
                   f"single-filing EC/LT dropped: {len(ec) + len(lt) - len(xdoc)}")
    d2, h2, _ = split_questions(dr)
    finalize("finrate_dr", corpus, d2, h2, src, notes=f"DR-QA single-document questions. missing-gold dropped: {m_dr}")


if __name__ == "__main__":
    main()
