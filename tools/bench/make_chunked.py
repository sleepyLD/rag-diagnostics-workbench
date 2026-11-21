"""Derive a chunked copy of a bench dataset (long documents -> overlapping word windows).

    python -m tools.bench.make_chunked --dataset multihop_rag --words 256 --overlap 64
Writes bench/data/<dataset>_c<words>/ : corpus = chunks (doc_id = <src>#<i>, source_doc_id = <src>),
queries copied unchanged (gold stays at the source-document level; run_retrieval max-pools chunks -> documents).
"""
import argparse, hashlib, json, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def chunks(words, size, overlap):
    step = max(1, size - overlap)
    for i in range(0, max(1, len(words) - overlap), step):
        yield i, words[i:i + size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--words", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--meta", default="", help="comma list of corpus fields put in a header line of every chunk, e.g. source,published_at")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    src = ROOT / "bench" / "data" / a.dataset
    dst = ROOT / "bench" / "data" / f"{a.dataset}_c{a.words}{a.suffix}"
    dst.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in open(src / "corpus.jsonl", encoding="utf-8"):
        d = json.loads(line)
        sid = d.get("source_doc_id") or d["doc_id"]
        head = " | ".join(f"{f}: {str(d.get(f))[:10] if f.endswith('_at') else d.get(f)}" for f in a.meta.split(",") if f and d.get(f))
        for j, (_, w) in enumerate(chunks(d["text"].split(), a.words, a.overlap)):   # contextual header (Anthropic Contextual Retrieval, 2024)
            rows.append({"doc_id": f"{sid}#{j}", "title": d.get("title", ""), "text": (head + "\n" if head else "") + " ".join(w), "source_doc_id": sid})
    sha = {}
    with open(dst / "corpus.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for p in src.glob("queries_*.jsonl"):
        shutil.copyfile(p, dst / p.name)
    for p in dst.glob("*.jsonl"):
        sha[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    m = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    m.update({"name": dst.name, "derived_from": a.dataset, "chunking": {"unit": "whitespace words", "size": a.words, "overlap": a.overlap,
              "title_kept_as_field": True, "meta_header": a.meta}, "sha256": sha})
    m.setdefault("checks", {})["n_corpus"] = len(rows)
    (dst / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{dst.name}: {len(rows)} chunks from {a.dataset}")


if __name__ == "__main__":
    main()
