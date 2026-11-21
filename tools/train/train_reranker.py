"""Fine-tune a cross-encoder reranker on (synthetic or real) train files: (q, positive)=1, (q, hard negative)=0.

    python -m tools.train.train_reranker --base BAAI/bge-reranker-v2-m3 --files neoqa_synth,hotpotqa_train --cap 1500,1500 --out models/rr-m3-neoqa-gpl
Chunked corpora: --chunk-words 256 turns each long positive into its best-matching 256-word window (BM25 overlap with q).
"""
import argparse, json, random, re, time
from tools.progress import ProgressCallback, start_parent_watch
from tools.train.common import CKPT_KW, ROOT, SEED, eval_question_set, latest_checkpoint, ptext


def best_window(q, text, size):
    w = text.split()
    if len(w) <= size:
        return text
    qw = set(re.findall(r"\w+", q.lower()))
    best, bi = -1, 0
    for i in range(0, len(w) - size + 1, size // 2):
        s = len(qw & set(x.lower().strip(".,;:!?\"'()") for x in w[i:i + size]))
        if s > best:
            best, bi = s, i
    return " ".join(w[bi:bi + size])


def mine_v2(rows, rnd, per_hard=2, per_rand=2):
    """BM25 ranks 10-60 (near misses, skipping likely unlabeled positives at the top) + random corpus docs (far negatives)."""
    from src.retrievers.bm25 import BM25Retriever
    from tools.bench.common import read_jsonl
    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["_ds"], []).append(r)
    for ds, rs in by_ds.items():
        corpus = read_jsonl(ROOT / "bench" / "data" / ds / "corpus.jsonl")
        text = {d["doc_id"]: d for d in corpus}
        hits = BM25Retriever().build(corpus, ds).search_batch([r["question"] for r in rs], 60)
        for r, h in zip(rs, hits):
            pos = {p["text"].strip() for p in r["positives"]}
            near = [text[d] for d, _ in h[10:] if text[d]["text"].strip() not in pos]
            far = [d for d in rnd.sample(corpus, min(50, len(corpus))) if d["text"].strip() not in pos]
            pick = rnd.sample(near, min(per_hard, len(near))) + far[:per_rand]
            r["negatives"] = [{"title": d.get("title", ""), "text": d["text"]} for d in pick]
        print(f"v2 negatives for {len(rs)} rows of {ds}", flush=True)


def build_val(val_rows, depth=30):
    """Each validation question: its positives + BM25 top-`depth` non-positives (incl. the very top), scored by hit@1 / MRR."""
    from src.retrievers.bm25 import BM25Retriever
    from tools.bench.common import read_jsonl
    by_ds = {}
    for r in val_rows:
        by_ds.setdefault(r["_ds"], []).append(r)
    for ds, rs in by_ds.items():
        corpus = read_jsonl(ROOT / "bench" / "data" / ds / "corpus.jsonl")
        text = {d["doc_id"]: d for d in corpus}
        hits = BM25Retriever().build(corpus, ds).search_batch([r["question"] for r in rs], depth)
        for r, h in zip(rs, hits):
            pos = {p["text"].strip() for p in r["positives"]}
            r["_cands"] = [(ptext(text[d]), 0) for d, _ in h if text[d]["text"].strip() not in pos] + [(ptext(p), 1) for p in r["positives"][:4]]   # positives last: ties never favour them


try:
    from transformers import TrainerCallback as _TCB
except Exception:   # CPU test env without transformers
    _TCB = object


class ValCallback(_TCB):
    """Every ~10% of training: rerank each validation question's candidates; save the best model (by MRR) to `best_path`."""

    def __init__(self, model, val_rows, best_path):
        self.model, self.rows, self.best_path = model, val_rows, best_path
        self.best, self.log, self.next = -1.0, [], 0.1

    def score(self):
        mrr = hit1 = 0.0
        for r in self.rows:
            c = r["_cands"]
            s = self.model.predict([(r["question"], t) for t, _ in c], batch_size=32, show_progress_bar=False)
            order = sorted(range(len(c)), key=lambda i: -float(s[i]))
            first = next(k for k, i in enumerate(order) if c[i][1] == 1)
            mrr += 1.0 / (first + 1); hit1 += first == 0
        self.model.model.train()
        return {"mrr": round(mrr / len(self.rows), 4), "hit1": round(hit1 / len(self.rows), 4)}

    def on_step_end(self, args, state, control, force=False, **kw):
        frac = state.global_step / max(1, state.max_steps)
        if not force and frac < self.next:
            return
        self.next = (int(frac * 10) + 1) / 10
        m = self.score()
        self.log.append({"step": state.global_step, **m})
        print("val", json.dumps(self.log[-1]), flush=True)
        if m["mrr"] > self.best:
            self.best = m["mrr"]
            self.model.save_pretrained(str(self.best_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--files", required=True)
    ap.add_argument("--cap", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--neg", type=int, default=4)
    ap.add_argument("--chunk-words", type=int, default=0)
    ap.add_argument("--mine-negatives", action="store_true", help="BM25 hard negatives for rows without negatives (dataset = file name before _train/_synth)")
    ap.add_argument("--header-fields", default="", help="unused placeholder for symmetry with make_chunked")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-len", type=int, default=512)
    # v2 recipe (after v1 collapsed: hard-negatives-only + pos_weight + batch 8 -> overconfident, ranks far junk high)
    ap.add_argument("--neg-recipe", default="v1", choices=["v1", "v2"], help="v2: 2 BM25 negatives from ranks 10-60 + 2 random corpus docs")
    ap.add_argument("--no-pos-weight", action="store_true")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--val-frac", type=float, default=0.0, help="hold out train questions; rerank positives+BM25 top-30 every 10%%, keep best")
    a = ap.parse_args()
    start_parent_watch()
    import torch
    from datasets import Dataset
    from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
    rnd = random.Random(SEED)
    rows = []
    for name, cap in zip(a.files.split(","), map(int, a.cap.split(","))):
        rs = [{**json.loads(l), "_ds": name.rsplit("_", 1)[0]} for l in open(ROOT / "bench" / "train" / f"{name}.jsonl", encoding="utf-8")]
        rnd.shuffle(rs)
        rows += rs[:cap]
    val_rows = []
    if a.val_frac > 0:
        rnd.shuffle(rows)
        nv = max(50, int(len(rows) * a.val_frac))
        val_rows, rows = rows[:nv], rows[nv:]
    if a.neg_recipe == "v2":
        mine_v2(rows, rnd)
    elif a.mine_negatives:
        from tools.train.train_embed import mine_bm25_negatives
        mine_bm25_negatives(rows)
    if val_rows:
        build_val(val_rows)
    evalq = eval_question_set()
    leak = sum(" ".join(r["question"].lower().split()) in evalq for r in rows)
    assert leak == 0, f"{leak} training questions also appear in dev/holdout"
    txt = (lambda q, p: best_window(q, ptext(p), a.chunk_words)) if a.chunk_words else (lambda q, p: ptext(p))
    Q, D, L = [], [], []
    for r in rows:
        for p in r["positives"]:
            Q.append(r["question"]); D.append(txt(r["question"], p)); L.append(1.0)
        for n in (r.get("negatives") or [])[:a.neg]:
            Q.append(r["question"]); D.append(txt(r["question"], n)); L.append(0.0)
    ds = Dataset.from_dict({"query": Q, "passage": D, "label": L}).shuffle(seed=SEED)
    model = CrossEncoder(a.base, num_labels=1, max_length=a.max_len, device="cuda")
    loss = BinaryCrossEntropyLoss(model) if a.no_pos_weight else BinaryCrossEntropyLoss(model, pos_weight=torch.tensor((len(L) - sum(L)) / max(1.0, sum(L))))
    args = CrossEncoderTrainingArguments(output_dir=str(ROOT / a.out / "_ckpt"), num_train_epochs=1, per_device_train_batch_size=a.batch, gradient_accumulation_steps=a.grad_accum,
                                         learning_rate=a.lr, warmup_ratio=0.1, bf16=True, logging_steps=50, **CKPT_KW, report_to="none", seed=SEED, dataloader_num_workers=0)
    t0 = time.time()
    cbs = [ProgressCallback()] if ProgressCallback else []
    vcb = None
    if val_rows:
        vcb = ValCallback(model, val_rows, ROOT / a.out / "_best")
        print("val before training:", json.dumps(vcb.score()), flush=True)
        cbs.append(vcb)
    tr = CrossEncoderTrainer(model=model, args=args, train_dataset=ds, loss=loss, callbacks=cbs)
    resume = latest_checkpoint(args.output_dir)
    if resume:
        print(f"resuming from {resume}", flush=True)
    tr.train(resume_from_checkpoint=resume)
    if vcb is not None:
        vcb.on_step_end(args, tr.state, None, force=True)
        if vcb.best_path.exists():   # keep the best validated checkpoint, not the last step
            import shutil
            for f in vcb.best_path.iterdir():
                shutil.copy2(f, ROOT / a.out / f.name)
        else:
            model.save_pretrained(str(ROOT / a.out))
    else:
        model.save_pretrained(str(ROOT / a.out))
    info = {"base": a.base, "files": a.files, "cap": a.cap, "n_questions": len(rows), "n_pairs": len(L), "leak_questions": leak,
            "chunk_words": a.chunk_words, "lr": a.lr, "batch": a.batch, "train_minutes": round((time.time() - t0) / 60, 1),
            "loss_log": [round(h["loss"], 4) for h in tr.state.log_history if "loss" in h],
            "neg_recipe": a.neg_recipe, "pos_weight": not a.no_pos_weight, "grad_accum": a.grad_accum,
            "val_log": vcb.log if vcb else None, "best_val": vcb.best if vcb else None}
    (ROOT / a.out / "train_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
