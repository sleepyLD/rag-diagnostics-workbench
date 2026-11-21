"""Fine-tune a bi-encoder on multi-hop train splits (MDR-style: hop-1 samples q->gold and hop-2 samples (q+gold_i)->gold_j).

    python -m tools.train.train_embed --base BAAI/bge-small-en-v1.5 --out models/bge-small-mh --query-prefix "Represent this sentence for searching relevant passages: "
"""
import argparse, json, random, time
from tools.progress import ProgressCallback, start_parent_watch
from tools.train.common import CKPT_KW, ROOT, SEED, eval_question_set, latest_checkpoint, load_train, path_ctx, ptext


def _words(s):
    import re
    return {w for w in re.findall(r"\w+", s.lower()) if len(w) > 2}


def reachable(question, pos):
    """Indices of gold docs the question can reach directly (MDR hop 1): every title word of length > 2 is in the question.
    If none qualifies, the doc with the largest word overlap with the question."""
    qw = _words(question)
    idx = [i for i, p in enumerate(pos) if _words(p.get("title") or "") and _words(p.get("title") or "") <= qw]
    if not idx:
        idx = [max(range(len(pos)), key=lambda i: len(_words(ptext(pos[i])) & qw))]
    return idx


def mine_bm25_negatives(rows, k=30, per_row=4):
    """Hard negatives for train rows that have none: BM25 top-k over the row's own dataset corpus, minus the positives."""
    import random
    from src.retrievers.bm25 import BM25Retriever
    from tools.bench.common import read_jsonl
    by_ds = {}
    for r in rows:
        if not r.get("negatives"):
            by_ds.setdefault(r["_ds"], []).append(r)
    rnd = random.Random(SEED)
    for ds, rs in by_ds.items():
        corpus = read_jsonl(ROOT / "bench" / "data" / ds / "corpus.jsonl")
        text = {d["doc_id"]: d for d in corpus}
        bm = BM25Retriever().build(corpus, ds)
        hits = bm.search_batch([r["question"] for r in rs], k)
        for r, h in zip(rs, hits):
            pos_txt = {p["text"].strip() for p in r["positives"]}
            cand = [text[d] for d, _ in h if text[d]["text"].strip() not in pos_txt][5:]   # skip the very top: may be unlabeled positives
            r["negatives"] = [{"title": d.get("title", ""), "text": d["text"]} for d in rnd.sample(cand, min(per_row, len(cand)))]
        print(f"mined BM25 negatives for {len(rs)} rows of {ds}", flush=True)


def build_samples(rows, qp, ctx_chars, sep, hop2, hop1_mode="all", task_instruct=False):
    rnd = random.Random(SEED)
    A, P, N = [], [], []
    for r in rows:
        if task_instruct:   # same per-dataset instruction the retriever uses at query time (src/retrievers/dense.task_instruction)
            from src.retrievers.dense import task_instruction
            qp = f"Instruct: {task_instruction(r['_ds'])}\nQuery:"
        pos, neg = r["positives"], r.get("negatives") or []
        if not pos or not neg:
            continue
        first = list(range(len(pos))) if hop1_mode == "all" else reachable(r["question"], pos)
        for i in first:
            p = pos[i]
            A.append(qp + r["question"]); P.append(ptext(p)); N.append(ptext(rnd.choice(neg)))
        if hop2 and len(pos) >= 2:
            pairs = [(i, j) for i in range(len(pos)) for j in range(len(pos)) if i != j and (hop1_mode == "all" or i in first)]
            for i, j in rnd.sample(pairs, min(2, len(pairs))):
                A.append(qp + path_ctx(r["question"], [pos[i]], ctx_chars, sep)); P.append(ptext(pos[j])); N.append(ptext(rnd.choice(neg)))
            if len(pos) >= 3:   # hop 3: two gold docs as context (the first one reachable in "reachable" mode)
                i = rnd.choice(first)
                j, l = rnd.sample([x for x in range(len(pos)) if x != i], 2)
                A.append(qp + path_ctx(r["question"], [pos[i], pos[j]], ctx_chars, sep)); P.append(ptext(pos[l])); N.append(ptext(rnd.choice(neg)))
    return A, P, N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", default="hotpotqa,2wikimultihopqa,musique,squad")
    ap.add_argument("--cap", type=int, default=15000, help="questions per dataset")
    ap.add_argument("--query-prefix", default="")
    ap.add_argument("--doc-prefix", default="")
    ap.add_argument("--no-hop2", action="store_true")
    ap.add_argument("--mine-negatives", action="store_true", help="rows without negatives get BM25 hard negatives from their dataset corpus")
    ap.add_argument("--hop1", default="all", choices=["all", "reachable"], help="reachable = MDR: hop-1 targets only docs the question reaches")
    ap.add_argument("--ctx-chars", type=int, default=400)
    ap.add_argument("--sep", default=" ; ")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--mini-batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-seq", type=int, default=256)
    ap.add_argument("--task-instruct", action="store_true", help="per-dataset 'Instruct: {task}\\nQuery:' prefix (Qwen3 / harrier style)")
    ap.add_argument("--lora", type=int, default=0, help="LoRA rank (0 = full fine-tune); adapters are merged into the saved model")
    ap.add_argument("--lora-alpha", type=int, default=0, help="default 2 x rank")
    a = ap.parse_args()
    start_parent_watch()
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
    from sentence_transformers.training_args import BatchSamplers
    rows = load_train(a.datasets.split(","), a.cap)
    evalq = eval_question_set()
    leak = sum(" ".join(r["question"].lower().split()) in evalq for r in rows)
    assert leak == 0, f"{leak} training questions also appear in dev/holdout"
    if a.mine_negatives:
        mine_bm25_negatives(rows)
    A, P, N = build_samples(rows, a.query_prefix, a.ctx_chars, a.sep, not a.no_hop2, a.hop1, a.task_instruct)
    if a.doc_prefix:
        P = [a.doc_prefix + x for x in P]; N = [a.doc_prefix + x for x in N]
    ds = Dataset.from_dict({"anchor": A, "positive": P, "negative": N})
    model = SentenceTransformer(a.base, device="cuda")
    model.max_seq_length = a.max_seq
    if a.lora:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=a.lora, lora_alpha=a.lora_alpha or 2 * a.lora, lora_dropout=0.05, bias="none",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        peft_model = get_peft_model(model[0].auto_model, cfg)   # injects LoRA layers in place into the model ST holds
        peft_model.print_trainable_parameters()
    loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=a.mini_batch)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(ROOT / a.out / "_ckpt"), num_train_epochs=a.epochs, per_device_train_batch_size=a.batch,
        learning_rate=a.lr, warmup_ratio=0.05, bf16=True, batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=50, **CKPT_KW, report_to="none", seed=SEED, dataloader_num_workers=0)
    t0 = time.time()
    tr = SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss, callbacks=[ProgressCallback()] if ProgressCallback else None)
    resume = latest_checkpoint(args.output_dir)
    if resume:
        print(f"resuming from {resume}", flush=True)
    try:
        tr.train(resume_from_checkpoint=resume)
    except Exception as e:          # e.g. a LoRA checkpoint the trainer cannot restore: start over rather than fail every retry
        if not resume:
            raise
        print(f"resume failed ({type(e).__name__}: {e}); training from scratch", flush=True)
        tr.train()
    if a.lora:
        peft_model.merge_and_unload()   # in place: LoRA layers folded back into plain Linear layers of the same model object
    model.save(str(ROOT / a.out))
    info = {"base": a.base, "datasets": a.datasets, "cap_per_dataset": a.cap, "n_questions": len(rows), "n_samples": len(A),
            "hop2": not a.no_hop2, "hop1_mode": a.hop1, "query_prefix": a.query_prefix, "ctx_chars": a.ctx_chars, "sep": a.sep, "batch": a.batch,
            "lr": a.lr, "lora": a.lora, "task_instruct": a.task_instruct, "epochs": a.epochs, "max_seq": a.max_seq, "train_minutes": round((time.time() - t0) / 60, 1),
            "leak_questions": leak, "loss_log": [h for h in tr.state.log_history if "loss" in h][-5:]}
    (ROOT / a.out / "train_info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
