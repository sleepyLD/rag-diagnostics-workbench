"""Fine-tune a cross-encoder for ChainRerankRetriever (Beam-Retrieval style, one model for both hops).

Per training question:  hop 1  (q, gold)=1, (q, distractor)=0 ;  hop 2  (q [SEP] gold_i, gold_j)=1, (q [SEP] gold_i, distractor)=0
    python -m tools.train.train_chain_ce --base cross-encoder/ms-marco-MiniLM-L6-v2 --out models/chain-minilm
"""
import argparse, json, random, time
from tools.progress import ProgressCallback, start_parent_watch
from tools.train.common import CKPT_KW, ROOT, SEED, eval_question_set, latest_checkpoint, load_train, path_ctx, ptext


def build(rows, n_neg, ctx_chars, sep):
    rnd = random.Random(SEED)
    Q, D, L = [], [], []
    for r in rows:
        pos, neg = r["positives"], r.get("negatives") or []
        if not pos or not neg:
            continue
        for p in pos:
            Q.append(r["question"]); D.append(ptext(p)); L.append(1.0)
        for n in rnd.sample(neg, min(n_neg, len(neg))):
            Q.append(r["question"]); D.append(ptext(n)); L.append(0.0)
        if len(pos) >= 2:
            pairs = [(i, j) for i in range(len(pos)) for j in range(len(pos)) if i != j]
            for i, j in rnd.sample(pairs, min(2, len(pairs))):
                ctx = path_ctx(r["question"], [pos[i]], ctx_chars, sep)
                Q.append(ctx); D.append(ptext(pos[j])); L.append(1.0)
                for n in rnd.sample(neg, min(2, len(neg))):
                    Q.append(ctx); D.append(ptext(n)); L.append(0.0)
        if len(pos) >= 3:                     # third hop: two gold docs as context (MuSiQue 3-4 hop, 2Wiki 4-doc)
            for _ in range(2):
                i, j, l = rnd.sample(range(len(pos)), 3)
                ctx = path_ctx(r["question"], [pos[i], pos[j]], ctx_chars, sep)
                Q.append(ctx); D.append(ptext(pos[l])); L.append(1.0)
                Q.append(ctx); D.append(ptext(rnd.choice(neg))); L.append(0.0)
    return Q, D, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", default="hotpotqa,2wikimultihopqa,musique")
    ap.add_argument("--cap", type=int, default=10000)
    ap.add_argument("--neg", type=int, default=3)
    ap.add_argument("--ctx-chars", type=int, default=600)
    ap.add_argument("--sep", default=" [SEP] ")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=384)
    a = ap.parse_args()
    start_parent_watch()
    import torch
    from datasets import Dataset
    from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
    rows = load_train(a.datasets.split(","), a.cap)
    evalq = eval_question_set()
    leak = sum(" ".join(r["question"].lower().split()) in evalq for r in rows)
    assert leak == 0, f"{leak} training questions also appear in dev/holdout"
    Q, D, L = build(rows, a.neg, a.ctx_chars, a.sep)
    ds = Dataset.from_dict({"query": Q, "passage": D, "label": L})
    model = CrossEncoder(a.base, num_labels=1, max_length=a.max_len, device="cuda")
    pos_w = (len(L) - sum(L)) / max(1.0, sum(L))
    loss = BinaryCrossEntropyLoss(model, pos_weight=torch.tensor(pos_w))
    args = CrossEncoderTrainingArguments(output_dir=str(ROOT / a.out / "_ckpt"), num_train_epochs=a.epochs, per_device_train_batch_size=a.batch,
                                         learning_rate=a.lr, warmup_ratio=0.05, bf16=True, logging_steps=100, **CKPT_KW, report_to="none", seed=SEED, dataloader_num_workers=0)
    t0 = time.time()
    tr = CrossEncoderTrainer(model=model, args=args, train_dataset=ds, loss=loss, callbacks=[ProgressCallback()] if ProgressCallback else None)
    resume = latest_checkpoint(args.output_dir)
    if resume:
        print(f"resuming from {resume}", flush=True)
    tr.train(resume_from_checkpoint=resume)
    model.save_pretrained(str(ROOT / a.out))
    info = {"base": a.base, "datasets": a.datasets, "cap_per_dataset": a.cap, "n_questions": len(rows), "n_pairs": len(L),
            "pos_ratio": round(sum(L) / len(L), 3), "ctx_chars": a.ctx_chars, "sep": a.sep, "batch": a.batch, "lr": a.lr,
            "epochs": a.epochs, "max_len": a.max_len, "train_minutes": round((time.time() - t0) / 60, 1), "leak_questions": leak,
            "loss_log": [h for h in tr.state.log_history if "loss" in h][-5:]}
    (ROOT / a.out / "train_info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
