"""Named retrieval methods for the bench. Query / document prefixes follow each model card."""
BGE_Q = "Represent this sentence for searching relevant passages: "
QWEN_Q = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"


def dense(name, model, **kw):
    return {"name": name, "type": "dense", "model": model, **kw}


METHODS = {
    # M0 baseline
    "bge-small": dense("bge-small", "BAAI/bge-small-en-v1.5", query_prefix=BGE_Q),
    "bm25": {"name": "bm25", "type": "bm25"},
    # M1 embedding swap
    "bge-base": dense("bge-base", "BAAI/bge-base-en-v1.5", query_prefix=BGE_Q),
    "bge-large": dense("bge-large", "BAAI/bge-large-en-v1.5", query_prefix=BGE_Q),
    "bge-m3": dense("bge-m3", "BAAI/bge-m3", batch_size=32),
    "gte-modernbert-base": dense("gte-modernbert-base", "Alibaba-NLP/gte-modernbert-base"),
    "e5-large-v2": dense("e5-large-v2", "intfloat/e5-large-v2", query_prefix="query: ", doc_prefix="passage: "),
    "qwen3-emb-0.6b": dense("qwen3-emb-0.6b", "Qwen/Qwen3-Embedding-0.6B", query_prefix=QWEN_Q, batch_size=32),
}
METHODS["rrf_bge-small_bm25"] = {"name": "rrf_bge-small_bm25", "type": "fusion", "mode": "rrf", "rrf_k": 60,
                                 "components": [METHODS["bge-small"], METHODS["bm25"]]}
PHASE_A = ["bge-small", "bm25", "rrf_bge-small_bm25", "bge-base", "bge-large", "bge-m3", "gte-modernbert-base", "e5-large-v2", "qwen3-emb-0.6b"]
DATASETS = ["hotpotqa", "2wikimultihopqa", "musique", "multihop_rag", "multihop_rag_c256", "neoqa", "squad", "scifact"]

# ---------------------------------------------------------------- Phase B (defined 11-18 03:35 after the Phase A offline analysis)
# offline weighted-RRF grid on Phase A top-100 lists: w_bm25 = 0.5 keeps bge-large's multi-hop wins and most of BM25's
# NeoQA / MultiHop-RAG wins, and gives the best all_gold_hit@100 (= rerank ceiling) on average.
METHODS["wrrf_large_bm25"] = {"name": "wrrf_large_bm25", "type": "fusion", "mode": "rrf", "rrf_k": 60, "weights": [1.0, 0.5],
                              "components": [METHODS["bge-large"], METHODS["bm25"]]}
METHODS["twohop_large"] = {"name": "twohop_large", "type": "twohop", "base": METHODS["bge-large"], "beam": 5, "per_beam": 20, "ctx_chars": 400}
METHODS["twohop_wrrf"] = {"name": "twohop_wrrf", "type": "twohop", "base": METHODS["wrrf_large_bm25"], "beam": 5, "per_beam": 20, "ctx_chars": 400}
METHODS["rr-bgev2m3_wrrf"] = {"name": "rr-bgev2m3_wrrf", "type": "rerank", "first": METHODS["wrrf_large_bm25"],
                              "model": "BAAI/bge-reranker-v2-m3", "depth": 100, "batch_size": 64, "max_length": 512}
METHODS["chain0-bgev2m3_wrrf"] = {"name": "chain0-bgev2m3_wrrf", "type": "chain", "first": METHODS["wrrf_large_bm25"],
                                  "model": "BAAI/bge-reranker-v2-m3", "pool": 30, "beam": 4, "ctx_chars": 600, "max_length": 512}
PHASE_B1 = ["wrrf_large_bm25", "twohop_large", "twohop_wrrf", "rr-bgev2m3_wrrf", "chain0-bgev2m3_wrrf"]

# ---------------------------------------------------------------- Phase B2 (defined 11-18 04:25)
# B1 finding: two-hop expansion fixes bridge questions (2Wiki 0.48 -> 0.74) but hurts single-hop / news; candidate-pool ceilings
# @50 are >= 0.90 except 2Wiki (0.86) and MuSiQue (0.68; 4-hop 0.29)  ->  3-hop beams, trained chain reranker with open-corpus expansion.
METHODS["bge-small-mh"] = dense("bge-small-mh", "models/bge-small-mh", query_prefix=BGE_Q)              # fine-tuned (015)
METHODS["wrrf_small-mh_bm25"] = {"name": "wrrf_small-mh_bm25", "type": "fusion", "mode": "rrf", "rrf_k": 60, "weights": [1.0, 0.5],
                                 "components": [METHODS["bge-small-mh"], METHODS["bm25"]]}
METHODS["mhbeam3_wrrf"] = {"name": "mhbeam3_wrrf", "type": "mhbeam", "base": METHODS["wrrf_large_bm25"], "hops": 3, "beam": 5, "per_beam": 10, "ctx_chars": 300}
METHODS["mhbeam3_small-mh"] = {"name": "mhbeam3_small-mh", "type": "mhbeam", "base": METHODS["bge-small-mh"], "hops": 3, "beam": 5, "per_beam": 10, "ctx_chars": 400}
METHODS["chain-minilm_wrrf"] = {"name": "chain-minilm_wrrf", "type": "chain", "first": METHODS["wrrf_large_bm25"], "model": "models/chain-minilm",
                                "pool": 30, "beam": 4, "hops": 2}
METHODS["chain3x-minilm_wrrf"] = {"name": "chain3x-minilm_wrrf", "type": "chain", "first": METHODS["wrrf_large_bm25"], "model": "models/chain-minilm",
                                  "pool": 30, "beam": 4, "hops": 3, "beam3": 3, "expand": METHODS["bge-small-mh"], "expand_k": 10}
METHODS["chain3x-minilm_fast"] = {"name": "chain3x-minilm_fast", "type": "chain", "first": METHODS["wrrf_small-mh_bm25"], "model": "models/chain-minilm",
                                  "pool": 20, "beam": 3, "hops": 3, "beam3": 2, "expand": METHODS["bge-small-mh"], "expand_k": 8}
METHODS["colbert-gtemodern"] = {"name": "colbert-gtemodern", "type": "colbert", "model": "lightonai/GTE-ModernColBERT-v1", "doc_length": 300}
METHODS["splade-cocondenser"] = {"name": "splade-cocondenser", "type": "sparse", "model": "naver/splade-cocondenser-ensembledistil"}
METHODS["qwen3-emb-0.6b-fp32"] = dense("qwen3-emb-0.6b-fp32", "Qwen/Qwen3-Embedding-0.6B", query_prefix=QWEN_Q, batch_size=16, fp16=False)  # audit of the fp16 result
PHASE_B2A = ["bge-small-mh", "wrrf_small-mh_bm25", "mhbeam3_wrrf", "mhbeam3_small-mh", "chain-minilm_wrrf", "chain3x-minilm_wrrf",
             "chain3x-minilm_fast", "colbert-gtemodern", "splade-cocondenser"]
METHODS["chain3x-gtemb_wrrf"] = {"name": "chain3x-gtemb_wrrf", "type": "chain", "first": METHODS["wrrf_large_bm25"], "model": "models/chain-gtemb",
                                 "pool": 30, "beam": 4, "hops": 3, "beam3": 3, "expand": METHODS["bge-small-mh"], "expand_k": 10, "batch_size": 64,
                                 "max_length": 512}
PHASE_B2B = ["chain3x-gtemb_wrrf"]

# ---------------------------------------------------------------- Phase B3 (defined 11-18 07:35)
# B2A: mhbeam3 with the MDR-tuned bge-small hits 2Wiki 0.934 / MuSiQue 0.514 at ~40 ms. Next: MDR-faithful hop-1 (v2), a bigger
# encoder, 4 hops for MuSiQue, BM25-heavy beam base for NeoQA (fictional names), contextual chunk headers for MultiHop-RAG.
DATASETS.append("multihop_rag_c256m")
for tag, path in (("small-mh2", "models/bge-small-mh2"), ("base-mh2", "models/bge-base-mh2")):
    METHODS[f"bge-{tag}"] = dense(f"bge-{tag}", path, query_prefix=BGE_Q)
    METHODS[f"mhbeam3_{tag}"] = {"name": f"mhbeam3_{tag}", "type": "mhbeam", "base": METHODS[f"bge-{tag}"], "hops": 3, "beam": 5, "per_beam": 10, "ctx_chars": 400}
    METHODS[f"mhbeam4_{tag}"] = {"name": f"mhbeam4_{tag}", "type": "mhbeam", "base": METHODS[f"bge-{tag}"], "hops": 4, "beam": 8, "per_beam": 10, "ctx_chars": 400}
METHODS["rrf_small-mh2_bm25"] = {"name": "rrf_small-mh2_bm25", "type": "fusion", "mode": "rrf", "rrf_k": 60, "weights": [1.0, 1.0],
                                 "components": [METHODS["bge-small-mh2"], METHODS["bm25"]]}
METHODS["mhbeam3_rrf-mh2-bm25"] = {"name": "mhbeam3_rrf-mh2-bm25", "type": "mhbeam", "base": METHODS["rrf_small-mh2_bm25"], "hops": 3, "beam": 5, "per_beam": 10, "ctx_chars": 400}
METHODS["fuse_rr_twohop"] = {"name": "fuse_rr_twohop", "type": "fusion", "mode": "rrf", "rrf_k": 60, "weights": [1.0, 1.0],
                             "components": [METHODS["rr-bgev2m3_wrrf"], METHODS["twohop_wrrf"]]}      # offline best for HotpotQA / SciFact
PHASE_B3 = ["bge-small-mh2", "mhbeam3_small-mh2", "mhbeam4_small-mh2", "bge-base-mh2", "mhbeam3_base-mh2", "mhbeam4_base-mh2", "mhbeam3_rrf-mh2-bm25"]

# ---------------------------------------------------------------- Phase B4: cascade rerank (defined 11-18 08:20)
# 2nd-stage ceilings of rr-bgev2m3_wrrf: @30 NeoQA 0.902 / MHR-c256m 0.942 / SciFact 0.947; @50 NeoQA 0.944.
QWEN_RR_INSTR = "Given a question that may need several documents, judge whether this document provides evidence needed to answer it"
for size in ("0.6B", "4B"):
    for depth in (30, 50):
        METHODS[f"casc-q3rr{size}@{depth}"] = {"name": f"casc-q3rr{size}@{depth}", "type": "rerank", "first": METHODS["rr-bgev2m3_wrrf"],
                                              "model": f"Qwen/Qwen3-Reranker-{size}", "scorer_type": "qwen3", "instruction": QWEN_RR_INSTR,
                                              "depth": depth, "max_length": 1024, "batch_size": 16 if size == "0.6B" else 4}

# ---------------------------------------------------------------- Phase B5: LLM decomposition (defined 11-18 10:05)
# MuSiQue: even 2-hop questions only 0.75 with the best beam; the bridge entity has to be *answered* before it can be searched.
METHODS["rr30_wrrf"] = {"name": "rr30_wrrf", "type": "rerank", "first": METHODS["wrrf_large_bm25"], "model": "BAAI/bge-reranker-v2-m3",
                        "depth": 30, "batch_size": 32, "max_length": 512}          # single-hop retriever for sub-questions
METHODS["decomp-q3-4b"] = {"name": "decomp-q3-4b", "type": "decomp", "base": METHODS["rr30_wrrf"], "llm": "qwen3:4b-instruct",
                           "max_steps": 4, "per_step": 3, "fallback": METHODS["mhbeam4_base-mh2"]}

# ---------------------------------------------------------------- Phase B6: evidence-by-evidence retrieval, 3 paper methods (11-18 11:35)
# Same single-hop retriever for every step (rr30_wrrf), same LLM (qwen3:4b-instruct), 4 steps max, 3 docs per step,
# same fallback (rr30_wrrf on the original question) -> only the "what is the next query" policy differs.
for style, nm in (("ircot", "ircot"), ("iterretgen", "iterretgen")):
    METHODS[f"{nm}-q3-4b"] = {"name": f"{nm}-q3-4b", "type": "iter", "style": style, "base": METHODS["rr30_wrrf"], "llm": "qwen3:4b-instruct",
                              "max_steps": 4, "per_step": 3, "fallback": METHODS["rr30_wrrf"]}
METHODS["selfask-q3-4b"] = {"name": "selfask-q3-4b", "type": "decomp", "base": METHODS["rr30_wrrf"], "llm": "qwen3:4b-instruct",
                            "max_steps": 4, "per_step": 3, "fallback": METHODS["rr30_wrrf"]}
PHASE_B6 = ["ircot-q3-4b", "iterretgen-q3-4b", "selfask-q3-4b", "rr30_wrrf"]

# ---------------------------------------------------------------- Phase D: medical / finance (defined 11-18 17:05)
DATASETS_DOMAIN = ["nfcorpus", "bioasq", "medqa", "fiqa", "finder", "finrate_dr", "finrate_xdoc"]
METHODS["bge-base-dom"] = dense("bge-base-dom", "models/bge-base-dom", query_prefix=BGE_Q)        # fine-tuned on NFCorpus/FiQA/BioASQ train
METHODS["wrrf_basedom_bm25"] = {"name": "wrrf_basedom_bm25", "type": "fusion", "mode": "rrf", "rrf_k": 60, "weights": [1.0, 0.5],
                                "components": [METHODS["bge-base-dom"], METHODS["bm25"]]}
METHODS["rr-m3dom_wrrfdom"] = {"name": "rr-m3dom_wrrfdom", "type": "rerank", "first": METHODS["wrrf_basedom_bm25"], "model": "models/rr-m3-dom",
                               "depth": 100, "batch_size": 64, "max_length": 512}
METHODS["rr-bgev2m3_wrrfdom"] = {"name": "rr-bgev2m3_wrrfdom", "type": "rerank", "first": METHODS["wrrf_basedom_bm25"], "model": "BAAI/bge-reranker-v2-m3",
                                 "depth": 100, "batch_size": 64, "max_length": 512}
PHASE_D1 = ["bge-base-dom", "wrrf_basedom_bm25", "rr-bgev2m3_wrrfdom", "rr-m3dom_wrrfdom"]


# Phase E: cross-entity split (Fin-RATE cross-company): one sub-query per company named in the question, interleaved
METHODS["entsplit_rr"] = {"name": "entsplit_rr", "type": "entsplit", "base": METHODS["rr-bgev2m3_wrrf"], "entity_field": "company"}
METHODS["entsplit_wrrf"] = {"name": "entsplit_wrrf", "type": "entsplit", "base": METHODS["wrrf_large_bm25"], "entity_field": "company"}
# v2: v1 found no entities on chunked corpora (metadata lives only in the header line) -> identical to its base. Fixed in EntitySplitRetriever.build.
METHODS["entsplit2_rr"] = {**METHODS["entsplit_rr"], "name": "entsplit2_rr"}
METHODS["entsplit2_wrrf"] = {**METHODS["entsplit_wrrf"], "name": "entsplit2_wrrf"}
# MultiHop-RAG: cross-source questions ("TechCrunch ... and The Verge ...") -> split by news source named in the question
METHODS["entsplit2_src_rr"] = {"name": "entsplit2_src_rr", "type": "entsplit", "base": METHODS["rr-bgev2m3_wrrf"], "entity_field": "source"}
# BioASQ: 4-5 gold passages per question; first-stage @100 = 0.856 -> give the reranker a deeper pool
METHODS["rr200_wrrf"] = {**METHODS["rr-bgev2m3_wrrf"], "name": "rr200_wrrf", "depth": 200}

# SciFact: offline RRF(rr, mhbeam3_base-mh2) 1:1 (default weights, not tuned) = 0.900 on dev
METHODS["fuse_rr_mhbeam3"] = {"name": "fuse_rr_mhbeam3", "type": "fusion", "mode": "rrf", "rrf_k": 60, "weights": [1.0, 1.0],
                              "components": [METHODS["rr-bgev2m3_wrrf"], METHODS["mhbeam3_base-mh2"]]}
# MultiHop-RAG: both source split (+0.016) and depth 200 (+0.020) helped separately -> combine
METHODS["entsplit2_src_rr200"] = {"name": "entsplit2_src_rr200", "type": "entsplit", "base": METHODS["rr200_wrrf"], "entity_field": "source"}
# MuSiQue: RRF(rr, beam) hurt; instead let the cross-encoder rerank the beam's own top-100 pool
METHODS["rr_on_mhbeam4"] = {"name": "rr_on_mhbeam4", "type": "rerank", "first": METHODS["mhbeam4_base-mh2"], "model": "BAAI/bge-reranker-v2-m3", "depth": 100}

# Question-only rerank of the beam pool hurt MuSiQue/2Wiki (2nd-hop docs share no words with the question) ->
# score whole paths (q, hop1 || hop2 || ...) with the same zero-shot cross-encoder
METHODS["mhbeam4_pathrr"] = {**METHODS["mhbeam4_base-mh2"], "name": "mhbeam4_pathrr", "path_rerank": "BAAI/bge-reranker-v2-m3"}
METHODS["mhbeam3_pathrr"] = {**METHODS["mhbeam3_base-mh2"], "name": "mhbeam3_pathrr", "path_rerank": "BAAI/bge-reranker-v2-m3"}
# reranker fine-tune v2 (balanced BCE, near+random negatives, effective batch 32, best-of-10 validation)
METHODS["rr-m3dom2_wrrf"] = {**METHODS["rr-bgev2m3_wrrf"], "name": "rr-m3dom2_wrrf", "model": "models/rr-m3-dom2"}

# BioASQ: 4-5 same-topic papers per question; reranker is not the limit (depth 200 = depth 100) and LLM decomposition
# does not apply -> dense pseudo-relevance feedback (Rocchio: q + 0.5*mean(top-5 doc vectors)) to pull in sibling papers
_bgel = [c for c in METHODS["wrrf_large_bm25"]["components"] if c["type"] == "dense"][0]
METHODS["bge-large-prf"] = {**_bgel, "name": "bge-large-prf", "prf_k": 5, "prf_beta": 0.5}
METHODS["wrrf_largeprf_bm25"] = {**METHODS["wrrf_large_bm25"], "name": "wrrf_largeprf_bm25",
                                 "components": [METHODS["bge-large-prf"] if c["type"] == "dense" else c for c in METHODS["wrrf_large_bm25"]["components"]]}
METHODS["rr_wrrf_prf"] = {**METHODS["rr-bgev2m3_wrrf"], "name": "rr_wrrf_prf", "first": METHODS["wrrf_largeprf_bm25"]}

# D1 set selection by requirement coverage (SetR-style, zero-shot: qwen3:4b requirements + bge-reranker coverage matrix)
METHODS["covsel_rr"] = {"name": "covsel_rr", "type": "covsel", "base": METHODS["rr-bgev2m3_wrrf"], "llm": "qwen3:4b-instruct", "pool": 30, "select": 10, "lam": 0.3}

# D2 (complementary merge, fixed 5/5 before looking at dev): LLM Self-Ask top-5, then the cheap method fills the rest
METHODS["merge_selfask_mhbeam4"] = {"name": "merge_selfask_mhbeam4", "type": "fusion", "mode": "head", "head": 5,
                                    "components": [METHODS["selfask-q3-4b"], METHODS["mhbeam4_base-mh2"]]}
METHODS["merge_selfask_rr"] = {"name": "merge_selfask_rr", "type": "fusion", "mode": "head", "head": 5,
                               "components": [METHODS["selfask-q3-4b"], METHODS["rr-bgev2m3_wrrf"]]}

# Phase F (11-20): newer embedding models, each with the query/document prompts shipped in its own config
METHODS["harrier-0.6b"] = dense("harrier-0.6b", "microsoft/harrier-oss-v1-0.6b", use_model_prompts=True, batch_size=32)
METHODS["qwen3-emb-0.6b-mp"] = dense("qwen3-emb-0.6b-mp", "Qwen/Qwen3-Embedding-0.6B", use_model_prompts=True, batch_size=32)
METHODS["qwen3-emb-4b"] = dense("qwen3-emb-4b", "Qwen/Qwen3-Embedding-4B", use_model_prompts=True, batch_size=8)
METHODS["jina-v5-small"] = dense("jina-v5-small", "jinaai/jina-embeddings-v5-text-small", use_model_prompts=True, trust_remote_code=True, batch_size=32)
METHODS["embeddinggemma-300m"] = dense("embeddinggemma-300m", "google/embeddinggemma-300m", use_model_prompts=True, batch_size=32, fp16=False)
PHASE_F = ["harrier-0.6b", "qwen3-emb-0.6b-mp", "jina-v5-small", "embeddinggemma-300m", "qwen3-emb-4b"]

# Phase F2 (11-21): put the new embeddings into the full pipeline (weighted RRF with BM25 x0.5, then bge-reranker-v2-m3 @100),
# same recipe as wrrf_large_bm25 / rr-bgev2m3_wrrf so the only change is the dense model
PHASE_F2 = []
for _m in ["harrier-0.6b", "qwen3-emb-0.6b-mp", "qwen3-emb-4b"]:
    METHODS[f"wrrf_{_m}_bm25"] = {**METHODS["wrrf_large_bm25"], "name": f"wrrf_{_m}_bm25", "components": [METHODS[_m], METHODS["bm25"]]}
    METHODS[f"rr_wrrf_{_m}"] = {**METHODS["rr-bgev2m3_wrrf"], "name": f"rr_wrrf_{_m}", "first": METHODS[f"wrrf_{_m}_bm25"]}
    PHASE_F2 += [f"wrrf_{_m}_bm25", f"rr_wrrf_{_m}"]
PHASE_F_MH = ["harrier-0.6b", "qwen3-emb-0.6b-mp", "qwen3-emb-4b"]

# Phase F3 (11-21): same models with a one-line per-dataset task instruction for the query (how MTEB scores are produced)
PHASE_F3 = []
for _m in ["harrier-0.6b", "qwen3-emb-0.6b-mp", "qwen3-emb-4b"]:
    METHODS[f"{_m}-ti"] = {**METHODS[_m], "name": f"{_m}-ti", "task_instruct": True}
    PHASE_F3.append(f"{_m}-ti")

# 11-21 05:55: PHASE_F2 redefined after the F3 results — use the task-instruction variants (harrier gains +0.03..+0.14 nDCG from it),
# swap qwen3-0.6b (weakest) for EmbeddingGemma-300m (strong on SciFact / MedQA). Gemma keeps its own model prompts.
PHASE_F2 = []
for _m in ["qwen3-emb-4b-ti", "harrier-0.6b-ti", "embeddinggemma-300m"]:
    METHODS[f"wrrf_{_m}_bm25"] = {**METHODS["wrrf_large_bm25"], "name": f"wrrf_{_m}_bm25", "components": [METHODS[_m], METHODS["bm25"]]}
    METHODS[f"rr_wrrf_{_m}"] = {**METHODS["rr-bgev2m3_wrrf"], "name": f"rr_wrrf_{_m}", "first": METHODS[f"wrrf_{_m}_bm25"]}
    PHASE_F2 += [f"wrrf_{_m}_bm25", f"rr_wrrf_{_m}"]

# 11-21 11:00: reranker on top of the strong dense model alone (no BM25 fusion, which drags qwen3-4b down on FiQA/SciFact/MedQA)
METHODS["rr_qwen3-emb-4b-ti"] = {**METHODS["rr-bgev2m3_wrrf"], "name": "rr_qwen3-emb-4b-ti", "first": METHODS["qwen3-emb-4b-ti"]}

# 11-21 12:00: zero-shot MDR-style beam with the new embeddings (no multi-hop fine-tuning), same beam settings as mhbeam3_base-mh2
for _m in ["qwen3-emb-4b-ti", "harrier-0.6b-ti"]:
    METHODS[f"mhbeam3_{_m}"] = {"name": f"mhbeam3_{_m}", "type": "mhbeam", "base": METHODS[_m], "hops": 3, "beam": 5, "per_beam": 10, "ctx_chars": 400}

# Phase G (11-21): LoRA fine-tunes of Qwen3-Embedding-0.6B (r=32, merged), queries use the per-dataset task instruction
METHODS["mhbeam3_qwen3-emb-0.6b-mp-ti"] = {"name": "mhbeam3_qwen3-emb-0.6b-mp-ti", "type": "mhbeam", "base": METHODS["qwen3-emb-0.6b-mp-ti"],
                                           "hops": 3, "beam": 5, "per_beam": 10, "ctx_chars": 400}   # zero-shot baseline for the MDR fine-tune
for _tag in ("mh", "dom"):
    METHODS[f"qwen06-{_tag}"] = dense(f"qwen06-{_tag}", f"models/qwen06-{_tag}-lora", use_model_prompts=True, task_instruct=True, batch_size=32)
for _h in (3, 4):
    METHODS[f"mhbeam{_h}_qwen06-mh"] = {"name": f"mhbeam{_h}_qwen06-mh", "type": "mhbeam", "base": METHODS["qwen06-mh"],
                                        "hops": _h, "beam": 5 if _h == 3 else 8, "per_beam": 10, "ctx_chars": 400}
PHASE_G_MH = ["qwen06-mh", "mhbeam3_qwen06-mh", "mhbeam4_qwen06-mh", "mhbeam3_qwen3-emb-0.6b-mp-ti"]
