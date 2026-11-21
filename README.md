# RAG 工作台與檢索方法比較

這個專案分成兩部分：

1. [**RAG 工作台**](#第一部分rag-工作台)：一個小型、可診斷的 RAG（Retrieval-Augmented Generation）系統。可以切換知識庫、檢索方法、生成模型和 Top-k，每一步的中間結果都會留下紀錄；評估流程能區分「檢索失敗」和「生成失敗」，並以 dense → hybrid 檢索為例，在 dev 與 holdout 上做前後對照。
2. [**檢索方法比較**](#第二部分檢索方法比較14-個資料集)：在 14 個單跳、多跳、跨文件、科學、醫療與金融資料集上，比較 BM25、embedding、混合檢索、重排、多跳檢索與 LLM 輔助檢索，並以沒有參與調參的 holdout 驗證。

## 第一部分：RAG 工作台

### 系統流程

```
事前：段落(.md) ──► bge-small 向量 ──► 索引快取 index/<Collection>__<模型>__<內容指紋>/
                └─► BM25（標題+段落，記憶體內）

每次查詢：問題 ──► 檢索 Top-k（dense / bm25 / hybrid=RRF）──► 組 Context（附段落 ID，超過上限整段捨棄）
             ──► Ollama 生成（只能依據段落、要附引用、證據不足就拒答）──► 解析答案與引用 ──► 評分、寫入 JSONL
```

三種模式並排比較：`no_rag`（只給問題）、`rag`（給檢索結果）、`oracle`（直接給正確段落）。oracle 答對、rag 答錯 → 問題在檢索；oracle 也答錯 → 問題在生成或題目。

### 資料

| Collection | 內容 | 題目 |
|---|---|---|
| `squad_v11_dev500` | SQuAD 1.1 dev 的 11 篇文章，534 段 | dev 300、holdout 100 |
| `squad_v11_b5` | 另外 5 篇主題不重疊的文章，185 段 | 用這 5 篇出題、去查 `squad_v11_dev500`，作為無答案題：dev 50、holdout 50 |
| `hotpot_dev400` | HotpotQA dev（distractor）400 題的所有段落，3,954 段 | dev 300、holdout 100 |
| `real_docs` | 網頁規格頁匯入工具的範例（1 份） | — |

資料都由 `tools/datasets/` 的腳本以固定 seed 產生，來源、雜湊、授權見 `data/README.md` 與 `eval/*_manifest.json`。SQuAD 和 HotpotQA 皆為 CC BY-SA 4.0。

### 主要結果（qwen3:4b-instruct，k = 3）

| | dev：dense | dev：hybrid | **holdout：dense** | **holdout：hybrid** |
|---|---|---|---|---|
| SQuAD 檢索 hit@1 | 0.743 | 0.843 | 0.700 | 0.780 |
| SQuAD 檢索 hit@3 | 0.887 | 0.947 | 0.890 | 0.910 |
| SQuAD 答對 | 0.833 | 0.893 | 0.830 | 0.850 |
| HotpotQA 檢索 hit@3（兩段都找到） | 0.617 | 0.490 | 0.790 | 0.580 |
| HotpotQA 答對 | 0.467 | 0.480 | 0.620 | 0.510 |
| 無答案題拒答（50 題） | 50/50 | 50/50 | 49/50 | 49/50 |

題數：dev 的檢索指標用 300 題，答題正確率用其中 150 題；holdout 兩者都是 100 題，開發期間沒有使用。dense = bge-small-en-v1.5，hybrid = dense 與 BM25 的等權重 RRF。hit@k = 正解段落出現在前 k 名的比例；「答對」為自動評分：回答包含標準答案，或去除與題目重複的詞後 token F1 ≥ 0.5。

- **混合檢索（等權重 RRF）不適合當預設方法。** 在 SQuAD 上穩定地把正確段落往前排（hit@1 +8～10 個百分點），但 k=3 時答案的改善在 holdout 上只有 +2 個百分點，不顯著；在 HotpotQA 上明顯有害（holdout 答對率 −11 個百分點）。HotpotQA 的干擾段落是用關鍵字相似度挑出來的，BM25 Top-3 裡的錯誤段落有 95% 正是這些干擾段落。詳見 [`reports/issue.md`](reports/issue.md)。
- **dev 高估了改善：** 改用混合檢索的想法來自 dev 的失敗案例，因此 dev 會高估效果：SQuAD 答對率在 dev 提升 6 個百分點，在 holdout 只提升 2 個。
- **生成模型比較**（`notebooks/02_collections_and_models.ipynb`）：1.7B 在 15 題無答案題中編造了 6 題；4B 和 9B 答對率相同，9B 較慢，且常把 Prompt 裡的格式範例原樣輸出，因此選用 4B。
- **No-RAG 答對率**（SQuAD 0.147、HotpotQA 橋接題 0.086）偏低，代表模型多半不是靠預先背過的知識在答題。
- **holdout 中唯一一題沒有拒答的無答案題**，是題目本身有歧義（「the network」在 `squad_v11_dev500` 裡也能指電視網，而且有原文依據），兩種方法表現相同。

### 快速開始（Windows）

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name rag-workbench
ollama pull qwen3:4b-instruct
```

**展示介面：**
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```
左側選知識庫、檢索方法、生成模型、Top-k、RAG 開關；可以挑範例題（會顯示標準答案和正確段落）或自己輸入，結果會列出答案、引用、取回的段落與分數，以及實際送給模型的 Prompt。

**評估：** 打開 `notebooks/03_benchmark.ipynb`。

**測試：** `.\.venv\Scripts\python.exe -m pytest -q`

## 第二部分：檢索方法比較（14 個資料集）

在 14 個資料集上比較 BM25、多種 embedding 模型、混合檢索、重排、多跳檢索與 LLM 輔助檢索，涵蓋單跳、多跳、跨文件新聞、科學、醫療與金融。全部在一張 RTX 5060 Ti 16GB 上執行。Fin-RATE 的單份與跨份題各算一個資料集。

每個資料集分成 dev 與 holdout：方法和參數只看 dev 決定，holdout 只用來確認。主要指標 **all@10** 是「一題需要的正解段落全部出現在前 10 名」的比例；多跳題要每一跳的證據都找到才算，有多組可行的正解時任一組滿足即可。兩個方法的差異用 paired bootstrap（重抽 5,000 次）的 95% 信賴區間判斷是否顯著。

### 資料集

| 資料集 | 類型 | 語料（段落數） | 題數 dev / holdout |
|---|---|---|---|
| SQuAD 1.1 | 單跳事實題 | 20,958 | 500 / 300 |
| HotpotQA | 2 跳（橋接、比較） | 40,003 | 500 / 300 |
| 2WikiMultiHopQA | 2–4 跳 | 40,002 | 500 / 300 |
| MuSiQue | 2–4 跳 | 21,093 | 500 / 300 |
| MultiHop-RAG | 跨多篇新聞 | 5,636 | 500 / 300 |
| NeoQA | 虛構新聞事件、跨文件 | 1,800 | 500 / 300 |
| SciFact | 科學主張查證 | 5,183 | 150 / 150 |
| NFCorpus | 營養與醫學 | 3,633 | 403 / 242 |
| MedQuAD（MTEB medical_qa） | 醫療問答 | 2,048 | 500 / 300 |
| BioASQ | 生醫問答，一題多篇佐證 | 27,974 | 500 / 300 |
| FiQA | 金融問答 | 57,600 | 500 / 300 |
| FinDER | 10-K 年報分析師問題 | 5,832 | 500 / 300 |
| Fin-RATE（單份 / 跨份） | SEC 財報，跨份題需比對多家公司或多個年度 | 216,480 | 500 / 300 各一組 |

SciFact、NFCorpus、FiQA 使用 BEIR 的完整語料，nDCG@10 可以和公開排行榜粗略對照；HotpotQA、2Wiki、MuSiQue、BioASQ 的語料是從原資料抽出的段落池（例如 HotpotQA 原本的 Wikipedia 語料有 520 萬篇），數字只適合用來比較這裡的方法。MultiHop-RAG 與 Fin-RATE 的文件先切成 256 個英文字（words）一段（見主要發現第 6 點）。

### 結果（holdout）

每個資料集列出已在 holdout 上確認過的方法中最好的一個（方法都先在 dev 上選出）。

| 資料集 | 最佳方法 | all@10 | nDCG@10 | 延遲 p50 |
|---|---|---|---|---|
| SQuAD | 混合檢索＋重排 | 0.983 | 0.940 | 0.68 s |
| 2Wiki | 多跳 beam（微調 bge-small） | 0.947 | 0.925 | 45 ms |
| MedQuAD | EmbeddingGemma-300M | 0.947 | 0.814 | 37 ms |
| HotpotQA | 重排＋兩跳檢索融合 | 0.913 | 0.895 | 0.58 s |
| SciFact | Qwen3-Embedding-4B | 0.900 | 0.777 | 59 ms |
| Fin-RATE 單份 | 切塊＋metadata 標頭＋混合檢索＋重排 | 0.887 | 0.640 | 0.96 s |
| FiQA | Qwen3-Embedding-4B | 0.853 | 0.571 | 59 ms |
| MultiHop-RAG | 依問題中的媒體拆查詢＋重排 | 0.827 | 0.870 | 5.3 s |
| NFCorpus | bge-large | 0.798 | 0.380 | 16 ms |
| NFCorpus | Qwen3-Embedding-4B | 0.785 | 0.398 | 60 ms |
| NeoQA | Self-Ask 前 5 名＋重排補滿 | 0.757 | 0.489 | 13 s |
| BioASQ | 混合檢索＋重排 | 0.717 | 0.839 | 0.86 s |
| MuSiQue | Self-Ask 前 5 名＋多跳 beam 補滿 | 0.640 | 0.766 | 12 s |
| Fin-RATE 跨份 | 切塊＋標頭＋依公司拆查詢＋重排 | 0.553 | 0.416 | 0.96 s |
| FinDER | 混合檢索＋重排 | 0.410 | 0.339 | 1.0 s |

NFCorpus 的兩個方法差距不顯著（all@10 −0.012 [−0.050, +0.025]），bge-large 的 all@10 較高，Qwen3-Embedding-4B 的 nDCG@10 較高。

方法說明：
- **混合檢索**：bge-large-en-v1.5 與 BM25 的加權 RRF（dense 權重 1.0、BM25 權重 0.5，k = 60）。
- **重排**：bge-reranker-v2-m3 對混合檢索的前 100 名重新排序。
- **重排＋兩跳檢索融合**：用第一輪的前幾名段落擴充查詢、再檢索一次，結果和重排結果以 RRF 合併。
- **多跳 beam**：仿 MDR（Multi-hop Dense Retrieval）。第一跳用問題檢索；之後把「問題＋已找到的段落」接成新的查詢再檢索，每一跳保留分數最高的 5 條路徑，最多 3 跳。encoder（bge-small、bge-base 或 Qwen3-Embedding-0.6B）用 HotpotQA、2Wiki、MuSiQue、SQuAD 的訓練集微調。
- **Self-Ask 前 5 名＋補滿**：qwen3:4b 把問題拆成子問題逐步檢索，取它找到的前 5 段，其餘名次由低成本的方法（重排或多跳 beam）補滿。
- **依實體拆查詢**：問題同時提到多家媒體或公司時，對每個實體各檢索一次，再輪流合併結果。
- **embedding 模型**：bge-large-en-v1.5（BAAI，335M）、Qwen3-Embedding-0.6B / 4B（Alibaba）、harrier-oss-v1-0.6B（Microsoft）、EmbeddingGemma-300M（Google）。Qwen3-Embedding 與 harrier 的查詢前面會加一句描述任務的指令，例如 SciFact 用「Given a scientific claim, retrieve documents that support or refute the claim」。

### 主要發現

**1. 加權混合檢索＋重排是通用場景的穩定基準。** 第一部分的等權重 RRF 會把 HotpotQA 中關鍵字相似的干擾段落排進來，因此不適合當預設。把 BM25 權重降到 0.5，再交給重排器後，結論就不同了：在下表 5 個資料集上都比單一檢索器（BM25、bge-large）好（dev all@10）：

| | BM25 | bge-large | 加權混合 | ＋重排 |
|---|---|---|---|---|
| SQuAD | 0.920 | 0.922 | 0.954 | **0.992** |
| HotpotQA | 0.594 | 0.806 | 0.828 | **0.876** |
| MultiHop-RAG | 0.624 | 0.554 | 0.610 | **0.792** |
| NeoQA | 0.608 | 0.340 | 0.432 | **0.730** |
| BioASQ | 0.526 | 0.472 | 0.570 | **0.708** |

代價是延遲：重排 100 段約 0.6–1.0 s，第一階段檢索只要 15–25 ms。

**2. 專業領域：強 embedding 單獨使用，比混合檢索＋重排更準，也快十倍以上。** holdout all@10，和混合檢索＋重排比較：

| 資料集 | 混合＋重排 | 強 embedding 單獨 | 差異 [95% CI] |
|---|---|---|---|
| FiQA | 0.723 | 0.853（Qwen3-Embedding-4B） | +0.130 [+0.087, +0.177] |
| SciFact | 0.820 | 0.900（Qwen3-Embedding-4B） | +0.080 [+0.027, +0.133] |
| NFCorpus | 0.727 | 0.785（Qwen3-Embedding-4B） | +0.058 [+0.012, +0.103] |
| MedQuAD | 0.907 | 0.947（EmbeddingGemma-300M） | +0.040 [+0.017, +0.067] |
| BioASQ | **0.717** | 0.533（Qwen3-Embedding-4B） | −0.183 [−0.237, −0.130] |

BioASQ 是例外：一題平均需要多篇摘要共同佐證，仍然要靠重排。

**3. 換成更強的 embedding 後，重排器反而成為上限。** 以 FiQA 為例（dev），把第一階段從加權混合檢索（bge-large＋BM25）換成 Qwen3-Embedding-4B 後，正解進入前 100 名的比例從 0.762 提高到 0.872，但交給 bge-reranker 重排後 nDCG@10 只從 0.433 變成 0.438，而 Qwen3-Embedding-4B 自己的排序是 0.586。在 SciFact、NFCorpus、FiQA、MedQuAD、BioASQ、HotpotQA、2Wiki、MuSiQue、MultiHop-RAG、NeoQA 上，只要最後一步是 bge-reranker，第一階段從加權混合檢索換成 Qwen3-Embedding-4B、harrier-0.6B 或 EmbeddingGemma（同樣再混入 BM25），nDCG@10 的差距多在 ±0.01 以內，最多 +0.016（MuSiQue，dev）。

任務指令對部分模型影響很大（dev）：harrier-0.6B 加上指令後，MedQuAD 的 nDCG@10 從 0.619 升到 0.756、HotpotQA 從 0.762 升到 0.846；Qwen3-Embedding 對指令則不敏感。

**4. 多跳：beam 檢索的關鍵是用多跳資料訓練，比放大模型更划算。** dev all@10：

| | 2Wiki | HotpotQA | MuSiQue | 延遲 |
|---|---|---|---|---|
| 單次檢索（bge-large） | 0.480 | 0.806 | 0.298 | 15 ms |
| beam，未微調 Qwen3-Embedding-0.6B | 0.704 | 0.792 | 0.360 | 150 ms |
| beam，未微調 Qwen3-Embedding-4B | 0.832 | **0.914** | 0.518 | 0.5 s |
| beam，微調 bge-base（110M） | 0.936 | 0.882 | 0.542 | 40 ms |
| beam，微調 Qwen3-Embedding-0.6B（LoRA） | **0.950** | 0.904 | **0.554** | 140 ms |

未微調的 4B 模型已經接近微調後的結果，但慢 3.5 倍以上。用同樣的訓練資料，微調後的 Qwen3-Embedding-0.6B 只比 bge-base 多 0.01–0.02，三個資料集都不顯著；在訓練時沒看過的 NeoQA 上則顯著較好（dev all@10 0.372 → 0.450，+0.078 [+0.034, +0.122]）。另外，2Wiki 上微調的 bge-small 也有 0.938（dev）、0.947（holdout），延遲 45 ms。多跳微調會讓其他領域退步：Qwen3-Embedding-0.6B 微調後 FiQA 從 0.714 降到 0.660（dev all@10，−0.054 [−0.082, −0.028]）。

**5. 跨文件題：把問題拆開，或讓 LLM 只負責前幾名。** MultiHop-RAG 的題目常比較兩家媒體的報導，依問題中的媒體拆成子查詢後，holdout all@10 從 0.807 升到 0.827。NeoQA 與 MuSiQue 交給 Self-Ask 找前 5 段、其餘由低成本方法補滿：NeoQA 從混合檢索＋重排的 0.697 升到 0.757，MuSiQue 從 4 跳 beam（微調 bge-base）的 0.573 升到 0.640（holdout all@10），代價是每題約 12 s。

**6. 長文件切塊：財報要先切小，並在每段前面加上公司與年度。** [Fin-RATE](https://arxiv.org/abs/2602.07294)（Jiang et al., 2026）原本以 SEC 財報的 item 為單位，平均一段約 2,649 個英文字，遠超過 embedding 模型 512 token 的輸入上限，後面的內容會被截掉。改成每段 256 字（重疊 64 字），並在每段前加一行 `company: … | fiscal_year: … | form: … | filed: … | section: …`。為了和原論文對照，這裡用全部 7,467 題，並採用論文的 Recall@10 定義（每題找回的正解段落比例，單位 %）：

| 設定 | 單份財報（DR） | 跨公司比較（EC） | 跨年度比較（LT） |
|---|---|---|---|
| 原論文各任務最佳檢索器 | 44.84 | 9.73 | 43.86 |
| 原始切法，BM25（重現原論文的 BM25 基準） | 43.8 | 4.7 | 45.7 |
| 原始切法，混合檢索＋重排 | 60.5 | 10.9 | 38.5 |
| 切塊＋標頭，混合檢索＋重排 | 86.7 | 28.6 | 68.0 |
| ＋依公司拆查詢 | **86.7** | **32.6** | **68.0** |

只看「有沒有找到正確的那份財報」，跨公司題找回 85.3% 的正解財報；段落層級的分數較低，是因為找到正確的財報，不代表該份財報裡每個正解段落都進了前 10 名（跨公司題平均有 4.4 個正解段落，分散在 2 份以上的財報）。原論文的主要指標是 LLM 評審的答題正確率，這裡只比較檢索。

**7. 拒答可以先看重排分數。** MultiHop-RAG 另外附有 301 題無答案題（問的事件不在語料中，不屬於上面的 dev / holdout），重排後第一名的分數中位數是 0.015，有答案的題目是 0.910（AUC 0.991）。以 0.1 為門檻，可以擋下 88.7% 的無答案題，只誤擋 1.4% 的有答案題，而且不需要呼叫 LLM。qwen3:4b 自己也能拒答 98.7% 的無答案題，但對有答案的比較題、時間題也拒答了 46.5%，這是此資料集答錯的主要來源。

**8. 資料本身的問題要先排除。** BioASQ 的原始語料有 12,220 段內容是字串 "nan"，dev 的正解有七成落在這些空段落上；清掉並重新對應後，all@10 從 0.23 變成 0.71。

**端到端答題**（qwen3:4b-instruct，給前 5 段，dev 200 題，評分方式同第一部分）：從 bge-small 單次檢索換成各資料集的最佳檢索後，答對率 HotpotQA 0.52 → 0.62、2Wiki 0.35 → 0.57、MuSiQue 0.07 → 0.18。

### 沒有效果的嘗試

數字為 dev all@10。

| 做法 | 結果 | 原因 |
|---|---|---|
| 用 NFCorpus / FiQA / BioASQ 訓練集微調 bge-base | 沒有提升，FiQA 反而退步 | 每個資料集只有幾千題訓練資料，微調後出現遺忘，通用檢索能力下降 |
| 用 NFCorpus / FiQA / BioASQ / SciFact 訓練集以 LoRA 微調 Qwen3-Embedding-0.6B | 只有 BioASQ 顯著提升（0.452 → 0.508，+0.056 [+0.018, +0.096]），其他三個訓練過的資料集沒有變化；沒訓練過的資料集退步：MedQuAD 0.888 → 0.854、HotpotQA 0.664 → 0.620、MultiHop-RAG 0.590 → 0.524（皆顯著）。微調後仍低於未微調的 Qwen3-Embedding-4B | 和 bge-base 相同：訓練資料少，微調後通用能力下降 |
| 用領域資料微調 bge-reranker | 只用困難負例訓練時排序崩壞（MedQuAD 0.87 → 0.12）；加入隨機負例後恢復，但仍略差於未微調版本 | 驗證時的候選（BM25 前 30）和實際評估（混合檢索前 100）分布不同 |
| 把只看問題的重排套在多跳 beam 的候選上 | MuSiQue 0.58 → 0.39、2Wiki 0.94 → 0.60 | 第二跳的段落和問題本身沒有字面關聯，只看問題就會被排掉 |
| Qwen3-Reranker 0.6B / 4B 疊在 bge 重排之後 | NeoQA 0.73 → 0.65、MultiHop-RAG 0.792 → 0.776，每題 2–8 s | 原因未確認 |
| LLM 迭代檢索（Self-Ask、IRCoT）單獨使用 | 只在 NeoQA 勝過重排，每題 8–12 s | 和低成本方法合併後才穩定有效（見主要發現第 5 點） |
| 偽相關回饋（把前 5 段的向量平均加回查詢）、重排深度從 100 加到 200、集合選擇（LLM 先列出回答需要的資訊，再挑能涵蓋這些資訊的段落組合） | 增益 ≤ 0.02 | 原因未確認 |

### 延遲與建索引成本

| 方法 | 每題延遲 p50 |
|---|---|
| BM25 | < 1 ms |
| bge-large | 15 ms |
| EmbeddingGemma-300M、harrier-0.6B、Qwen3-Embedding-0.6B | 40–50 ms |
| Qwen3-Embedding-4B | 55–65 ms |
| 多跳 beam（bge-base / Qwen3-Embedding-0.6B） | 40 / 140 ms |
| 混合檢索＋重排（前 100） | 0.55–1.0 s |
| 依實體拆查詢＋重排 | 0.96–5.3 s（視子查詢數與重排深度而定） |
| Self-Ask 前 5 名＋補滿 | 12–13 s |

建索引時，5.8 萬段的 FiQA 用 bge-large 約 4 分鐘，用 Qwen3-Embedding-4B 約 40 分鐘。

### 依題型的選擇

- 一般單跳問答：加權混合檢索＋重排。
- 專業領域問答（科學、醫療、金融）：Qwen3-Embedding-4B 或 EmbeddingGemma 單獨使用，不加 BM25、不重排；延遲 40–60 ms。
- 一題需要多篇佐證（BioASQ）：仍用混合檢索＋重排。
- 可以一步步追下去的多跳題：用多跳資料微調過的 beam 檢索，40–140 ms；要更準且能接受十幾秒，再加上 Self-Ask 前 5 名。
- 比較多家公司或媒體的題目：依問題中的實體拆查詢。
- 長財報：先切成 256 字並加上公司、年度標頭。
- 無答案題：重排第一名分數低於 0.1 時直接拒答。

### 重現

環境：`py -3.11 -m venv .venv-auto`，再依 `requirements-bench.txt` 安裝（torch 要裝對應 CUDA 的版本）。生成與 Self-Ask 用本機 Ollama 的 `qwen3:4b-instruct`。

```powershell
# 準備資料：通用資料集一次建立；其他資料集各有 tools/bench/prepare_*.py（固定 seed，會記錄來源與雜湊）
.\.venv-auto\Scripts\python.exe -m tools.bench.prepare_all
# 跑一組 資料集 × 方法（方法定義在 tools/bench/methods.py；已完成的組合會跳過）
.\.venv-auto\Scripts\python.exe -m tools.bench.run_matrix --datasets scifact,fiqa --methods bge-large,rr-bgev2m3_wrrf,qwen3-emb-4b-ti
# holdout 確認
.\.venv-auto\Scripts\python.exe -m tools.bench.run_matrix --split holdout --allow-holdout --datasets fiqa --methods qwen3-emb-4b-ti
# 兩個結果做 paired bootstrap
.\.venv-auto\Scripts\python.exe -m tools.bench.metrics results\bench\<run_a> --compare results\bench\<run_b>
# 多跳 beam 的 encoder 微調（LoRA）
.\.venv-auto\Scripts\python.exe -m tools.train.train_embed --base Qwen/Qwen3-Embedding-0.6B --task-instruct --lora 32 --lr 1e-4 --out models/qwen06-mh-lora --hop1 reachable
```

常用方法代號（定義在 `tools/bench/methods.py`）：`bm25`、`bge-large`、`wrrf_large_bm25`（加權混合檢索）、`rr-bgev2m3_wrrf`（加權混合檢索＋bge-reranker-v2-m3）、`qwen3-emb-4b-ti`（Qwen3-Embedding-4B，`-ti` 表示加任務指令）、`embeddinggemma-300m`、`mhbeam3_base-mh2`（3 跳 beam，encoder 為用多跳資料微調過的 bge-base；mh2 是第二版多跳訓練資料）。訓練指令的 `--hop1 reachable` 表示第一跳只用問題裡直接提到的段落當正例。

每次執行的設定與指標在 `results/bench/<run_id>/`（`config.json`、`metrics.json`），總表是 `results/bench/_index.jsonl`。長時間的實驗由 `auto/runner.py` 依序執行佇列中的工作：每個工作都會存進度，當機或關機後重新啟動會從中斷處接續（見 `auto/RECOVERY.md`）。

## 專案結構

```
app.py                     Streamlit 展示介面
src/rag_core.py            載入、索引、檢索（dense/BM25/hybrid）、Context、Ollama、解析、評分
notebooks/
  01_rag_squad.ipynb       50 題的入門範例（程式內嵌，方便逐步執行）
  02_collections_and_models.ipynb   Collection 隔離、無答案題、換生成模型
  03_benchmark.ipynb       主要評估：dev / holdout、dense / bm25 / hybrid、和基準並排比較
tools/datasets/            產生知識庫與題目的腳本
tools/ingest/              網頁規格頁 → Markdown 的匯入與驗證工具
eval/                      題目（jsonl）與 manifest
results/<run_id>/          每次執行的設定與完整紀錄
reports/                   人工核對、失敗案例、混合檢索分析（issue.md）
tests/                     pytest

檢索方法比較：
src/retrievers/            BM25、dense、加權 RRF、重排、多跳 beam、依實體拆查詢、Self-Ask 等檢索器
tools/bench/               資料集準備、方法定義（methods.py）、跑分矩陣、指標與 paired bootstrap
tools/train/               embedding（含 LoRA）與 reranker 微調
auto/runner.py             依序執行長時間實驗的佇列，當機後自動接續
results/bench/             每次執行的設定與指標，總表 _index.jsonl
```

## 已實作 / 只有介面 / 未來方向

**已實作且測試過**
- 第一部分：三個評估用知識庫（另有一個網頁匯入範例）的切換與隔離、dense / BM25 / hybrid 檢索、Top-k、No-RAG / RAG / oracle、引用檢查、JSONL 紀錄、Ollama 本地模型切換、人工核對流程、Streamlit 介面。
- 第二部分：14 個資料集的檢索評估，包含加權 RRF、重排、多跳 beam、依實體拆查詢、LLM 輔助檢索與 embedding 微調（含 LoRA）。

**只有介面**
- 雲端 OpenAI 相容 API（`OllamaGenerator` 的介面可以照樣實作一個）。
- Streamlit 展示介面目前只接第一部分的檢索方法。

**未來方向**
- 降低比較題、時間題的誤拒答（允許推論的 Prompt）。
- 依題型自動選擇檢索方法。
- HotpotQA 人工核對。

## 限制

**第一部分**
- 這是小樣本的工程診斷，不是通用 benchmark。
- 「答對」是自動評分（回答包含標準答案，或去除與題目重複的詞後 token F1 ≥ 0.5），只在 SQuAD 的 20 題上和人工判斷對照過；HotpotQA 沒有人工核對。
- 同設定、同呼叫順序可以完全重現；呼叫順序不同時，約 1% 的題目對錯會翻轉。
- 生成模型可能看過 Wikipedia；檢索模型的訓練資料是否包含這些資料集無法驗證。

**第二部分**
- 題目是從各資料集抽樣（每個資料集 dev 150–500 題、holdout 150–300 題）；部分資料集的語料是縮小過的段落池，只有 SciFact、NFCorpus、FiQA 的數字能和公開排行榜粗略對照。
- 答題正確率只在 HotpotQA、2Wiki、MuSiQue、MultiHop-RAG 上用 qwen3:4b 自動評分測過，沒有人工核對。
