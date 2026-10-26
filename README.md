# RAG 工作台

一個小型、可診斷的 RAG（Retrieval-Augmented Generation）系統：可以切換知識庫、檢索方法、生成模型和 Top-k，每一步的中間結果都會留下紀錄。另外附有一組評估流程，用來區分「檢索失敗」和「生成失敗」，並完成一次有前後對照的改善實驗。

## 系統流程

```
事前：段落(.md) ──► bge-small 向量 ──► 索引快取 index/<Collection>__<模型>__<內容指紋>/
                └─► BM25（標題+段落，記憶體內）

每次查詢：問題 ──► 檢索 Top-k（dense / bm25 / hybrid=RRF）──► 組 Context（附段落 ID，超過上限整段捨棄）
             ──► Ollama 生成（只能依據段落、要附引用、證據不足就拒答）──► 解析答案與引用 ──► 評分、寫入 JSONL
```

三種模式並排比較：`no_rag`（只給問題）、`rag`（給檢索結果）、`oracle`（直接給正確段落）。oracle 答對、rag 答錯 → 問題在檢索；oracle 也答錯 → 問題在生成或題目。

## 主要結果（qwen3:4b-instruct，k = 3）

| | dev：dense | dev：hybrid | **holdout：dense** | **holdout：hybrid** |
|---|---|---|---|---|
| SQuAD 檢索 hit@1 | 0.743 | 0.843 | 0.700 | 0.780 |
| SQuAD 檢索 hit@3 | 0.887 | 0.947 | 0.890 | 0.910 |
| SQuAD 答對 | 0.833 | 0.893 | 0.830 | 0.850 |
| HotpotQA 檢索 hit@3（兩段都找到） | 0.617 | 0.490 | 0.790 | 0.580 |
| HotpotQA 答對 | 0.467 | 0.480 | 0.620 | 0.510 |
| 無答案題拒答（50 題） | 50/50 | 50/50 | 49/50 | 49/50 |

dev：檢索 300 題、答對 150 題；holdout：各 100 題，開發期間沒有看過。

- **混合檢索（等權重 RRF）不適合當預設方法。** 在 SQuAD 上穩定地把正確段落往前排（hit@1 +8～10），但 k=3 時答案的改善在 holdout 上只有 +2，不顯著；在 HotpotQA 上明顯有害（holdout 答對 −11）。HotpotQA 的干擾段落是用關鍵字相似度挑出來的，BM25 Top-3 裡的錯誤段落有 95% 正是這些干擾段落。詳見 [`reports/issue.md`](reports/issue.md)。
- **dev 高估了改善：** 假設是看 dev 的失敗案例想出來的，SQuAD 的答案改善在 dev 是 +6、在 holdout 只剩 +2。
- **生成模型比較（02）：** 1.7B 在沒有答案時會編造（6/15）；4B 和 9B 答對率相同，9B 較慢且常照抄範本格式 → 選 4B。
- **No-RAG 答對率**（SQuAD 0.147、HotpotQA 橋接題 0.086）偏低，代表模型多半不是靠預先背過的知識在答題。
- **holdout 無答案題的 1 題「亂答」** 是題目本身有歧義（「the network」在 A 裡也能指電視網，而且有原文依據），兩種方法表現相同。

## 快速開始（Windows）

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

## 資料

| Collection | 內容 | 題目 |
|---|---|---|
| `squad_v11_dev500` | SQuAD 1.1 dev 的 11 篇文章，534 段 | dev 300、holdout 100 |
| `squad_v11_b5` | 另外 5 篇主題不重疊的文章，185 段 | 拿去問 A 當作無答案題：dev 50、holdout 50 |
| `hotpot_dev400` | HotpotQA dev（distractor）400 題的所有段落，3,954 段 | dev 300、holdout 100 |
| `real_docs` | 網頁規格頁匯入工具的範例（1 份） | — |

資料都由 `tools/datasets/` 的腳本以固定 seed 產生，來源、雜湊、授權見 `data/README.md` 與 `eval/*_manifest.json`。SQuAD 和 HotpotQA 皆為 CC BY-SA 4.0。

## 專案結構

```
app.py                     Streamlit 展示介面
src/rag_core.py            載入、索引、檢索（dense/BM25/hybrid）、Context、Ollama、解析、評分
notebooks/
  01_rag_squad.ipynb       最早的 50 題實驗（程式內嵌，方便逐步看）
  02_collections_and_models.ipynb   Collection 隔離、無答案題、換生成模型
  03_benchmark.ipynb       主要評估：dev / holdout、dense / bm25 / hybrid、和基準並排比較
tools/datasets/            產生知識庫與題目的腳本
tools/ingest/              網頁規格頁 → Markdown 的匯入與驗證工具
eval/                      題目（jsonl）與 manifest
results/<run_id>/          每次執行的設定與完整紀錄
reports/                   人工核對、失敗案例、Stage 3 報告
tests/                     pytest
```

## 已實作 / 只有介面 / 未來方向

| 已實作且測試過 | 只有介面 | 未來方向 |
|---|---|---|
| 三個知識庫切換與隔離、dense / BM25 / hybrid 檢索、Top-k、No-RAG / RAG / oracle、引用檢查、JSONL 紀錄、Ollama 本地模型切換、人工核對流程、Streamlit 介面 | 雲端 OpenAI 相容 API（`OllamaGenerator` 的介面可以照樣實作一個） | 加權 RRF、多跳檢索（HotpotQA 橋接題）、允許推論的 Prompt（比較題）、HotpotQA 人工核對、虛構內容的資料集（例如 NEOQA） |

## 限制

- 這是小樣本的工程診斷，不是通用 benchmark。
- 「答對」是自動評分（回答包含標準答案，或刪掉題目字後 F1 ≥ 0.5），只在 SQuAD 的 20 題上和人工判斷對照過；HotpotQA 沒有人工核對。
- 同設定、同呼叫順序可以完全重現；呼叫順序不同時，約 1% 的題目對錯會翻轉。
- 生成模型可能看過 Wikipedia；檢索模型的訓練資料是否包含這些資料集無法驗證。
