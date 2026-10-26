# 資料來源

## SQuAD 1.1（dev split）

- 檔案：`raw/squad/dev-v1.1.json`
- 下載自：https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json （SQuAD 官方 GitHub repo）
- 下載日期：2026-10-22
- sha256：`95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9`
- 內容：48 篇 Wikipedia 文章、2,067 個段落、10,570 題
- 授權：CC BY-SA 4.0。引用：Rajpurkar et al., *SQuAD: 100,000+ Questions for Machine Comprehension of Text*, EMNLP 2016.

只用 dev split，不用 train split（約 30 MB，這個專案用不到）。

## HotpotQA（dev，distractor setting）

- 檔案：`raw/hotpotqa/validation-00000-of-00001.parquet`（**不在版控裡**，27 MB，需自行下載）
- 下載自：https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet （HotpotQA 官方帳號在 Hugging Face 的 parquet 版本；原始 JSON 的官方下載點 `curtis.ml.cmu.edu` 在這個環境連不到）
- 下載日期：2026-10-25
- sha256：`c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6`
- 內容：7,405 題，每題 10 段（2 段正確段落 + 8 段干擾段落）
- 授權：CC BY-SA 4.0。引用：Yang et al., *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*, EMNLP 2018.
- 轉換（需要 `pyarrow`）：`python -m tools.datasets.prepare_hotpotqa data/raw/hotpotqa/validation-00000-of-00001.parquet` → `collections/hotpot_dev400/`（3,954 段，不在版控裡）與 `eval/hotpot_dev.jsonl`、`eval/hotpot_holdout.jsonl`

## 重建所有知識庫與題目

```powershell
python -m tools.datasets.prepare_squad data/raw/squad/dev-v1.1.json            # Collection A + 最早的 dev/holdout 各 50
python -m tools.datasets.prepare_squad_b data/raw/squad/dev-v1.1.json          # Collection B + 15 題
python -m tools.datasets.expand_squad_questions data/raw/squad/dev-v1.1.json   # dev 300 / holdout 100
python -m tools.datasets.expand_noanswer data/raw/squad/dev-v1.1.json          # 無答案題 dev 50 / holdout 50
python -m tools.datasets.prepare_hotpotqa data/raw/hotpotqa/validation-00000-of-00001.parquet
```

全部用固定 seed，重跑結果相同。重建某個知識庫前，先刪除對應的 `collections/` 資料夾。

## 評估時要注意

- **`answer_also_in_docs`**：100 題中有 37 題的答案字串也出現在其他段落。這只是字串比對，不代表那些段落真的回答了這題（例如 "Anglican" 出現在別的段落，但講的不是同一所學校）。這些題目檢索「沒命中 gold 段落」時，請先人工看一下取回的段落，再判定是不是 retrieval failure。
- **多個標準答案**：52 題有 2 個以上的答案寫法，計算 EM/F1 時要取對所有答案中的最高分。
- **公開資料**：SQuAD 和 Wikipedia 很可能出現在模型訓練資料裡。No-RAG 答對不代表題目太簡單，RAG 答對也不代表檢索有效，要同時看 gold 段落有沒有進 Top-k。
- **holdout 題目**：開發期間不要打開 `eval/*holdout*.jsonl` 調整設定，最後驗收才跑。
