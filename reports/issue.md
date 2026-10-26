# Stage 3：混合檢索（dense + BM25）

> 所有數字來自 `results/` 下的 run，可用 `03_benchmark.ipynb` 的 `LOAD_RUN` 重現表格。

**一句話結論：** 等權重的混合檢索在 SQuAD 上把正確段落往前排（hit@1 +8～10 個百分點），但 k=3 時答案的改善在 holdout 上只剩 +2、落在隨機範圍內；在 HotpotQA 上則明顯有害（holdout 答對 0.62 → 0.51）。**不建議當成預設檢索方法。**

## 1. 問題

SQuAD 開發集 50 題中，向量檢索（bge-small）有 6 題沒把正確段落排進 Top-3。人工核對後確認，其中 4 題是真正的檢索失敗（D7），模型在這些題目上只能拒答或答錯。

## 2. 可重現的案例

都在 `squad_v11_dev500` 上，k = 3。

| 題目 | 標準答案 | dense 名次 | BM25 名次 | 4B 的回答（dense） |
|---|---|---|---|---|
| What are the two ABC affiliates for Tampa, Florida? | WFTS-TV and WWSB | 4 | 1 | 拒答 |
| What chess grandmaster is also a university alumni? | Samuel Reshevsky | 5 | 1 | 拒答 |
| What was the name of the experiment that tested how life originated? | Miller–Urey experiment | 6 | 1 | 拒答（拿到 3 段講氧氣的段落） |
| Besides Britain and North America, where else did Huguenot refugees settle? | Holland, Prussia, and South Africa | 12 | 2 | 把其他段落的 20 多個地名全部列出 |

## 3. 根因假設

這些題目靠**罕見的專有名詞**（Tampa、chess、Miller–Urey）才能定位到正確段落。小型向量模型把語意壓縮成 384 維時，這類字詞的權重不夠，會被同一篇文章裡主題相近的段落蓋過。關鍵字檢索（BM25）對罕見字給很高的權重，正好可以補上。

**預期副作用：**
- BM25 可能因為常見字相同，撈到不相關段落 → 原本找得到的題目被「弄壞」。
- 撈到關鍵字相同但無關的段落，可能讓模型在**沒有答案的題目上亂答**。

## 4. 唯一改變的變數

| | 修改前 | 修改後 |
|---|---|---|
| 檢索 | dense（bge-small-en-v1.5，cosine） | **hybrid**：dense 名次和 BM25 名次用 RRF 融合 |

**RRF：** 融合分數 = 1/(60 + dense 名次) + 1/(60 + BM25 名次)，兩邊權重相同。用名次而不是原始分數，因為兩種分數的尺度不同，不能直接相加。BM25 比對的文字是「標題 + 段落」。

**固定不變：** Embedding 模型、索引、生成模型（qwen3:4b-instruct，temperature 0）、Prompt、k = 3、Context 上限、題目與評分方式。

**評分：**「答對」= 回答包含標準答案，或 f1_qstrip ≥ 0.5（D8，經 20 題人工核對選定）。

## 5. 結果（dev）

Run：dense `20261026_001846_bench_dense`；hybrid 檢索 `20261026_024324_bench_hybrid`、完整問答 `20261026_155225_bench_hybrid`。

### 5.1 檢索（全部 300 題，不呼叫 LLM，結果每次都相同）

| | dense | hybrid | 變化 |
|---|---|---|---|
| SQuAD hit@1 | 0.743 | 0.843 | +0.100 |
| SQuAD hit@3 | 0.887 | 0.947 | +0.060（救回 23、弄壞 5） |
| SQuAD MRR | 0.825 | 0.896 | +0.071 |
| HotpotQA hit@3（兩段都找到） | 0.617 | 0.490 | −0.127（救回 16、弄壞 54） |
| HotpotQA hit@10 | 0.850 | 0.850 | 0 |

第 2 節的 4 個案例在 hybrid 下的名次：Tampa 4 → 1、西洋棋 5 → 2、生命起源實驗 6 → 2、Huguenot 12 → 7。Huguenot 這題 BM25 單獨排第 2，融合後卻是第 7：dense 排太後面，把平均拉低了。等權重 RRF 的限制在 SQuAD 上也看得到。

### 5.2 完整問答（各 150 題）

| | dense | hybrid | 答對：救回／弄壞 |
|---|---|---|---|
| SQuAD 答對（RAG） | 0.833 | **0.893** | 13／4 |
| HotpotQA 答對（RAG） | 0.467 | 0.480 | 16／14 |
| 無答案題拒答（15 題） | 15/15 | 15/15 | 沒有新增亂答 |
| 無答案題拒答（擴充後 50 題，`20261026_172612` / `173052`） | 50/50 | 50/50 | 沒有新增亂答 |

**這些差距可信嗎？** oracle 模式兩次拿到的段落完全相同，答案的對錯只翻轉 0～2 題（150 題中）。所以 SQuAD 淨 +9 題是真實改善；HotpotQA 淨 +2 題在隨機範圍內，視為持平。

## 6. 為什麼 HotpotQA 沒有改善

- HotpotQA 每題的 8 個干擾段落，是原作者**用 bigram TF-IDF（關鍵字比對）挑出來的**，刻意設計成字面上很像。
- 驗證：150 題中，BM25 Top-3 裡的錯誤段落有 **93%** 是該題自己的干擾段落；dense 是 76%；隨機只有 0.2%。BM25 正好掉進資料集為關鍵字檢索設下的陷阱。
- RRF 的特性讓情況更糟：一段「兩邊都給中等名次」的干擾段落（3, 3 → 0.0317），會贏過「只有 dense 排第 1、BM25 排第 20」的正確段落（0.0289）。
- 答案沒有跟著掉，是因為 hybrid 比較常找到**含答案的那一段**：只找到一段正確段落的題目，hybrid 答對 30%，dense 19%。所以「兩段都要找到」的 hit@k 和能不能答對並不完全一致。

## 7. Holdout 驗證

SQuAD 100 題、HotpotQA 100 題、無答案題 50 題，開發期間都沒有看過；設定和 dev 完全相同，只把 `SPLIT` 改成 `"holdout"`。
Run：dense `20261026_185546_bench_dense_holdout`、hybrid `20261026_191449_bench_hybrid_holdout`。

| | dense | hybrid | 救回／弄壞 |
|---|---|---|---|
| SQuAD hit@1 | 0.700 | 0.780 | |
| SQuAD hit@3 | 0.890 | 0.910 | 6／4 |
| SQuAD MRR | 0.795 | 0.854 | |
| SQuAD 答對（RAG） | 0.830 | 0.850 | 6／4 |
| HotpotQA hit@3 | 0.790 | **0.580** | 1／22 |
| HotpotQA 答對（RAG） | 0.620 | **0.510** | 1／12 |
| HotpotQA 答對（bridge／comparison） | 0.671／0.458 | 0.579／0.292 | |
| 無答案題拒答（50 題） | 49/50 | 49/50 | 同一題 |

**怎麼讀：**
- **隨機差異：** oracle 兩次拿到的段落相同，200 題中對錯只翻轉 1 題。
- **SQuAD：** 改善集中在前 1～2 名（hit@1 +8、hit@2 +10、MRR +0.06）。但 dense 的 hit@3 已經 0.89，k=3 時只多救回淨 2 題，答案 +2 個百分點（6 對 4），**不足以說明有改善**。dev 上的 +6 在 holdout 上縮小了。
- **HotpotQA：** 檢索弄壞 22 題、只救回 1 題，答案淨少 11 題，**明顯有害**，比 dev 更明顯。原因和 dev 相同：BM25 Top-3 裡的錯誤段落有 **95%** 是該題自己的干擾段落（dense 82%）。
- **無答案題的 1 題「亂答」不是副作用：** 題目「What things did the network concentrate on」原本問的是封包交換的網路，脫離原文後也能讀成「電視網」。A 裡的 ABC 段落確實寫了這家電視網重視行為與人口統計資料，模型有依據。兩種方法在這題的表現一樣。
- **dev 和 holdout 的絕對數字差很多**（例如 HotpotQA dense hit@3：dev 0.617、holdout 0.790），是 100 題樣本之間的差異。所以要比較的是**同一批題目上兩種方法的差**，不是跨批次的絕對值。

## 8. 結論

- **SQuAD：方向成立，但幅度比 dev 看到的小。** 混合檢索穩定地把正確段落往前排（兩批題目的 hit@1 都 +8～10），所以在**只給模型 1～2 段**的情境下有價值；k=3 時 dense 已經很少漏，答案的改善在 holdout 上不顯著。
- **dev 高估了改善幅度。** 這個假設是看 dev 前 50 題的失敗案例想出來的，dev 上的結果自然偏樂觀。holdout 的意義就在這裡。
- **HotpotQA：假設不成立，而且有害。** 原因可以用資料集的建構方式解釋：干擾段落是用關鍵字相似度挑的。
- **整體判斷：** 不把等權重混合檢索當預設方法。是否採用要看資料特性：題目靠專有名詞定位、而且只能給模型很少段落時有幫助；資料中有大量字面相似的干擾內容時有害。

## 9. 限制

- 小樣本工程診斷（dev 每個資料集 150～300 題、holdout 100 題），**不能外推成通用的 RAG benchmark 結果**。100 題的樣本之間差異就有 10～17 個百分點（見第 7 節）。
- 「答對」是自動評分，只在 SQuAD 的 20 題上和人工判斷對照過；HotpotQA 沒有人工核對，已知有評分誤差（例如「Ed Roland」對上「Edgar Eugene Roland, Jr.」）。
- RRF 的權重和常數（1:1、60）沒有調整；調整會是看過 dev 結果後的第二輪實驗，必須另外用 holdout 確認。
- 兩個資料集都來自 Wikipedia，模型可能看過。No-RAG 答對率低（SQuAD 0.147、HotpotQA 橋接題 0.086），生成端受影響有限；檢索模型的訓練資料無法驗證。
