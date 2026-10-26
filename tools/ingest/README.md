# 網頁規格 → Markdown 匯入工具

把「只寫在官網、沒有 PDF」的規格頁，轉成 `collections/real_docs/` 用的 `.md`，並自動檢查轉換有沒有漏字或改錯數字。不使用 OCR。

## 安裝（Windows）

```powershell
.\.venv\Scripts\python.exe -m pip install requests beautifulsoup4 pytest
```

## 每一頁的流程

```powershell
# 1. 存原始快照（之後不可修改）
python -m tools.ingest.snapshot "https://www.belkin.com/support-article/?articleNum=335765" `
    --doc-id belkin_wch014_faq --profile belkin-faq

# 2. 轉成 Markdown（只取文章區塊，丟掉導覽列與頁尾）
python -m tools.ingest.html_to_md raw/belkin_wch014_faq.html `
    --out collections/real_docs/belkin_wch014_faq.md --manufacturer Belkin --model WCH014

# 3. 人工處理：打開圖片寫標註、把原文疑點記在 known_source_issues

# 4. 驗證，全部 PASS 才算完成
python -m tools.ingest.verify collections/real_docs/belkin_wch014_faq.md
```

若第 1 步顯示 `no <h1> found` 或 `no <table> after <h1>`，代表頁面是 JavaScript 動態載入。請用瀏覽器開啟該頁，「另存新檔 → 網頁，僅 HTML」，再加上 `--from-file 存下來的.html` 重跑。

## verify 檢查什麼

| 檢查 | 抓得到的錯誤 |
|---|---|
| `raw_sha256` | 原始快照在轉換後被改動 |
| `numbers` | 數字漏掉、改錯、自行加入原文沒有的數字、悄悄「更正」原文 |
| `entries` | 整筆 Q/A 被漏掉 |
| `sentences` | 某格文字沒有完整出現在 .md |
| `images` | 圖片還沒有人工標註 |
| `manual` | 標註裡還留著 TODO |

人工標註的格式如下。標註內容不參與數字與文字比對，但 metadata 會記錄是誰、哪天寫的：

```markdown
<!-- manual:start image=sf335765-001_en_v2.png by=yourname date=2026-10-22 -->
> [人工描述] ...
<!-- manual:end -->
```

## 目前的限制

- `belkin-faq` profile 是依照我透過網頁讀取工具看到的頁面結構寫的，**尚未在真實伺服器 HTML 上跑過**（我的環境連不到 belkin.com）。測試用的是重建的範例 `tests/fixtures/belkin_faq_sample.html`。第一次在你電腦上跑時，請確認 Q/A 數量與網頁一致。
- `generic` profile 只做數字、圖片、雜湊檢查，沒有逐格文字與條目數比對。其他廠商的頁面若結構固定，建議各加一個專屬 profile。
- 數字比對看的是「數字＋單位」，像 `5V – 21V` 會拆成 `5V`、`21V` 兩個 token。範圍符號本身（–）是否保留，由 `sentences` 檢查負責。
- 自動檢查只證明「轉換忠於原文」，不證明原文本身正確。
