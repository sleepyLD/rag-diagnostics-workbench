# 當機 / 關機後怎麼恢復

**只要重新雙擊 `auto\start_runner.cmd`。** 其餘都會自動處理：

| 中斷時正在做什麼 | 重開後 |
|---|---|
| 檢索評估（一格 dataset × method） | 從 `results/bench/_partial/` 接續：已完成的題目不重跑（每批寫入後 fsync） |
| 跑分矩陣（多格） | 已完成的格子跳過（`results/bench/_index.jsonl`），只補沒完成的 |
| 微調（embedding / cross-encoder / reranker） | 從 `models/<name>/_ckpt/checkpoint-*` 最新一份接續（每 10% 步數存一次，只留最新一份） |
| 合成問題生成（GPL） | 從最後寫入的 anchor 之後繼續 |
| 下載資料集 / 模型 | Hugging Face 會續傳 |
| 寫向量索引寫到一半 | 不會被誤用：`meta.json`（或 `ok`）最後才寫，缺它就重建；檔案都是先寫 `.tmp` 再改名 |

- 被中斷的工作會排回佇列原位，最多嘗試 3 次（`auto/recovery_log.jsonl` 有紀錄；日誌接在 `auto/logs/<id>.log` 後面）。
- 乾淨地停：在 `auto\` 放一個名為 `STOP` 的空檔案（5 秒內停止，下次啟動會自動刪掉）。
- 不要在工作管理員只結束 runner 的 python.exe（子程序可能留下來佔 GPU）；要停就關視窗或用 STOP。
- Ollama 的模型由 Ollama 自己管理，閒置 5 分鐘後釋放 VRAM；`ollama stop <model>` 可立即釋放。

## 卡死監控（watchdog）
- 每個工作都會回報進度到 `auto/progress/<job>.json`（`auto/heartbeat.json` 的 `progress` 欄位也看得到：完成數、百分比、預估剩餘分鐘）。
- **40 分鐘內既沒有任何輸出、也沒有進度更新** → 視為卡死（例如無窮迴圈），runner 結束整棵程序樹（含孫程序），自動重試（總共最多 3 次）。個別工作可在 job 檔設 `"stall_minutes"`。
- 存檔點：評估每 ≤10% 題目寫一次；訓練每 10% 步數存 checkpoint；合成問題每題寫入。
- runner 重開時，若上次的工作程序還活著（孤兒），會先把它結束再重排。
- 子程序會監看父程序，父程序消失就自行退出，不會留在 GPU 上。

建議（需使用者自己設定）：
- Windows「電源與睡眠」→ 插電時「睡眠：永不」，避免長時間訓練被睡眠打斷。
- 想開機自動恢復：在 `shell:startup` 資料夾放一個 `start_runner.cmd` 的捷徑。
