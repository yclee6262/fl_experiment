# Stage 3 模型查詢快取：低品質情境工程 pilot

## 問題與改動

原五候選初始 α 方法要各自執行一次 Stage 3。`best_of` 中的 SLSQP 與 Custom 搜尋會在同一次求解中反覆查詢部分相同的 local-model 輸入。本次先檢查能否少呼叫模型，並要求交付 X、選解及 pooled 評分不變。此處的「API 請求」是程式對 `AgentNode.api_predict` 的呼叫；實驗使用本機模型，尚未量測網路計費或延遲。

新增 `run_validation_calibrated_alpha.py --cache-stage3`，只在**單一候選 Stage 3 求解期間**快取每名 `AgentNode` 的預測。鍵使用形狀與模型實際接收的 `float32` 輸入位元；模型權重固定時，相同鍵的預測可重用。每次候選求解都重新建立快取，完成或異常退出均恢復原 API。選解器及 pooled evaluator 不使用快取，也不參與 α 或 X 的決策。預設關閉，舊輸出與成本口徑不變。開啟時 `stage3_requests`／`stage3_input_samples` 計實際模型呼叫；新增 `stage3_cache_hits`／`stage3_cache_misses` 保存重用情況。對非 `AgentNode` 的 API，此選項會明確拒絕。

## 診斷與 pilot

使用已封存的低品質正式資料 seeds 2400–2404、五個候選、25 次求解，`audit_stage3_query_reuse.py` 重播模型與 proposal；在每次候選內按模型可見 `float32` 輸入統計，原呼叫 11,166 次，其中約 **10.24%** 可在候選內重用；若跨候選共用快取，另有約 2.90%，但這次未實作跨候選共用。若鍵要求原始陣列位元完全相同，候選內可重用約 1.64%；`float32` 口徑之所以合理，是因 `AgentNode.api_predict` 本來就在預測前把輸入轉為 `float32`。這是保存模型上的成本診斷，不是新的品質比較。

真正流程 pilot 使用低品質情境 seeds 0–4，比較既有未快取的 `alpha_three_candidate_lowquality_pilot_v1_20261004/` 與新 `alpha_stage3_cache_pilot_v2_20261004/`。兩者同 seed 使用相同資料、local-model checkpoints、proposal 與 pooled evaluator；五組 α 均執行，另外保存 A、B、原五候選與三候選方法摘要。結果：

| 驗收項目 | 結果 |
|---|---:|
| 候選／方法摘要數 | 25／20，兩批鍵集合一致 |
| 候選 X／α 最大差 | 0／0 |
| 候選／摘要 pooled 分數最大差 | 0／0 |
| 方法摘要交付 X 最大差 | 0 |
| Stage 3 原／快取後模型呼叫 | 8,334／7,305 |
| 省下的 Stage 3 模型呼叫 | 1,029，約 12.35% |
| 重新載入交付與選解驗證 | 通過；5 seeds、10 個選解變體 |

快取後的 `stage3_cache_hits=1,029`、`stage3_cache_misses=7,305`，與原始請求數相加核對一致。這是 **Stage 3 呼叫數**的節省，不是整個實驗或真實部署總成本的 12.35%；共同訓練、資格檢查、proposal、初始化、標記取得與外部評分另列。`float32` 輸入鍵只適用於目前 `AgentNode` 的固定模型行為；模型更新、非確定性 API 或不同輸入轉換都必須另外驗證。

第一個 `alpha_stage3_cache_pilot_v1_20261004/` 暴露 runner 參數傳遞缺口：命令列旗標未進入候選求解，因此沒有啟用快取。該批保留為失敗診斷，**不計入快取驗收或節省數字**。修正後於新目錄執行 v2，沒有覆蓋舊結果。

## 研究判讀與下一步

本次完成的是固定五候選方法的**等解成本優化**，並沒有找到 pooled 品質更好的新初始 α，也沒有證明貨幣淨收益。5-seed pilot 的實際呼叫減幅與保存模型診斷方向一致；若要把快取納入新的正式方法成本，需固定新版本並在獨立 seeds 上核對交付等價、失敗率及實際供應方計費單位。原五候選的品質比較與先前負面結論維持原版本解讀。

可重播診斷：`python audit_stage3_query_reuse.py alpha_weight_outputs/alpha_three_candidate_lowquality_formal_v1_20261004 --key-mode model_float32 --output <new-json-path>`。流程 pilot 命令：`python run_validation_calibrated_alpha.py --output-dir alpha_weight_outputs/alpha_stage3_cache_pilot_v2_20261004 --scenario low_quality_25pct --seeds 0,1,2,3,4 --portfolio-three --cache-stage3`。重評命令：`python verify_validation_calibrated_alpha.py alpha_weight_outputs/alpha_stage3_cache_pilot_v2_20261004`。
