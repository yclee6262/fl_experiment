# 100 筆可信盲測的局部風險初始 α：基本情境驗證

## 問題與方法

原五候選用 A、B、MSE power 1.5、Corr precision 0.5、Target-local precision 各求解一次 Stage 3，並用原 20 筆 Stage 0 盲測上的 B 權重 ridge 有號殘差校正分數選一個可行 X。新法保留原五個候選與選解規則，只新增 `local_bias_risk_100`：使用與原 Stage 0 前 20 筆完全相同的 100 筆合成真值標記，在各 local model 殘差上擬合 ridge 局部偏差，將 proposals 平均位置的預測偏差平方加回歸後變異作風險，正規化風險倒數為第六組初始 α，再多做一次 Stage 3。原 Stage 0 的 `MSE_i` 與內部選解校正仍只使用 20 筆；pooled evaluator 與 truth 都只在交付後評分。

這個資訊採購假說付出的資源是額外 80 筆可信標記、初始化時的 local-model 批次請求與第六次求解。合成環境可由真實函數產生標記；實際使用時其取得價格未知。α 是資訊權重，不是付款或經濟公平保證。

## 開發、pilot 與正式區分

在已用於多次挑法的 510 個 seeds 上，這個六候選相對五候選的描述性 pooled 平均差約 -0.000637；各 cohort 方向不一致。這是選法後的開發材料，不作確認檢定。來源為 `alpha_weight_outputs/alpha_extended_selector_first30_development_20261004.csv`、`alpha_extended_selector_replication60_development_20261004.csv`、`alpha_expanded_audit_development_700_20261004.csv`、`alpha_expanded_audit_development_800_1279_20261004.csv`。

seeds 0–4 pilot：20／20 摘要條件、10／10 選解重算、全部有效；原 A、B、五候選 X、α 與 pooled 分數和舊 pilot 完全相同。五候選／新六候選 pooled 均值同為 0.014757，五個 seed 均未選中新增候選。新法平均多 406.2 次模型請求及 80 筆標記。pilot 只驗流程與成本計數。

正式執行前，在 `docs/alpha_expanded_audit_formal_preregistration_20261004.json` 固定全新 seeds 1300–1479、來源 14 個檔案雜湊、選解與成本口徑。主要門檻是：180 個完整可行的配對，`E_new-E_five` 的 10,000 次 seed-bootstrap 雙側 95% 區間上限小於 0，且新法無更多失敗；通過才擴到複雜情境。模型請求、批次輸入、本地時間與 80 筆標記另報，不預設實際貨幣價格。

## 獨立 180-seed 正式結果

正式輸出 `alpha_weight_outputs/alpha_expanded_audit_formal_v1_20261004/` 完整；兩方法各 180／180 個有效可行解，180／180 配對，360／360 選解變體可由保存模型與盲測資料重算，來源 14／14 雜湊相符，重載外部評分檢查通過。原始資料、模型、manifest、preregistration 與 source snapshot 保留。

| 指標 | 原五候選 | 100 標記新六候選 |
|---|---:|---:|
| 平均 pooled 絕對誤差 | 0.014168694 | 0.013765638 |
| 平均 truth 絕對誤差（診斷） | 0.011920712 | 0.011142495 |
| 平均模型請求／方法 | 1891.517 | 2279.217 |
| 平均本地方法時間（秒） | 0.122217 | 0.157312 |
| 額外可信標記／條件 | 0 | 80 |

新減舊 pooled 配對平均差 **-0.000403056**；95% 區間 **[-0.001273042,+0.000408086]**，跨零。較好／相同／較差為 **25／135／20**；45／180 個 seed 選中新增 `local_bias_risk_100`。pooled 與 truth 的逐 seed 優劣方向衝突 **18／180**。新法平均多 **387.7 次請求**（約 20.50%），且每條件需額外 80 筆可信標記。本地方法時間平均多約 0.0351 秒；此時間不是網路延遲。

若每降低一單位 pooled 誤差的價值是 `v`、每次 local-model 請求價格為 `c`、每筆新增可信標記價格為 `l`，只按這批觀察均值算的損益平衡條件是 `v×0.000403056 > c×387.7 + l×80`。由於品質改善未達統計門檻，且 `c`、`l`、`v` 沒有實際估值，不能宣稱淨收益；這個條件僅說明所需價值比例。

**決策：正式品質門檻未通過。** 開發訊號在新 seeds 上沒有形成穩定優勢；保留五候選為對照，依事前規則不啟動複雜情境。不得用這 180 seeds 再挑同法參數後稱為獨立複驗。若提出不同方法，需定義新版本與未使用的 seeds。

## 重現與欄位

- 執行命令、環境版本、seed 清單、主判準與來源 SHA-256 在 `docs/alpha_expanded_audit_formal_preregistration_20261004.json`；同版來源在 `docs/alpha_expanded_audit_source_snapshot_20261004/` 與正式輸出的 `source_snapshot/`。
- `reference_clean/summary.csv` 每 seed 有 A、B、原五候選、新六候選；`candidate_rows.csv` 每 seed 六個單獨求解；`shared_costs.jsonl` 保存原 20 與擴充 100 筆標記、proposals、模型路徑及共用成本。
- `trusted_label_count`、`extra_trusted_labels`、`trusted_label_source` 明示標記數與來源；`initialization_requests`、`stage3_requests` 只計模型 API 請求，新增標記費用另列。`external_global_error` 是主要 pooled 誤差，`final_target_error` 是 truth 診斷，`final_box_feasible` 與 `max_box_violation` 記交付限制。
- 驗證結果在 `selection_verification.json`，固定規則判定在 `expanded_audit_decision.json`。執行前 81 項回歸測試通過，來源雜湊於完成後再核對一致。
