# 乾淨情境的初始 α：校正式多起點結果

## 問題與方法

本實驗固定二維、target 0、參與者 IDs 1／2／20 與 box [-1, 1]，只比較靜態初始 α；不做動態更新、pruning 或付款。主要品質是交付 X 的 pooled evaluator 絕對目標誤差。pooled 與合成真實函數只在選擇 X 後評分，不進入方法決策。

基準 A 是 Uniform；B 依 Stage 0 盲測 MSE 倒數正規化。新方法先建立 A、B、MSE 倒數 1.5 次方、0.5 殘差相關性修正、目標附近誤差加權五組 α，各自執行 Stage 3。在 20 筆 Stage 0 盲測標記上，將 B 加權集成的有號預測殘差對 [1, x] 做 ridge 校正，懲罰為 [0.01, 1, 1]；從可行候選 X 中選取校正後預測最接近 target 的解。這是初始 α 的選擇流程，需五次倒推，並非低成本的單一公式。品質與模型查詢成本都列入判讀。

## 證據層級

舊 seeds 300–329、400–429 及 pilot seeds 0–4 曾參與方法開發，不能作新方法的獨立優勢證據。首次獨立正式比較使用 seeds 500–529；方法平均 pooled 誤差低於 A、B，但預登錄的兩基準 Bonferroni 同時區間跨零。保持程式與公式不變後，另外註冊 seeds 600–659 作獨立複驗，並要求該批資料單獨通過兩個 97.5% 配對 bootstrap 區間完全低於零、完整有效配對且無較多失敗。

| Cohort | 方法 | 有效數 | 平均 pooled 誤差 | 平均真實函數誤差 | 每條件平均模型請求 |
|---|---|---:|---:|---:|---:|
| 首次正式，500–529 | A | 30/30 | 0.019111 | 0.015207 | 339.8 |
| 首次正式，500–529 | B | 30/30 | 0.018444 | 0.015631 | 394.7 |
| 首次正式，500–529 | 校正式多起點 | 30/30 | 0.014730 | 0.010347 | 1923.6 |
| 獨立複驗，600–659 | A | 60/60 | 0.016663 | 0.018045 | 341.75 |
| 獨立複驗，600–659 | B | 60/60 | 0.016652 | 0.015049 | 341.60 |
| 獨立複驗，600–659 | 校正式多起點 | 60/60 | 0.013275 | 0.012824 | 1782.45 |

配對差定義為 E_多起點 − E_基準，負值表示多起點較好。各區間以 seed 為單位重抽樣 10,000 次，固定分析 RNG；兩基準比較各使用 97.5% 區間作 Bonferroni 同時判斷。

| Cohort | 比較 | 平均配對差 | 97.5% 同時區間 | 判定 |
|---|---|---:|---:|---|
| 首次正式 | 對 A | -0.004380 | [-0.010167, +0.000494] | 未通過 |
| 首次正式 | 對 B | -0.003714 | [-0.008329, +0.000319] | 未通過 |
| 獨立複驗 | 對 A | -0.003388 | [-0.006443, -0.000491] | 通過 |
| 獨立複驗 | 對 B | -0.003377 | [-0.006405, -0.000930] | 通過 |

複驗 180/180 條摘要條件有效且 box 可行，60/60 個內部選擇可由保存的模型、盲測資料、候選解獨立重算。多起點對 A、B 的 pooled／truth 個案改善方向各衝突 13/60。target 附近 pooled evaluator 的測試樣本中位數 258.5（範圍 232–291），target-near truth MSE 中位數約 0.000101；這是合成資料中的局部診斷，不能推成真實部署可靠性。隔離重播 seed 600 的 coalition、proposal、盲測資料、local models、三方法的 α／X／pooled 分數與選擇均一致，數值最大差 0。

在複驗中，多起點平均模型請求約為 B 的 5.22 倍；方法內耗時約 0.0728 秒，B 約 0.0136 秒。時間為本地方法計時，不含每 seed 共用的訓練、資格、proposal 與 pooled evaluator 建立；若 local model API 具有實際網路或收費成本，效益是否值得額外請求需另定價評估。α 依事前預測品質與系統性殘差挑選，具有資訊配置上的解釋，尚未證明誘因相容、分潤公平或淨經濟效益。

相對 B，平均多用 1440.85 次模型請求，平均 pooled 誤差少 0.003377。若每次請求成本為 c、每降低 0.001 誤差的應用價值為 v，僅按這兩項計算的損益平衡條件是 v > 426.65c。此換算只是把未知價格顯性化；不能把合成誤差直接當成實際貨幣利益。

## 讀取與重現

- 首次正式：`alpha_weight_outputs/alpha_calibrated_multistart_formal_v1_20261003/`；事前規則：`docs/alpha_calibrated_multistart_formal_preregistration_20261003.json`。
- 獨立複驗：`alpha_weight_outputs/alpha_calibrated_multistart_replication_v1_20261003/`；事前規則：`docs/alpha_calibrated_multistart_replication_preregistration_20261003.json`。
- 主要結果：各 cohort 的 `decision.json`；逐條件 X、α、外部分數、失敗與成本見 `reference_clean/summary.csv`，五個候選見 `candidate_rows.csv`，共享成本與 blind test 見 `shared_costs.jsonl`。
- 驗收：`verification.json` 檢查外部模型重載、X 與來源／artifact；`selection_verification.json` 重新計算校正係數、候選內部分數、選擇及查詢成本。各 cohort 的 `run_manifest.json` 與 `source_snapshot/` 保存環境、參數與來源雜湊。
- 隔離重播：`alpha_weight_outputs/alpha_calibrated_multistart_replay_seed600_v1_20261003/`。

封存 runner 的 `suite_manifest.json/statistical_rule` 說明文字仍寫「30 seed／95%」，是首次正式批次沿用的硬編碼標籤；複驗的實際 `seeds` 為 600–659、`expected_conditions` 為 180。共用分析器的 `analysis.json/formal_advantage` 也只處理 30-seed 單基準規則，故複驗中顯示 `false`。這兩個泛用欄位**不作複驗判定**；複驗啟動前封存的 `docs/alpha_calibrated_multistart_replication_preregistration_20261003.json` 與按該規則產生的 `decision.json` 才是 60-seed 雙基準 97.5% 同時區間判定依據。原始檔案保留，避免改寫歷史。

本結論只支持此乾淨合成設定中，多起點在獨立複驗批次的 pooled 品質比 A、B 好；不聲稱跨資料、維度、污染、參與者集合或真實應用的普遍優勢。首次正式批次未通過預定門檻，必須與通過的複驗並列保留。
