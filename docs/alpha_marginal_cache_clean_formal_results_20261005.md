# 邊際資訊初始 α ＋ Stage 3 精確快取：乾淨基本情境正式結果

更新：2026-10-05。此檔記錄已完成的測試；後續研究依使用者要求暫停。

## 問題、方法與事前判準

問題是：維持五次倒推與原有 20 筆可信 Stage 0 標記時，能否同時改善交付解的 pooled 品質與 local-model 查詢成本？對照法為原五候選：A（Uniform）、B（Stage 0 逆 MSE）、MSE power 1.5、Corr precision 0.5、Target-local precision，各用自己的初始 α 執行一次 Stage 3，最後用相同的 B20 ridge 內部分數選一個可行 X。測試法保留前四組，把 Target-local 換成 Marginal information 50，並在每個 Stage 3 候選的 local-model 預測中啟用精確快取。兩法都只交付選中的一個 X；pooled evaluator 與合成真實函數不參與 α、倒推或選解。

Marginal information 50 的定義：先用 20 筆 Stage 0 盲測求 B 權重集成的 MSE。對每位參與者 i，暫時將其 B 權重設為 0，重新正規化其餘權重，計算 `gain_i = MSE_without_i - MSE_full`；把正增益 `g_i=max(gain_i,0)` 正規化後，令 `α_marginal = 0.5 B + 0.5 norm(g)`。若沒有正增益則退回 B 並記錄。此值衡量 i 對**既定 B 集成在盲測上的邊際預測資訊**，不是完整移除參與者的倒推貢獻、Shapley 或分潤份額。

快取鍵包含 agent/model 身分、送入模型的 float32 輸入形狀及逐位元內容；每個候選 Stage 3 求解開始時重設。只有完全相同的同模型查詢才重用預測；不同 seed、模型或候選求解之間不共享。這是計算成本機制，不改 α 定義或選解規則。正式對照採**原五候選不快取**，測試採**邊際資訊五候選加快取**，故品質差來自候選 α 的替換，請求節省同時包含替換和快取，不能把兩者的成本效果混稱為 α 單獨帶來。

事前登錄 `alpha_marginal_cache_clean_formal_preregistration_20261005.json` 固定獨立 seeds 3900–4079 共 180 個、乾淨合成二維 target 0、固定 agent IDs 1／2／20、box `[-1,1]` 與容差 `1e-8`、相同 local models／proposals／凍結 pooled evaluator。主要差值 `Δ=E_pool(測試)-E_pool(對照)`；負值較好。10,000 次以 seed 為單位的配對 bootstrap 95% 區間，事前成功條件為 180／180 有效可行配對、區間上限嚴格小於零、方法獨立執行的平均模型請求與平均輸入樣本數都減少。無動態 α、pruning 或付款。五 seed pilot 3800–3804 的 pooled 平均差 +0.003213，僅用來檢查流程，沒有因此改方法或門檻。

## 正式結果

| 指標 | 原五候選／無快取 | 邊際資訊五候選／快取 | 判讀 |
|---|---:|---:|---|
| 有效可行交付 | 180／180 | 180／180 | 無失敗；全部符合 box |
| 平均 pooled 絕對目標誤差 | 0.016114904 | 0.014981525 | 配對平均差 -0.001133379 |
| 配對 pooled 差 95% CI | — | [-0.002255332, -0.000168896] | 上限小於 0；通過品質門檻 |
| pooled 較好／近乎相同／較差 | — | 32／120／28 | 相同採 `1e-7` 容差；平均差不能用勝場數取代 |
| 平均方法模型請求 | 1888.133 | 1680.367 | 平均少 207.767 次，約 11.0% |
| 配對請求差 95% CI | — | [-230.7675, -184.6817] | 成本方向一致 |
| 平均方法模型輸入樣本數 | 2059.133 | 1851.367 | 平均少 207.767 點，約 10.1% |

兩法初始化與內部選解平均各 24 次模型請求；Stage 3 原始平均請求對照 1864.133、測試 1869.533，測試套用快取後 Stage 3 為 1656.367。兩法仍各做五次倒推，不增加可信標記。成本欄位是方法**獨立執行**的 local-model 請求與批次輸入點數；共同資料／模型訓練、資格檢查、proposal、evaluator 建置及事後評分另計，不能重複加到各方法。時間與實際 API 價格未作正式經濟估值。

`verify_validation_calibrated_alpha.py` 對基線通過：180 seeds、360 個交付選解重算、0 錯誤。正式重播共 1080 個候選 Stage 3，快取與不快取的 X、pooled 分數與 box 違規量最大差皆為 0；每個候選的原始請求數等於快取後請求數加 cache hits。所有候選合計原始 Stage 3 請求 400,764、快取後 355,155、命中 45,609。這證明在本次固定的 deterministic local models 與已測查詢路徑上，快取減少重複查詢且沒有改變交付數值；不保證模型版本變動或非確定性服務仍相同。

補充判讀：對照／測試的 pooled 中位數 0.014521／0.013358，Q1–Q3 為 0.005867–0.022052／0.005592–0.021500。最好的單 seed 改善約 -0.043833（seed 3963），最大的惡化約 +0.021221（seed 4009）；不能只展示均值。邊際資訊候選在 41／180 seeds 被最後選中。合成真實函數的配對平均差約 -0.000474，但其 95% 區間跨零；pooled／truth 方法優劣方向衝突 25／180，因此不能將 pooled 成功延伸成 truth 的穩定改善。target 附近 `|y-target|≤0.25` 的 evaluator 測試樣本每 seed 至少 217、中位數 262，該區真實函數 MSE 平均約 0.000123；這是覆蓋診斷，不保證每個交付 X 的排序正確。

## 判定與限制

四項事前門檻全部通過，因此**在乾淨二維、target 0、固定三人、靜態模型的合成基本情境**，目前有獨立證據支持「邊際資訊初始 α＋Stage 3 精確快取」同時改善 pooled 平均品質和模型請求成本。這是組合方法的結論。前一輪邊際資訊 α 不帶快取在樣本量不平衡與低品質 local labels 情境的同時區間均跨零；本輪未重測快取組合在這些情境的品質，故不宣稱跨情境穩定優勢。α 是資訊權重，不是價格、報酬或付款；沒有真實 API 單價、可信標記價格、延遲與品質價值資料，無法計算貨幣淨收益或宣稱誘因相容。

## 核對來源與欄位

- 事前規格：`docs/alpha_marginal_cache_clean_formal_preregistration_20261005.json`；凍結程式：`alpha_weight_outputs/alpha_marginal_cache_clean_formal_source_snapshot_20261005/`。
- 基線資料、模型、proposals、CSV 與 manifest：`alpha_weight_outputs/alpha_marginal_cache_clean_formal_baseline_3900_4079/`。
- 逐 seed、逐候選快取重播及成本：`alpha_weight_outputs/alpha_marginal_cache_clean_formal_replay_3900_4079.json`。`raw_stage3_requests` 為不快取請求，`cached_stage3_requests` 為啟用快取請求，`cache_hits` 為重用預測次數；`portfolios` 中 `raw_method_requests`、`cached_method_requests` 是初始化＋選解＋各自五次 Stage 3 的總數。
- 事前門檻判定與配對 CI：`alpha_weight_outputs/alpha_marginal_cache_clean_formal_assessment_3900_4079.json`。`valid_pairs` 是兩法均可行且 pooled 分數有限的 seed 數；`gates.registered_success` 只表示已登錄的本情境門檻通過。
- evaluator coverage：`alpha_weight_outputs/alpha_marginal_cache_clean_formal_evaluator_coverage_3900_4079.csv`。

使用者要求在完成本方法測試與 Notion 更新後暫停；不啟動新的 α 方法或 seed 批次。
