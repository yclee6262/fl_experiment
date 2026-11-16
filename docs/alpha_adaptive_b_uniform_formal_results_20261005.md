# B 後選擇性增加 Uniform：低品質情境正式結果

## 研究問題與事前規則

本輪檢查一種有經濟動機的初始 α **計算策略**：先以既有 Stage 0 反 MSE 權重 B 求解一次；用 20 筆 Stage 0 blind labels 擬合 B20 ridge residual calibrator。若 B 求解失敗、無法內部評分，或 B 的校正後內部 target 誤差**嚴格大於 0.01**，才增加 Uniform 初始 α 的第二次求解。兩候選以同一 B20 內部評分選解，平手留 B。兩種 Stage 3 均使用 `|Σ_i α_i f_i(X)−T|`，固定三人 `[1,2,20]`、二維 target 0、box `[-1,1]`、`best_of` solver、相同 proposals 和凍結 pooled evaluator。pooled 與 truth 不參與自適應方法的 α、觸發或選解。

在新資料執行前，[事前登錄](/Users/yclee/Documents/GitHub/fl_experiment/docs/alpha_adaptive_b_uniform_formal_preregistration_v2_20261005.json)固定低品質 25% 情境、seeds **3400–3579**、0.01 觸發門檻、來源 SHA-256、180/180 有效配對要求與品質／成本門檻。前版登錄有一處敘述過廣：次要五候選參考程式在選解前載入 pooled evaluator 並計算候選分數，雖然選解函式不讀這些分數。v2 在自適應與五候選正式重播**之前**修正了資訊邊界說明，未更動方法、seeds 或判定門檻；原版與 v2 皆保留。先前看過的開發 seeds 2600–2629 和流程 pilot 3300–3304 不計入本輪正式樣本。

事前成功必須同時符合：180/180 B 與自適應交付 X 都有效可行，沒有較高失敗數；以 seed 為配對單位、10,000 次固定 RNG bootstrap 所得的 pooled 平均差 `E_adaptive−E_B` 雙側 95% CI 上界嚴格小於 0；自適應的 Stage 3＋選擇 API 請求平均小於五候選的一半。新目標五候選是成本與品質的次要參考，非另一項事後替換的主要基準。這些請求數沒有換算貨幣，也沒有預設一單位誤差值多少錢。

## 完整性、失敗與重評

基礎 runner 正常完成 180 個 seeds，但 **seed 3567 的固定參與者 20 未通過倒推資格測試**：記錄的 inverse loss 為 0.641326，該 seed 沒有固定三人 coalition、Stage 3 候選或交付 X。其訓練 4.405 秒、qualification 1.054 秒、proposal 0.128 秒，以及已發生的 qualification **40 次 API 請求／420 個輸入樣本**均保留；Stage 3 請求為 0。這是共同招募／資格階段的研究失敗，不是把缺失品質補 0，也不是直接當成自適應 solver failure。未更換 seed、固定參與者或資格門檻。

原 runner 的重載驗收回報交付驗證通過、179 個 seed 的選解可重播；總 `passed=false` 的五項 candidate coverage 錯誤都指向 seed 3567。其餘 **179/179** 個 seed 的 B、自適應和五候選有效；新目標五候選共 **895 個候選、179 次選解**由保存模型與 evaluator 獨立重算，pooled 分數、α 與內部分數最大差皆為 **0**。自適應交付 X、pooled 分數及 Stage 3 請求與保存候選逐 seed 核對，最大差亦為 **0**。正式來源 15 個檔案與登錄快照雜湊全數一致。所有 179 個有效 X 均符合 `1e-8` box 容差。

## 事前判定與觀察結果

下表的 pooled 誤差與請求量只在 **179 個有效配對**上計算；缺失的 seed 3567 另外保留在失敗率（1/180，0.56%）與準備成本。不能用這 179 筆的區間替代 180/180 成功門檻。

| 指標 | 單次 B | 自適應 B→Uniform | 新目標五候選 |
|---|---:|---:|---:|
| 有效 X／預期 seeds | 179／180 | 179／180 | 179／180 |
| pooled 平均絕對誤差 | 0.015754 | 0.015059 | 0.014519 |
| pooled 誤差中位數 | 0.012154 | 0.012027 | 0.010291 |
| Stage 3＋選擇平均 API 請求 | 346.24 | 531.02 | 1,736.41 |

自適應法在 179 個有效配對中觸發 Uniform **88** 次，最後選 Uniform **33** 次；相對 B 為 **22 改善、146 相同、11 惡化**。平均配對 pooled 差 `自適應−B` 為 **−0.000695**，事前指定的 95% CI 為 **[−0.001478, +0.0000366]**，上界跨零。相對五候選平均差為 **+0.000540**，描述性 95% CI 為 [−0.000442, +0.001520]。自適應法請求量為五候選的 **30.58%**（少 69.42%），但比單次 B 多約 **184.78 次／seed**（多 53.37%）。上述數字是 local-model API 呼叫量，不等於端到端時間或遠端帳單。

自適應法對 B 的 pooled 配對差第 95 百分位退步為 **+0.000903**，最差退步 **+0.020537（seed 3473）**；改善最大為 **−0.033083（seed 3439）**。truth 誤差平均差 −0.000889 僅作合成診斷，pooled／truth 對改善方向有 10 個逐 seed 衝突。179 個 evaluator 的 target-near（距 target 0.25 內）測試點數最少 **230**、中位數 **260**，相對 truth 平均 MSE 約 **0.000121**，沒有 `<30` 稀疏旗標；這不能保證每個候選的排序可靠。seed 3567 未留下 evaluator artifact，因此 coverage 分母為 179。

**事前正式結論：未通過。**完整有效配對門檻因 seed 3567 失敗；179 個有效配對的 pooled CI 也跨零。成本門檻在有效配對上觀察到通過，但不能把它與未通過的品質門檻合成成功主張。即使忽略資格失敗，這批資料也不足以確認自適應法品質穩定優於 B。按有效配對的觀察平均，若每次額外模型呼叫成本為 `c`、每降低一單位 pooled 絕對誤差價值為 `v`，自適應相對 B 的簡化收支臨界為 `c/v < 3.76×10⁻⁶`；這不是已知真實價格，也因品質區間跨零而不是預期收益保證。

## 後續研究判斷與重現

目前保留單次 B 為低成本基準，不將 0.01 自適應規則提升為主要 α 方法。若要繼續改善，優先分析 B20 何時把 Uniform 錯選為較好，以及 seed 3567 的固定 coalition 資格失敗率；任何新 gate、候選集合或資格處理都需另立版本與未使用 seeds，不能用本批正式資料重新挑參數後再聲稱獨立驗證。若要建立經濟收益，先定義實際 API 單價、可信標記費用與可接受的品質尾端損失。

原始資料與可重算輸出：

- `alpha_weight_outputs/alpha_adaptive_baseline_formal_3400_3579/`：資料、模型、evaluator、失敗狀態及舊流程候選。
- `alpha_weight_outputs/alpha_adaptive_new_five_formal_valid179_3400_3579.json`、`alpha_weight_outputs/alpha_adaptive_b_uniform_formal_valid179_3400_3579.json`：179 個有效 seed 的候選與決策重播。
- `alpha_weight_outputs/alpha_adaptive_b_uniform_formal_assessment_3400_3579.json`：含 seed 3567 的正式門檻判定及逐項驗證。
- `alpha_weight_outputs/alpha_adaptive_evaluator_coverage_formal_3400_3579.csv`：evaluator 區域覆蓋。
- `alpha_weight_outputs/alpha_adaptive_b_uniform_formal_source_snapshot_20261005/`：執行來源快照。

重評入口：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python verify_validation_calibrated_alpha.py alpha_weight_outputs/alpha_adaptive_baseline_formal_3400_3579`（因 seed 3567 的缺候選而預期整體 `passed=false`，需核對其餘 delivery verification 與錯誤全集）；`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python verify_ensemble_target_replay.py alpha_weight_outputs/alpha_adaptive_baseline_formal_3400_3579 alpha_weight_outputs/alpha_adaptive_new_five_formal_valid179_3400_3579.json`；正式判定可用 `assess_adaptive_alpha_formal.py`、v2 登錄與三個結果檔在新輸出路徑重算。
