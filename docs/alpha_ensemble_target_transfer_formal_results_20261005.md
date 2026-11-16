# 加權集成 Stage 3 目標的跨情境正式比較

## 問題與事前規則

低品質情境曾在獨立 180-seed 比較中通過 pooled 品質與 Stage 3 查詢量門檻。本研究檢查同一變更能否轉移到**乾淨基本**與**4 倍樣本量不平衡**情境。變更的是 Stage 3 如何使用 α：原目標為 `Σ_i α_i |f_i(X)−T|`，新目標為 `|Σ_i α_i f_i(X)−T|`。五組事前初始 α、固定參與者 `[1,2,20]`、二維 target 0、local models、proposals、box `[-1,1]`、`best_of` solver 和 B20 ridge 選解均維持相同。新舊方法**都不開快取**，不增加可信標記。凍結 pooled evaluator 只在交付決策後評分，合成 truth 只作診斷。

在正式執行前，`docs/alpha_ensemble_target_transfer_formal_preregistration_20261005.json` 固定兩情境、每情境新 seeds 3000–3179、17 個來源 SHA-256、完整指令與門檻。查核當時 238 份相關 manifest／登錄文件，沒有發現正式 seeds 重複使用。每情境以 seed 為配對單位，主要差為 `E_new−E_old`，負值有利新目標。使用固定 RNG 的 10,000 次配對 bootstrap，除一般雙側 95% 區間，兩項確認比較各用雙側 **97.5% Bonferroni 同時區間**。單一情境成功需同時具備：180／180 有效可行配對且新候選無失敗、同時區間上界嚴格小於零、平均 Stage 3 local-model API 請求不增加。跨兩情境主張需兩者都成功。未依正式資料重新選參數。

## 完整性與獨立重評

兩情境的原流程 runner 均正常結束，各保存 180／180 seeds、540／540 有效方法摘要，既有選解各重播 180／180 通過。新目標評估亦各保存 180／180 有效配對、零候選失敗。`verify_ensemble_target_replay.py` 從保存的模型、X、α 與 evaluator 獨立重算兩情境共 **1,800 個新候選**及 **360 次選解**；pooled 分數、α、內部分數最大差均為 0。事前登錄與正式 assessment 的雜湊相同，17／17 來源快照雜湊相符。原始逐 seed 紀錄、assessment manifest、來源快照及驗證報告保存在 `alpha_weight_outputs/alpha_ensemble_target_transfer_formal_assessment_3000_3179/`。

凍結 pooled evaluator 的 target-near 測試點數：乾淨情境最少 230、中位數 261.5；不平衡情境最少 218、中位數 260.5；兩情境各 180 個 evaluator 均沒有 `<30` 稀疏標記。target-near 相對合成 truth 的平均 MSE 依序約 0.000126、0.000127。這只描述本合成評估者的覆蓋與預測誤差，不能保證每個候選的優劣排序正確。

## 正式結果

| 情境 | 原目標平均 pooled 誤差 | 新目標平均 pooled 誤差 | 平均配對差 `新−舊` | 95% 配對區間 | 97.5% 同時區間 | 新較好／較差 |
|---|---:|---:|---:|---:|---:|---:|
| 乾淨基本 | 0.013640 | 0.012510 | −0.001131 | [−0.002618, +0.000321] | [−0.002837, +0.000514] | 103／77 |
| 樣本量不平衡 4× | 0.015442 | 0.013905 | −0.001537 | [−0.003037, −0.000034] | [−0.003271, +0.000182] | 101／79 |

| 情境 | 原／新 Stage 3 平均請求 | 平均請求差 | 降幅 | 請求差的描述性 95% 區間 | 正式品質／成本門檻 |
|---|---:|---:|---:|---:|---|
| 乾淨基本 | 1,778.82／1,693.27 | −85.55 | −4.81% | [−134.59, −35.58] | 品質未通過；成本通過 |
| 樣本量不平衡 4× | 1,749.05／1,623.67 | −125.38 | −7.17% | [−171.70, −79.72] | 品質未通過；成本通過 |

**事前主判定：兩情境皆未通過完整的品質＋成本門檻。**乾淨情境連一般 95% 品質區間都跨零；不平衡情境的一般 95% 區間上界雖略低於零，但事前指定的 Bonferroni 同時區間仍跨零，不能事後改用較寬鬆的區間宣稱跨情境確認性優勢。兩情境的 Stage 3 local-model 請求平均下降成立；這不等於整體實驗、遠端 API 計費或真實貨幣成本同比例下降。既有低品質情境的正式成功仍只適用於其原本的合成條件，不因本輪結果擴大成通用結論。

## 尾端退步與 truth 診斷

乾淨情境的 pooled 配對差中位數約 −0.000688，第 95 百分位退步 +0.013414，最差退步 **+0.029399（seed 3073）**。該 seed 原、新流程均選 `corr_precision_0p5`，但新目標將 pooled 誤差從 0.001438 提高到 0.030837；因此即使初始 α 與選中候選相同，改變 Stage 3 目標也可能使 X 惡化。不平衡情境中位數約 −0.000869，第 95 百分位退步 +0.011915，最差 **+0.043691（seed 3074）**：新流程選 `mse_power_1p5`，其 pooled 誤差 0.046204；同批新候選中的 Uniform 事後 pooled 誤差為 0.009001，顯示內部選解可能錯排。pooled 最佳候選只作離線診斷，沒有回饋選解器。

truth 平均差依序為 −0.002430、−0.000696；其**描述性** 95% 配對區間依序約 [−0.003714, −0.001204]、[−0.001917, +0.000516]。pooled／truth 對逐 seed 改善方向衝突依序有 57／180、46／180，其中兩邊差幅都超過 0.001 的衝突為 40、38。truth 的區間不是事前主要成功門檻，不取代 pooled 判定；衝突案例說明只看整體 evaluator MSE 不足以保證每個 X 的排序可靠。

上述 evaluator coverage、truth 區間及極端案例是完成正式主分析後的診斷彙整，機器可重算結果保存在 `alpha_weight_outputs/alpha_ensemble_target_transfer_formal_assessment_3000_3179/posthoc_diagnostics.json`；它不修改事前成功門檻。

## 決策與下一步

新集成目標在低品質情境已有**情境限定的正式品質＋請求量證據**；本輪證明它在乾淨與不平衡情境平均少用 Stage 3 模型呼叫，但**未建立事前要求的穩定 pooled 品質優勢**。因此不將新目標提升為所有情境的預設方法，也不將這些結果說成「找到較佳初始 α」。若研究重心仍是具經濟邏輯的初始 α，下一輪應提出可事前取得的新資訊或直接減少付費模型查詢的機制，將資訊／查詢成本完整計價；同一 20 筆 Stage 0 標記上的小幅權重調整已有多次負面結果，不宜直接消耗新正式 seeds。

重算入口：

```bash
python verify_validation_calibrated_alpha.py alpha_weight_outputs/alpha_ensemble_target_transfer_clean_formal_3000_3179
python verify_validation_calibrated_alpha.py alpha_weight_outputs/alpha_ensemble_target_transfer_imbalance_formal_3000_3179
python verify_ensemble_target_replay.py alpha_weight_outputs/alpha_ensemble_target_transfer_clean_formal_3000_3179 alpha_weight_outputs/alpha_ensemble_target_transfer_formal_assessment_3000_3179/reference_clean/replay.json
python verify_ensemble_target_replay.py alpha_weight_outputs/alpha_ensemble_target_transfer_imbalance_formal_3000_3179 alpha_weight_outputs/alpha_ensemble_target_transfer_formal_assessment_3000_3179/sample_imbalance_4x/replay.json
python analyze_ensemble_target_transfer_diagnostics.py docs/alpha_ensemble_target_transfer_formal_preregistration_20261005.json alpha_weight_outputs/alpha_ensemble_target_transfer_formal_assessment_3000_3179 --output <new-diagnostics-json-path>
```
