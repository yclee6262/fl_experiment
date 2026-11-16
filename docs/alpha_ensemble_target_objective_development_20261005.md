# α 加權集成目標的 Stage 3 開發與正式驗證規格

## 問題與方法

原 Stage 3 對給定的初始權重 α 最小化 `Σ_i α_i |f_i(X)−T|`，使每個 local model 個別接近目標。新的假說改為最小化 `|Σ_i α_i f_i(X)−T|`：先組合各模型預測，再要求組合值接近目標。兩種目標均用相同的五組事前 α（Uniform、Stage 0 inverse-MSE、MSE power 1.5、Corr precision 0.5、Target-local）、同一 coalition、proposal、box 限制及 `best_of` 求解器。最後仍用原 20 筆 Stage 0 標記建立 B 權重的全域 ridge 殘差校正，從五個可行 X 中選一個交付。pooled evaluator 與合成真實函數只在交付決策後作評估；本探索重播程式避免在求解歷程中呼叫真實函數。

這是**改變 α 在 Stage 3 中的使用方式**，不是新的初始 α 公式，也不是付款或公平性機制。新目標允許 local model 的正負偏差互相抵銷，可能改善 pooled 近似，也可能造成較差的交付 X；必須直接比較外部品質及尾端退步。

## 開發資料，不作正式優勢主張

`explore_ensemble_target_objective.py` 從既有保存的 local-model checkpoints、proposal、α 與 evaluator 重播新目標，不重新訓練。舊方法使用同 seed 已保存的交付解，並重載 evaluator 核對舊 pooled 分數。這些資料已進入方法設計，區間僅供判斷是否值得另立正式實驗。

| 情境／資料 | n | 新−舊 pooled 平均差 | 描述性 95% 配對區間 | 新法較好／相同／較差 | 新−舊平均 Stage 3 模型請求 |
|---|---:|---:|---:|---:|---:|
| 乾淨基本，既有 seeds 1900–1929 | 30 | −0.001612 | [−0.004539, +0.001092] | 14／0／16 | −149.4 |
| 樣本量不平衡，既有 seeds 1900–1929 | 30 | +0.001638 | [−0.001837, +0.004913] | 11／0／19 | −59.8 |
| 低品質，既有 seeds 1900–2019 | 120 | −0.001923 | [−0.003881, −0.000043] | 63／1／56 | −110.55 |
| 低品質，新開發 seeds 2600–2629 | 30 | −0.001908 | [−0.004992, +0.000848] | 15／0／15 | −118.2 |

新開發 seeds 的前五筆平均 pooled 差為 **+0.003192**，含單一退步 **+0.014618**；後 25 筆平均為 **−0.002928**。合併 30 筆的區間仍跨零，truth 平均差約 **+0.000108**，且有 10／30 筆 pooled／truth 對改善方向相衝突。原低品質 120 筆的最大 pooled 退步約 **+0.020183**。因此不能宣稱新目標普遍更好，也不能只展示負的平均值。新 5-seed pilot 的既有基準重載驗證通過，後 25-seed 開發批次的既有基準亦通過 25／25 選解重播；新解均由相同保存模型重新求解。

## 正式門檻與執行狀態

`docs/alpha_ensemble_target_objective_formal_preregistration_20261005.json` 在正式執行前固定低品質單一情境、未見於 243 份既有 manifests 的 seeds 2700–2879、15 個來源檔案雜湊、完整設定及命令。正式比較不使用 Stage 3 快取，以隔離目標函數效果。成功需同時滿足：180／180 完整可行配對且新方法沒有候選失敗、`E_new−E_old` 的 seed 配對平均差 95% bootstrap 區間上界小於零、平均 Stage 3 模型請求不增加。α 初始化及五候選選解呼叫兩法相同，沒有額外可信標記。truth、pooled／truth 衝突、尾端退步與 evaluator coverage 另報。

正式基準在 `alpha_weight_outputs/alpha_ensemble_target_baseline_formal_2700_2879/` 逐 seed 保存，重載驗證通過：180／180 個原選解可重播、交付有效。`assess_ensemble_target_objective.py` 在 `alpha_weight_outputs/alpha_ensemble_target_formal_assessment_2700_2879/` 使用同一批模型與 proposal 重播新目標，180／180 個新舊配對有效、沒有候選失敗。獨立驗證器另重新載入 **900 個候選 X 的 pooled 評分、180 次內部選解**及 α，最大差均為 0；來源 15／15 個事前雜湊相符。

## 正式結果與限制

| 指標 | 原五候選目標 | 新五候選集成目標 |
|---|---:|---:|
| 平均 pooled 絕對 target 誤差 | 0.018702 | 0.016495 |
| pooled 中位數 | 0.014434 | 0.013249 |
| pooled 誤差第 95 百分位 | 0.048218 | 0.037479 |
| 平均 truth 絕對 target 誤差（診斷） | 0.015082 | 0.014033 |
| 平均 Stage 3 local-model 請求 | 1817.35 | 1685.83 |
| 有效交付／候選失敗 | 180／0 | 180／0 |

主要配對差 `E_new−E_old` 的平均為 **−0.002207**，固定 RNG、10,000 次 seed bootstrap 的 95% 區間 **[−0.003829, −0.000625]**，上界低於零。每 seed 平均少 **131.52 次 Stage 3 模型請求**，約 **7.24%**；請求差的描述性 95% 區間 **[−175.12, −86.70]**。α 初始化及五候選選解所需資訊不變，沒有增加可信標記。依事前登錄的「完整有效配對、pooled 品質、平均請求不增加」三道門檻，**本低品質情境正式比較通過**。

這不是逐 seed 保證：新法改善／相同／變差為 **93／0／87**；配對差中位數僅 **−0.000261**，第 95 百分位退步 **+0.013929**，最差退步 **+0.026118**（seed 2740）。新法也有大幅改善案例（最大約 −0.041539）。truth 平均差為 **−0.001048**，但其描述性 95% 區間 **[−0.002436, +0.000314]** 跨零；pooled／truth 對改善方向衝突 **37／180**，其中兩邊差幅皆超過 0.001 的衝突為 **23**。因此正式證據支持**平均 pooled 品質與模型請求的情境限定改善**，不能宣稱 truth 穩定改善、每個案例都更好或實際營運價值已算出。

凍結 pooled evaluator 的 target-near 測試樣本數最少 220、中位數 262，沒有 `<30` 稀疏標記；target-near 對合成 truth 的平均 MSE 約 0.000120。這提供評估者覆蓋資訊，不能消除逐案例排序衝突。交付後使用 truth 評分，沒有把 truth 或 pooled 分數回饋 α、Stage 3 或選解。

本結果**改良的是 Stage 3 使用 α 的目標函數**，並未找到更好的新初始 α 公式。且正式比較只含合成低品質、二維、target 0、固定 IDs `[1,2,20]`；開發資料中的樣本量不平衡情境平均反而變差。下一步若仍以「新初始 α」為研究目標，可在**已用過的開發 seeds** 比較單次 B、現有 `ensemble_shrink` 等權重在新目標下的品質與成本，再決定是否值得用全新 seeds 正式驗證；不得用 seeds 2700–2879 挑新 α 後再稱為獨立證據。

正式產物：事前登錄 `docs/alpha_ensemble_target_objective_formal_preregistration_20261005.json`；基準的 `selection_verification.json`；新法 `analysis.json`、`seed_rows.jsonl`、`assessment_manifest.json`、`source_snapshot/` 與 `independent_verification.json`。重評命令：`python verify_ensemble_target_assessment.py alpha_weight_outputs/alpha_ensemble_target_baseline_formal_2700_2879 alpha_weight_outputs/alpha_ensemble_target_formal_assessment_2700_2879 docs/alpha_ensemble_target_objective_formal_preregistration_20261005.json`。

## 後續初始 α 探索：單次堆疊權重與第六候選

正式結果確認的是 Stage 3 目標，**不是新初始 α**。因此只在先前已用於開發的低品質 seeds 1900–2019 與 2600–2629 檢查三種單次 α，沒有碰正式 seeds 2700–2879 作選法：

- 先用 20 筆 Stage 0 標記與各 local model 預測，將集成盲測 MSE 最小化，求非負且總和為 1 的 `w_fit`。
- `uniform_stacked_half = 0.5 A + 0.5 w_fit`：現有 `ensemble_shrink`。
- `B_stacked_half = 0.5 B + 0.5 w_fit`：以 Stage 0 inverse-MSE 品質權重作收縮基準。
- `stacked_full = w_fit`：直接使用擬合權重。

各方法都用新的集成目標只做一次 Stage 3。擬合需 3 次 local-model 批次查詢（每次 20 個輸入），單次求解平均約 331–355 次請求，明顯少於五候選。然而相對**原五候選舊目標**的 pooled 平均差如下；區間皆為開發描述，不能作正式優勢：

| 方法 | 舊 120 seeds 平均差 | 新開發 30 seeds 平均差 | 主要風險 |
|---|---:|---:|---|
| Uniform 堆疊半混合 | +0.002979 | +0.000882 | 舊批次平均較差。 |
| B 堆疊半混合 | −0.000265 | +0.003614 | 新批次平均轉差；單 seed 最大退步約 +0.115。 |
| 純堆疊 | +0.000330 | +0.001783 | 新批次平均較差；單 seed 最大退步約 +0.116。 |

再把 `B_stacked_half` 當作新集成目標五候選的**第六個初始 α 候選**，仍用原 B20 選解規則：舊 120 seeds 被選中 14 次，六減五平均 pooled 差約 **−0.000053**，描述性 95% 區間跨零；新開發 30 seeds 合計被選中 9 次，後 25 筆平均差反而 **+0.000157**。每 seed 還要多約 319–362 次 Stage 3 請求及 3 次初始化批次查詢。缺少跨批次一致品質收益，**現階段不把此新 α 加入正式方法，也不為它啟動新 seeds**。

這組探索支持一個限制：新 Stage 3 目標的正式改善，不能自動歸因於堆疊 α；候選選解及多起點搜尋仍有作用。下一輪若要得到真正新的初始 α，需提出比同一 20 筆盲測資料更多的可用資訊，或對取得新資訊的成本與可行性做明確約束，避免在已使用 seeds 上反覆挑公式。

## 跨情境正式追蹤（seeds 3000–3179）

後續另以事前登錄的新 seeds 3000–3179，各測 180 個乾淨與 4 倍樣本量不平衡配對，詳見 `docs/alpha_ensemble_target_transfer_formal_results_20261005.md`。兩情境平均 Stage 3 請求依序少 4.81%、7.17%，但 pooled 配對平均差的事前 97.5% Bonferroni 同時區間均跨零；**跨情境正式品質門檻未通過**。低品質情境原先通過的結論不變，但不能擴大為新目標在所有資料條件都較好的主張。該追蹤仍未提出新初始 α 公式。
