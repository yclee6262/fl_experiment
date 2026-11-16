# P5 單因子情境擴充 pilot（2026-09-23）

## 問題、範圍與主要判準

本階段檢查 P2 的主要基準 A（Uniform、α 不更新）與研究對照 D（Uniform、Guarded、無進展 patience 2）在單一因素改變時的最終 X 品質、可行性與成本。主要品質判準固定為 frozen pooled evaluator 的
`E_global = |f_global(X)-Y*|`；合成真實函數誤差為交叉檢查。

本輪每個情境只有 seeds 0–4，屬方向探索 pilot。所有差異均為描述性結果，不作 95% 信賴區間或穩定改善宣稱。

## 實驗設計

共同條件為 20 agents、每人基準 500 samples、30 epochs、最多 20 calibration rounds、pruning 關閉。每個 seed 的 A／D 共用資料、local models、候選點與 frozen evaluator。8 個單因子情境如下：

| 情境 | 相對基準的單一改動 |
|---|---|
| reference_clean | 2D、target 0、乾淨資料 |
| heterogeneity_0p6 | agent 特徵分布中心依序偏移至 ±0.6 |
| sample_imbalance_4x | agent 樣本數依幾何序列配置，最大／最小約 4 倍且平均維持約 500 |
| low_quality_25pct | 後 25% agents 的 label noise std 由 0.05 提高至 0.3 |
| controlled_poison | 5 人固定 coalition，強制 1 名污染者，trimmed 內部 evaluator |
| dimension_5 | 維度由 2 提高至 5 |
| controlled_coalition_5 | 5 人固定乾淨 coalition |
| hard_target_2p5 | target 由 0 提高至 2.5 |

低品質情境中的 pooled evaluator 使用保存的 `clean_y`；此處的 `clean_y` 仍包含該情境設定的隨機 label noise。因此 evaluator 是對該資料生成分布的集中模型，而不是無噪聲 oracle。樣本不平衡情境直接 pooled 全部樣本，因此大樣本 agent 對 evaluator 訓練的影響較大。

## 實作與消融範圍

- `dataset.py` 支援逐 agent 樣本數、feature shift 與 noise std，並將實際資料設定存入 artifact metadata。
- `alpha_weight_experiments.py` 新增 P5 A／D 比較套件與上述資料控制參數。
- `p5_scenario_suite.py` 固定並依序執行 8 個情境。
- `analyze_p5_results.py` 保留全部 seed、成對計算 D−A 品質及成本差。
- `reevaluate_saved_solutions.py` 重載各 seed 的 evaluator artifact 重評 accepted X。

本輪消融依 P2／P3 已有證據限縮：A 與 D 在本輪重跑；無 Guarded 的固定更新 C 已由 P2 正式 30-seed 比較；pruning 因 P3 未建立 q 對完整參與者效果的有效性而不啟用；完整參與者移除只保留為 P3 離線參考，不作部署用更新近似。這些停止決策避免把未獲證據支持的機制包成「完整方法」。

## Pilot 結果

| 情境 | A 平均 pooled error | D 平均 pooled error | 平均 D−A | D 較佳 seeds | D/A 查詢倍率 |
|---|---:|---:|---:|---:|---:|
| reference_clean | 0.02328 | 0.02294 | -0.00034 | 2/5 | 11.61× |
| heterogeneity_0p6 | 0.00803 | 0.00893 | +0.00090 | 4/5 | 9.17× |
| sample_imbalance_4x | 0.01577 | 0.01464 | -0.00113 | 3/5 | 8.16× |
| low_quality_25pct | 0.01824 | 0.03315 | +0.01490 | 1/5 | 6.34× |
| controlled_poison | 0.01746 | 0.01748 | +0.00002 | 1/5 | 7.71× |
| dimension_5 | 0.03492 | 0.03304 | -0.00188 | 3/5 | 7.53× |
| controlled_coalition_5 | 0.01586 | 0.02103 | +0.00517 | 2/5 | 8.34× |
| hard_target_2p5 | 0.01567 | 0.02477 | +0.00910 | 0/5 | 7.51× |

負值代表 D 的 pooled error 較低。D 在部分情境有小幅平均改善，但所有情境都增加 6.34–11.61 倍查詢量；低品質、五人 coalition 與高目標情境的平均 pooled error 明顯惡化。這些結果不支持把動態 Guarded 升為 P5 主方法，A 仍是目前主方案。

pooled 與真實函數對 D−A 的 seed 級方向在異質性等情境有衝突，已完整保留於 `paired_seed_results.csv`。外部 evaluator 的平均 truth RMSE 約為 0.0106–0.0322，其中 5D 最高；高目標情境的 evaluator truth RMSE 約 0.0120。這些值衡量 evaluator 對資料分布的擬合，不能替代交付 X 的 target error。

## 可行性發現與驗收

80 個 accepted X 均成功從 frozen evaluator 重評，pooled 與真實函數保存值皆為 `mismatches=0`。7 個情境的 70 個解均通過 box 檢查。

`hard_target_2p5` 的 seed 2 在 A 與 D 都交付相同的
`[1.1459243641, 0.6452849545]`，超出預定 `[-1,1]` box。這不是 evaluator 問題，而是目前 Stage 3 solver 未把 box 當成硬限制；`final_box_feasible` 正確保留失敗。正式 P5 前必須建立新的 solver-feasibility 版本並重新跑受影響比較，不能刪除 seed 2 或把事後檢查當成求解器保證。

P5 pilot 驗收結果：

- [x] A／D 有獨立 seed 級效果與成本比較；P2／P3 已停止的消融有明確依據。
- [x] 低品質、五人 coalition、高目標等負面案例完整保留。
- [x] 高目標另報 evaluator truth MSE，並揭露 box failure。
- [x] 結論明確限定為 5-seed pilot。
- [x] 8 個 manifest、40 組 paired results、80 個 accepted X、pooled data、local checkpoints 與 evaluator checkpoints均已保存並可重評。
- [ ] 尚未形成論文正式 P5 結果；須先修正硬限制，再使用預先登錄的新 seeds 執行正式比較。

## 執行與證據

```sh
/Users/yclee/anaconda3/bin/python p5_scenario_suite.py \
  --output-dir alpha_weight_outputs/p5_one_factor_pilot_20260923 \
  --seeds 0,1,2,3,4 --external-evaluator-epochs 100

/Users/yclee/anaconda3/bin/python analyze_p5_results.py \
  alpha_weight_outputs/p5_one_factor_pilot_20260923 \
  --output-dir alpha_weight_outputs/p5_one_factor_pilot_analysis_20260923
```

主要輸出：

- `alpha_weight_outputs/p5_one_factor_pilot_20260923`
- `alpha_weight_outputs/p5_one_factor_pilot_analysis_20260923`
- `report_figures/p5_one_factor_pilot_20260923.png`

## 決策

保留 A 作主方法，D 只作研究對照。下一個正式 P5 版本先使 Stage 3 對 box constraint 提供硬保證，再決定 seeds 與正式情境；不得直接把本 pilot 擴寫為論文穩定性結論。
