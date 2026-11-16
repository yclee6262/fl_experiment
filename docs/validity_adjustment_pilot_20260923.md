# 實驗有效性修正與新版 pilot 驗收

文件日期：2026-09-23

本文件記錄限制處理、P5 分析口徑、情境控制與 P6 預算設計修正後的實作與 seeds 0–4 pilot。這是正式 30-seed 實驗前的有效性驗收，不是新的正式效果證據；本輪未進行 α 調參、pruning、Shapley 或分潤。

## 1. 結論與決策

1. 新版 `box_constrained` 流程已通過功能測試與 pilot。舊版高目標 seed 2 的兩筆違規交付可由保存結果重現；新版 P5 的 80 筆與 P6 的 90 筆成功交付解均符合 `[-1, 1]` box，最大允許違規量為 `1e-8`。
2. P5 八個情境的 40 組 A／D 配對全部有效。Guarded 只在 clean 與高目標情境得到很小的平均 pooled-error 改善；其餘六個情境平均持平或惡化，且查詢量為 A 的 6.40–12.98 倍。維持 **A：Uniform、不更新 α** 為主要方法。
3. 受限版 P4 再驗證顯示，patience 2 在 clean／controlled poison 分別節省 36.9%／62.8% 查詢量，10 組 accepted state 全部一致。無進展停止繼續保留。
4. P6 的三種預算確實形成二人、二至三人、三人的取捨。Exact 在每個 seed 對同目標 greedy 的 selection objective 均不低，但六個預算／多樣性條件的平均 pooled error 都沒有優於該條件中表現最好的 greedy 方法。將「招募代理目標是否能預測 final X 品質」列為後續研究問題。
5. 限制、配對、保存與重評驗收已通過，可以另行固定正式實驗版本與 seeds 100–129；正式版本不得再用這批 pilot 結果調參。

## 2. 實作範圍

### 2.1 限制模式與有效交付

`HostServer` 現在明確支援兩種模式：

- `legacy_unconstrained`：保留舊實驗語義，供重現使用。
- `box_constrained`：新版研究使用。manifest 保存模式、上下界與 `1e-8` 可行性容差。

受限求解保留原候選子空間 `X = candidates.T @ beta`：

- 原 BFGS 路徑在受限模式下改由 SLSQP 解 beta，使用線性限制 `lower <= candidates.T @ beta <= upper`，並保存實際 solver 名稱。若 warm start 不可行，先以線性規劃尋找可行 beta。
- Custom 路徑以可行步長限制 tangent finite difference 與每次更新，不使用逐座標 clipping，因此不改變候選方向。warm start、recovery 與 exclusion 共用同一限制處理。
- `best_of` 只在數值有限、box 可行且可交付的候選中選擇。所有候選無效時保存 solver failure，不交付違規 X。
- 數值停止狀態、solver success、box feasibility、最大違規量與 valid delivery 分開保存。最終 solver audit 取自 accepted state；被拒絕的最後 proposal 不會覆蓋交付狀態。

### 2.2 保存、分析與情境控制

每次執行保存 manifest、根目錄 Python 來源快照與 SHA-256、資料／模型 artifacts、accepted X、歷程、限制資訊、各階段查詢量與耗時。新版分析分開報告：

- 全部執行狀態與診斷用全體平均。
- 有效交付解的平均數、中位數與四分位範圍。
- box infeasible、solver failure 與缺失配對原因。
- 僅在雙方皆有效時成立的 paired quality comparison。
- pooled evaluator 與 truth 的方向衝突。
- target 附近 evaluator 樣本數與 truth MSE；樣本少於 30 時標示 sparse。

P5 的資料／異質性／維度／目標情境固定 participant IDs `[1, 2, 20]`；五人 coalition 使用 `[1, 2, 3, 4, 20]`。污染情境也使用同一五人集合，只改 agent 20 的污染狀態。低品質噪聲只作用於 local observed labels，外部 evaluator 維持乾淨噪聲設定。

P6 保存 10 名候選的 qualification 結果、殘差、報價、proposal、selection objective、solver 與交付狀態。Exact 與 objective greedy 使用完整相同候選全集、目標與限制；legacy 使用相同預算與人數上限，但明列為不同選人規則。`selection_precompute` 使用獨立計數器。招募承諾與基本報酬是同一筆負債，未重複計費；分潤狀態為未執行。

## 3. 驗收證據

### 3.1 限制與回歸測試

- 舊結果 `p5_one_factor_pilot_20260923/hard_target_2p5` 重現 2 筆違規交付（A／D、seed 2）。這只描述舊版語義。
- `test_box_constraints.py` 涵蓋邊界解、單一候選、秩不足候選、無可行子空間、custom 查詢邊界、warm start、recovery、exclusion 與 `best_of` 丟棄不可行 engine。
- accepted-state 回歸涵蓋 rejected proposal 不覆蓋 accepted X。
- `python -m unittest test_box_constraints test_alpha_weight_experiments`：46 項全部通過。
- `py_compile` 與 `git diff --check` 通過。
- 跨階段驗收器輸出：legacy failures 2、P4 paired seeds 10、P5 results 80、P6 results 90。

### 3.2 P5：受限版八情境 pilot

所有情境均有 A 與 D、seeds 0–4，共 80 筆有效交付；solver failure、box infeasible 與缺失 quality pair 均為 0。80 筆從磁碟重載 evaluator 後，保存分數差異超過 `1e-7` 的筆數為 0。

| 情境 | D−A pooled error 平均差 | D／A 查詢倍數 | pooled／truth 方向衝突 | 判讀 |
|---|---:|---:|---:|---|
| Clean reference | -0.003994 | 12.98 | 0/5 | 小幅改善，成本大增 |
| Heterogeneity 0.6 | +0.009255 | 6.40 | 0/5 | 惡化 |
| Sample imbalance 4x | +0.008307 | 8.02 | 2/5 | 惡化且有 evaluator 衝突 |
| Low quality 25% | +0.004543 | 8.80 | 0/5 | 惡化 |
| Controlled poison | +0.010430 | 7.89 | 0/5 | 惡化 |
| Dimension 5 | +0.000049 | 8.34 | 1/5 | 幾乎持平，成本大增 |
| Controlled coalition 5 | +0.000458 | 9.19 | 2/5 | 幾乎持平，成本大增 |
| Hard target 2.5 | -0.000067 | 9.58 | 1/5 | 幾乎持平，成本大增 |

Hard target 在 target 附近平均只有 23.2 筆 evaluator 樣本，五個 seeds 全部標示 sparse；其他情境平均 190.8–263.2 筆。高目標結果只能視為低資料密度區域的診斷，不能單獨支持品質優勢。

Pooled／truth 方向衝突共 6/40 pairs，已保存在 `scenario_comparisons.csv`；正式判讀仍依預先固定的 pooled evaluator，truth 只作診斷。

### 3.3 P4：受限版無進展停止

| 情境 | Seeds | 觸發 seeds | accepted-state mismatch | Control 平均查詢 | Patience 2 平均查詢 | 節省 |
|---|---:|---:|---:|---:|---:|---:|
| Clean | 5 | 4 | 0 | 30,137.4 | 19,006.8 | 36.9% |
| Controlled poison | 5 | 5 | 0 | 150,996.0 | 56,148.0 | 62.8% |

四個條件共 20 筆 accepted solutions 的 evaluator 重載均為 0 mismatch。停止原因與 convergence 分開，沒有把無進展誤標成收斂。

### 3.4 P6：預算取捨 pilot

每組均有 5 seeds；90 筆結果全部符合預算、最多三人與 box 限制，solver failure 為 0，重載 evaluator mismatch 為 0。每個方法／seed 都使用完整 10 人候選池，selection precompute 為 10 次查詢。

| 預算 | eta | Exact 平均人數 | Exact objective | Exact pooled error | Objective-greedy pooled error | Legacy pooled error |
|---:|---:|---:|---:|---:|---:|---:|
| 2.5 | 0.0 | 2.0 | 0.401097 | 0.022704 | 0.022704 | 0.020498 |
| 2.5 | 0.5 | 2.0 | 0.528132 | 0.024419 | 0.023659 | 0.022704 |
| 3.5 | 0.0 | 2.8 | 0.519854 | 0.018520 | 0.016980 | 0.016980 |
| 3.5 | 0.5 | 3.0 | 0.830423 | 0.016348 | 0.015139 | 0.016980 |
| 8.0 | 0.0 | 3.0 | 0.569144 | 0.013166 | 0.013166 | 0.011751 |
| 8.0 | 0.5 | 3.0 | 0.943648 | 0.014539 | 0.013782 | 0.013166 |

Exact 的 selection objective 逐 seed 不低於 objective greedy，驗證了實作正確性；但較高代理目標沒有轉化為較低 pooled error。寬預算讓三人集合都可行，卻仍沒有保證 final X 更好。選人目標與 final X 品質必須保持兩個不同的研究指標。

## 4. 產出位置

- P5 原始結果：`alpha_weight_outputs/p5_validity_v5_20260923/`
- P5 分析：`alpha_weight_outputs/p5_validity_v5_20260923_analysis/`
- P5 圖：`report_figures/p5_validity_v5_20260923.png`
- P4 結果與分析：`alpha_weight_outputs/p4_constrained_no_progress_v1_20260923/`
- P6 原始結果：`alpha_weight_outputs/p6_budget_validity_v4_20260923/`
- P6 分析：`alpha_weight_outputs/p6_budget_validity_v4_20260923_analysis/`
- P6 圖：`report_figures/p6_budget_validity_v4_20260923.png`

`p5_validity_v3/v4` 與 `p6_budget_validity_v3` 是開發中間產物；正式引用本輪 pilot 時只使用上述 v5／v4 路徑。既有 P2–P6 輸出均未覆蓋。

## 5. 下一步

1. 將目前程式、manifest schema、來源快照與環境固定成新的正式版本。
2. 預先登錄正式 seeds 100–129、主要 pooled 指標、失敗率、有效配對規則與 bootstrap 單位。
3. 正式 P5 以 A 為主方法；D 僅保留必要對照，不再新增 α 調參。
4. P6 若進入正式研究，先把「代理目標與 final pooled quality 的關聯」定為研究問題；exact 只作小型離線上界，不以 objective 較高宣稱 X 較好。
5. Pruning、Shapley 與分潤維持暫停，待核心品質與招募代理目標的證據更明確後再評估。
