# P6 招募選人與預算 pilot（2026-09-23）

## 問題與範圍

本階段把招募選人定義成可重現的受預算集合最佳化，並檢查精確法、使用相同目標的 greedy 近似、以及既有 Stage 1 greedy 所選 coalition 的最終 X 品質。這是 5-seed 小型 pilot，不是經濟公平性或正式商業可行性證明。

資格測試 inverse target 固定為 0.0；正式倒推 target 固定為 0.5，兩者在 manifest、候選資料與結果中分開保存。選人完成後，每個 coalition 使用靜態 Uniform α 求解，主要品質判準為 frozen pooled evaluator。

## 明確選人問題

候選人先通過既有盲測與 inverse qualification。對候選集合 S 定義：

`V(S) = sum_i quality_share_i + eta * normalized_sum_pairwise_residual_dissimilarity`

其中 `quality_share` 為 Stage 0 正規化品質權重；殘差多樣性為盲測 residual 的 pairwise cosine dissimilarity 絕對相似度版本；pairwise sum 以最大 coalition 的 pair 數正規化。限制為：

- S 非空。
- selected bid 總和不超過 selection budget 8.0。
- coalition size 不超過 3。

`exact_select` 枚舉最多 20 位候選人的可行子集；`objective_greedy` 每步加入相同 V(S) 邊際增益最大的可行候選人。兩者共用完全相同的目標與限制。`legacy_greedy` 只作既有行為對照，將其已選集合用同一 V(S) 評分，不宣稱它原本就是在最佳化同一集合函數。

## 保存與成本／付款語義

每個 seed 保存所有合格候選人的 MSE、quality share、bid、是否被既有方法選取，以及針對正式 target 的 proposal。結果分開保存：

- `recruitment_bid_commitment`：入選者 reservation bids 總和。
- `base_reward_liability`：若完成招募，基礎報酬負債；本 pilot 設為同一 bid 總和。
- `unallocated_surplus_pool`：total budget 減去基礎報酬。
- `contribution_settlement_status`：固定為 `not_run_in_recruitment_benchmark`。

因此本實驗沒有產生 q_pay 或最終貢獻分潤，也不把合理付款描述為完整公平性保證。

## Pilot 結果

| 方法 | 平均選人目標 | 平均 pooled error | pooled error 中位數 | 平均 true error | 平均 bid | 預算違規 | box failure |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exact | 0.94365 | 0.01455 | 0.01492 | 0.00734 | 3.660 | 0/5 | 0/5 |
| Same-objective greedy | 0.93696 | 0.01380 | 0.01492 | 0.00706 | 3.724 | 0/5 | 0/5 |
| Existing legacy greedy | 0.93459 | 0.01474 | 0.01540 | 0.01138 | 3.660 | 0/5 | 0/5 |

Same-objective greedy 在 4/5 seeds 與 exact 選到相同集合；seed 1 的目標差為 0.03345。Legacy 在 3/5 seeds 與 exact 相同，另外兩個 seeds 的目標差為 0.03349 與 0.01181。

Exact 保證本次定義的選人目標最高，但平均 pooled error 並非最低；same-objective greedy 的平均 pooled error 反而略低 0.00076。這不代表 greedy 品質較佳，因為只有 5 seeds；它證明目前選人目標與最終 X 品質並非單調等價。後續若要改善招募規則，應直接驗證目標權重與 delivered-X 品質的關係，不能只報集合目標值。

## 驗收

- [x] 15 個 coalition 全部未超過 selection budget。
- [x] 三種方法都保存並比較最終 X 的 pooled、true、可行性與 Stage 3 查詢量。
- [x] Exact 與 same-objective greedy 共用目標、budget 與 size constraint；單元測試驗證回報目標可由同一函式重算。
- [x] 資格 target 0.0 與正式 target 0.5 分開記錄。
- [x] 招募承諾、基本報酬負債、剩餘池與未執行分潤分開記錄。
- [x] 15 個交付 X 重載 frozen evaluator 後 pooled 與真實分數皆 `mismatches=0`。
- [x] 報告未提出完整公平性保證。
- [ ] 僅完成小型 5-seed pilot；尚未驗證更多預算、報價分布、coalition sizes 或正式 seeds。

## 執行與證據

```sh
/Users/yclee/anaconda3/bin/python p6_recruitment_experiment.py \
  --output-dir alpha_weight_outputs/p6_recruitment_pilot_v2_20260923 \
  --seeds 0,1,2,3,4

/Users/yclee/anaconda3/bin/python analyze_p6_recruitment.py \
  alpha_weight_outputs/p6_recruitment_pilot_v2_20260923 \
  --output-dir alpha_weight_outputs/p6_recruitment_pilot_v2_analysis_20260923

/Users/yclee/anaconda3/bin/python reevaluate_p6_solutions.py \
  alpha_weight_outputs/p6_recruitment_pilot_v2_20260923 \
  --output alpha_weight_outputs/p6_recruitment_pilot_v2_20260923/reevaluation.csv
```

主要輸出：

- `alpha_weight_outputs/p6_recruitment_pilot_v2_20260923`
- `alpha_weight_outputs/p6_recruitment_pilot_v2_analysis_20260923`
- `report_figures/p6_recruitment_pilot_20260923.png`

`v2` 另在 manifest 保存 Git commit、工作目錄狀態、9 個來源檔 SHA-256、Python 與套件版本；其 `selection_results.csv` 與先前未含完整 provenance 的首輪執行逐位元相同，首輪目錄只保留作執行歷程。

## 決策

P6 的可重現比較框架與共同目標基準已建立。Exact 適合作為小型離線驗證基準；目前沒有證據支持僅憑較高集合目標即可宣稱 final X 較好。若進入正式 P6，應先登錄預算與 eta 網格，以 final pooled error、失敗率及成本共同決策。
