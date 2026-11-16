# P3 貢獻估計外部效度正式實驗登錄

登錄日期：2026-10-06。正式結果產生前固定本文件；不得依 seeds 3600–3629 的結果調整方法或門檻。

## 問題與定義

本實驗驗證 operational candidate-only internal contribution 是否能預測完整移除參與者後的 pooled-evaluator 品質變化。完整移除同時刪除 candidate、local engine 與 alpha，正規化剩餘 alpha，從完整 coalition 候選點平均的相同起點重跑 Stage 3；不重訓 local model，也不招募替代者。

- `C_op = L_eval(X_without_proposal) - L_eval(X_full)`。
- `C_candidate_pooled = E_pooled(X_without_proposal) - E_pooled(X_full)`。
- `C_full = E_pooled(X_without_agent) - E_pooled(X_full)`。

pooled evaluator 僅在解產生後評分，不得進入 selection、Stage 3、alpha 或 pruning。真實函數只作診斷。本實驗不估計 Shapley value，也不執行 alpha 更新、pruning 或付款。

## 固定設計

- Seeds：3600–3629；pilot seeds 0–4 不納入正式樣本。
- Clean：Stage 1 選出的三人 coalition、uniform static alpha、uniform internal evaluator。
- Controlled poison：固定五人且一名污染者、uniform static alpha、trimmed 20% internal evaluator。
- 2D、target 0、box constrained `[-1,1]`、可行性容差 `1e-8`。
- `best_of` solver，Custom 30 iterations；每 seed/scenario 共用 frozen clean pooled evaluator。
- 預期 60 個 seed-scenarios、240 個 participant effects。

## 主要分析與門檻

以 seed 為抽樣單位，分情境計算 `C_op` 對 `C_full` 的 Spearman 與三分類方向一致率。使用固定 RNG 20261006、10,000 次 seed bootstrap，對兩情境乘兩主要指標提供 98.75% Bonferroni simultaneous CI。

主要 dead zone 為 `1e-7`；`1e-6` 與 `1e-4` 僅為敏感度分析。

兩情境均需滿足以下條件，才判為 `usable_external_validity`：

- Spearman CI 下界大於 0，且平均點估計至少 0.5。
- 方向一致率 CI 下界大於 0.5，且平均點估計至少 0.70。
- 預期 seeds 與 participant effects 完整，成功解全部可行，保存解重評一致。

若統計下界通過但實質門檻未通過，判為 `limited_signal`；其他情況為 `external_validity_not_established`。失敗與缺漏不得刪除，缺少任一預期 seed 時不得作正式有效性主張。

## 次要結果

報告 candidate-pooled 對 full-removal 的排名相關、負 `C_op` 移除改善率、invalid q、pooled/truth 衝突、污染者移除效果、查詢量、耗時與代表案例。次要結果不得取代主要判準。

## Pilot 啟動證據

修補後 pilot 位於 `alpha_weight_outputs/p3_external_validity_pilot_v2_20261006`：Clean 5 scenarios/15 effects、Controlled poison 5 scenarios/25 effects；120 筆保存解重評零 mismatch；正式驗收器通過。Pilot 結論仍為 `external_validity_not_established`，只用於驗證流程，不作正式研究結論。
