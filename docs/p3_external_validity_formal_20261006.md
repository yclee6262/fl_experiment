# P3 貢獻估計外部效度正式實驗（2026-10-06）

## 結論

正式實驗未建立現行 operational contribution 的外部效度。現行量應繼續稱為「候選點排除下的內部影響」，不能解釋成完整參與者貢獻，也不應直接用於自動 pruning 或公平分潤。

這是完整且通過驗收的負面研究結果，不是因為缺漏、不可行解或 evaluator 重載失敗而無法判讀。

## 問題與方法

主要問題是：只移除 agent 的候選點、保留其 local engine 與 alpha 後得到的內部貢獻，能否預測同時移除 candidate、engine 與 alpha 後，最終解在 frozen clean pooled evaluator 上的品質變化。

- `C_op = L_eval(X_without_proposal) - L_eval(X_full)`。
- `C_candidate_pooled = E_pooled(X_without_proposal) - E_pooled(X_full)`。
- `C_full = E_pooled(X_without_agent) - E_pooled(X_full)`。

完整移除後將剩餘 alpha 正規化，從相同固定起點重跑 Stage 3。local models 不重訓，也不重新招募替代者。pooled evaluator 只在解產生後評分，沒有參與 Stage 3、alpha、選人或 pruning。

正式使用獨立 seeds 3600–3629：Clean 為 Stage 1 選出的三人 coalition、uniform static alpha 與 uniform internal evaluator；Controlled poison 為固定五人含一名污染者、uniform static alpha 與 trimmed 20% internal evaluator。兩組皆使用 2D、target 0、`box_constrained [-1,1]` 與 `best_of` solver。

## 完整性與重現驗收

- 60/60 seed-scenarios 完成。
- Clean 90/90、Controlled poison 150/150 participant effects 完成。
- 所有 baseline、candidate-only 與 full-removal 成功解符合 box。
- 720 筆保存解評分紀錄重載 evaluator 後零 mismatch；其中 baseline 在同 seed 的 participant rows 重複保存，對應 540 個唯一解。
- Pilot seed 0 隔離重播的選人、固定起點、baseline X、q 與核心評分差異均為 0。
- 正式驗收器通過，沒有重複鍵、缺 seed、缺 artifact 或非有限主要指標。

## 主要結果

以 seed 為單位、10,000 次 bootstrap，以下為事前登錄的 98.75% Bonferroni simultaneous CI：

| 情境 | 平均 Spearman | 98.75% CI | 方向一致率 | 98.75% CI | 判定 |
|---|---:|---:|---:|---:|---|
| Clean | 0.083 | [-0.250, 0.400] | 40.0% | [27.8%, 52.2%] | 未建立外部效度 |
| Controlled poison | -0.047 | [-0.290, 0.197] | 48.7% | [37.3%, 60.0%] | 未建立外部效度 |

事前成功門檻要求兩情境的 Spearman CI 下界大於 0、點估計至少 0.5，且方向一致率 CI 下界大於 0.5、點估計至少 70%。兩情境均未達標。

dead zone 改為 `1e-6` 或 `1e-4` 時，Clean 方向一致率為 43.3%；Controlled poison 分別為 48.0% 與 43.3%。敏感度分析不改變結論。

## 次要結果

| 指標 | Clean | Controlled poison |
|---|---:|---:|
| Candidate-pooled 對 full-removal 平均 Spearman | 0.004 | 0.050 |
| 被判為負貢獻的人數 | 25 | 87 |
| 負貢獻者完整移除後 pooled 品質改善率 | 32.0% | 42.5% |
| Invalid q scenarios | 2/30 | 4/30 |
| Pooled／truth 方向衝突 | 23/90 | 42/150 |
| 污染者完整移除後 pooled 品質改善率 | 不適用 | 53.3%（16/30） |
| Candidate exclusion 平均查詢量／scenario | 1,020.4 | 5,061.7 |
| Full removal 平均查詢量／scenario | 573.3 | 3,537.6 |

即使將 candidate-only restricted solution 改由 pooled evaluator 評分，它與 full-removal 的排序關係仍接近零；主要落差並非只由內部與外部 evaluator 尺度不同造成，而是「只拿掉候選點」和「完整拿掉參與者」本身是不同反事實問題。

污染者移除改善率只有 53.3%，再次顯示污染標記不等同於該參與者在特定倒推問題中必然造成負面 delivered-X 影響。

## 研究決策

1. 保留現行 `C_op` 作內部診斷，不再把它描述為完整 participant contribution。
2. 不使用現行負 `C_op` 自動 pruning；正式結果中，多數被判負的人完整移除後並未改善 pooled 品質。
3. 不以現行 q 支持公平分潤主張；若保留付款流程，必須明示操作性定義與限制。
4. 若後續仍需要 participant-level contribution，研究方向應改為 full-removal estimator 的低成本近似，並另立新實驗；本結果不支持單純替換 evaluator 後繼續沿用 candidate-only exclusion。
5. 本實驗不是 Shapley value，未定義完整 coalition value、空集合價值或所有 coalition permutations。

## 產出

- 正式登錄：`docs/p3_external_validity_formal_preregistration_20261006.md`
- 結果：`alpha_weight_outputs/p3_external_validity_formal_v1_20261006`
- 正式分析：`analysis_v3/analysis_manifest.json`、`seed_metrics.csv`、`aggregate_metrics.csv`
- 衝突案例：`analysis_v3/pooled_truth_conflicts.csv`
- 圖表：`analysis_v3/p3_external_validity_summary.png`
