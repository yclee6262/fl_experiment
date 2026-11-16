# P3 貢獻有效性 Pilot（2026-09-22）

## 問題與比較語義

本階段檢查目前的候選點排除貢獻，能否代表「完整移除參與者」後的 delivered X 外部品質變化。

兩種反事實必須分開命名：

1. **Operational candidate-only exclusion**：移除一個候選點，但保留該參與者的 local engine 與現有 α；以目前 Stage 3 目標重新搜尋，再用固定內部 `L_eval` 計算原始 `C_i`。
2. **Full participant removal**：同時移除該參與者的候選點、local engine 與 α，將剩餘 α 正規化，從與完整 coalition 相同的固定 feature-space 起點重跑 Stage 3。local models 不重新訓練，也不重新招募替代者。

兩種解完成後才由相同的 frozen pooled evaluator 評分。主要完整移除貢獻定義為：

`C_full,i = E_global(X_without_i) - E_global(X_full)`

正值代表移除後誤差增加，該參與者對 delivered X 有幫助；負值代表移除後誤差降低。這是本次固定倒推流程下的反事實影響，不是完整資料市場的 Shapley value。

## Pilot 設計與驗收

- seeds 0–4，2D，target 0。
- 使用 P2 決策後的 A：Uniform、靜態 α。
- Clean：Stage 1 選出的 3 人 coalition，uniform 內部 evaluator。
- Poison：固定 5 人且包含 1 名污染者，trimmed 內部 evaluator。
- 每個 scenario 的 baseline 與所有完整移除案例使用同一個固定起點：完整 coalition 候選點平均。
- 每個 scenario 只建立一個 frozen pooled evaluator，所有 participant counterfactual 共用。
- 保存 baseline、operational restricted solution、full-removal solution、剩餘 agent IDs、正規化後 α、原始 `C_i`、外部貢獻與查詢成本。
- Clean／Poison manifest 均為 `complete`；共 10 scenarios、40 個 participant effects。Baseline、candidate-only 與 full-removal 解全部通過相同的 box feasibility 檢查。
- 從磁碟重新載入 frozen evaluator 後，Clean 45 個與 Poison 75 個保存解全部重評一致，合計 120 個解、`mismatches=0`。
- 核心回歸測試確認完整移除同時刪除 candidate、engine、α，且完成後恢復原 coalition 狀態。

輸出：

- `alpha_weight_outputs/p3_candidate_vs_full_clean_20260922_v2`
- `alpha_weight_outputs/p3_candidate_vs_full_poison_20260922_v2`
- `alpha_weight_outputs/p3_candidate_vs_full_analysis_20260922_v2`

第一版目錄保留不覆蓋；v2 新增三類 solution 的 box feasibility 欄位，數值結果不變。

## 結果

| 指標 | Clean | Controlled poison |
|---|---:|---:|
| Seeds | 5 | 5 |
| Participant effects | 15 | 25 |
| 三分類符號完全一致率 | 46.7% | 60.0% |
| 兩者皆非零時的方向一致率 | 50.0%（7/14） | 60.0%（15/25） |
| 每 seed `C_i`／完整移除效果 Spearman 平均 | 0.10 | 0.24 |
| 目前判為負貢獻的人數 | 4 | 14 |
| 負貢獻者移除後 pooled 品質改善率 | 50.0%（2/4） | 50.0%（7/14） |
| 無有效 q scenarios | 1/5 | 1/5 |
| 候選點排除平均查詢數／scenario | 948.6 | 6,992.0 |
| 完整參與者移除平均查詢數／scenario | 614.0 | 4,969.6 |

Poison 組每個 seed 固定有一名污染者；5 名污染者中只有 2 名在完整移除後改善 pooled 品質。這表示「污染標記」與「在特定倒推問題上造成負面 delivered-X 影響」也不能直接視為同一件事。

即使把 operational candidate-only restricted solution 改用 pooled evaluator 評分，其對完整移除效果的平均排名相關仍只有 Clean 0.20、Poison 0.14。因此差異不只來自內部 `L_eval` 尺度；保留被排除參與者的 engine 與 α，確實改變了反事實問題。

完整移除在本次模型查詢計數中反而較少，原因是每次反事實的 objective 少一個 local engine；這不代表真實部署的完整移除必然便宜。本 pilot 不包含 local model 重訓、重新招募、資料搬移或 coalition 重新選擇成本。

## 判讀與決策

本 pilot 不支持把目前 operational `C_i` 或 q 解釋成完整參與者貢獻：

1. 符號一致率只約一半至六成，被判為負貢獻者的移除成功率只有 50%。
2. 排名相關低且跨 seed 波動大；Clean 各 seed Spearman 從 -1.0 到 1.0，Poison 從 -0.1 到 0.9。
3. 兩組都出現無有效 q，不能以均勻分布代替或刪除案例。
4. P2 已顯示 α–q 一致性改善沒有建立外部 X 品質改善；P3 進一步顯示 q 與完整移除效果的關係偏弱。

因此後續限制如下：

- 論文與文件將現有估計稱為「候選點排除下的內部影響」，不稱為完整 participant LOO 或 Shapley contribution。
- 暫停以目前負 `C_i` 自動 pruning；否則 pilot 中約一半負貢獻判斷會導向未改善的完整移除。
- 付款若沿用此量，必須明示其操作性定義與 fallback，不能宣稱完整公平性。
- 暫不進行 Shapley，因 coalition value 與空集合價值尚未固定，且基礎近似尚未顯示足夠外部效度。
- 若後續要把完整移除估計納入方法，需另做獨立 30-seed 版本與成本預算；不得把本 5-seed pilot 當成正式穩定性結論。

## 下一步

依 P2／P3 證據，P4 不應先調整 α 更新或啟用 pruning。可獨立實作且風險最低的項目是「無進展停止」：辨識 Guarded 在最小步長、持續拒絕且 accepted state 不再改善的狀態，將它與 fixed-point convergence 分開記錄，並用既有 P2 seeds 比較是否降低查詢成本而不改變 accepted X。
