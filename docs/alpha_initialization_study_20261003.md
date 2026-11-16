# 初始 α 的靜態倒推比較：預先登錄與驗收

## 研究問題與邊界

本輪只問「在同一批已合格參與者、相同 local models 與 Stage 2 候選點下，哪種事前 α 能讓單次倒推的最終 X 在凍結的 pooled evaluator 上更接近 target？」α 是 Stage 3 的預期資訊權重，不代表報價、承諾報酬或事後分潤。報價仍屬 Stage 1 招募限制。這個實驗不執行動態 α 更新、貢獻估計、pruning 或付款。

pooled evaluator 僅在方法產生有效 X 後評分；合成真實函數僅作診斷。兩者不參與初始化、求解器選擇或交付輪次選擇。

## 事前固定方法

設 coalition 人數為 n、Stage 0 blind-test MSE 為 m_i，品質份額 `s_i = (m_i + 1e-5)^-1 / sum_j (m_j + 1e-5)^-1`，Uniform 為 `u_i=1/n`。

| 方法 | α 定義 | 可解釋性與限制 |
|---|---|---|
| A_uniform | u | 主要基準，無初始化模型查詢。 |
| B_stage0 | s | 現有品質基準；小型 blind test 的誤差可能不穩。 |
| quality_shrink | `(u+s)/2` | 將品質訊號往平等權重收縮，限制單一參與者支配。 |
| ensemble_shrink | `(u+w)/2`，w 為以 Stage 0 blind labels 擬合的非負、總和為 1 的最小平方權重 | 衡量互補性，但 20 筆 blind test 可能過度擬合；額外查詢每位模型一次、每次 20 筆。 |
| target_peer_shrink | `(u+t)/2`，`t_i ∝ s_i exp(-e_i / scale)`，e_i 是其他模型對 i 的 Stage 2 proposal 預測離 target 的絕對差中位數，scale 是所有有效 peer error 的中位數加 `1e-5` | 事前衡量品質與目標相容性；peer 可能共享偏差，並不保證 pooled 品質。 |

若 MSE 無效、擬合失敗或 peer 資料全無效，記錄原因並回退 Uniform；不把回退視為該候選方法的成功證據。固定 coalition 順序按 participant ID，α、MSE 與 proposal 同序保存。任何方法都不讀 pooled 評分來配置 α。

## 配對設計與主要分析

Pilot seeds `0–4`；正式 seeds `300–329`，正式啟動前重查這些 seeds 尚未用於本研究。參考情境 `reference_clean`，確認性擴充為 `sample_imbalance_4x`、`low_quality_25pct`、`dimension_5`、`controlled_coalition_5`。`hard_target_2p5` 僅允許探索性檢查，不列確認性優勢。過往 P5 的異質性與污染情境曾出現固定參與者資格失敗，因此本輪先不把它們納入完整配對主張。

每個 seed 同情境五方法共用資料、資格測試、fixed participant IDs、local models、proposal、pooled evaluator；僅改初始 α。預設 box `[-1,1]`、容差 `1e-8`，Stage 3 `best_of`，每方法只倒推一次。有效交付必須 X 有限且可行；失敗、缺漏與 fallback 分開報告。主要品質 `|f_pooled(X)-target|`，truth error 是診斷值。初始化 API 查詢、Stage 3 API 查詢、共享準備、evaluator 時間分開計帳。

對每個候選方法以 seed 配對差 `E_candidate-E_A` 比較，負值較好。使用 10,000 次 seed-level 配對 bootstrap；三候選的同一確認性情境使用 Bonferroni 調整的 95% 同時區間。B 亦作靜態品質基準，候選對 B 的差為補充分析。正式優勢需完整有效配對、同時區間上界小於 0，且觀察失敗數未增加。若仍跨 0，記為證據不足；不得依 pilot 調整公式後仍把正式比較稱作相同版本。

本輪的確認性情境彼此仍有多重比較風險，因此跨情境「普遍優勢」不得僅憑一個情境的區間作宣稱。最清楚的主要判斷是 reference_clean 的三候選對 A；其他情境檢查適用範圍。若 pooled 與 truth 的差異方向相反，列出衝突案例，不選擇有利的一邊。

## 實作、產物與執行指令

`initial_alpha_methods.py` 定義公式；`run_alpha_initialization_study.py` 做配對、保存與外部評分；`analyze_alpha_initialization_study.py` 產生覆蓋與統計；`verify_alpha_initialization_study.py` 重新載入 pooled evaluator 驗證 X 與分數。

```bash
python -m pytest -q test_initial_alpha_methods.py
python run_alpha_initialization_study.py --output-dir alpha_weight_outputs/alpha_initial_pilot_v1_20261003 --seeds 0,1,2,3,4
python verify_alpha_initialization_study.py alpha_weight_outputs/alpha_initial_pilot_v1_20261003
python analyze_alpha_initialization_study.py alpha_weight_outputs/alpha_initial_pilot_v1_20261003
python run_alpha_initialization_study.py --output-dir alpha_weight_outputs/alpha_initial_formal_v1_20261003 --seeds 300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329
```

每情境 `summary.csv` 的一列是一個方法／seed 條件；`shared_costs.jsonl` 每 seed 一列共享成本、資格報告與 evaluator coverage。`artifacts` 存 pooled 資料切分、local models、pooled evaluator；`run_manifest.json` 存來源快照與 SHA-256。最外層 `suite_manifest.json` 存方法、情境、seed、分析規則及執行狀態。`analysis.json` 與 `verification.json` 是分析與重評紀錄。舊結果不覆蓋。

重要欄位：`status` 為執行結果（valid、qualification_failure、solver_failure、invalid_output、interrupted）；`delivery_status=valid` 才有可評分 X；`initial_alpha`、`agent_ids` 與 `stage0_mse` 同序；`alpha_audit` 含原始分數與 fallback 原因；`actual_solver` 是 best_of 真正採用的求解器；`final_box_feasible` 與 `max_box_violation` 是交付限制檢查；`external_global_error` 是主要 pooled 誤差；`final_target_error` 是 truth 診斷；`initialization_requests`、`stage3_requests` 分別為模型 API 呼叫數，`*_input_samples` 是傳入模型的樣本數。沒有交付的品質欄位留空。

## 驗收關卡

- [x] Pilot 125 條件（5 情境 × 5 seeds × 5 方法）無缺漏或重複。
- [x] 每個有效 X 都符合 box，且重新載入的 pooled 分數差不超過 `1e-7`。
- [x] 每個 seed 的五方法共用相同 artifacts、coalition 與候選點；隔離重播 seed 0 的差異為 0。
- [x] 失敗、fallback、資料缺失的保存路徑與分析規則已建立；本輪實際 750 條件均有效、沒有 fallback，因此尚無正式失敗案例可作經驗驗證。
- [x] Pilot 通過後執行正式 750 條件（5 × 30 × 5）；正式前封存來源與設定，正式結果未回饋調參。
- [x] 報告 pooled 品質、truth 衝突、失敗率、初始化與求解成本；只在符合預定統計規則時作正式優勢主張。詳見 `alpha_initial_formal_results_20261003.md`。

此文件的勾選僅由實際跑出的 `verification.json`、`analysis.json` 與 manifest 支持，不把程式測試等同研究結果。
