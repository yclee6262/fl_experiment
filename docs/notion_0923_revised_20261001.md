# 聯邦式倒推研究進度與驗收（截至 2026-10-01）

> 本頁以已保存的實驗資料與 2026-09-25 更正後分析為準。P2 是舊版求解流程；P4／P5／P6 是修正限制與失敗處理後的新版流程，兩版數字不可合併。主要品質判準是凍結 pooled evaluator 對**最終有效交付 X**的絕對目標誤差；合成真實函數僅作診斷。

## 先看結論

- **主方法維持 A：Uniform α、不更新。**新版 P5 的五個可進行完整品質配對的確認情境，Guarded（D）相對 A 的 Bonferroni 同時 95% 信賴區間均跨 0，沒有建立穩定品質優勢；D 的 Stage 3 加貢獻估計查詢量約為 A 的 8.7–10.0 倍。
- **無進展停止可保留。**P4 新版 clean pilot 的五組配對中，patience 2 與停用早停的 accepted X、α、pooled 分數相同，查詢量減少 36.9%。這是成本改善證據，不是品質改善證據。
- **現有貢獻估計不可解釋為完整參與者貢獻。**P3 的候選點排除與完整移除反事實關係偏弱；暫緩依負貢獻自動 pruning，也不宣稱 Shapley 或付款公平性。
- **招募代理目標仍需改良。**P6 Exact 可最大化已定義的選人目標，但較高選人分數不保證較低的最終 pooled 誤差。
- **兩個確認情境沒有品質結論。**P5 的 heterogeneity 0.6 與 controlled poison 固定 coalition，在正式 30 seeds 中 A、D 均於資格檢查失敗；這些列保留為失敗，不算成 X 品質配對。

## 研究問題、方法與判讀口徑

研究問題是：在不共建 global engine 的倒推流程中，調整參與者權重 α 是否能改善最後交付 X 的品質、可行性與計算成本？研究用 pooled evaluator 由乾淨資料聯集訓練並凍結，只在演算法完成後評分，不參與 α 更新、招募或 pruning。

- **A（Uniform static）**：入選者 α 均勻，單次倒推，不更新。
- **B（Stage 0 static）**：依 Stage 0 品質設定 α，單次倒推，不更新；只在舊版 P2 比較。
- **C（Fixed-step）**：從 Uniform 起始，以固定步長更新 α；只在舊版 P2 比較。
- **D（Guarded）**：從 Uniform 起始，依候選點排除得到的 q 提議更新 α；若同 coalition 的內部 evaluator 未接受新解，就保留 accepted state 並縮小步長。新版最多 20 rounds，啟用 patience 2 無進展停止。
- **主要品質**：`E_pooled(X) = |f_pooled(X) − target|`，越低越好。配對差 `Δ = E_D − E_A`；負值代表 D 較好。僅在同 seed 的 A、D 都有有限、box 可行的有效交付時比較品質。
- **輔助診斷**：`E_true(X) = |f_true(X) − target|`、α–q 距離、內部 `L_eval`、停止原因、各階段查詢量及時間。pooled 與 truth 對方法方向衝突時，逐案揭露。
- **統計單位**：seed。正式 P5 以 seed 配對 bootstrap 10,000 次；七個確認情境提供 Bonferroni 同時 95% 區間。target 2.5 只報探索性一般 95% 區間。

## 已完成實驗與結果

### P2｜舊版四種 α 策略（2026-09-22）

**目的**：先檢查動態更新是否比真正不更新的基準交付較好的 X。A、B、C、D 在 clean 與受控污染情境各用 seeds 100–129 配對，二維、target 0、不 pruning；同 seed 共用資料、local models、候選點與 evaluator。

**結果**：B／C／D 相對 A 的 pooled-error 配對 95% 區間在兩個情境全部跨 0。C／D 雖縮小 α–q 距離，沒有轉成穩定的外部品質收益。D 的總查詢量約為 A 的 16.4–16.5 倍。Clean C 的一筆 box 違規仍保留在舊結果，成為後續限制修正的觸發案例。此批屬 `legacy_unconstrained` 語義，不與新版受限結果混合。詳見 `docs/p2_formal_20260922.md`。

### P3｜候選點影響與完整參與者移除（5-seed pilot）

**目的**：檢查現有 signed contribution `C_i` 是否能代表「移除整個參與者」對最終 X 的影響。候選點排除只移除其 proposal，仍保留其 local engine 與 α；完整移除則同時移除 proposal、engine 與 α，從固定起點重跑倒推。後者不重訓模型，也不重新招募。

**結果**：clean／受控污染各五個 seeds，共 40 個 participant effects。`C_i` 與完整移除效果的每 seed 平均 Spearman 分別為 0.10／0.24；依現有 `C_i` 判為負貢獻的人，移除後 pooled 品質改善比例兩組皆為 50%。兩組各有 1/5 情境沒有有效 q。故暫緩用負 `C_i` 自動 pruning，亦不把它稱為完整 participant LOO 或 Shapley。此為 pilot，不是 30-seed 穩定性證明。詳見 `docs/p3_contribution_pilot_20260922.md`。

### 工程與實驗有效性修正

- 新版指定 `box_constrained`，座標上下界 `[-1, 1]`、可行性容差 `1e-8`；保留 `legacy_unconstrained` 供舊結果重現。受限求解、warm start、recovery、exclusion 及 `best_of` 均檢查可行性；全無有效候選時記錄 solver failure，不交付違規 X。
- 最終解取 **accepted state**；被拒絕的最後 proposal 不覆蓋它。結果分開保存 solver 數值狀態、最大 box 違規、交付狀態及失敗階段。
- Manifest 保存條件全集、設定、資料／模型 artifacts、來源 SHA-256；外部評分可從保存的 X 與 evaluator 重載。成功交付的重評容差為 `1e-7`。無 X 的列標示不適用，不把品質或未知成本填 0。
- 新版 P4／P5／P6 明確停用模擬結算，付款狀態為 `not_run_by_configuration`。舊流程曾執行模擬結算；舊欄位 `payment_requests=0` 不代表未結算。
- 成本按訓練、資格檢查、proposal、殘差預計算、選人、Stage 3、貢獻估計與外部評分分開；批次 API requests 和輸入樣本數分列。每 seed 的共用準備成本獨立保存，不重複加到每個方法。

### P4｜Guarded 無進展停止（新版 5-seed pilot）

**方法**：同一 Guarded 設定比較 `patience=0`（停用早停）與 `patience=2`（最小步長後連續拒絕即停止），clean 與 controlled poison 各 seeds 0–4，共 20 條件。

**結果**：clean 的五組有效配對，accepted X、α、pooled 分數全部相同，patience 2 減少 36.9% Stage 3 加貢獻估計查詢。controlled poison 的固定 Agent 20 在五個 seeds 均未通過資格，兩方法共十列均於 Stage 3 前停止；不可引用早期未正確執行資格檢查的 62.8% 節省數字作新版污染情境證據。詳見 `alpha_weight_outputs/p4_repair_v4_20260925`。

### P5｜八情境 A 與 D 正式比較（新版 seeds 200–229）

**方法**：先以 seeds 0–4 完成修補版 80 條件 pilot，之後封存方法與判讀規則，再用獨立 seeds 200–229 執行八情境 × A／D × 30 seeds，共 480/480 條件。一般三人情境固定 IDs `[1, 2, 20]`；五人／污染情境固定 `[1, 2, 3, 4, 20]`。污染比較只改 Agent 20 的標籤；低品質噪聲只改 local observed labels，pooled evaluator 維持乾淨設定。

| 情境與單一改動 | 有效配對 | 平均 Δ（D−A） | Bonferroni 同時 95% CI | pooled／truth 衝突 |
|---|---:|---:|---:|---:|
| Clean reference：二維、target 0 | 30/30 | −0.00129 | [−0.00507, 0.00265] | 4/30 |
| Heterogeneity 0.6：資料中心分散 | 0/30 | 不適用 | 不適用 | 無品質配對 |
| Sample imbalance 4×：樣本量不平衡 | 30/30 | +0.00070 | [−0.00195, 0.00403] | 4/30 |
| Low quality 25%：部分 local labels 噪聲提高 | 30/30 | +0.00219 | [−0.00363, 0.00924] | 6/30 |
| Controlled poison：五人中 Agent 20 污染 | 0/30 | 不適用 | 不適用 | 無品質配對 |
| Dimension 5：由二維增至五維 | 30/30 | +0.00175 | [−0.00929, 0.01436] | 10/30 |
| Coalition 5：由三人增至五人 | 30/30 | +0.000003 | [−0.00504, 0.00483] | 8/30 |

**確認性判讀**：五個有品質配對的情境，區間均跨 0；heterogeneity 與 controlled poison 的 A、D 各有 30/30 qualification failure，不能得出品質等效結論。D 的 Stage 3 加貢獻估計平均查詢量約為 A 的 8.7–10.0 倍。

**探索性情境**：Hard target 2.5 的 30/30 配對平均 Δ 為 −0.000076，普通 95% CI `[−0.00321, 0.00249]`；不納入確認性優勢主張。target 附近評估資料較稀疏，須另看 evaluator coverage 與 pooled／truth 衝突。

**驗收**：480/480 條件無重複或缺失鍵；全部有效交付通過 box 與重載評分檢查。隔離重播 reference clean seed 0，資料切分、模型、X／α／q、分數與離散決策一致。正式引用 `alpha_weight_outputs/p5_formal_v1_20260925/analysis_v3`。最初分析漏計 qualification failure，`analysis_v3` 已修正失敗數；有效配對誤差及 CI 未變。

### P6｜招募預算與代理目標（新版 5-seed pilot）

**方法**：每 seed 固定十名候選、最多選三人；qualification target 0，正式倒推 target 0.5，入選後使用靜態 Uniform。比較 Exact 枚舉、同一集合目標的 objective greedy，以及不同規則的 legacy greedy；預算 2.5／3.5／8.0，多樣性係數 `eta=0／0.5`，共 90 條件。Exact 與 objective greedy 使用相同全集、目標與限制。

**集合目標**：入選者的 `quality_share` 總和，加上 `eta ×` 成對 residual cosine 差異的正規化總和。它是招募時的代理目標，**不是**最終 X 的 pooled 品質。Exact 的 objective 不低於同目標 greedy，僅證明此小型枚舉的目標最優；legacy 規則不同，只比較交付品質與成本。

**結果**：90/90 條件符合預算、三人上限及 box，重評分數吻合。預算 2.5 主要選二人、3.5 允許部分三人組合、8.0 可選三人。六個 budget／eta 組合中，Exact 平均 pooled error 沒有優於該組表現最好的 greedy 方法。

**離線代理目標檢查**：每 seed 枚舉全部一至三人的 175 個集合，共 875 個，均有效可行。按 seed 彙總，選人 objective 與負 pooled error 的平均 Spearman 約 0.40–0.52；兩種前五名集合的平均重疊率約 4%–32%；依 objective 所選集合相對最佳有效集合的平均 pooled-error 差距約 0.012–0.022。五個 seeds 是統計樣本單位，875 個集合不可當成獨立樣本。離線最佳 pooled 集合只作診斷，不回饋招募。詳見 `alpha_weight_outputs/p6_repair_v1_20260925/analysis` 與 `alpha_weight_outputs/p6_enum_v1_20260925/analysis`。

## 讀取結果欄位時的共同字典

| 欄位／符號 | 意義與判讀 |
|---|---|
| `scenario`, `seed`, `method` | 情境、隨機種子、方法；P6 條件另含 `budget`、`eta`。同 seed 才能配對。 |
| `alpha`、`C_i`、`q_raw`、`q_update`、`q_pay` | 依序為最佳化權重、帶符號原始候選點影響、正貢獻正規化分布、更新分布、結算分布；不得混稱為 q。新版研究未執行結算。 |
| `final_solution`／accepted X | 最終通過接受規則的 X；只有 `delivery_status=valid` 才可當有效交付。 |
| `delivery_status`、`failure_stage`、`failure_reason` | 區分有效交付、資格失敗、無可行招募、solver failure、不可行輸出及其階段／原因；失敗列不可刪掉。 |
| `solver_success`、`actual_solver` | 數值狀態與實際 solver；成功訊號不取代有限性、box 與交付檢查。 |
| `box_feasible`、`max_box_violation` | X 是否在容差 `1e-8` 內、超界最大量；有效交付需重新由 X 計算。 |
| `external_global_error`、`true_error` | pooled 主要誤差與合成真實函數診斷誤差；無有效 X 時留空。 |
| `valid_pairs`、`mean_delta_D_minus_A`、`ci95` | 雙方有效的 seed 數、D−A 配對平均誤差、以 seed bootstrap 的區間；負差才代表 D 較好，區間跨 0 不支持穩定優勢。 |
| `stage3_requests`、`contribution_requests`、`input_samples`、`wall_seconds` | 倒推／貢獻 API 呼叫數、批次輸入樣本數、牆鐘時間；共用準備成本另列，不能重複加總。 |
| `payment_status` | 新版 P4／P5／P6 為 `not_run_by_configuration`；`payment_requests=0` 本身不足以判斷是否結算。 |
| `selection_objective`、`selection_budget`、`recruitment_bid_commitment` | 招募代理目標、預算、入選報價總和；報價承諾與基本報酬是同一筆負債，不重複計。 |
| `target_near_samples`、`target_near_truth_mse` | evaluator 測試資料中靠近 target 的樣本數與相對 truth 的 MSE。少於 30 只標為描述性稀疏，並非經驗證的可靠性門檻或 OOD 證明。 |

## 目前限制與下一步

1. **先處理資格／招募限制**：查明 heterogeneity 與受控污染固定 coalition 全部資格失敗的原因，設計能取得有效 coalition 的獨立版本；不能改寫既有正式失敗列。
2. **檢驗代理目標**：在不使用外部分數做線上決策的前提下，研究招募分數與最終 X 品質落差的原因，再提出新目標並用新 seeds 驗證。
3. **評估困難 target 的可靠性**：逐距離區間報 evaluator 樣本數、誤差及 pooled／truth 衝突；真實資料泛化尚未驗證。
4. **維持研究界線**：現階段不新增 α 調參、pruning、Shapley 或付款公平性結論；若恢復這些方向，需要重新定義貢獻與獨立驗證。

## 版本與來源

- 研究定義：`docs/research_execution_and_acceptance.md`
- P2 舊版正式：`docs/p2_formal_20260922.md`
- P3 貢獻 pilot：`docs/p3_contribution_pilot_20260922.md`
- P4／P5／P6 修補與驗收：`docs/repair_and_formal_p5_p6_execution_20260925.md`
- P5 正式登錄：`alpha_weight_outputs/p5_formal_preregistration_20260925.json`
- P5 正式分析：`alpha_weight_outputs/p5_formal_v1_20260925/analysis_v3`
- P6 離線枚舉：`alpha_weight_outputs/p6_enum_v1_20260925/analysis/budget_eta_summary.csv`

舊版 2026-09-23 Notion 敘述含未完成的預告與早期試跑數字，已由本頁現況版本取代；舊原始輸出及本地歷史文件仍保留，不能把版本間結果拼成同一批樣本。
