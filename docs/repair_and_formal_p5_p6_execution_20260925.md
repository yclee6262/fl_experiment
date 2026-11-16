# 缺漏修補與正式 P5／P6 代理目標驗證紀錄

更新日期：2026-09-25。本文依據已核准的修補計畫記錄程式、驗收和實驗狀態。`完成` 指程式與輸出通過對應檢查；`研究結果未支持` 仍可構成有效交付；`未執行` 不得引用成結果。

## 目前狀態

| 工作 | 狀態 | 證據位置或下一步 |
|---|---|---|
| 失敗列、成本與分析工具修補 | 已實作；55 個測試通過 | 根目錄 `research_analysis.py`、P5/P6 分析器與驗收器 |
| P4 修補版 pilot | 已完成並驗收 | `alpha_weight_outputs/p4_repair_v4_20260925` |
| P5 修補版 pilot | 已完成並驗收 | `alpha_weight_outputs/p5_repair_v3_20260925`；80/80 條件、seeds 0–4 |
| seed 0 隔離重播 | 已通過 | `alpha_weight_outputs/p5_replay_seed0_20260925/replay_verification.json` |
| P6 budget/eta pilot | 已完成並驗收 | `alpha_weight_outputs/p6_repair_v1_20260925`；90/90 條件 |
| P6 全 coalition 枚舉 | 已完成並重評 | `alpha_weight_outputs/p6_enum_v1_20260925`；875/875 coalition |
| P5 seeds 200–229 正式比較 | 已完成；無穩定 D 優勢 | `alpha_weight_outputs/p5_formal_v1_20260925/analysis_v3`；480/480 條件 |
| Notion「0923」頁更新 | 已同步並回讀 | 最新完成狀態置於原有歷史章節前；舊內容明確標為 2026-09-23 快照 |

## 決策與統計口徑

- A（Uniform、不更新 α）維持主要方法，D（Guarded）為比較方法。
- 主要品質判準為凍結 pooled-data evaluator 的 `|f_pooled(X)-target|`。合成真實函數只作診斷，絕不回饋更新或選人。
- 只對 A、D 都有有效、有限、box 可行交付的 seed 計算品質差 `E_D-E_A`；負值代表 D 較佳。qualification failure、solver failure、無可行選人、不可行輸出和有效交付分開列示。
- 所有新版研究停用模擬結算，結果必須寫 `payment_status=not_run_by_configuration`。一般舊入口預設維持結算，供歷史重現；歷史的 `payment_requests=0` 不代表未結算。
- box 可行容差為 `1e-8`，從磁碟重載分數容差為 `1e-7`。缺品質值或未知成本留空，不以 0 代替。
- Bootstrap 以 seed 為單位，10,000 次、固定分析 RNG seed `20260924`。正式 P5 前七個情境以 family-wise 95% Bonferroni 區間；`hard_target_2p5` 為探索性。

## 已完成程式修補

1. `--no-settle-payment` 可關閉結算；預設仍是舊版行為。P4/P5 runner 明確停用付款，沒有付款列，並回報 `not_run_by_configuration`。
2. `StageCounter` 現可安全深拷貝；資料建立、資格檢查、proposal、殘差預計算、Stage 3、貢獻估計、付款及外部評分以共用／方法獨立欄位分開計時或計數。共享成本在獨立檔案按 seed 列出，分析時不在方法總額重複加總。
3. 固定 participant IDs 仍須通過資格檢查；未通過者在 Stage 3 前保存為 `qualification_failure`，不再偷偷強制進 coalition。成功交付保留 X、accepted state 與模型；失敗的品質欄位留空。
4. P5/P6 分析按 manifest 的方法、情境、seed、budget 和 eta 檢查重複鍵、缺漏鍵及多餘鍵。結果缺列會輸出診斷，但不形成正式優勢判定。
5. P4/P5/P6 重評器遇到沒有交付 X 的列，寫「不適用」及原因，不把空向量送入 evaluator；成功解會直接重算 box 違規與外部分數。
6. Evaluator artifact 新增 target 距離區間 `[0,0.1]`、`(0.1,0.25]`、`(0.25,0.5]`、`(0.5,1]`、`>1` 的測試樣本數與 truth MSE。`<30` 是描述性稀疏標記，不是經驗驗證的可靠性門檻，也不代表 OOD。
7. P4、P6 pilot seed 已參數化；P5 沿用 `--seeds`。run manifest 保存來源與 artifact SHA-256。P5 分析包含 paired error、Bonferroni 判讀、pooled/truth 方向衝突、α–q 描述性資料、逐輪品質／查詢資料與代表案例。
8. 新增 `pytest.ini` 排除輸出目錄中的來源快照，避免 pytest 重複收集同名測試檔。

## P4 修補版 pilot（seeds 0–4）

### 目的與操作

固定 Uniform 初始 α、Guarded 更新率 0.5、最多 20 rounds、無 pruning。clean 與 controlled-poison 各比較 patience 0 與 2；同 seed 使用相同資料與模型。驗收聚焦 accepted X／α 是否一致及停止規則能節省多少 Stage 3／contribution 查詢，不將無進展停止解讀為品質提高。

### 結果

| 情境 | 完整 seed 配對 | 可比較 accepted-state 配對 | qualification failure | 查詢／狀態結果 |
|---|---:|---:|---:|---|
| clean | 5 | 5 | 0 | X、α 與 pooled 分數全部一致；patience 2 節省 36.9% 查詢 |
| controlled poison | 5 | 0 | 5 | 五個 seed 的固定污染參與者均未通過 Phase 1；兩組都在 Stage 3 前停止，無 X 可評分 |

clean 10 筆重評分數吻合，最大 box 違規為 0。controlled-poison 不能作為 Guarded／停止機制的 Stage 3 結果；它只能說明現有 qualification gate 拒絕這個設定下的污染參與者。P4 驗收器：`verify_research_pilots.verify_p4` 通過。原 `p4_repair_v2_20260924` 是未加資格門檻的舊試跑，雖保留其 solver 結果，但不作資格控制後的證據。

## P5 修補版 pilot

八個事前指定情境、A/D、seeds 0–4，共 80 個預期方法×seed 條件。七個非高目標情境作確認性設計的 pilot；target 2.5 僅探索。P5 須保留固定 coalition 與資格結果；A/D 相同 upstream qualification failure 是可報告的結果，但不能當成有效 X 品質配對。

**狀態：pilot 與正式比較均完成。**修補版 pilot 根目錄為 `alpha_weight_outputs/p5_repair_v3_20260925`。80/80 條件完整；重評器及 manifest 驗收器均通過，所有成功交付都可行，付款狀態為 `not_run_by_configuration`。參考情境 5/5 配對有效；heterogeneity 與 controlled-poison 各有 5/5 seed 的兩方法資格失敗，因此不計 X 品質配對。其餘情境均有 5/5 有效配對。

Pilot 參考情境的 `E_D-E_A` 平均值為 -0.00399，但 Bonferroni family-wise 95% 區間為 [-0.00939, 0.00034]，跨過零，不支持 D 有穩定改善。sample imbalance、low-quality、dimension 5、五人 coalition 的 Bonferroni 區間也都跨零；pooled/truth 方向衝突分別在 sample imbalance 2/5、dimension 5 1/5、五人 coalition 2/5 配對出現。target 2.5 僅探索，不納入確認性主張。完整結果見 `analysis/scenario_comparisons.csv`、`analysis/coverage_audit.csv`、`analysis/diagnostics/` 及 `analysis/evaluator_coverage.csv`。

### P5 正式比較（seeds 200–229）

正式開始前掃描 104 份舊 run/suite manifests，未發現 seeds 200–229；480 個方法×seed 條件全數完成，coverage 無重複、缺失或額外鍵。所有有效交付經 pooled evaluator 與 truth 重評，評分差不超過 `1e-7`，直接重算 box feasibility 全數通過。source/artifact hash 與付款停用驗收通過。seed 0 隔離重播另見下節。

| 確認情境 | 有效配對 | 平均 `E_D-E_A` | 配對 95% CI | 七情境 Bonferroni 95% CI | pooled/truth 方向衝突 |
|---|---:|---:|---:|---:|---:|
| reference_clean | 30/30 | -0.00129 | [-0.00403, 0.00152] | [-0.00507, 0.00265] | 4/30 |
| heterogeneity_0p6 | 0/30 | — | 不適用 | 不適用 | 0/30 |
| sample_imbalance_4x | 30/30 | 0.00070 | [-0.00129, 0.00307] | [-0.00195, 0.00403] | 4/30 |
| low_quality_25pct | 30/30 | 0.00219 | [-0.00213, 0.00721] | [-0.00363, 0.00924] | 6/30 |
| controlled_poison | 0/30 | — | 不適用 | 不適用 | 0/30 |
| dimension_5 | 30/30 | 0.00175 | [-0.00624, 0.01073] | [-0.00929, 0.01436] | 10/30 |
| controlled_coalition_5 | 30/30 | 0.000003 | [-0.00359, 0.00352] | [-0.00504, 0.00483] | 8/30 |

七項確認比較的 Bonferroni 區間都未完全小於零；對有完整品質配對的五情境，D 全部是 `no_stable_improvement`。heterogeneity 與 controlled_poison 的 A、D 各有 30/30 qualification failure，沒有有效 X 品質樣本，故不得據此說兩方法品質相同或某方較好。target 2.5 為探索性：30/30 配對，平均差 -0.0000761，95% CI [-0.00321, 0.00249]，不納入確認性結論。

配對平均品質以外，D 的 Stage 3 加貢獻估計平均查詢量約為 A 的 8.7–10.0 倍（高目標探索情境約 11.2 倍）；因此本批沒有證據支持以高計算成本換取穩定品質改善。各階段 requests、輸入樣本數、耗時及 shared costs 分列於 `analysis_v3/method_results.csv` 與 `shared_costs_per_seed.csv`。

**分析修正紀錄：**第一版正式分析只把 solver failure／不可行列算入「觀察失敗」，漏列 qualification failure。原始輸出保留；`analysis_v3/` 依 `delivery_status != valid` 計算完整失敗數並加入 status breakdown。此修正不改變有效配對誤差、bootstrap CI、Bonferroni CI 或 truth 衝突數；以 `analysis_v3/` 為正式引用版本，修正原因和範圍記在 `alpha_weight_outputs/p5_formal_v1_20260925/analysis_revision_notice.json`。

`p5_repair_v1_20260924` 已在執行中發現舊流程會把未通過資格的 Agent 20 強制放入 coalition，故中止並保留為無效診斷；不得沿用該批的表面全成功品質結論。

## P6 pilot 與全 coalition 代理目標枚舉

P6 pilot 固定十名合格候選、最多三人、資格 target 0、正式 target 0.5、Uniform α；budgets 為 2.5、3.5、8.0，eta 為 0、0.5，方法為 exact、objective greedy、legacy greedy，五個 pilot seeds。每個條件需保存預算／人數、選人目標、proposal、Stage 3 solver、交付狀態、品質、成本及 `not_run_by_configuration`。

全枚舉每 seed 評估所有一至三人 coalition，十人時最多 175 組；每組只倒推一次。以每 seed／budget／eta 為彙整單位，報 Spearman(objective, -pooled error)、top-five 集合重疊，以及 objective 所選可行集合相對最佳有效集合的 pooled-error gap。失敗 coalition 保留，不把 coalition 數當獨立 seed。最佳 pooled 集合是離線診斷，不回饋部署選人。

**狀態：兩項工作均完成。**P6 budget/eta pilot 有 90/90 完整條件，90 個條件均取得符合 budget、三人上限與 box 限制的有效交付；重載 evaluator 後 90 筆 pooled/truth 分數完全吻合。來源與 artifacts 驗收通過。

全枚舉每個 seed 恰為 175 組、五 seed 共 875 組，875 組均成功且可行；重評 875 筆沒有分數差異，artifact/source hash 驗證無錯。所有預算與 eta 條件皆有 5/5 seeds。seed-level 平均 Spearman(objective, -pooled error) 約為 0.40–0.52；top-five 集合重疊率平均約 0.04–0.32；代理目標所選集合相對最佳有效集合的 pooled-error regret 平均約 0.012–0.022。各 seed/條件結果在 `objective_validity.csv`，按 seed 彙總於 `analysis/budget_eta_summary.csv`。這些結果說明招募代理目標與最終 X 品質只有有限關聯，最佳品質集合是離線參考，不回饋選人規則。

P5 reference_clean seed 0 隔離重播通過：pooled 資料及切分完全一致；模型權重、X／α／q、分數最大差均為 0，離散決策一致。紀錄在 `alpha_weight_outputs/p5_replay_seed0_20260925/replay_verification.json`。重播驗收器已修正為按 seed 對應原始 5-seed artifact，避免把 5-seed pilot 和單 seed 重播誤判為條件鍵缺失。

## P5 正式比較啟動條件

正式固定 seeds `200–229`，開始前掃描 104 份舊 run/suite manifests，沒有重複 seed。8×2×30＝480 個方法條件已完成。來源雜湊、方法、情境、主要指標、統計及判讀門檻登錄在 `alpha_weight_outputs/p5_formal_preregistration_20260925.json`；更正後的最終分析在 `alpha_weight_outputs/p5_formal_v1_20260925/analysis_v3`。執行結果：

1. P4/P5/P6 pilot 的完整 coverage、來源／artifact hash、付款停用、失敗分類與重評均通過。
2. 隔離 seed 0 重播通過；資料切分、模型、離散決策一致，X／α／q／評分最大差為 0。
3. 正式設定依封存規格執行完成，沒有依正式結果調參。
4. 七情境 CI 均未支持 D 穩定優勢；A 維持主要基準。兩個資格失敗情境沒有品質判決，需列為資格／招募限制。

## 測試與本地驗收指令

使用研究環境執行：

```bash
PYTHONPATH='.venv/lib/python3.11/site-packages:.' python -m pytest -q
python -m py_compile alpha_weight_experiments.py p4_constrained_no_progress_pilot.py p5_scenario_suite.py p6_budget_pilot.py p6_coalition_enumeration.py analyze_p5_results.py analyze_p6_recruitment.py verify_research_pilots.py
```

pilot 完成後再執行：

```bash
python verify_research_pilots.py --p4 <P4_DIR> --p5 <P5_DIR> --p6 <P6_DIR>
```

本地文件及 Notion「0923」均已記錄正式結果與其限制；Notion 更新後重新 fetch，確認最新狀態與 analysis_v3 可見，且舊內容保留為歷史快照。

### P5 cost-schema interruption note

`p5_repair_v2_20260925` was stopped after the implementation review found that query counts did not include the number of input samples per batched API call. Its qualification-gating rows remain diagnostic only. The final pilot will record both request counts and input sample totals by stage.
