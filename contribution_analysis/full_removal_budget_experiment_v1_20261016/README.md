# Full-removal 預算穩定性研究 v1

本輪研究：相同模型、coalition、固定完整候選平均起點，對比 best_of 的 custom 30／120 次，SLSQP 設定不變。只有解保存後才用 frozen pooled evaluator 評分；truth 僅診斷。不做 α 更新、pruning、付款或 Shapley。

## 來源與執行狀態

- `registration.json`、`full_removal_budget_preregistration_20261016.md`：pilot 前凍結的設計、種子、成本與驗收。
- `source_snapshot/`：8 個歷史來源與5個本輪研究／測試檔案，SHA256 在登錄中。
- `pilot/`：2 seeds×2情境。每個 scenario 的 preparation.json 保存資料／local model／evaluator artifact hash、coalition.json 保存起點與成員。solves/ 保存 A、B、第一個 seed 的 replay_A。
- `pilot_gate.json`、`pilot_independent_check.json`：初始 pilot／重播／重載技術驗收，50 次求解重載差0。
- `confirmation/`：10 個另行保留的 seed×2情境，與pilot不混合統計。
- `execution.json`、`analysis/`：原執行因 OpenMP 共享記憶體問題停止，保留原始 incomplete 狀態。不能引用這裡的缺失零筆彙總作完整確認結果。
- `recovery_registration.json`、`recovery_source_snapshot/`：任何確認組 pooled 評分前保存的基礎設施修訂；不改種子、模型、求解或統計方法。詳見 `../../docs/full_removal_budget_recovery_amendment_20261016.md`。
- `recovery_execution.json`：從既有成功輸出恢復後的排程狀態；calls／samples／solver_seconds 包含一次失敗程序的保守預留，不是全為實測。
- `recovery_analysis/`：恢復完成後的完整驗收與登錄分析。原 analysis/ 不覆蓋。
- `independence_and_intervention_audit.json`：額外唯讀來源／資料獨立性與 A/B 介入一致性核對；不是新增假說檢定。
- `stop_label_audit/`：初版觀測器誤判內層break造成7筆停止標籤錯誤（確認3筆），以原trace還原後的獨立診斷。原數值及標籤不覆寫。停止原因應讀此處，原recovery_analysis的custom_stop欄位與彙總有這項已知限制。

完整繁體中文結果：`../../docs/full_removal_budget_results_20261016.md`。

## 分析檔案

`recovery_analysis/acceptance.json` 是完整性、可行性、重載與配對驗收；`results.json` 是門檻判定與成本，兩者不可混為一個「有效」標籤。

`participant_effects.csv` 保存 pilot／confirmation 明確分欄的 C_A、C_B、幅度改變、pooled／truth 與 X 變化；正式本輪確認只看 phase=confirmation。`seed_metrics.csv` 保存主 δ=1e-4 與兩個敏感度，不將參與者當獨立樣本。`solver_records.csv` 保存逐解引擎、success、停止理由、calls、輸入樣本與時間；`reload_verification.csv` 與 `replay_verification.csv` 保存驗收差值。

本輪是新的直接完整移除預算研究，不回填成既有 P3 的確認結果。任何有限穩定性結果均不等於全域最優、所有起點穩定、全訓練歷程貢獻或部署可用刪人訊號。
