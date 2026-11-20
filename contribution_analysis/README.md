# 角色研究的必要實作與凍結依賴

此處只保留角色分析與冷快取重播實際匯入的模組。一般使用者從 [experiments/contribution](../experiments/contribution/README.md) 的三個入口執行。

- `senior_three_packages_v1`：固定端點下三種評分器、角色分數、全任務參照及影子付款。
- `peer_exclusion_cost_v1`：四個方案的 peer-scored 冷快取重播。
- `model_step_decision_v1`：可重播單輪與恢復狀態的方向引擎。
- 其餘保留模組提供上述程序的數值容差、branch、cache、ledger 與固定端點支援；不是額外的論文比較組。
- 三個 `*_experiment_*/source_snapshot/` 是目前重播載入的凍結來源。其內容保留，以維持歷史端點和 solver 行為。

歷史核心函式保留原路徑，避免破壞保存案例中的載入位置。非必要探索套件與工作報告位於忽略的 `.local_archive/`。本地 registration、records、模型與結果不納入 Git，也不以程式碼假裝附帶這些資料。
