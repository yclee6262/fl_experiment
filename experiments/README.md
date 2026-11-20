# 實驗入口

從專案根目錄執行 `python run.py <入口> [參數]`。輸出使用新資料夾。

| 入口 | 用途 | 詳細說明 |
|---|---|---|
| `stages` | 資格、招募、反推、搜尋、可選剪枝與結算 | [shared](shared/README.md) |
| `baseline` | FedAvg 訓練反推與協作流程比較 | [baselines](baselines/README.md) |
| `initialization` | U/Q/E/S 與六種兩兩混合的十法重播 | [alpha](alpha/README.md) |
| `alpha-summary` | 彙整十法既有結果與配對比較 | [alpha/analysis](alpha/analysis/README.md) |
| `coalition` | 枚舉一至三人聯盟及預算比較 | [coalition](coalition/README.md) |
| `roles` | 固定端點下的錨點／模型／聯合效果與分潤 | [contribution](contribution/README.md) |
| `sampling` | 既有搜尋路徑的抽樣分配差異 | [contribution](contribution/README.md) |
| `contribution` | CA+V、CA+CM、J、完整 LOO 冷快取成本重播 | [contribution](contribution/README.md) |
| `calibration` | 獨立的 alpha 校正迴圈研究 | [alpha_calibration](alpha_calibration/README.md) |

初始化與角色實驗固定案例，和包含招募、剪枝及付款的完整流程是不同設計。alpha 校正研究獨立保留於 `alpha_calibration/`；比例搜尋與其他探索位於本機封存區。
