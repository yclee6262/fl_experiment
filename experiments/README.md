# 論文實驗入口

從專案根目錄執行 `python run.py <入口> [參數]`，使用新輸出資料夾。完整命令見 [操作指南](../docs/reorganization_20261116/README.md)。

| 入口 | 用途 | 實作 |
|---|---|---|
| `stages` | 原法五各Stage，含剪枝及原分潤 | `shared/run_experiments.py` |
| `baseline` | 可設定維度的FedAvg與法五數值執行比較 | `baselines/compare_fedavg.py` |
| `initialization` | 1112的U/Q/E/S/EU/EQ/SU/SQ；重用合格模型 | `alpha/compare_initializations.py` |
| `contribution` | CA+V、CA+CM、J、完整LOO獨立冷快取成本重播 | `contribution/replay_cost_comparison.py` |
| `calibration` | 0917的α校正與固定／guarded等策略 | `alpha/calibration/alpha_weight_experiments.py` |

`core/pipeline.py`只封裝資格、提案與求解；完整Stage3.5／4仍由stages入口執行。校正另選入口，沒有默默納入1112固定初始化研究。

`contribution/packages` 保留20個報告相關套件的相對索引，其中包括方向投影／改善分帳、抽樣研究支援、錨點與模型分支、分池與結算。其實體仍在 `contribution_analysis/`，以維持動態載入、模型及基準端點路徑。1112抽樣CA最終分析在 `artifacts/anchor_sampling_peer_20261111/`；本次保留其結果與程式，沒有重寫固定路徑分析腳本。

早期探索與主文外擴充（SE/QU、C/P、比例搜尋等）在 `archive/legacy/experiments/alpha/`；必要的共用模組仍保留於主要區域。可用原根目錄名稱或 `run.py` 的舊分類路徑啟動歷史程式，來源hash檢查仍有效；部分舊runner固定輸出位置，不應直接對完成結果再執行。

FedAvg數值比較保留並修正多維誤差公式與缺參數呼叫。MedMNIST／Flower版本仍在 `baselines/`，保留作傳統比較候選；本次未安裝它們的額外依賴或宣稱已完成醫療資料集重跑。
