# 整理後的執行方式

完成核對：2026-11-17。從專案根目錄執行；新入口以repo為工作目錄。舊入口使用自動產生的 `.local_scripts/` 平面視圖，常用輸出／來源路徑會轉成repo絕對路徑。環境依賴見根目錄requirements；本次使用既有環境，未驗證乾淨安裝。以下輸出位置請選不存在的新資料夾。

## 五個入口

### 法五完整Stage

```sh
python run.py stages --output-dir /tmp/method5-stages-new --seeds 42 --dimensions 2 --rhos 0 --num-agents 3 --samples-per-agent 32 --epochs 1 --mse-threshold 1000000 --disable-inverse-check --custom-iterations 3
```

此為已實跑的小型接線設定，會執行Stage0/1→2→3→3.5剪枝→4結算，輸出summary、selection、pruning、payment等CSV。刻意放寬資格並停用反推能力gate；不能作正式品質結果。正式設定請由 `stages --help` 指定。此歷史入口仍使用其原有約束設定；不能當作受限α研究的相同設定。

### 傳統FedAvg與法五數值比較

```sh
python run.py baseline --output-dir /tmp/method5-fedavg-new --dimensions 5 --agents 3 --samples 32 --rounds 1 --mse-threshold 1000000 --disable-inverse-check
```

輸出參數、執行紀錄與 `report.json`，含FedAvg每輪反推誤差及法五最終解。保留相同生成資料，法五本地訓練epoch數與FedAvg各節點總epoch數一致；模型起始狀態、選人和反推預算仍不同。這是可執行比較入口，尚不是訓練／通訊／查詢全部公平配對的正式benchmark。預設資格門檻為0.1且開啟inverse check，正式研究應明確固定協定；範例只供接線檢查。

舊數值版本 `baselines/fl_comparison_numerical.py`保留FedAvg／Krum；Flower／MedMNIST版本另需 `flwr`、`medmnist`、`torchvision`及資料下載，未列入本次通過的執行保證。

### 1112八種α初始化

```sh
python run.py initialization --source alpha_weight_outputs/four_alpha_v1_20261014/formal_batch_1/source --seeds 20000 --output-dir /tmp/method5-initialization-new
```

預設三情境，可用 `--scenarios reference_clean` 選一個。用 `--seeds 20000,20001`擴大重播，seed必須存在於來源資料。每方法獨立求解；所有決策封存至 `decisions.jsonl` 後才載入pooled evaluator評分。`results.jsonl`含方法、α、X、pooled與solver成本。共享盲測／支持訊號的準備未併入每方法solver成本，不能直接當完整部署總成本。來源是已見過的模型，不是新獨立確認。

此入口呼叫既有方法定義，不改歷史fallback或求解規則。舊正式批次仍有source hash驗收：整理改動過的檔案可能不符合舊規約，應使用封存快照，不可移除檢查。

### 1112分潤角色方法成本重播

```sh
python run.py contribution --output-dir /tmp/method5-contribution-new
```

預設重播全部24個既有解析模型團隊×四方案；可加 `--limit 1`做快速檢查。輸出每隊每臂的端點、分數、cache事件及呼叫成本，最後manifest標示complete。沿用歷史runner的逐端點／baseline檢查，輸出到新位置，保護原結果。比較臂AV／ACM／J／LOO分別對應CA+V／CA+CM／J／完整LOO。

此為解析模型成本重播，不會執行真實付款。完整角色分池／結算與方向投影的其他研究保留在 `experiments/contribution/packages/`；固定路徑runner應先閱讀規約。抽樣CA分析與1112分配誤差資料保留在 `artifacts/anchor_sampling_peer_20261111/`。

### 0917 α校正（保留）

```sh
python run.py calibration --output-dir /tmp/method5-calibration-new --seeds 42 --dimensions 2 --poison-ratios 0 --initializations uniform --evaluators uniform --update-rates 0.5 --update-policies fixed,guarded --max-calibration-rounds 2 --num-agents 3 --samples-per-agent 32 --epochs 1 --mse-threshold 1000000 --disable-inverse-check --custom-iterations 3 --no-settle-payment --constraint-mode box_constrained
```

實跑兩種策略、各至多兩輪；輸出summary、rounds、checkpoint、pooled評分及來源快照。範例不結算，完整參數見 `calibration --help`。本研究程式保留，論文主線仍以0604／1112為準。

## 其他入口與測試

`python run.py list`列主要與依賴程式。新的完整路徑或舊分類路徑仍可作參數，透過 `.local_scripts/` 內自動產生的相容連結執行；不在主要清單的112支程式已實體移入archive。

```sh
python -m pytest -q
python -m pytest -q contribution_analysis/model_step_decision_v1/test_engine.py
python docs/reorganization_20261116/verify_layout.py
```

各歷史套件分開啟動pytest，避免同名kernel／engine／study互相覆蓋。通用測試只收集tests；其餘結果快照不遞迴測試。

## 交接

正式程式保持追蹤，本機相容連結會依layout.json自動建立。分享時保留實體程式、project_support、對應歷史模型／資料與快照；僅傳新的主要程式子目錄不足以重現。Git未追蹤的proposal、checkpoint及結果需另行包含。這次沒有Git提交或對外傳送，原論文與歷史研究資料保留原內容。

## 根目錄清理（2026-11-17）

根目錄214個相容腳本已集中至 `.local_scripts/`，整個目錄忽略。它包含可重建的連結，正式程式仍在分類目錄追蹤。`run.py`、pytest的conftest及套件匯入會自動準備此視圖；不要直接使用已移除的 `python alpha_weight_experiments.py` 等舊命令，改用 `python run.py calibration ...`。

需要直接操作平面歷史工具時，可先執行 `python -c 'from project_support.legacy import prepare; prepare()'`，再將 `.local_scripts` 加入Python搜尋路徑。舊runner仍須遵守原規約與來源快照要求。
