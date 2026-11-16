# 過程統計校準：實作交付與停止執行記錄

日期：2026-10-16。最新使用者指示：「先完成實驗實作就好，先不要執行實驗」。

## 目前交付

工作程式位於`contribution_analysis/trajectory_calibration_v1/`。已完成從資料準備、零新增查詢記錄器、規則樹、seed分組開發，到獨立確認與驗收的程式入口。修正版屬`v2_saved_start`，預計未來使用全新輸出目錄，尚未建立或執行新一輪。

沒有修改歷史求解演算法、舊P3結果、full-removal預算研究結果或其凍結來源，也沒有修改Notion。方法仍是未驗證的研究假說。

## 在停止指示前已發生什麼

原實驗目錄：`contribution_analysis/trajectory_calibration_experiment_v1_20261016/`。

- 已凍結原v1來源、案例及標籤清單，尚未擬合任何規則。
- 曾啟動第一個技術pilot案例，做了2次warmup及1次control baseline求解；共837 model calls／samples，保存的求解時間約0.0354秒。
- 第一次control與舊保存X最大差9.4140619e-7，超過1e-7，程式依規則停止。其狀態為`stopped_technical_failure`。
- 記錄版配對、完整pilot成本評估、80個開發baseline重播、校準樹訓練、全新LOO確認均未開始。
- 使用者要求停止後，沒有再次執行上述工作，也沒有進行修正版的數值或單元測試。

原始`extraction_execution.json`與凍結v1快照保留，不回寫成通過或覆蓋失敗。

## 起點修正與驗證狀態

舊P3與後續預算資料的起點生成曾有float64／float32差異。原v1載入序列化候選後，以float64重算平均X0，可能改變求解路徑。因此工作版改為：

1. 資料準備直接保存來源原始`fixed_initial_solution`或`fixed_start`。
2. 記錄器接受明確`initial_solution`，原樣傳入solver，不重新求平均。
3. 新確認baseline與完整移除共用同一保存X0。
4. 缺少或不一致起點標記失敗，不用放寬容差補救。

這是針對可疑來源的修正，尚未重播確認它是否完全解決差異。不能稱數值問題已驗收通過。

## 實作涵蓋的階段

| 階段 | 程式與輸出 | 執行條件 |
|---|---|---|
| 準備 | `prepare.py`：查種子、凍結來源／文件、案例與標籤分檔、SHA256 | 未來明確啟動；本身不求解 |
| 技術／成本pilot | `experiment.py extract`：4舊案例、零額外查詢／不改X、15組交替成本配對 | 新輸出目錄；通過成本才進開發提取 |
| 特徵提取 | 同上：80舊baseline、320 effects、8固定特徵 | 技術與成本閘門通過 |
| 獨立重載核對 | `experiment.py audit`：來源／模型hash、baseline品質與X、calls、特徵列數 | 全部特徵先保存，不將pooled送入記錄器 |
| 開發 | `experiment.py evaluate`：5折seed分組、0／常數／兩消融／完整規則樹、模型凍結 | 重載核對通過；兩情境MAE均優於常數且完整線上成本達標才進階 |
| 新案例預測 | `predict_new.py`：一次正常求解、鎖定所有方法預測與模型hash | 不載入pooled／LOO資料 |
| 離線確認 | `confirm.py`：20新seed、兩批、40情境／160 effects、200原定求解 | 明確呼叫，且開發與成本閘門通過；先預測、再LOO、後評分 |
| 確認驗收 | `analyze_confirmation.py`：重算預測與特徵、模型重評、seed CI、方向／MAE／負值／成本 | 明確呼叫，不由前一步自動觸發 |

所有方法限制深度2、至多4葉、每個子節點至少5個不同seed、seed與情境平衡權重。ID、污染標記、pooled、truth與LOO結果不得進入8項特徵。新確認預測先寫入不可覆蓋檔案，才允許產生移除解；驗收器再獨立核對預測與凍結模型一致。

API calls、input samples、solver時間與準備時間設上限；缺值不填0、失敗不換seed、不在原失敗目錄原地恢復。沒有自動啟動新資料或新LOO的背景程序。

## 檢查狀態

停止指示前的原v1有5項單元測試通過。工作版新增保存起點、異常還原與確認統計的測試，目前共10項測試定義，**修正版尚未執行**。

本次只以AST解析檢查語法、入口保護、測試定義與歷史來源hash。這些靜態檢查不代表執行成功，也不代表零新增查詢、10%耗時目標、規則預測能力或整合流程已驗證。

操作順序與各CLI用途見`contribution_analysis/trajectory_calibration_v1/README.md`。下一次取得明確執行指示後，應先跑單元測試，再在全新v2輸出目錄重做技術pilot；目前不執行。
