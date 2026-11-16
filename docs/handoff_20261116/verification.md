> 本文件記錄第一次搬移；目前位置與最新驗證請見 [第二次整理](../reorganization_20261116/README.md)。

# 整理驗證紀錄

2026-11-16。這是檔案整理、相容性與小型流程驗證，不是重跑所有研究或重新確認論文結論。

| 檢查 | 結果 |
|---|---|
| 搬移前通用測試 | 252 passed，22 warnings |
| 搬移後最終通用測試 | 252 passed，22 warnings；警告為既有 SciPy SLSQP 邊界裁切警告 |
| 貢獻套件抽查 | local_anchor_step 11、combined layer2 6、model_step_decision 12、j_peer 9，共38 passed |
| 小型主流程 | 3個合成Agent、2維、seed42、1訓練epoch；資格→提案→best-of完成，輸出有限且符合box |
| 213份原程式 | 204份搬移前後SHA-256相同；9份只調整來源根目錄定位 |
| 科學計算內容 | 對9份調整檔，移除定位helper、將舊新路徑表達式正規化後，AST與搬移前完全一致 |
| 論文原稿 | 36檔逐一SHA-256相同；未改寫或編譯 |
| 相容路徑 | 所有移轉清單入口與59個貢獻套件連結解析至指定目標 |
| 命令列入口 | 分類啟動器執行 α runner 的 `--help` 成功；主流程smoke成功 |
| 平面來源快照 | P3的8檔snapshot在專案外以isolated Python匯入成功，定位到snapshot自身，無新增helper依賴 |

最初回歸發現分析器區域變數與新增定位函式同名，已修正為 `_source_workspace`，最後252項全部通過。貢獻套件有同名的 `kernel`／`study` 等平面匯入，不能混在同一個importlib收集程序；本次依套件分開啟動測試，38項全部通過。這是原套件的執行限制，沒有修改其程式。

第一次快照檢查在已載入數值庫的父程序下啟動子程序，遇OpenMP共享記憶體錯誤；改為獨立程序、單執行緒且 `KMP_USE_SHM=0` 後成功。此項只驗匯入與路徑，不宣稱已完成完整歷史重播。

主流程smoke停用inverse gate且放寬MSE門檻；不驗證資格、品質或經濟機制。新主流程封裝沒有啟用剪枝、動態α或付款；舊研究程式的設定仍原樣保留。

## 可回查檔案

- [移轉與雜湊清單](migration_manifest.json)
- [結構檢查](structure_verification.json)
- [貢獻套件測試](contribution_test_results.json)
- [環境](environment.json)
- [測試前紀錄](tests_before.log)、[測試後紀錄](tests_after.log)
- [小型流程紀錄](smoke.log)、[小型流程結果](smoke_report.json)
- [快照檢查](snapshot_verification.json)

可再次執行：`python docs/handoff_20261116/verify_layout.py`。此檢查不啟動訓練／正式實驗；若日後正常修改程式或論文，雜湊不一致表示與這次交接版本不同，不自動代表錯誤。

## 保留的範圍與限制

- 歷史結果、模型與source snapshots未搬動或寫入；本次沒有逐檔重算整個龐大結果庫的hash，不能把此紀錄當作所有歷史證據的全面稽核。
- 新舊位置仍透過根目錄連結相容，尚非可只複製一個資料夾的獨立安裝套件；跨平台分享須保留連結。詳見[重現指南](reproduction.md)。
- 論文落差以專案LaTeX為基準，不代表所有外部PDF版本皆已比對；後續初稿正文尚未補寫。
- 未重新安裝乾淨Python環境；未跑整批實驗、未修改Notion、未提交Git、未傳送資料給學長。
