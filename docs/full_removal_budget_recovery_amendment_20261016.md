# Full-removal v1：共享記憶體中斷的技術修訂

日期：2026-10-16。此修訂在任何確認組 pooled 評分產生前保存；不改研究問題、求解演算法、種子、資料、neutral 閾值、統計門檻或比較次數。

## 原始中斷

Pilot 四情境、50 次含重播求解完成且重載驗收通過。確認組第一個 Clean seed 2026101610 的 A_2 程序因 Intel OpenMP `Error #179: Function Can't open SHM failed` 終止，只留下 log，沒有結果 JSON。排程依原規則停止，沒有填零或補 seed。

原 `execution.json` 與原 `analysis/` 永久保留 stopped／incomplete_or_failed_acceptance。這份中斷不是成功完成的原登錄流程，不能從報告刪除。

## 恢復範圍

同一已授權實驗允許本機計算程序使用沙箱外的共享記憶體，沿用相同 interpreter、套件、執行緒設定與凍結 solver。新增恢復排程器只跳過已有成功輸出、重用保存模型與資料，對尚未產生輸出的既定工作繼續排程。舊結果與失敗 log 均不覆蓋。

恢復前已保存 `recovery_registration.json`，明列原狀態、缺失工作、所有修訂來源 SHA256。`recovery_source_snapshot/run_study_recovery.py` 的變更限於恢復既有輸出、新 log／execution／pilot gate 路徑與保守成本預留。worker.py 與原 solver 檔逐位元不變。

失敗程序沒有可靠 counters，記為 unknown，不能聲稱實測為 0。資源帳保守預留單次上限 20,000 calls／input_samples 及 60 solver seconds，仍受原全輪 1,000,000 calls／samples、1,800 秒求解上限控制。原定 250 個唯一 best_of 工作不變，含一次中斷的 worker 嘗試上限改為251。這是明示的技術修訂，不冒充原登錄未中斷。

恢復完成時，`recovery_execution.json` 的 calls／samples／solver_seconds 是成功結果實測值加上上述保守預留，不是純實測總數。分析中的逐工作／A/B 成本則只加總保存的成功結果；中斷成本另列。

## 結果分析與判讀

恢復分析器與原凍結分析器只有兩處路徑差異：改讀 recovery_execution.json、改寫 recovery_analysis/。公式、CI、資格與門檻不變。原 analysis/ 不覆蓋。

最終結果須分列原中斷狀態與技術恢復後驗收。若恢復再失敗，停止並交付不足結果，不靜默追加重試。即使恢復完整且有限穩定性門檻通過，也只回答 custom 30／120 的配對預算問題，不回答最優性、起點穩定性或貢獻估計的普遍有效性。
