# Full-removal 效果對求解預算穩定性：事前登錄 v1

登錄日期：2026-10-16。使用者已授權依固定研究文件進行本輪實驗。此文件與 `registration.json` 在產生 pilot 或確認資料前凍結；不得看結果後改成功門檻或替換 seed。

研究根目錄：`contribution_analysis/full_removal_budget_experiment_v1_20261016/`。
本輪只處理固定文件的 RQ1。P3 正式結果仍獨立保存。

## 固定反事實與方法

`C_i^k=|f_pooled(X_without_i^k)−0|−|f_pooled(X_full^k)−0|`，k 為 A 或 B。參與者完整移除包括候選、local model、α，剩餘 α 正規化。不重訓被移除後模型、不補人、不重新選 coalition。

Clean：沿用 Stage 1 的三人 coalition。Controlled poison：沿用歷史固定五人、一名污染者程序。兩者均 2D、target 0、box [−1,1]、static uniform α。20 個 local models、每個 500 筆、30 epochs；盲測 20 點、原資格設定與候選生成不變。凍結 pooled evaluator 使用原設定 100 epochs。原來源八檔 SHA256 與 P3 快照核對一致，複製到本輪 source_snapshot 後才執行。

A=best_of(custom 30, SLSQP ftol=1e-9/maxiter=200)；B 僅把 custom 改 120。baseline 與每位移除解在兩設定各自獨立重求。引擎按內部 L_opt 選，不要求原 solver success=True 才交付，另完整記錄 success、status、停止原因，不把交付可行稱收斂。

每次呼叫的指定 X0 都是完整候選平均。SLSQP 記 beta0 與實際映射 X0；custom 記原 X0。映射殘差>1e-8、不可行或兩引擎搜尋域不一致為語義資格失敗，停止後續，不換 seed、不修歷史演算法補救。

部署與評估分離：準備、求解、評分各用獨立程序；求解程序只載入 local_models 與 coalition，不載入 pooled evaluator。全部解先保存，另一程序才載入 evaluator 評分，truth 僅診斷。這不是部署刪人、α 更新、pruning、付款或 Shapley 研究。

## 種子與階段

- pilot：2026101600–2026101601，兩情境，共 4 scenarios／16 effects／40 best_of 呼叫。
- pilot 第一個 seed 的兩情境另從 artifacts 在獨立程序重播 A，共 10 次呼叫。不重訓／不新增資料，驗證相同模型、coalition、候選、α、X0 可重播。
- 確認：2026101610–2026101619，兩情境，共 20 scenarios／80 effects／200 呼叫。
- 全輪最多 250 best_of 呼叫，順序執行。各 seed 的 A/B 共用 local models、資料與 evaluator；不同 seed 新生成資料與模型。Clean/poison 同 seed 有共同生成基礎，是配對情境，不視為獨立重複。
- 登錄前已搜尋全專案既有 manifest／registration，無上述種子實驗紀錄；固定文件中的文字預留不是既有實驗。準備後另記 semantic data hash，跨 seed 不得重複。
- pilot 閘門只檢查技術完整性、重播、重載及成本投影，不以觀察到的穩定性決定是否跑確認組。不改方法／門檻即繼續確認。失敗停止、缺失保留、不替換樣本。

## 記錄與獨立驗收

獨立 wrapper 觀測原引擎輸出、scipy minimize 回傳與 custom frame 的停止資訊，不更改原 objective 或搜尋決策，也不額外呼叫 objective。以合成 fixture 測 instrumentation 前後 X／loss 完全相等、calls 對帳；另測 API 上限中止及 rank 映射失敗保存。來源快照、測試結果在 pilot 前保存。

保存 baseline／移除的 IDs、α、候選、X0、投影、rank／奇異值、X、內部 loss、逐引擎 success/message/iterations、calls／input_samples／seconds、來源與 artifact hash。custom `success=True` 只表示歷史程式回傳搜尋完成，另按觀察到的 break／iteration limit 記停止理由。

所有被評解有限且 box violation≤1e-8。獨立分析器重新載入凍結 local models 與 evaluator，核對相同剩餘 α／模型的 loss 和 pooled error 差≤1e-7；公式、移除 IDs、候選／起點、hash、calls 對帳。pilot 重播 X／loss／pooled 差≤1e-7。完整資料未通過不作穩定性主張；缺值不能填零。

## 指標與固定門檻

主 neutral δ=1e-4，|C|≤δ 為 neutral；1e-7、1e-3 只敏感度。幅度容差 τ=1e-3。這是固定文件已提出且本次採用的研究尺度，不是精確數值真值或應用普遍門檻。

每 seed：a_s 為三分類方向一致比例；b_s 為 |C_A−C_B|≤τ 比例。兩情境各自平均 a、b 都≥90%。10,000 seed bootstrap、RNG20261016、98.75% Bonferroni CI（四個主要情境×指標），所有下界≥80%。bootstrap 對每項以同一固定 RNG 初始化，沒有跨參與者獨立抽樣。另報 max／median |ΔC|、baseline／移除的 ΔE／ΔX、嚴格正負反轉、neutral、solver 與成本。

每情境至少 5/10 seeds 有 A 或 B 非 neutral 的效果；所有此類配對的方向一致比例≥90%。不足時不能用大量 neutral 宣稱可靠。全條件通過才標記 `limited_budget_stability_established`；完整驗收通過但統計／方向條件未過為 `budget_stability_not_established`；缺失或技術失敗為 `incomplete_or_failed_acceptance`。不因 CI 太寬追加資料。

## 成本與停止

全輪求解 calls≤1,000,000、input_samples≤1,000,000、solver wall time≤1800 秒，準備（local 訓練／資格／候選／evaluator）合計≤3600 秒；單次 best_of≤20,000 calls 或60秒。每次 API 前查 calls／samples 上限、signal wall timer 中止當次。外部評分、程序啟動與紀錄 overhead 另計，不混入搜尋 calls。這裡 solver seconds 是含觀測紀錄器的 wall time，不聲稱純演算法速度。

pilot 成本乘6作全輪保守投影，超上限即停。任一語義、來源、重播或資源條件失敗立即保存狀態並停止後續排程，不補 seed、不臨時降低預算。確認完成後即使穩定性未建立也停止本輪，另作下一步提案。結果不回填 P3，不改 Notion 或其他研究線。
