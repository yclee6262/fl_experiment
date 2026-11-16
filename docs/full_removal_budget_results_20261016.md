# 直接 full-removal 效果：求解預算穩定性結果

日期：2026-10-16。研究狀態：原執行曾因環境中斷；經明列技術修訂恢復後，全部資料與驗收完成。新研究，不回填 P3 正式結果。

## 結論

**在這批獨立確認資料中，直接完整移除效果對 custom 30／120 次預算改動建立了有限穩定性證據。** Clean 與 Controlled poison 的參與者效果方向全部一致，幅度改變也全部小於事前容差。這次未觀察到增加 custom 預算會明顯改變參與者被判正、負或 neutral 的結果。

這只驗證固定模型、coalition、完整候選平均起點、2D target=0 與目前 best_of 下的單一預算擾動。**尚未建立起點穩定性、全域最優性或低成本估計器有效性。** 既有候選排除對完整移除的 P3 負面結果維持不變。

## 測了什麼

同時刪除參與者的候選點、local model 與 α，剩餘 α 正規化，以相同完整候選平均 X0 重求。local models 不重訓、coalition 不重選、不補人。每一預算都獨立重求完整 baseline 和所有移除解：

`C_i^k = E_pooled(X_without_i^k) − E_pooled(X_full^k)`。

A 為 best_of(custom 30, SLSQP)；B 僅將 custom 上限改成120。SLSQP 仍 ftol=1e-9、maxiter=200。求解程序不載入 pooled evaluator，解保存後才由另一程序評分；truth 只診斷。

- Pilot：2 個新 seed×2情境，16 effects；另重播第一個 seed 的 A，共50次求解。只驗證技術流程與成本，未混入主要分析。
- 確認：10 個另行保留 seed×2情境，共20 scenarios、80 effects。Clean 每 seed 3人，污染每 seed 5人。抽樣單位是 seed，不是80個獨立人次。
- 全輪250個原定求解工作完成。另有一次失敗程序，詳見技術修訂；沒有替換 seed、補樣本或調門檻。

## 確認結果

主 neutral δ=1e-4，幅度容差 τ=1e-3。方向一致率含正／neutral／負三分類。依登錄對兩情境×兩主要指標使用10,000次 seed bootstrap、RNG20261016、98.75% Bonferroni CI。

| 指標 | Clean，10 seeds／30 effects | Controlled poison，10 seeds／50 effects |
|---|---:|---:|
| 方向一致率 | 100%，CI [100%,100%] | 100%，CI [100%,100%] |
| 幅度差≤1e-3比例 | 100%，CI [100%,100%] | 100%，CI [100%,100%] |
| 非 neutral effects／方向一致率 | 28／100% | 41／100% |
| 有非 neutral effects 的 seeds | 10/10 | 10/10 |
| 嚴格正負反轉 | 0 | 0 |
| median abs(C_A−C_B) | 0 | 0 |
| max abs(C_A−C_B) | 0 | 0.000006199 |
| C 正／neutral／負，A與B相同 | 19／2／9 | 27／9／14 |

兩情境皆通過事前 mean≥90%、CI下界≥80%，以及非 neutral 識別充分度門檻。登錄狀態為 `limited_budget_stability_established`。

敏感度 δ=1e-7 或1e-3時，兩情境方向一致率仍全部100%。非 neutral 數分別為 Clean 30／26、污染50／36。這些敏感度未取代主δ。

CI端點全100%是因10個seed的觀察比例全相同，經驗bootstrap沒有抽出變異；**不表示未來案例有100%穩定機率**，也不排除未測情境或起點的失敗。

## 為何預算增加幾乎沒改結果？

本輪確認共有100組A/B子問題：20個baseline加80個移除問題。所有SLSQP配對輸出完全相同，custom共同前綴歷程也相同，支持這輪確實只改custom上限。

依保存的原始迭代／break行號核對後：A的100次custom中，97次因找不到改善步驟停止，3次用完30次上限；B的這3次分別繼續到33、39、42次，之後也因找不到改善步驟停止，沒有跑滿120。

只有一個被best_of選出的最終解改變：污染seed2026101611的完整baseline。80個移除解全部不變。因此該seed五位參與者的C一起位移6.1989e-6，不能當成五個獨立的不穩定事件。

| 污染seed2026101611完整baseline | A：30次 | B：120次上限 |
|---|---:|---:|
| 實際custom迭代 | 30 | 39 |
| 內部L_opt | 0.01621773 | 0.01620304 |
| pooled誤差 | 0.00491849 | 0.00492469 |
| X | (−0.04955297,0.05868563) | (−0.04949344,0.05863346) |

內部目標稍微改善，pooled誤差卻稍微變大；不能把增加迭代視為外部品質必然更好。其他99個被選解完全相同。找不到改善步驟是這個搜尋程序的停止條件，不是全域最優或收斂證書。

## 代表案例

下列正／負／neutral案例各情境按seed與agent ID取第一筆，另加最大預算差案例。是事後說明，不是新增假說檢定；不按truth是否支持篩選。

| 情境／seed／agent | 完整誤差A | 完整移除誤差A | C_A | C_B | 解讀 |
|---|---:|---:|---:|---:|---|
| Clean／2026101610／8 | 0.01318374 | 0.02047910 | +0.00729536 | +0.00729536 | 移除變差，原參與有幫助 |
| Clean／2026101611／2 | 0.01240929 | 0.00015589 | −0.01225340 | −0.01225340 | 移除後pooled誤差改善 |
| Clean／2026101610／6 | 0.01318374 | 0.01313506 | −0.00004868 | −0.00004868 | 在主δ內，判neutral |
| 污染／2026101610／15 | 0.00447378 | 0.00981106 | +0.00533728 | +0.00533728 | 此污染標記者移除反而變差 |
| 污染／2026101610／4 | 0.00447378 | 0.00026375 | −0.00421003 | −0.00421003 | 移除後pooled誤差改善 |
| 污染／2026101610／2 | 0.00447378 | 0.00443460 | −0.00003918 | −0.00003918 | neutral |
| 污染／2026101611／15 | 0.00491849 | 0.00308125 | −0.00183724 | −0.00184344 | 最大預算差之一，方向不變 |

最後一例的truth差為A +0.00057787、B +0.00056443，與pooled方向相反。本輪pooled/truth三分類衝突為Clean 8/30、污染16/50，A/B相同。這是評估尺限制的診斷，不用truth重選引擎、篩除案例或決定部署行為。

負C代表在本次固定程序及pooled評分定義下，移除後誤差較小；這與「移除改善」是同一件事，不是獨立預測命中，也不代表該參與者惡意、應被刪除或應如何付款。

## 成本

以下只含確認組搜尋，每情境10 seeds；baseline與所有移除都包含，pooled評分／獨立驗收／程序啟動另計。每次API輸入都是單一X，因此calls與輸入樣本數相同。秒數含觀測記錄器，不代表未加觀測的純演算法效能。

| 情境／預算 | 求解次數 | model calls／input samples | 求解秒數 |
|---|---:|---:|---:|
| Clean A | 40 | 9,504 | 0.886 |
| Clean B | 40 | 9,504 | 0.958 |
| 污染 A | 60 | 46,973 | 3.657 |
| 污染 B | 60 | 48,032 | 3.889 |

Clean calls不變，污染B比A多1,059 calls（2.25%），遠低於上限倍數4，因多數custom提早停止。短時間耗時差不作普遍部署速度主張。

含pilot與重播，250次成功工作實測141,884 calls／samples、11.669秒求解，local訓練／資格／候選／evaluator準備269.947秒。失敗程序成本unknown，額外保守預留20,000 calls／samples及60 solver秒；保守資源帳為161,884 calls／samples、71.669 solver秒，仍在原全輪上限內。程序載入、排程與額外驗收未混成搜尋耗時。

## 驗收、技術修訂與記錄錯誤

**數值與來源驗收完成：**20/20確認scenarios、80/80確認effects，無替換seed、缺失或不可行交付。250筆成功解全部重新載入local models與evaluator核對；pooled最大差0，內部loss最大差6.94e-18，皆小於1e-7。10次pilot重播X／loss／pooled差0。舊輸入、凍結來源及artifacts雜湊核對通過；12個新seed資料與P3的30個舊seed無重複，Clean/poison同seed共用生成基礎屬配對設計。

須保留兩項限制：

1. **OpenMP環境中斷與恢復。** 原確認第一個seed的一次程序未產生JSON，原execution及原analysis保留stopped／incomplete。修訂在任何確認pooled評分前登錄，只允許共享記憶體存取並恢復缺失工作，沿用已保存模型／資料及原solver／門檻。總共251次程序嘗試對應250個成功原定工作，沒有補樣本。完整結果應稱「經明列技術修訂恢復的本輪確認」，不稱原流程無中斷。
2. **停止標籤記錄器錯誤。** 初版把線搜尋內層break誤當成外層停止，250筆中7筆標籤需修正，其中確認組3筆。原始frame行號、iteration與limit已保存，可確定還原。修正診斷另存`stop_label_audit/`，未修改原JSON、X、loss、pooled、C或主要統計；工作版記錄器已修正並新增測試，6項測試通過。原`recovery_analysis/results.json`中的custom_stops屬初版標籤，停止原因請以stop_label_audit為準。

## 已完成、未建立有效性、下一步提案

- **已完成：**直接完整移除的A/B配對求解、獨立資料、重播／重載／來源驗收及登錄分析。這兩個合成情境、固定起點下，達到有限預算穩定性門檻。
- **未建立有效性：**跨起點穩定、全域最優、完整歷史參與者因果貢獻、低成本近似器，或可部署的刪人／付款規則。沒有改變P3對候選排除的負面結論。
- **下一步提案：**另登錄RQ2的起點敏感性。先比較完整候選平均與剩餘候選平均，明確保存兩引擎實際可表示X0與搜尋域，再確認直接基準對起點是否可靠。本輪不自動開始RQ2，也不因這輪通過就採用影響函數、線性近似或支持度估計。

## 檔案入口

- 固定設計／進度：`docs/contribution_analysis_research.md`。
- 原事前登錄：`docs/full_removal_budget_preregistration_20261016.md`。
- 技術修訂：`docs/full_removal_budget_recovery_amendment_20261016.md`。
- 本輪資料：`contribution_analysis/full_removal_budget_experiment_v1_20261016/README.md`。
- 完整主要結果：同目錄`recovery_analysis/results.json`、`participant_effects.csv`、`seed_metrics.csv`。
- 完整驗收：`recovery_analysis/acceptance.json`、`reload_verification.csv`、`replay_verification.csv`。
- 獨立來源／介入核對：`independence_and_intervention_audit.json`。
- 更正後停止診斷：`stop_label_audit/summary.json`與`records.csv`。

本輪沒有修改歷史演算法、舊結果或Notion；沒有推進α初始化／選擇、α更新、pruning、付款、Shapley或商模。
