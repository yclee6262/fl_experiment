# 自適應誤差相關初始化：執行與驗收紀錄

本輪沿用唯一主文件「α 初始化研究總紀錄｜方法、各輪結果與下一步」，不新增單輪 Notion 頁。執行前設計已固定於 adaptive_correlation_pipeline.py，正式啟動前登錄於 adaptive_corr_preregistration_20261010.json。

## 研究範圍

- A、B、原保守品質、固定 Corr precision 0.5、新自適應相關權重。
- 乾淨、低品質、樣本量不平衡；二維、target 0、固定 IDs 1／2／20、三人 coalition。
- 單次倒推；無 α 更新、pruning、結算或額外可信標記；box 容差 1e-8，重載 1e-7。
- 原 20 筆資格資料五折16/4切分，λ固定0／0.25／0.5／0.75／1，平手容差1e-12偏好較大λ。
- 以未中心化殘差二階矩保留偏差；僅留出混合MSE調 λ；最後全20筆重估一次α。
- pooled／truth 只在交付後評分，不回饋權重或 solver。

## 依序執行

1. 實作與25項相關回歸測試通過；新固定0.5與舊法完全一致，λ1與B一致。
2. 歷史模型 pilot：4800–4804，75條件；保存、可行性、重載驗收通過。
3. 獨立於 pilot 的歷史模型開發：4810–4819，150條件；六組配對SD只用於事前N規劃，不調方法、不加入正式。
4. 研究解析度0.005、名義80%常態近似、六比較Bonferroni；最大開發SD0.00834808743，N=34。這不是保證power或業務門檻。
5. 查核337份manifests，正式5000–5033未見重疊；五方法×三情境×34共510條件。
6. 凍結來源、重新訓練資料／local／pooled模型；正式比較與重載。
7. 每個正式新法條件的25折內矩陣／留出損失直接核對，固定舊法與B端點驗證；六組新方法／固定相關fresh與cache實際倒推一致性。
8. 完整重播seed5000的三情境十五條件，重產資料、模型、資格、proposal、λ、solver、α與X，差異須不超過1e-7。
9. 失敗分析測試、報告、同一Notion主文件整合與回讀；完成逐項審核後才標示研究完成。

## 可重現入口

下列命令須用獨立且未存在的輸出目錄；既有完成目錄不允許覆蓋。模型訓練使用本地 Python，限制每種運算庫一個執行緒。

```text
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /Users/yclee/anaconda3/bin/python adaptive_correlation_pipeline.py
/Users/yclee/anaconda3/bin/python analyze_adaptive_correlation.py alpha_weight_outputs/adaptive_corr_formal_v1_20261010 --registration docs/adaptive_corr_preregistration_20261010.json
/Users/yclee/anaconda3/bin/python alpha_weight_outputs/adaptive_corr_extra_acceptance_20261010.py
/Users/yclee/anaconda3/bin/python alpha_weight_outputs/adaptive_corr_full_replay_20261010.py
/Users/yclee/anaconda3/bin/python alpha_weight_outputs/adaptive_corr_report_20261010.py
/Users/yclee/anaconda3/bin/python alpha_weight_outputs/adaptive_corr_completion_audit_20261010.py
```

pipeline首次執行會建新目錄；已有完整實驗時只重跑分析、驗收與report。完整重播已有目錄時使用 `--verify-existing`；更換seed或輸出建立新版本，不刪既有結果。

## 保存與成本界線

主結果包含每seed／方法實際α、交付X、solver、可行性、pooled/truth、初始與Stage3呼叫／输入點數／秒數、外部評分秒數。完整fold audit在results.jsonl，CSV僅精簡欄位。各scenario／seed有artifact_reference與模型、資料、split、evaluator雜湊。

模型訓練／資格／proposal共同準備成本在來源shared_costs.jsonl每seed一次。重新載入後的資格預測快取建立在calibration.jsonl，各方法重用同一批。額外舊OOBfresh/reuse、來源Uniform求解、完整重播及驗收是研究開銷，不加到某方法部署成本，也不冒稱不存在。

初始化＋Stage3倍數不能稱完整流程倍數；本機calls不等於已支付遠端API費。沒有品質貨幣價值設定，不宣稱經濟淨收益。

## 完成證據

最终狀態以 adaptive_corr_requirement_audit_20261010.json 為準，需與實際完整結果、測試、重播及回讀Notion證據一起檢查。負面品質結果不妨礙完成研究交付。

## 最終交付狀態：完成

pipeline與postprocess所有階段returncode0。正式510／510條件有效可行；重載與完整重播差0；25項回歸與4項分析失敗測試通過；102新法條件2550折內驗收通過。來源模型／資料／快照雜湊、旧固定法、B端點及實際fresh/cache核對均通過。λ1與B的最大浮點差2.220446049250313e-16，在既定1e-7容差內；固定0.5與舊法差0。

六項校正確認只有乾淨對B通過，新法未穩定勝固定0.5對照。保留A/B，不升格新法；後續限定三法乾淨複驗，尚未啟動。

Notion已更新同一主頁並回讀，沒有建立新頁；含方法、正式結果、CI、成本、coverage、欄位及兩份CSV下載。主文件本地鏡像同步；回讀證據adaptive_corr_notion_publication_20261010.json，完成13項requirement audit。
