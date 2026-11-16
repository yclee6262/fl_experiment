# 留出驗證的重抽樣＋收縮：研究規格與驗收

狀態：本輪研究與交付驗收完成。正式126-seed、1,890條件全部有效可行；六項確認比較均未建立校正後穩定優勢。重載、378組fresh/reuse及seed4400完整重播通過；不全面取代A/B。完整結果見oob_quality_shrink_results_20261008.md。

## 問題與固定研究範圍

在不修改Stage 3目標、solver、coalition及候選點的條件下，使用原20筆Stage 0盲測，加入留出驗證決定收縮程度，是否能穩定改善交付X的pooled誤差？主要對照A Uniform及B反MSE品質權重；固定0.5簡單收縮與原bootstrap作機制對照。

論文來源：Wachslander與Setzer，2025，*Ensembling Shrunk Weight Estimations in Forecast Combination*，https://doi.org/10.1007/978-3-031-92575-7_6。出版社公開摘要的核心為子集估權重、留出資料決定收縮，再平均權重。本輪是非負反MSE品質權重改編，沒有將其稱為完整重現論文的optimal combination weight演算法；亦未將預測組合的理論直接移植到倒推。

## 詳細演算法

每seed/scenario有20筆盲測、固定IDs 1/2/20的逐筆模型預測P及標記y。

1. 用本地RNG同時為三人抽20個行號，有放回，共1,000輪；相同行號保留三模型配對。
2. 第b輪只用抽到行號上的逐模型平方誤差，計算MSE及q_b=normalize(1/(MSE+1e-5))。
3. 留出集合O_b為完全沒有抽到的行號，與in-bag disjoint；少於2行的輪次跳過並保存數量，不以訓練行替補。
4. 在留出P/y上令r=P q_b-y，d=P Uniform-P q_b，lambda_b=clip(-dᵀr/(dᵀd),0,1)。dᵀd=0時令lambda=1，明確偏好Uniform的平手規則。
5. w_b=(1-lambda_b)q_b+lambda_b Uniform。各輪留出MSE僅用於tuning，不稱為無偏品質估計。
6. 對所有合格輪次的w取平均，再正規化。若全部無合格輪次，Uniform fallback並記原因；報告fallback率。
7. 只用最終α進行一次Stage 3；best_of內含SLSQP beta與Custom30路徑，成本全部計入。不做1,000次倒推，不更新α、不pruning、不結算。
8. 交付後才載用凍結pooled evaluator評分，truth僅診斷。兩者都不參與q、lambda、solver或選解。

lambda是Uniform比例，0代表保留品質權重，1代表完全均勻。它由混合預測MSE決定；倒推目標仍是sum alpha_i abs(f_i(X)-target)，兩個目標不同，必須外部驗證。

## 實驗控制

三情境：乾淨；低品質25%（全20人最後5人label SD升至0.3，固定coalition只有ID20受影響）；樣本量不平衡4倍。二維、target0、三人固定IDs、box[-1,1]、可行容差1e-8、重評容差1e-7。local30epochs、pooled100epochs validation checkpoint、proposal500steps。外部模型沿用乾淨訓練設定。不同方法同seed共用資料、模型、原20盲測、proposal及evaluator。

5-seed保存驗收使用4100–4104，共75條件；10-seed開發使用4100–4109，共150條件。兩者重疊，不能相加當成15個獨立seeds。開發使用已封存模型；正式另重新產生資料、模型與evaluator。

正式seeds4400–4525，126個×3情境×5方法=1,890條件。328份既有manifest/preregistration核對未見所選seeds。開發六項A/B配對差最大SD0.0161191；依常態近似、六項Bonferroni、nominal80% power、delta0.005估得126，事前下限116上限300。此delta是合成研究解析度，非业务成功門檻；近似不保證power，小幅改善仍可能不夠精確。

## 分析與判讀

主要指標E_pool(X)=abs(f_pool(X)-target)。品質、失敗、成本分開。每scenario主方法−A／B，共六個confirmatory contrasts；seed為抽樣單位，10,000次配對bootstrap，analysis RNG20260924，普通95% CI與各99.1667% Bonferroni區間並列（family simultaneous95%）。僅當完整有效配對、驗收通過、校正上界小於0且觀察失敗率不高於基準，才建立該情境／基準穩定優勢。通用優於A/B需六項全部符合；部分通過只能限定情境。對固定收縮及原bootstrap的比较屬描述性／探索性，不轉成新的確認優勢。

所有條件落盤；失敗保留耗費成本；非預期程式錯誤中斷並停止。缺漏或重複時仍產生診斷，不允許正式優勢判定。模型訓練、原資格、proposal、初始化、Stage3、pooled評分及重評開銷分開。資格預測重用不增加方法的API呼叫或標記；重新核對cache的呼叫為研究驗證開銷，不隱藏為零。資料準備用額外Uniform求解不計為任一方法單次部署，但整批研究成本保留。沒有貨幣價格，不宣稱金額收益。

## 交付與驗收

- 方法、測試、正式登錄及程式快照；舊輸出不覆蓋。
- 結果JSONL及CSV，完整配對差CSV、α／lambda診斷、成本、平均／中位數／Q1／Q3、pooled/truth衝突及代表案例。
- manifest完整條件、來源／artifact雜湊、重載α及pooled分數、全部X直接可行檢查。
- fresh/reuse實際α/X/Stage3成本配對及完整seed重播；不得放寬1e-7容差。
- 正式結果完整後產出報告與圖，逐情境判斷；負面研究也可完成交付。

正式登錄：docs/oob_quality_shrink_formal_preregistration_v1_20261008.json。正式管線：alpha_weight_outputs/oob_quality_shrink_pipeline_v1_20261008/status.json。
