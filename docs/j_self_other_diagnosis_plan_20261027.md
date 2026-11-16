# 當輪 j 的自己／其他模型評分拆解：分析規約

2026-10-27。依使用者授權實作與執行。資料已揭露，這是後續探索診斷，不是未見資料正式確認。

## 固定資料與問題

全部20舊開發隊、5起點、5450個共同移除端點對。使用 j_loo_start_stability_experiment_v1_20261027 保存的原始模型回覆、cases、pooled_points，以及 j_local_quality_analysis_v1_20261027 的逐輪j／j_E。核對兩個來源receipt的全部hash。無新增搜尋、模型推論、pooled推論或訓練。

主要問題：原j與同端點j_E反號時，若只在事後評分中扣掉該參與者自己模型的部分，其他模型合計是否常已與j_E同號？這是固定端點的代數解釋，不能識別該模型對搜尋方向或接受決策的因果貢獻。

## 精確定義

對移除參與者i的分支，delta_k=|f_k(x_minus)−target|−|f_k(x_full)−target|，target=0。

own_i=w_i*delta_i；other_i=Σ(k!=i)w_k*delta_k；j_i=own_i+other_i。w保留原完整團隊正規化權重，不對other重新正規化。兩者可正可負。這是「誰在評分」的拆解，不是錨點角色與模型角色的拆解。

另存other_mean=other/(1−w_i)只作同尺度診斷，明確標示不是原j、不採用為新方法。累積OWN=Σown、OTHER=Σother，必須加回原J。

精確評分落差：gap=j−j_E；own_gap=w_i*(delta_i−j_E)；other_gap=Σ(k!=i)w_k*(delta_k−j_E)。兩部分加回gap。自身落差絕對占比=|own_gap|/(|own_gap|+|other_gap|)，分母<=1e-4不計。這不是因果百分比。

## 指標與分母

epsilon=1e-4僅分類，原分數不裁切。主要反號集是sign(j)與sign(j_E)相反且兩邊皆非零。分成互斥三類：其他模型與j_E同號、其他模型仍與j同號、其他模型近零。第一類稱「自己的評分在固定端點上改變合計方向」；不是主觀灌分或操縱證據。另報所有分支own與other異號率、own絕對占比、own_gap絕對占比。

公平比較：固定同一分母abs(j_E)>epsilon，對j、own、other、other_mean列同號／反號／預測近零率；不能刪掉other難判案例。另按舊|j|>epsilon分母保留對照。逐輪跨人rho及可排名輪覆蓋、累積分數對原同端點J_E的rho作次要診斷。完整LOO不是本輪估計目標，不重新驗證全程替代。

評分模型表：對每個實際模型k，在同一批分支上比較delta_k與j_E；先同隊同起點內平均，再同隊跨起點平均。是選定端點集合上的條件表現，不代表一般預測準確率。每分支保存完整逐模型delta與加權部分，以辨認是少數模型還是面板普遍分歧。不按模型或起點挑成功，不據此更新alpha或刪人。

## 聚合、優先門檻與案例

每項指標先同隊同起點，再同隊有效起點，最後情境10隊等權；bootstrap10000次、seed20261027、95%探索區間。保留有效隊、起點與分母。只把團隊當重抽單位。

若要把「去除自己評分」列為下一輪候選提案，需兩情境都有>=8有效隊、>=20反號分支，反號集中other已同號比例>=30%且下界>10%；同一外部分母上other_mean相對j的同號率配對改善>=10百分點且下界>0。這只是研究優先門檻，不是採用或效度門檻。另以同號>=85%且下界>70%、rank>=.8且下界>.5、可排名輪覆蓋>=80%作局部品質探索篩選。均不可稱正式確認。

案例每情境各選：other已與外部同號的反號例、other仍與原j同號的反號例，各以min(|j|,|j_E|)最大選取；若沒有則保留空。平手team/start/agent/t升序。另原樣核對前輪兩個固定最大反號例。不得改用較好故事。

## 驗收及產物

獨立從原回覆重建每模型delta、own/other/j與gap恒等式；從精確float64端點查pooled；累積吻合原J與J_E，容許atol/rtol1e-10。獨立math.fsum、手寫rank、分類、聚合與bootstrap核對。所有來源hash前後一致；缺資料即停止，不補推論。最多300秒／100MB，新推論0。

新程式 contribution_analysis/j_self_other_v1/；新輸出 contribution_analysis/j_self_other_diagnosis_v1_20261027/；報告 docs/j_self_other_diagnosis_results_20261027.md。保留歷史與原J；Notion、alpha、pruning、付款、Shapley皆不處理。分列已完成、未建立有效性、下一步提案。
