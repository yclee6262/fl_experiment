# R6｜乾淨情境相關權重：獨立複驗結果
更新：<mention-date start="2026-10-11"/>。本輪追加於唯一α研究主文件，不建立新頁，也不與R5合併樣本。
**正式40個新seeds（6000–6039）、三方法，共120條件。三項確認比較有0項通過校正判準。**
## 目的：把兩個問題分開
1. R5「自適應相關在乾淨情境勝B」是否在新資料／模型重現？
2. 相較固定相關0.5，自適應選強度是否帶來額外品質收益？另直接比較固定相關對B，不能只用兩者均值接近就宣稱固定法等效。
## 方法、例子與固定控制
- **B：品質反比。**原20筆資格資料每人MSE加0.00001後倒數、正規化到總和1。例如MSE0.01／0.04／0.09，權重約0.735／0.184／0.082。
- **固定相關0.5。**保留逐筆殘差R，S＝R轉置×R／20，D為每人MSE加0.00001的對角矩陣。以0.5S＋0.5D求非負、總和1的權重，使组合風險最小；SLSQP沿用舊法，沒有更換solver或公式。
- **自適應相關。**原20筆固定一次五折，每折16筆估S／D，測λ0／0.25／0.5／0.75／1；矩陣(1−λ)S＋λD。每折w只由訓練資料產生，在另4筆評混合MSE；平均五折選λ，1e−12內平手偏好較大λ，最後全20筆重估一次α。
S是未中心化殘差二階矩，保留偏差；不能稱純共變異數。λ1為忽略非對角資訊的B方向，λ0充分考慮相關性；λ不是往Uniform混合的比例。若两模型一起高估，其錯誤資訊可能重複；互補訊號有分散風險直覺，但這不是付款公平或經濟最適保證。
自適應失敗候選λ不可選，全部無效或全量重估失敗回退B並保存原因；任何失敗不靜默刪除。
二維、target0、固定IDs1／2／20、三人coalition、box−1至1；可行性1e−8，重載1e−7；乾淨local標記noise SD0.05，各人500筆，20筆原資格標記。同seed三法共用資料、local／pooled模型、proposal與資格快取。單次Stage3，無α更新、pruning、結算或新增標記。
pooled只在真正交付X後評分，為主品質；truth只作診斷。外部分數不選λ、權重、solver、coalition或交付輪次。資格混合MSE與Stage3各local目標絕對誤差加權不同，不能直接推論倒推品質。
## 登錄、樣本與控制驗收
歷史pilot5000–5004共15條件，與R5同三方法的α／X／pooled／truth最大差0，Stage3查詢及狀態一致；只作實作門檻，不加入本輪正式。
事前以R5乾淨34配對的三組SD最大值、研究解析度0.005、名義80%常態近似與三比較Bonferroni估N40。只用舊變異規劃N，不改公式／網格；0.005不是業務達標門檻，沒有保證power。正式開始前查核347份manifests，6000–6039未見重疊。來源、環境及設定已封存，未因正式結果重新調參或增加N。
## 交付品質與失敗
pooled error＝凍結evaluator在交付X的預測與target的絕對差，越低越好。平均／中位／Q1／Q3只在有效可行解上統計；失敗獨立揭露。
<table header-row="true" fit-page-width="true">
<tr><td>方法</td><td>平均pooled</td><td>中位（Q1～Q3）</td><td>失敗／執行</td></tr>
<tr><td>B：品質反比</td><td>0.01514423</td><td>0.01256272（0.00546227～0.02107432）</td><td>0/40</td></tr>
<tr><td>固定相關0.5</td><td>0.01561011</td><td>0.01538243（0.00559342～0.02088197）</td><td>0/40</td></tr>
<tr><td>自適應相關</td><td>0.01475673</td><td>0.01101996（0.00350904～0.02087548）</td><td>0/40</td></tr>
</table>
## 三項事前確認比較
Δ＝方法pooled error−基準error；負值有利方法。40seed配對bootstrap10,000次，RNG20260924；三項家族0.05，Bonferroni同時區間98.3333%，另報一般95%。優勢需完整有效配對、校正區間全小於0、未增加觀察失敗且驗收通過。跨零只表示證據不足，不證明等效、非劣或沒有作用。
<table header-row="true" fit-page-width="true">
<tr><td>比較</td><td>Δ平均（有效n）</td><td>95%CI</td><td>98.3333%同時CI</td><td>優勢</td></tr>
<tr><td>自適應相關−B：品質反比</td><td>-0.00038750（40）</td><td>-0.00168320 ～ 0.00128821</td><td>-0.00191390 ～ 0.00178530</td><td>未建立</td></tr>
<tr><td>固定相關0.5−B：品質反比</td><td>0.00046587（40）</td><td>-0.00138060 ～ 0.00269709</td><td>-0.00174612 ～ 0.00327203</td><td>未建立</td></tr>
<tr><td>自適應相關−固定相關0.5</td><td>-0.00085338（40）</td><td>-0.00219629 ～ 0.00026097</td><td>-0.00256726 ～ 0.00045990</td><td>未建立</td></tr>
</table>
自適應相關對B：品質反比：pooled／truth方向衝突5/40，排除0。
固定相關0.5對B：品質反比：pooled／truth方向衝突7/40，排除0。
自適應相關對固定相關0.5：pooled／truth方向衝突9/40，排除0。
完整逐seed差值、衝突seed及各方法X保存在analysis與JSONL；不只選有利評估者。
## 方法成本與λ
<table header-row="true" fit-page-width="true">
<tr><td>方法</td><td>初始化秒</td><td>Stage3秒</td><td>合計秒</td><td>Stage3 calls</td></tr>
<tr><td>B：品質反比</td><td>0.00005297</td><td>0.01175499</td><td>0.01180796</td><td>391.20000000</td></tr>
<tr><td>固定相關0.5</td><td>0.00159581</td><td>0.01061011</td><td>0.01220592</td><td>357.52500000</td></tr>
<tr><td>自適應相關</td><td>0.03110085</td><td>0.01083875</td><td>0.04193960</td><td>360.52500000</td></tr>
</table>
自適應初始化＋Stage3為固定相關約3.44倍；不是含訓練的完整流程倍數。λ次數{"0": 16, "0.25": 9, "0.5": 5, "0.75": 1, "1": 9}。各方法fallback：B：品質反比0/40；固定相關0.50/40；自適應相關0/40。
本機耗時不等於遠端API價格。API calls與批次輸入點數分開保存；三法重用資格快取，初始化calls為0，不代表資格服務免費。共同模型／訓練／資格／proposal每seed一次在來源shared_costs.jsonl；重載快取建立每seed3batch／60输入在calibration.jsonl。額外來源Uniform求解、舊OOB重用核對、驗收及重播另作研究開銷，不加入某方法部署，也不刪除。沒有品質貨幣價值設定，不宣稱經濟淨收益。
## Evaluator coverage
<table header-row="true" fit-page-width="true">
<tr><td>距target區域</td><td>樣本數平均（範圍）</td><td>區域truth MSE平均</td><td>少於30的seed數</td></tr>
<tr><td>[0,0.1]</td><td>104.0（80～126）</td><td>0.00015062</td><td>0/40</td></tr>
<tr><td>(0.1,0.25]</td><td>158.4（129～182）</td><td>0.00013805</td><td>0/40</td></tr>
<tr><td>(0.25,0.5]</td><td>271.5（231～302）</td><td>0.00013411</td><td>0/40</td></tr>
<tr><td>(0.5,1]</td><td>732.9（699～782）</td><td>0.00014383</td><td>0/40</td></tr>
<tr><td>>1</td><td>233.3（197～265）</td><td>0.00017275</td><td>0/40</td></tr>
</table>
分區用合成truth輸出對target的距離，僅供事後覆蓋診斷；少於30是稀疏提示，不是可靠門檻，不直接叫OOD。一般／hard區域完整指標在凍結evaluator payload。
## 驗收與重現
- 正式120/120條件、有效可行120；α、pooled與資格預測重載最大差0.0／0.0／0.0。
- 40新法條件、1000折內矩陣與heldout loss逐項核對；固定0.5與舊法最大差0.0，λ1與B最大差1.1102230246251565e-16。
- 實際fresh／cache兩組（固定、自適應）α／X／Stage3查詢一致；fresh3calls60輸入、cache0。額外驗收成本126calls／2520輸入、0.593秒，不混入部署。
- 完整重播seed6000，重產資料／切分／local／pooled／資格／proposal／λ／solver／α／X／評分，3條件最大差0.0。時間不要求相同。
- 20項回歸及4項失敗分析測試通過；缺漏／重複阻止正式主張，全部失敗時CI不可計算，失敗成本保留。完整模型、資料與來源快照hash通過。
## 欄位如何讀
<table header-row="true" fit-page-width="true">
<tr><td>欄位</td><td>值代表什麼</td></tr>
<tr><td>seed／scenario／method</td><td>本輪完整條件鍵；seed6000–6039、scenario固定reference_clean。</td></tr>
<tr><td>alpha／X／status／max_box_violation</td><td>固定權重、真正交付解、執行狀態、由X重算的違規量。</td></tr>
<tr><td>selected_lambda／candidate_cv_mse</td><td>矩陣對角強度、五折留出損失；不是Uniform比例或pooled分數。</td></tr>
<tr><td>fold_audits／final_fit／fallback_reason</td><td>完整16/4索引、矩陣、留出損失、全量solver與退回B原因，在JSONL。</td></tr>
<tr><td>pooled_error／truth_error</td><td>主要品質與診斷品質，均為對target絕對差。</td></tr>
<tr><td>CI95／CI_simultaneous_3／n</td><td>一般95%、三比較校正区間、有效seed配對數；五折不算五個N。</td></tr>
<tr><td>initialization_*／stage3_*／scoring_seconds</td><td>各階段本機秒／calls／输入點；外部評分不加入Stage3。</td></tr>
</table>
登錄：`docs/corr_replication_preregistration_20261011.json`；完整結果：`alpha_weight_outputs/corr_replication_formal_v1_20261011/`；runner與分析：`run_correlation_replication.py`、`analyze_correlation_replication.py`。CSV精簡條件與配對差值，不包含完整fold矩陣；完整JSONL在本地保存。
## 採用與下一步決策
本輪未重現可確認的對B優勢，也未建立自適應相對固定的收益。R5單輪正面訊號與本輪負面結果分開保留，不合併挑選有利結論。維持A/B；暫停新增初始化變體，下一步優先調查資格代理損失與倒推品質的關係。
所有后續研究尚未執行。沒有非劣性或等效設計，不能因區間跨零就宣稱方法相同。
