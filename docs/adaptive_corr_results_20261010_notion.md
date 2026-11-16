# R5｜自適應誤差相關權重：正式結果
更新：<mention-date start="2026-10-10"/>。本輪追加到同一主文件；沒有新增單輪Notion頁。
**完成34個新seeds、三情境、五方法，共510條件。六項確認比較有1項通過預登錄校正判準。**負面結果同樣保留；本輪不與任何舊seeds合併。
## 研究目的與固定設計
檢驗資格資料估計的誤差相關性，是否值得納入α；並測試由留出資料選強度是否優於已做過的固定0.5相關權重。主要品質是實際交付X的pooled error，不是資格混合MSE。
二維、target0、固定三人IDs1／2／20、box−1至1、可行性容差1e−8。乾淨／local低品質／樣本量不平衡的資料控制沿用前輪；pooled始終使用乾淨評估資料。同seed五方法共用資料、models、20筆資格資料、proposal與evaluator。單次Stage3，無α更新、pruning、結算或新增可信標記。
Pilot：4800–4804，75條件；開發：4810–4819，150條件，均使用歷史模型作實作與變異估計，不是獨立新品質證據，不併入正式。正式：5000–5033，啟動前查核337份manifests，未見重疊。
事前固定研究解析度0.005、名義80%常態近似與六比較家族0.05；取開發六組配對SD最大值約0.00834809，計得N34。10seed變異估計不精確，不能宣稱保證power或業務達標門檻。沒有根據正式結果修改網格、參數或N。
## 五種方法如何做
- **A：Uniform。**三人各1/3，無品質區分。
- **B：品質反比。**原20筆每人MSE加0.00001、倒數、正規化。
- **原保守品質。**共同bootstrap1000次，每人取MSE第90百分位，倒數正規化；分位數不是90%信賴上界。
- **固定相關0.5。**以殘差二階矩S與加epsilon的個別MSE對角矩陣D各一半，求非負、總和1的w，使w轉置×矩陣×w最低。與舊Corr precision原目標及solver保持一致。
- **新自適應相關。**固定五折，每折16筆估S/D，測λ0／0.25／0.5／0.75／1：矩陣為(1−λ)S＋λD。每個λ的w只由訓練16筆計算，再於另4筆評估混合MSE；平均五折，最低者勝，1e−12內平手偏好較大λ。最後全20筆以選定λ重估一次w，作單次倒推α。
λ1忽略非對角相關性，對應B式品質方向；λ0全面使用相關性。λ0.5對應固定相關方法。沒有在w上另加Uniform收縮，也沒有1000輪重抽樣擬合新法。
S是**未中心化殘差二階矩**，保留系統偏差，不是純共變異數。SLSQP不直接求反矩陣；任一折求解無效則該λ不可選，全部無效或最後重估失敗才回退B並記錄。
**經濟直覺與例子。**兩個模型若經常一起高估，兩份訊號可能重複；第三模型較獨立則可能分散風險。不能僅由三個平均MSE推導相關權重，要逐筆殘差。此為資訊配置理由，不是付款比例、公平性或經濟最適保證。
**限制。**留出混合MSE與Stage3各local目標絕對誤差的加權和不同。pooled／truth只在交付後評分，不參與λ、α、solver、coalition或交付選擇。
## 品質：同輪配對判斷
平均／中位／四分位只含有效可行解；完整失敗另列，不用刪除失敗改善平均。pooled error越低越好。
### 乾淨
<table header-row="true" fit-page-width="true">
<tr><td>方法</td><td>平均pooled</td><td>中位（Q1～Q3）</td><td>失敗／執行</td></tr>
<tr><td>A：Uniform</td><td>0.02549730</td><td>0.02203459（0.01537975～0.03250602）</td><td>0/34</td></tr>
<tr><td>B：品質反比</td><td>0.02134368</td><td>0.01811094（0.00976263～0.02890281）</td><td>0/34</td></tr>
<tr><td>原保守品質</td><td>0.01900313</td><td>0.01635789（0.00688121～0.02773706）</td><td>0/34</td></tr>
<tr><td>固定相關權重0.5</td><td>0.01781306</td><td>0.01580909（0.00944045～0.02624331）</td><td>0/34</td></tr>
<tr><td>新自適應相關權重</td><td>0.01705781</td><td>0.01485371（0.00800644～0.02602764）</td><td>0/34</td></tr>
</table>
### 低品質
<table header-row="true" fit-page-width="true">
<tr><td>方法</td><td>平均pooled</td><td>中位（Q1～Q3）</td><td>失敗／執行</td></tr>
<tr><td>A：Uniform</td><td>0.02498395</td><td>0.01907798（0.01142116～0.03510162）</td><td>0/34</td></tr>
<tr><td>B：品質反比</td><td>0.02481737</td><td>0.01913886（0.00927204～0.03195529）</td><td>0/34</td></tr>
<tr><td>原保守品質</td><td>0.02508102</td><td>0.02206984（0.00962673～0.03055979）</td><td>0/34</td></tr>
<tr><td>固定相關權重0.5</td><td>0.02313259</td><td>0.01903735（0.01096039～0.03094706）</td><td>0/34</td></tr>
<tr><td>新自適應相關權重</td><td>0.02335430</td><td>0.01931178（0.01124639～0.03123485）</td><td>0/34</td></tr>
</table>
### 樣本量不平衡
<table header-row="true" fit-page-width="true">
<tr><td>方法</td><td>平均pooled</td><td>中位（Q1～Q3）</td><td>失敗／執行</td></tr>
<tr><td>A：Uniform</td><td>0.02058292</td><td>0.01675086（0.00760001～0.03188121）</td><td>0/34</td></tr>
<tr><td>B：品質反比</td><td>0.01434093</td><td>0.01063296（0.00357717～0.02375595）</td><td>0/34</td></tr>
<tr><td>原保守品質</td><td>0.01582828</td><td>0.01401206（0.00360452～0.02503937）</td><td>0/34</td></tr>
<tr><td>固定相關權重0.5</td><td>0.01691858</td><td>0.01531209（0.00716398～0.02435273）</td><td>0/34</td></tr>
<tr><td>新自適應相關權重</td><td>0.01464932</td><td>0.01000708（0.00324605～0.02451777）</td><td>0/34</td></tr>
</table>
## 新法的配對比較
Δ為新法pooled error−基準error，負值較好。以seed做10,000次paired bootstrap，RNG20260924；新法對B及固定相關各三情境共六項，Bonferroni每項99.1667%同時區間。需完整有效配對、區間全小於0、未增加觀察失敗及驗收通過。A／保守是探索，不計入六項確認結論。
### 乾淨
<table header-row="true" fit-page-width="true">
<tr><td>基準／類型</td><td>平均Δ（有效n）</td><td>95%CI</td><td>六項同時CI</td></tr>
<tr><td>A：Uniform／探索</td><td>-0.00843949（34）</td><td>-0.01354984 ～ -0.00302498</td><td>不作確認校正</td></tr>
<tr><td>B：品質反比／確認</td><td>-0.00428588（34）</td><td>-0.00755725 ～ -0.00120529</td><td>-0.00872433 ～ -0.00021686</td></tr>
<tr><td>原保守品質／探索</td><td>-0.00194532（34）</td><td>-0.00460291 ～ 0.00081153</td><td>不作確認校正</td></tr>
<tr><td>固定相關權重0.5／確認</td><td>-0.00075525（34）</td><td>-0.00395431 ～ 0.00234365</td><td>-0.00507952 ～ 0.00336902</td></tr>
</table>
A：Uniform：未建立確認優勢，pooled／truth衝突5/34，排除0；B：品質反比：通過，pooled／truth衝突4/34，排除0；原保守品質：未建立確認優勢，pooled／truth衝突10/34，排除0；固定相關權重0.5：未建立確認優勢，pooled／truth衝突7/34，排除0。
### 低品質
<table header-row="true" fit-page-width="true">
<tr><td>基準／類型</td><td>平均Δ（有效n）</td><td>95%CI</td><td>六項同時CI</td></tr>
<tr><td>A：Uniform／探索</td><td>-0.00162965（34）</td><td>-0.00748192 ～ 0.00408895</td><td>不作確認校正</td></tr>
<tr><td>B：品質反比／確認</td><td>-0.00146306（34）</td><td>-0.00745925 ～ 0.00261979</td><td>-0.00997576 ～ 0.00344448</td></tr>
<tr><td>原保守品質／探索</td><td>-0.00172672（34）</td><td>-0.00771061 ～ 0.00281692</td><td>不作確認校正</td></tr>
<tr><td>固定相關權重0.5／確認</td><td>0.00022171（34）</td><td>-0.00223906 ～ 0.00282079</td><td>-0.00312171 ～ 0.00375204</td></tr>
</table>
A：Uniform：未建立確認優勢，pooled／truth衝突7/34，排除0；B：品質反比：未建立確認優勢，pooled／truth衝突7/34，排除0；原保守品質：未建立確認優勢，pooled／truth衝突8/34，排除0；固定相關權重0.5：未建立確認優勢，pooled／truth衝突9/34，排除0。
### 樣本量不平衡
<table header-row="true" fit-page-width="true">
<tr><td>基準／類型</td><td>平均Δ（有效n）</td><td>95%CI</td><td>六項同時CI</td></tr>
<tr><td>A：Uniform／探索</td><td>-0.00593360（34）</td><td>-0.01170573 ～ -0.00103400</td><td>不作確認校正</td></tr>
<tr><td>B：品質反比／確認</td><td>0.00030839（34）</td><td>-0.00184823 ～ 0.00287390</td><td>-0.00247796 ～ 0.00379692</td></tr>
<tr><td>原保守品質／探索</td><td>-0.00117896（34）</td><td>-0.00455995 ～ 0.00214261</td><td>不作確認校正</td></tr>
<tr><td>固定相關權重0.5／確認</td><td>-0.00226926（34）</td><td>-0.00612598 ～ 0.00078049</td><td>-0.00768139 ～ 0.00164570</td></tr>
</table>
A：Uniform：未建立確認優勢，pooled／truth衝突9/34，排除0；B：品質反比：未建立確認優勢，pooled／truth衝突8/34，排除0；原保守品質：未建立確認優勢，pooled／truth衝突8/34，排除0；固定相關權重0.5：未建立確認優勢，pooled／truth衝突5/34，排除0。
## 成本、λ及fallback
時間為本機wall time；倍數只涵蓋初始化＋Stage3，不含共享訓練與資格成本，也不是遠端API價格。API呼叫與輸入樣本數另存；batch20點不能只算1個輸入。共享資格快取重建為每scenario/seed三次batch、60輸入，方法間共享；初始化重用時calls為0，不代表整個部署免資格費。
### 乾淨
<table header-row="true" fit-page-width="true">
<tr><td>方法</td><td>初始化秒</td><td>Stage3秒</td><td>合計秒</td><td>Stage3 calls</td></tr>
<tr><td>A：Uniform</td><td>0.00002749</td><td>0.01033633</td><td>0.01036382</td><td>337.67647059</td></tr>
<tr><td>B：品質反比</td><td>0.00005747</td><td>0.01140791</td><td>0.01146538</td><td>376.50000000</td></tr>
<tr><td>原保守品質</td><td>0.00057367</td><td>0.01117349</td><td>0.01174716</td><td>364.05882353</td></tr>
<tr><td>固定相關權重0.5</td><td>0.00156957</td><td>0.01127806</td><td>0.01284763</td><td>375.08823529</td></tr>
<tr><td>新自適應相關權重</td><td>0.03168839</td><td>0.01211369</td><td>0.04380208</td><td>390.97058824</td></tr>
</table>
新法λ次數：{"0": 15, "0.25": 7, "0.5": 4, "0.75": 2, "1": 6}；fallback 0/34。
### 低品質
<table header-row="true" fit-page-width="true">
<tr><td>方法</td><td>初始化秒</td><td>Stage3秒</td><td>合計秒</td><td>Stage3 calls</td></tr>
<tr><td>A：Uniform</td><td>0.00002916</td><td>0.01182381</td><td>0.01185296</td><td>387.26470588</td></tr>
<tr><td>B：品質反比</td><td>0.00005585</td><td>0.01053588</td><td>0.01059173</td><td>348.97058824</td></tr>
<tr><td>原保守品質</td><td>0.00055495</td><td>0.00973024</td><td>0.01028520</td><td>324.26470588</td></tr>
<tr><td>固定相關權重0.5</td><td>0.00148694</td><td>0.01325011</td><td>0.01473705</td><td>356.82352941</td></tr>
<tr><td>新自適應相關權重</td><td>0.02973361</td><td>0.01017124</td><td>0.03990484</td><td>331.58823529</td></tr>
</table>
新法λ次數：{"0": 20, "0.25": 1, "0.5": 3, "0.75": 1, "1": 9}；fallback 0/34。
### 樣本量不平衡
<table header-row="true" fit-page-width="true">
<tr><td>方法</td><td>初始化秒</td><td>Stage3秒</td><td>合計秒</td><td>Stage3 calls</td></tr>
<tr><td>A：Uniform</td><td>0.00002706</td><td>0.01009758</td><td>0.01012464</td><td>332.38235294</td></tr>
<tr><td>B：品質反比</td><td>0.00005654</td><td>0.01096573</td><td>0.01102227</td><td>360.35294118</td></tr>
<tr><td>原保守品質</td><td>0.00054018</td><td>0.01229718</td><td>0.01283736</td><td>398.64705882</td></tr>
<tr><td>固定相關權重0.5</td><td>0.00159180</td><td>0.01091096</td><td>0.01250276</td><td>359.82352941</td></tr>
<tr><td>新自適應相關權重</td><td>0.03145089</td><td>0.00969297</td><td>0.04114386</td><td>319.05882353</td></tr>
</table>
新法λ次數：{"0": 12, "0.25": 4, "0.5": 9, "0.75": 2, "1": 7}；fallback 0/34。
## Evaluator coverage與衝突案例
### 乾淨
<table header-row="true" fit-page-width="true">
<tr><td>距target區域</td><td>測試樣本數平均（範圍）</td><td>區域truth MSE平均</td><td>少於30的seed數</td></tr>
<tr><td>[0,0.1]</td><td>101.9（85～120）</td><td>0.00012537</td><td>0/34</td></tr>
<tr><td>(0.1,0.25]</td><td>158.2（132～185）</td><td>0.00010958</td><td>0/34</td></tr>
<tr><td>(0.25,0.5]</td><td>274.3（247～302）</td><td>0.00011830</td><td>0/34</td></tr>
<tr><td>(0.5,1]</td><td>734.4（686～763）</td><td>0.00015986</td><td>0/34</td></tr>
<tr><td>>1</td><td>231.2（204～257）</td><td>0.00016290</td><td>0/34</td></tr>
</table>
### 低品質
<table header-row="true" fit-page-width="true">
<tr><td>距target區域</td><td>測試樣本數平均（範圍）</td><td>區域truth MSE平均</td><td>少於30的seed數</td></tr>
<tr><td>[0,0.1]</td><td>101.9（85～120）</td><td>0.00012537</td><td>0/34</td></tr>
<tr><td>(0.1,0.25]</td><td>158.2（132～185）</td><td>0.00010958</td><td>0/34</td></tr>
<tr><td>(0.25,0.5]</td><td>274.3（247～302）</td><td>0.00011830</td><td>0/34</td></tr>
<tr><td>(0.5,1]</td><td>734.4（686～763）</td><td>0.00015986</td><td>0/34</td></tr>
<tr><td>>1</td><td>231.2（204～257）</td><td>0.00016290</td><td>0/34</td></tr>
</table>
### 樣本量不平衡
<table header-row="true" fit-page-width="true">
<tr><td>距target區域</td><td>測試樣本數平均（範圍）</td><td>區域truth MSE平均</td><td>少於30的seed數</td></tr>
<tr><td>[0,0.1]</td><td>102.9（91～124）</td><td>0.00013951</td><td>0/34</td></tr>
<tr><td>(0.1,0.25]</td><td>159.0（131～181）</td><td>0.00012085</td><td>0/34</td></tr>
<tr><td>(0.25,0.5]</td><td>276.1（255～302）</td><td>0.00010876</td><td>0/34</td></tr>
<tr><td>(0.5,1]</td><td>731.7（701～758）</td><td>0.00012587</td><td>0/34</td></tr>
<tr><td>>1</td><td>230.3（208～246）</td><td>0.00016421</td><td>0/34</td></tr>
</table>
距離按合成truth輸出與target分區，只有事後coverage；少於30是資料稀疏提示，不是經驗證可靠門檻，不直接稱OOD。完整一般／hard區域指標留在凍結evaluator payload。所有pooled／truth衝突的seed、Δ及X保留於analysis及完整JSONL，不選擇性呈現。
## 驗收、封存與重播
- 正式完整性510/510；有效可行510。α／pooled／資格預測重載最大差：0.0／0.0／0.0。
- 每一新方法條件的五折資料分離、矩陣與留出損失直接核對：102條件、2550折內擬合；固定0.5與舊法、λ1與B核對102組，舊法最大差0.0。
- 新法與固定相關實際fresh／cache重算6組，α、交付X與Stage3查詢一致；fresh3次／60輸入，cache0。額外audit成本324calls、6480輸入、1.441秒，未混入方法部署成本。
- 完整重播seed5000，三情境15條件；重產資料／切分、local與pooled模型、資格選擇、proposal、λ、solver、α／X／分數。最大差0.0；時間不要求相同。
- 回歸測試25項通過，另有分析失敗處理驗收；相關命令與證據保存在execution及audit文件。舊結果不覆蓋，模型／資料與來源快照hash核對。
## 欄位字典與重現資料
<table header-row="true" fit-page-width="true">
<tr><td>欄位</td><td>代表什麼</td></tr>
<tr><td>scenario／seed／method</td><td>同輪完整識別键，不把不同輪結果混成配對。</td></tr>
<tr><td>alpha／X／status／max_box_violation</td><td>初始固定權重、真正交付解、執行狀態、由X直接重算的最大違規量。</td></tr>
<tr><td>selected_lambda／candidate_cv_mse</td><td>矩陣對角收縮比例、各λ五折留出損失；不是Uniform比例或pooled分數。</td></tr>
<tr><td>fold_audits／second_moment／risk_matrix</td><td>訓練與留出索引、訓練殘差未中心化二階矩、實際solver矩陣。</td></tr>
<tr><td>final_fit／fallback_reason</td><td>全20筆重估solver狀態、退回B原因；權重fallback與交付失敗分開。</td></tr>
<tr><td>pooled_error／truth_error</td><td>事後主品質與診斷品質，均為對target绝對差。</td></tr>
<tr><td>CI95／CI_simultaneous_6／confirmatory</td><td>一般区間、六比較校正区間、是否事前確認；探索不冒稱確認。</td></tr>
<tr><td>initialization_*／stage3_*／scoring_seconds</td><td>本地各階段秒／模型calls／輸入點；外部分數耗時不混入Stage3。</td></tr>
<tr><td>artifact_reference</td><td>模型／資料／split／evaluator凍結路徑及hash索引。</td></tr>
</table>
完整結果：`alpha_weight_outputs/adaptive_corr_formal_v1_20261010/`。正式登錄：`docs/adaptive_corr_preregistration_20261010.json`。來源：`adaptive_correlation_alpha.py`、runner與analyzer；正式結果含results.jsonl、每條件完整fold audit、analysis、verification、extra_acceptance及replay_verification。CSV下載提供精簡條件與配對Δ；大矩陣與折內資料在完整JSONL，不冒稱CSV含全部細節。
## 本輪決策
通過範圍：乾淨對B：品質反比。僅限本輪實際通過情境；尚無跨全部情境的通用保證。
新法沒有在任何情境穩定勝過固定0.5相關對照，因此尚未證實「自適應選λ」這項新增機制有價值。保留A/B，不把新法升格主要方法。下一步優先以新的獨立seeds在乾淨情境配對複驗B／固定相關0.5／新自適應三法，確認是否固定相關已足夠；不依本輪結果調λ網格或增加變體。正式複驗需另行固定版本與N，本輪只完成目前研究交付。
