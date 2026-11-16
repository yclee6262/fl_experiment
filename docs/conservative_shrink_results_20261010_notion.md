# 本輪結論

完成30個新seeds、三情境、五方法，共450條件。新方法對四個基準／診斷方法的12項確認比較中，有0項符合預登錄穩定優勢判準。

通過比較：沒有。平均誤差較低不等於已證明穩定改善。

# 目的與固定控制

檢驗保守品質權重加入留出收縮後，是否讓最終倒推X更好、更穩定。不是只看資格預測MSE降低，也不預設新方法必勝。

二維、target0、固定participant IDs1、2、20與三人coalition。每seed五方法共用乾淨評估資料、local／pooled模型、資格盲測20筆及proposal；單次倒推，無α更新、pruning與結算。低品質僅改local observed labels，評估者維持乾淨；樣本量不平衡控制保持既有四倍設定。

Pilot沿用歷史seeds4600–4604模型作實作驗收，不能當新獨立品質證據、不併入正式。正式新seeds4800–4829，啟動前查核324份manifests未見重疊。樣本數固定30，沒有保證檢定力。設定在正式開始前固定，正式結果不能用來調參。

本頁q是資格資料估計的品質分布，與貢獻q_raw、更新q_update、付款q_pay不同。

# 每個方法如何做

**A：Uniform。**三人各1/3；沒有從資格誤差區分可信度。

**B：Stage0 inverse-MSE。**原20筆資格資料的各人平均平方誤差加0.00001、取倒數、正規化到總和1。

**原保守品質。**原20筆共同重抽樣1000次，得到每人MSE分布，取第90百分位數，加0.00001後倒數正規化。這是固定風險分位規則，不是90%信賴上界，也不是付款或公平性保證。

**切分保守品質、不收縮（診斷）。**20次隨機排列，每次分成五折；每折16筆估計、4筆留出。只在16筆共同bootstrap1000次，取得第90百分位MSE並轉成q；不選λ，直接平均100個q。與新方法使用完全相同切分與內部bootstrap亂數，分辨效果是否僅來自切分與平均。

**新保守品質＋留出收縮。**與上述診斷一樣先用16筆估q，再只用另外4筆決定λ，最小化留出資料上的混合預測MSE。每折權重=(1−λ)q＋λUniform，λ範圍0–1，平坦損失選λ1。平均100個收縮向量成為交付倒推的初始α。

每筆資格資料在每次五折恰好驗證一次，共被留出20次；同一折的留出列不能參與風險分位估計。每折MSE分位的1000次bootstrap只抽該16筆，不是先用全20筆算保守權重再切分。

**例子。**某折q=(0.7,0.2,0.1)，留出4筆選λ0.3，該折權重約=(0.59,0.24,0.17)。仍較信任第一人，但減少集中。最終α是100折的平均，不挑資格或pooled最好的一折。

**經濟直覺。**較高風險誤差降低資訊權重；當留出資料不足以支持強烈集中時，可向均勻分散。這是待驗證的資訊配置理由，不等於已證明經濟最適或貢獻公平。

留出損失是混合預測MSE；Stage3求的是各local模型對target的加權絕對誤差，兩者不同。資格預測變好不保證倒推X變好。

# 品質：越低越好

pooled error為凍結評估模型在交付X的預測與target的絕對差。truth error為合成真實函數診斷，不能用來調λ、選α、求解或挑交付輪次。品質只在有效解上統計，全部失敗另外揭露。

## 乾淨

<table header-row="true" fit-page-width="true">
<tr><td>方法</td><td>平均pooled</td><td>中位（Q1～Q3）</td><td>失敗/執行</td></tr>
<tr><td>A：均勻</td><td>0.02146820</td><td>0.01823242（0.00915530～0.03020745）</td><td>0/30</td></tr>
<tr><td>B：品質反比</td><td>0.01653306</td><td>0.01715215（0.00658864～0.02134249）</td><td>0/30</td></tr>
<tr><td>原保守品質</td><td>0.01646993</td><td>0.01457160（0.00596306～0.02528837）</td><td>0/30</td></tr>
<tr><td>切分保守品質（不收縮）</td><td>0.01584449</td><td>0.01376452（0.00571158～0.02549789）</td><td>0/30</td></tr>
<tr><td>新保守品質＋留出收縮</td><td>0.01666363</td><td>0.01567717（0.00575194～0.02668581）</td><td>0/30</td></tr>
</table>

## 低品質

<table header-row="true" fit-page-width="true">
<tr><td>方法</td><td>平均pooled</td><td>中位（Q1～Q3）</td><td>失敗/執行</td></tr>
<tr><td>A：均勻</td><td>0.02129734</td><td>0.01576438（0.00706499～0.03334695）</td><td>0/30</td></tr>
<tr><td>B：品質反比</td><td>0.02240855</td><td>0.02183923（0.01252143～0.03410289）</td><td>0/30</td></tr>
<tr><td>原保守品質</td><td>0.02183963</td><td>0.02109316（0.01350995～0.02958711）</td><td>0/30</td></tr>
<tr><td>切分保守品質（不收縮）</td><td>0.02191345</td><td>0.02112683（0.01309624～0.02962414）</td><td>0/30</td></tr>
<tr><td>新保守品質＋留出收縮</td><td>0.02080588</td><td>0.01837419（0.01105323～0.03041066）</td><td>0/30</td></tr>
</table>

## 樣本量不平衡

<table header-row="true" fit-page-width="true">
<tr><td>方法</td><td>平均pooled</td><td>中位（Q1～Q3）</td><td>失敗/執行</td></tr>
<tr><td>A：均勻</td><td>0.01657951</td><td>0.01207057（0.00411093～0.02735534）</td><td>0/30</td></tr>
<tr><td>B：品質反比</td><td>0.01805604</td><td>0.01208928（0.00556126～0.01942049）</td><td>0/30</td></tr>
<tr><td>原保守品質</td><td>0.01724391</td><td>0.01254014（0.00515473～0.01898580）</td><td>0/30</td></tr>
<tr><td>切分保守品質（不收縮）</td><td>0.01772461</td><td>0.01302196（0.00677878～0.01924547）</td><td>0/30</td></tr>
<tr><td>新保守品質＋留出收縮</td><td>0.01698626</td><td>0.01210426（0.00689974～0.01887641）</td><td>0/30</td></tr>
</table>

# 配對比較與判讀

差值=新方法pooled error−基準error，負值較好。只保留雙方都有效的品質配對，失敗與排除獨立保存。每seed是抽樣單位，100折、1000次內層bootstrap不能當獨立樣本數。

以seed做10,000次配對bootstrap，分析RNG20260924。12項比較採Bonferroni家族0.05，每項同時區間信賴水準99.5833%；另列一般95%。穩定優勢要求同時區間完全小於0、完整有效配對、無較高失敗率及保存驗收通過。CI跨零表示證據不足，不證明等效。

## 乾淨

<table header-row="true" fit-page-width="true">
<tr><td>比較基準</td><td>平均差（有效n）</td><td>95%區間</td><td>12項同時區間</td></tr>
<tr><td>A：均勻</td><td>-0.00480457（30）</td><td>-0.01126659 ～ 0.00015192</td><td>-0.01518935 ～ 0.00182548</td></tr>
<tr><td>B：品質反比</td><td>0.00013057（30）</td><td>-0.00222276 ～ 0.00241960</td><td>-0.00340196 ～ 0.00341026</td></tr>
<tr><td>原保守品質</td><td>0.00019371（30）</td><td>-0.00237236 ～ 0.00247406</td><td>-0.00366627 ～ 0.00336938</td></tr>
<tr><td>切分保守品質（不收縮）</td><td>0.00081915（30）</td><td>-0.00161506 ～ 0.00306350</td><td>-0.00276204 ～ 0.00387502</td></tr>
</table>

A：均勻：未建立穩定優勢，pooled/truth衝突7/30，排除0seed；B：品質反比：未建立穩定優勢，pooled/truth衝突5/30，排除0seed；原保守品質：未建立穩定優勢，pooled/truth衝突5/30，排除0seed；切分保守品質（不收縮）：未建立穩定優勢，pooled/truth衝突9/30，排除0seed

## 低品質

<table header-row="true" fit-page-width="true">
<tr><td>比較基準</td><td>平均差（有效n）</td><td>95%區間</td><td>12項同時區間</td></tr>
<tr><td>A：均勻</td><td>-0.00049146（30）</td><td>-0.00542181 ～ 0.00429240</td><td>-0.00789641 ～ 0.00646246</td></tr>
<tr><td>B：品質反比</td><td>-0.00160267（30）</td><td>-0.00474220 ～ 0.00110309</td><td>-0.00630330 ～ 0.00217672</td></tr>
<tr><td>原保守品質</td><td>-0.00103375（30）</td><td>-0.00375644 ～ 0.00146222</td><td>-0.00513055 ～ 0.00246371</td></tr>
<tr><td>切分保守品質（不收縮）</td><td>-0.00110757（30）</td><td>-0.00370231 ～ 0.00108178</td><td>-0.00502065 ～ 0.00182607</td></tr>
</table>

A：均勻：未建立穩定優勢，pooled/truth衝突4/30，排除0seed；B：品質反比：未建立穩定優勢，pooled/truth衝突6/30，排除0seed；原保守品質：未建立穩定優勢，pooled/truth衝突5/30，排除0seed；切分保守品質（不收縮）：未建立穩定優勢，pooled/truth衝突5/30，排除0seed

## 樣本量不平衡

<table header-row="true" fit-page-width="true">
<tr><td>比較基準</td><td>平均差（有效n）</td><td>95%區間</td><td>12項同時區間</td></tr>
<tr><td>A：均勻</td><td>0.00040674（30）</td><td>-0.00408302 ～ 0.00501613</td><td>-0.00627448 ～ 0.00735569</td></tr>
<tr><td>B：品質反比</td><td>-0.00106978（30）</td><td>-0.00525478 ～ 0.00266627</td><td>-0.00739915 ～ 0.00453732</td></tr>
<tr><td>原保守品質</td><td>-0.00025765（30）</td><td>-0.00458350 ～ 0.00396560</td><td>-0.00667147 ～ 0.00621819</td></tr>
<tr><td>切分保守品質（不收縮）</td><td>-0.00073835（30）</td><td>-0.00487292 ～ 0.00320448</td><td>-0.00679164 ～ 0.00495499</td></tr>
</table>

A：均勻：未建立穩定優勢，pooled/truth衝突7/30，排除0seed；B：品質反比：未建立穩定優勢，pooled/truth衝突8/30，排除0seed；原保守品質：未建立穩定優勢，pooled/truth衝突7/30，排除0seed；切分保守品質（不收縮）：未建立穩定優勢，pooled/truth衝突7/30，排除0seed

pooled/truth衝突指同seed比較的新方法與基準，兩評估來源的改善方向相反；逐seed差值與衝突保留，不選擇性採用有利來源。

# 成本與收縮行為

下表只包括初始化＋Stage3，不包括資料準備、訓練、資格與proposal或事後評分；不可稱整個流程成本。共用資格預測，初始化API為零仍有本地計算。新方法每次做100組×1000bootstrap，遠多於原保守品質一次1000。

## 乾淨

<table header-row="true" fit-page-width="true">
<tr><td>方法</td><td>初始化秒</td><td>Stage3秒</td><td>合計秒</td><td>Stage3 API次</td></tr>
<tr><td>A：均勻</td><td>0.00002639</td><td>0.01184295</td><td>0.01186935</td><td>390.90000000</td></tr>
<tr><td>B：品質反比</td><td>0.00005462</td><td>0.01105283</td><td>0.01110745</td><td>372.90000000</td></tr>
<tr><td>原保守品質</td><td>0.00056473</td><td>0.01127711</td><td>0.01184185</td><td>379.80000000</td></tr>
<tr><td>切分保守品質（不收縮）</td><td>0.04134788</td><td>0.01095987</td><td>0.05230775</td><td>368.30000000</td></tr>
<tr><td>新保守品質＋留出收縮</td><td>0.04598669</td><td>0.01172144</td><td>0.05770813</td><td>390.90000000</td></tr>
</table>

平均λ=0.39890739；λ表示均勻比例，越大越接近A，不能當成pooled品質保證。

## 低品質

<table header-row="true" fit-page-width="true">
<tr><td>方法</td><td>初始化秒</td><td>Stage3秒</td><td>合計秒</td><td>Stage3 API次</td></tr>
<tr><td>A：均勻</td><td>0.00002493</td><td>0.01808539</td><td>0.01811031</td><td>369.10000000</td></tr>
<tr><td>B：品質反比</td><td>0.00005472</td><td>0.01237240</td><td>0.01242712</td><td>373.80000000</td></tr>
<tr><td>原保守品質</td><td>0.00052991</td><td>0.01036277</td><td>0.01089269</td><td>348.40000000</td></tr>
<tr><td>切分保守品質（不收縮）</td><td>0.04150144</td><td>0.01024174</td><td>0.05174318</td><td>344.30000000</td></tr>
<tr><td>新保守品質＋留出收縮</td><td>0.04583775</td><td>0.01065152</td><td>0.05648927</td><td>356.10000000</td></tr>
</table>

平均λ=0.23738175；λ表示均勻比例，越大越接近A，不能當成pooled品質保證。

## 樣本量不平衡

<table header-row="true" fit-page-width="true">
<tr><td>方法</td><td>初始化秒</td><td>Stage3秒</td><td>合計秒</td><td>Stage3 API次</td></tr>
<tr><td>A：均勻</td><td>0.00002472</td><td>0.01009430</td><td>0.01011902</td><td>340.90000000</td></tr>
<tr><td>B：品質反比</td><td>0.00005403</td><td>0.01083552</td><td>0.01088955</td><td>362.90000000</td></tr>
<tr><td>原保守品質</td><td>0.00052998</td><td>0.01003968</td><td>0.01056967</td><td>336.20000000</td></tr>
<tr><td>切分保守品質（不收縮）</td><td>0.04173942</td><td>0.00876042</td><td>0.05049984</td><td>296.60000000</td></tr>
<tr><td>新保守品質＋留出收縮</td><td>0.04606377</td><td>0.01146385</td><td>0.05752762</td><td>386.40000000</td></tr>
</table>

平均λ=0.38886378；λ表示均勻比例，越大越接近A，不能當成pooled品質保證。

# 保存、驗收與可重現

每條件即時JSONL保存，完整X、α、solver、原因、成本及artifact路徑；無解品質留空，失敗不刪。來源與模型／資料雜湊、全部資格預測、α、pooled及truth重新載入驗收；直接從X檢查box容差1e−8，重載容差1e−7。

```json
{
  "passed": true,
  "errors": [],
  "coverage": {
    "duplicate_keys": [],
    "missing_keys": [],
    "unexpected_keys": [],
    "complete": true
  },
  "expected": 450,
  "observed": 450,
  "valid": 450,
  "max_alpha_reload_diff": 0.0,
  "max_pooled_reload_diff": 0.0,
  "max_blind_prediction_reload_diff": 0.0,
  "reuse_pairs_checked": 90
}
```

# 欄位定義

**scenario／seed／method**識別情境、獨立seed與方法；draw=-1沿用原資格資料，blind_n=20、target=0。

**status／X／max_box_violation**為執行狀態、二維交付解及超出box的最大距離。**alpha**按IDs1、2、20依序列出非負且總和1的權重，不是付款比例。**mse**是三人的原20筆平均平方誤差。

**pooled_error／truth_error**為交付X的兩種目標絕對誤差，pooled是主要指標，truth只是診斷。平均、中位、Q1、Q3分別是有效解上的平均、50%、25%、75%分位；失敗另列全部執行分母。

**initialization_seconds／stage3_seconds／scoring_seconds**為初始化、求解、事後評分秒數。**requests**是API呼叫數，批次仍計一次；**input_samples**另外計輸入點數。共用訓練／資格成本每seed保存一次，不重複加在每方法。

**fold_audits**保存每次train／heldout索引、risk_mse、risk_weight、λ、weight和heldout_losses。risk_mse是該16筆內bootstrap MSE第90百分位；risk_weight是倒數正規化q；weight是該折收縮後權重。lambda_mean是100折的平均，fits100不是獨立seed數。

**配對平均差／n／CI／衝突**是新方法減基準、雙方有效seed數、seed重抽樣信賴區間、兩評估來源方向不同的seed數。

# 範圍與後續

證據只涵蓋本輪二維、target0、固定三人與三情境。不得依此宣稱所有目標、高維或真實資料都更好；沒有穩定優勢時保留A/B和原保守品質基準。

方法架構參考[Wachslander與Setzer（2025）](https://link.springer.com/chapter/10.1007/978-3-031-92575-7_6)的留出收縮與權重平均想法；風險分位與重複五折為本研究改編，不是原論文完整重現或效果保證。

本地完整資料：alpha_weight_outputs/conservative_shrink_formal_v1_20261010；登錄：docs/conservative_shrink_preregistration_20261010.json。原輪次結果不覆蓋、不合併。

# 完整重播與資料分離驗收

49項回歸測試通過。全部正式新方法與不收縮方法的每折索引核對：train16與heldout4不重疊，每列恰被留出20次；同fold兩方法q完全一致。fresh API與cache實際α、X及Stage3呼叫數一致。

完整重播seed4800的三情境五方法，共15條件。重新生成資料切分、local與pooled模型、資格、proposal、離散solver決策、α、X與評分，最大差異0.0；時間不要求一致。

# 三情境等權平均（描述性）

以下各情境權重1/3，只有觀察平均排序，不是新的正式確認比較；不能把同seed三情境當90個獨立seeds。

<table header-row="true" fit-page-width="true">
<tr><td>方法</td><td>三情境等權平均pooled</td></tr>
<tr><td>A：均勻</td><td>0.01978168</td></tr>
<tr><td>B：品質反比</td><td>0.01899922</td></tr>
<tr><td>原保守品質</td><td>0.01851782</td></tr>
<tr><td>切分保守品質（不收縮）</td><td>0.01849418</td></tr>
<tr><td>新保守品質＋留出收縮</td><td>0.01815192</td></tr>
</table>

# 評估者target附近涵蓋

<table header-row="true" fit-page-width="true">
<tr><td>情境</td><td>target±0.25平均點數</td><td>最低點數</td><td>少於30的seeds</td><td>附近truth MSE平均</td></tr>
<tr><td>乾淨</td><td>258.50000000</td><td>223</td><td>0</td><td>0.00012191</td></tr>
<tr><td>低品質</td><td>258.50000000</td><td>223</td><td>0</td><td>0.00012191</td></tr>
<tr><td>樣本量不平衡</td><td>265.06666667</td><td>239</td><td>0</td><td>0.00010391</td></tr>
</table>

少於30只是固定稀疏標記，不是已驗證可靠性門檻；點數不能保證評估者準確，也不能把稀疏直接稱為OOD。

# 精簡CSV補充欄位

下載保存全部450條件的精簡列；每折索引及完整risk/λ audit留在本地原始results.jsonl。lambda_mean、fits、risk_quantile、shrink_enabled與fallback_reason取自alpha_audit；不適用方法的值留空。fits是估計折數，shrink_enabled是否啟用收縮，fallback_reason是權重退回來源原因，不能與solver失敗混用。