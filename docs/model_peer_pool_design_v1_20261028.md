# 模型池：排除自評的加權平均 v1 實驗設計

日期2026-10-28。使用者授權實作並驗證效果。主代理規劃與判讀，GPT-6.1 Sol負責實作及實驗執行。新目錄隔離；不改歷史solver、alpha、錨點池、舊輸出、Notion或實際付款。本設計在新研究執行前凍結。

## 1. 固定算法與分配契約

每原輪t針對owner i，保持全部錨點、a、起點、模式、初始eta及原接受規則，只在minus分支移除i所有模型評分，剩餘v重新正規化。full/minus均比較已接受位置；無更新用原位置。各owner的full必須一致。minus不沿新路徑繼續，非完整LOO。

先鎖共同25probe=[−1,−.5,0,.5,1]²及模型版本，s_m=Q90(loss_m)−Q10(loss_m)，線性分位數。原搜尋不使用s。端點鎖定後d_mit=(loss_m(minus_i)−loss_m(full))/s_m。

主候選P_it=Σ_{m不屬於i}v_m*d_mit / Σ_{m不屬於i}v_m。排除的是owner的所有模型；按剩餘原v作加權平均，不是總和或owner等票。S_i=Σ_t P_it，門檻tau=max(1e−8,n*1e−10)，w_i=S_i若S_i>tau否則0，q=w/Σw。固定每案500研究單位，最大餘數取整、同分owner字典序。只分此模型池，基本保障不因負值扣減。

沿用共識池完整名冊／輪次／版本／端點一致驗證：缺槽／重複／無minus／非法loss／剩餘權重0／正權重尺度<=1e−12等整池held，q/amounts=null；全無可靠正值reserved，q=null、amounts=0、預算全保留。零原權重模型不評分，各參與owner須具正總權重。技術上可重用共識池validate/settle但不改原檔；核心及reference必須明確以peer而非min累積。

對照：原raw C（未標準化）、完整團隊標準化C、已實作min共識保守分數；四者各自有號累積、相同500單位門檻／取整規則。尺度不同影響門檻可比性，須揭露，不能把分配差全歸因品質。此輪一個情境就是一輪獨立任務，不把不同任務串成多輪來宣稱完整路徑累積有效。

## 2. 資料及角色

A. 原model_decision_validity_v1的30個已讀2模型解析設定，補齊兩位owner各自minus（60個owner作用）。它們全部為舊開發診斷，連原heldout_parameter也不可再稱未讀保留。

B. 16個事前固定新3模型設定，四類各4案，前2開發、後2參數保留；不是新實務獨立樣本。固定poly(b,q)=1000+b*x0+q*x0²，start=(0,0)、anchors=(1,0),(−1/4,0)、a=(.5,.5)、eta=.1，box[−1,1]²。

b依序=3/5,17/20,11/20,9/10；q依序=6/5,12/5,8/5,14/5。
- truthful_minority：模型poly(1,0),poly(−b,0),poly(−b,0)，v=(3/4,1/8,1/8)，reference=poly(1,0)。
- majority_correct：與上類同可觀測資料，reference=poly(−1,0)。此配對中多數模型的方向與診斷答案一致；報告仍須直接核對D。
- useful_restraint：模型poly(1,2q),poly(−1,0),poly(−1,0)，v=(1/2,1/4,1/4)，reference=poly(0,q)。
- false_restraint：與上類同可觀測資料，reference=poly(−1,0)。
各模型對應不同owner，48個owner作用；另用單元測試驗多模型同owner的排除，不能把3模型例當多人多模型全部情境已覆蓋。

總共46任務、108個owner比較。先鎖全部fixtures、sources、probe，再執行舊30與新開發8並封存checkpoint，最後新保留8；不依中途結果調公式。所有owner分支及共同評分保存raw events。可沿用T3 execute每target原控制/full/minus共3求解，因此最多324次單輪求解；full重跑屬研究對照成本，不冒稱最佳部署成本。

## 3. 技術與有效性驗收

原solver控制與觀察full查詢／端點一致；每owner的full相同；共同loss只在端點鎖定後取得；minus不能呼叫被移除模型。固定v/a不得混淆。獨立Fraction按raw endpoint／probe重算P、各對照、S、q和取整。獨立解析單輪驗證可使用沿x0的一維精確多項式：g=linear+.375*quadratic；至多10次半步嚴格下降判斷；檢查兩分支g、位置、每trial接受及loss。不得使用production結果填expected sign或更改舊T3 expected字段來假裝通過原驗證。

技術測試含缺失／held、負分有號抵銷、全零reserved、單一positive取得整池、確切門檻與同分、owner多模型全排除、排序、同owner等價拆分、nonfinite、尺度退化。

固定端點尺度變體：每任務每模型probe與所有端點loss一致乘.1/10/100，共324個變體，核對peer各owner分數（atol1e−10+rtol1e−9）及可靠正負資格；付款比例容差1e−8，整數差異如在理論同分邊界須列出，不掩蓋float輸入換算造成的tie。不得因此修改門檻或重搜。own-endpoint-only控制：只改受評者自己的minus loss、不改其他回覆，該owner的P須不變，不能把此主張擴大成整個q不可操弄。

另做一個不跑solver的「間接分配」帳務反例：2owner、probe尺度1、full loss各1，minus_a與minus_b原loss各1.1，peer兩人+.1分50/50；僅把評價a在minus_b的loss改成.9，其餘不改，a自己P不變但b變負，a比例升到100%。這是假設可以任意回覆的威脅示範，不是證明鎖定模型能通過准入作弊。要明說排除自評仍無法保證分配比例對自身評分無依賴。

品質只在scores與shadow settlement封存後用D=reference(minus_i)−reference(full)離線診斷。reference不傳給計分器，不用於排序分配，不引入pooled。報告每方法每類／每資料階段符號符合（±1e−10）、有益負分／歸零、有害正分、正score及正amount對D的precision、所有owner覆蓋、保留預算率、發給D<−1e−10的單位比例。零D得到獎金另列，不混作有害。

必要品質gate：所有新保留案例內非零D作用符號皆符合，零D近零，且無D<−deadzone者領取研究單位；任一反例不通過就不主張可部署品質分潤。這是反例篩選的必要條件，非代表性準確率驗證。品質失敗保存完整結果；不調評價權重、不直接換真實團隊大實驗。內部peer分數仍可作他評作用指標。

## 4. 成本與封存

新主研究上限324單輪求解、30000搜尋／共同評分API、2700probe、180秒、100MB。測試／獨立核對另列；禁止新神經模型訓練或推論，pooled／實際付款0。probe必先鎖定；記錄查詢量、各階段成本、校準及診斷解析運算，勿只報牆鐘或說低成本已成立。

來源與舊receipt事前核對；產物exclusive-create，失敗保留，必要工程修正另開新attempt，不改原失敗輸出。新程式 model_peer_pool_v1/，新結果 model_peer_pool_experiment_v1_20261028/，報告docs/model_peer_pool_results_20261028.md。主代理獨立審查並更新研究固定入口；不改舊Notion。
