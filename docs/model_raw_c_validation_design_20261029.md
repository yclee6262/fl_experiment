# 固定完整內部loss模型池：raw C多輪與尺度驗證設計

日期2026-10-29。使用者明確定義品質為降低原完整團隊內部loss；本輪不研究自評操弄。主代理規劃與判讀，GPT-6.1 Sol子代理實作執行。全部既有78案均已讀探索，沒有新保留樣本，不回填任何舊確認結果。先固定本設計再計算。

## 1. 主要定義及來源

來源 `contribution_analysis/model_direction_pool_experiment_v1_20261029_attempt_02/`。78案、157個原輪、441 owner-round，其中16案實際多輪。從原events建立(model,exact point)→loss字典，核對重複回覆一致，取得全模型對full/minus接受端點的loss；缺漏held，不重新呼叫model或solver，不插入reference、pooled或truth。

F(x)=sum_j v_j loss_j(x)，v使用原權重正規化，兩個端點使用同一組完整模型與v。C_it=F(minus_i)-F(full)，不排除受評者、不除模型尺度。逐輪有號累積S_i=sum_t C_it。保留直接F差與加權逐模型差兩種算法核對，但都只用原始loss。固定epsilon_round=1e-10；累積tau=max(1e-8,T*1e-10)。同一共同尺計算一致性不代表外部品質或full LOO有效性。

## 2. 分類與帳務

每個owner保留每輪C、正向總量Gplus=sum max(C,0)、負向量Gminus=sum max(-C,0)、net、實際T，並列round正負零數。累積按±tau分正/負/近零。若Gplus及Gminus都>tau，記抵銷；區分抵銷後仍正／近零／負。不得將無變化稱永久無價值。

所有round近零時，再分全部full/minus端點距離<=1e-10（本路徑冗餘）或存在不同端點但共同F近同（等值結果）；有顯著正負但net近零為抵銷；其餘微量累積在死區單列。負C表示移除後原固定完整F反而更低，可伴隨原full仍比起點改善，兩者不矛盾。

模型池每案500研究單位，先有號累積再對S>tau分配，最大餘數整數法owner ID打破平手。全員無正則reserved，缺owner×round或非有限資料held且amount/reserve=null。不作實際支付。

## 3. 固定補充對照

U採同輪實際probe(point,h,unit)精確去重後最近鄰斜率RMS；依既有診斷模組，拒絕相同key不同g，不以相同g合併不同幾何。Ub=.1U/(1+.1U)，I=norm(full-minus)/pre.eta，Ib=I/(1+I)，N=Ub*Ib。

比較C-only及C+Ub、C+Ib、C+N，beta固定.1/.2/.3，主.2。主額度(1-beta)*500按可靠正net C，補充額度beta*500按跨輪Ub/Ib/N累積；無正額度不互轉。U/I/N表示不同面向作用，不宣稱內部改善量。

指標：C正／負／零owner領獎數與金額、整池保留與部分保留、補充是否為net負作用提供獎金、是否只獎勵有位移但等值的結果；同時列多輪子集、全部、舊46，原new_heldout標籤只用來源識別，不能稱本輪新保留。品質相對C的符號一致是定義檢查，不是獨立有效性證明。

## 4. 尺度敏感度：固定端點

三組factor=.1/10/100，不重跑搜尋：

1. 全模型loss同乘factor、v固定：理論C等比例，正分相對比例不變；絕對tau可能影響極小值資格，逐一記錄例外。
2. 單模型loss乘factor、v固定：這改變完整目標F中相對重要性，預期可能改符號或分配；當成偏好／量尺敏感度，不判為算錯，也不當原F之下的品質證據。
3. 單模型loss乘factor、配合v'_j proportional v_j/a_j：F'=F/Z，Z=sum v_j/a_j。理論共同F僅整體縮放，分配應保持，仍檢查tau邊界。此為離線單位換算控制，不是建議部署更新alpha／權重，不改原搜尋。

保留變換後的原始endpoint loss與v／Z／C，核對預期比例；U與s同尺度換算時g不變，I端點固定，但主報尺度指標以C-only為準。另人工微量閾值例子揭露絕對tau的單位敏感性，不為通過而改門檻。

## 5. 驗收、成本與停止

先核對封存receipt、鎖sources及設計。raw C舊46須對照歷史peer實驗raw方法之分數/整數分配（允許數值容差1e-10+1e-9*abs），原40分配/6保留若不符先查原因。獨立Fraction由原loss重算C及整數分配，完整矩陣/非有限/缺漏held、正負抵銷、預算守恆、無正保留、閾值例外單元測試。

純離線最多200MB/120秒，每次exclusive-create；保留失敗與修正版本。0新模型/solver/reference查詢。沒有外部D門檻；若補充獎勵net負C，明確報告與「只獎勵降低F」目標的衝突，不調beta或換案例掩蓋。多輪正分與保留率為探索證據，不能推廣實務。依結果決定是否保留C-only為主候選、U/I/N作解釋，不自動啟動新實驗或改主算法。
