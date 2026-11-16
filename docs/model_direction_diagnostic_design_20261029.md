# 封存方向資料純離線診斷事前設計

2026-10-29；全部78案均已讀探索，沒有新保留資料。輸入固定為 attempt_02 封存；不呼叫 solver、模型、reference evaluator，不變更 alpha、錨點池、主公式或歷史檔。

1. 精確觀測採白名單：fixture 的 models/weights/owners/anchors/anchor_weights/start/iterations，以及 iterations/full_result/full/branches/directions/rows/probe/scale/events。剔除 reference/id/category/split、成本、cache context 等執行註記；events 完整保留（本封存無context/id）。JSON 字典排序、有限float以 hex 表示，整數保留型別；不四捨五入或容差合組。保存每案 canonical 觀測與SHA256。比對所有相同觀測組、逐owner封存累積D符號（1e-12近零），保存所有配對差異。另對正確/錯誤少數及有益/錯誤制衡名稱配對輸出完整逐欄差異，即使未能精確同組也誠實保留。
2. 幾何key為實際point、h（探測距離）、unit的精確canonical值；同一round固定base。不依g相等去重；不同距離割線不合併；切線原anchor distance只屬來源，若實際point/h/unit完全相同則去重。完全幾何相同卻g不同則失敗。U維持最近同場peer斜率向量RMS；只移除重複欄。逐一原方向將該欄總共出現1/3/9次，另測全部欄同倍複製及倒序。保存原U與去重U、全部誤差及排序（含平手），驗證去重後精確不變。浮點計算比較額外記錄1e-12容差，精確不變由排序key後運算保證。
3. N只沿用固定I與Ub=(.1U)/(1+.1U)，離線P+N beta=.2對照：P/路徑/owner固定、500單位、400/100分帳；沿用既有allocation函數且不執行搜尋。保存每案帳務。只描述本批探索性變動，不能推論品質改善、抗身份複製、一般理論不可能或部署有效性。
4. 新隔離程式、單元測試、輸出exclusive mkdir與檔案x模式。輸入所有封存檔及程式/設計hash在執行前記錄，完成receipt再次核對。輸出完整canonical觀測、配對差異、group-owner診斷、每輪幾何重複與複製控制、分配與摘要；成本列0新solver/0新模型/0新reference計算，另列離線時間與bytes。主代理另行審核。

## 首次離線執行後技術修訂

首run採anchor distance作key，未正確合併相同切線probe，首輸出保留作技術attempt，不作最終結果。修正key移除原anchor distance，只保留實際point/h/unit，再以exclusive attempt_02重跑；無新樣本、查詢或solver。補有限值與空集合邊界測試。
