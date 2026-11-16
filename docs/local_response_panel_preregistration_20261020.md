# 本次局部回應面板：開發登錄

2026-10-20。前9族204個global-free候選均未通過；在地32點神經模型殘差線性校正也失敗。另做的truth診斷不是新估計器，也不把truth當分數；所有歷史負面與確認失敗保留。

## 新方法與新增資料假設

提供者在自己本次已存在的觀測(X, observed_y)上回傳局部回應摘要，不依賴歷史LOO或其他任務資料。研究容器按agent_ids隔離原本client資料，只存取X、observed_y、agent_ids，不讀clean_y、poison flags或pooled預測。本族使用client資料回饋，不能混稱只從原搜尋statistics免費配分；部署API與隱私保證尚未實作。

固定以既有Stage1 mse最小3個提供者作評分面板（MSE同分agent_id排序）；它不改solver alpha或coalition。每client在正常X_full最近128個自己的觀測上作加權局部多項式回歸，基底包含intercept、各維線性；degree2另含所有平方與兩兩交互。這是一般局部回歸，不從truth公式挑係數或歷史LOO訓練。degree1、degree2均事前固定，不能因結果再改階數。

h=第128近點距離、至少1e-6，phi以(X-X_full)/h標準化，w=exp(-dist²/(2h²))。ridge=1e-6×斜率Gram平均對角，intercept不懲罰。若client少於128觀測、Gram非有限或資料不足6倍基底維數，留資格失敗／棄權，不少算不利人次。保存coefficients、h、樣本索引、條件數；不讀原模型權重或pooled模型來擬合。

各proxy端點由原已鎖定四設定產生，不用新回歸模型選點或求解。每個局部摘要評分端點，聚合三個預測後用四種Q：abs_mean_signed、abs_median_signed、mean_abs、median_abs。score=Q(proxy端點)-Q(正常解)。2degree×4Q×4proxy=32候選，完整揭露。沒有新的神經模型查詢或solver，但每案3×128=384個client觀測處理、回歸、摘要傳輸成本必須列出，不稱零成本。

## 驗收與獨立資料

開發只用最初20個舊團隊（2026101610–19）。已揭露的40新團隊現在屬已用診斷資料，不能再作本方法獨立確認。主比較仍原custom30完整參與者LOO pooled品質，不改reference或符號。

分數先鎖定再讀LOO。團隊內Spearman平均、seed bootstrap10000/RNG20261020；兩情境都>0.5、有效覆蓋>=90%才可選一個候選。最小情境相關最大，平手5格、linear proxy、degree1、Q名稱順序。開發最多處理8000個client觀測、零額外神經模型samples、300秒。所有方法、先前204候選及4個offline診斷全數保留。

若有通過者，先凍結單一方法與代理設定，再新建全新seeds2026102300–2026102319×兩情境作40團隊獨立確認，不用上輪新資料挑公式。確認兩情境平均>0.5、95%下界>0、覆蓋>=90%；額外資料需求、模型calls、樣本、CPU、傳輸、同normal+LOO範圍端到端成本完整報告。<=70%模型查詢樣本的門檻只屬一項成本，不掩蓋client資料掃描及擬合成本。

不更新alpha、不pruning、不付款、不Shapley、不Notion。不將2D局部多項式結果外推為複雜高維場景已有效；若只有degree2通過，這個模型假設與樣本維度限制必須列為實質限制。
