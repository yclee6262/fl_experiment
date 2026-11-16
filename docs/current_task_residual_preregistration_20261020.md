# 本次任務的在地殘差回饋：開發登錄

2026-10-20。前8族172個global-free候選未在兩情境同時超過0.5；不再只組合相同預測統計。本族加入明確的新資訊假設：模型提供者可以在**自己的本次既有觀測資料**上計算局部預測殘差摘要。它不需要歷史LOO標籤，也不需要複製其他任務資料；但不再是僅由正常搜尋統計零額外資訊估計。應把這個邊界明列，不能假裝已建置部署API或隱私保證。

## 資料界線

仍不讀pooled預測、clean_y、truth、is_poisoned或LOO作分數。研究容器模擬各client持有資料：只從原artifact pooled_data.npz讀X、observed_y、agent_ids，按agent_id隔離自身資料。host只需得到每client局部殘差摘要／校正預測，正式部署接口尚未建置。觀測y是原client本次任務既有測量，含雜訊或污染，不是真值oracle；新假設可能無法在沒有觀測資料回饋的部署使用。

## 固定方法與成本

沿用原MSE最低3個local models（只用既有phase1_report的mse、agent_id；不更新alpha），每client選距正常X_full最近的32個自身觀測位置。原model逐點回覆，r=observed_y-f(X)。不跨client共享原資料、不新增量測，不查外部歷史資料。用dist第32近點作h、h至少1e-6，權重exp(-dist²/(2h²))。

兩固定校正：加權殘差平均（intercept）；局部線性phi=[1,(X-X_full)/h]，解加權ridge（斜率ridge=1e-3×斜率Gram平均對角，intercept不懲罰）。不依效果調ridge、鄰居數或選模型。端點改正預測f(z)+phi(z)beta；不改solver，也不把beta當貢獻。

各校正四種品質Q：abs(mean(signed corrected f))、abs(median(signed corrected f))、mean(abs(corrected f))、median(abs(corrected f))。score_i=Q(proxy endpoint)-Q(X_full)。2校正×4Q×4既有proxy設定=32候選，全報告。新增模型samples至多每案3×32＋3×(4m+1)，兩proxy解析度的成本各自分列；開發上限10000 samples、0真實solver、300秒。

另保留校正矩陣條件數、端點距離/h、模型IDs、樣本索引與原始觀測殘差、hash供獨立驗證。不用外插案例消失製造高相關；不可用clean_y或pooled訓練殘差。資料不足32或非有限輸出留失敗／棄權，不補0。

## 開發、確認與驗收

同20舊開發團隊，分數先鎖定再讀LOO；團隊內Spearman平均、seed bootstrap10000/RNG20261020。兩情境均>0.5、覆蓋>=90%可選；最小情境相關最大，平手5格、linear代理、intercept校正、品質名稱排序。

若通過，在新40團隊任何LOO／pooled效果揭露前凍結單一方法，才可獨立確認；本次新資料已生成但未揭露品質，不能查看其效果再修改。確認兩情境點估計>0.5、95%下界>0、覆蓋>=90%，全成本與<=70%查詢門檻另驗收。結果即使通過，仍是本次局部觀測回饋＋代理LOO模擬的近似，不是嚴格非LOO，也不是只累積原錨點單步statistics的原構想。
