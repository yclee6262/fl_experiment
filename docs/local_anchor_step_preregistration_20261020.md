# 單步錨點增益累積v1：執行登錄

日期2026-10-20。使用者授權依`local_anchor_step_value_proposal_20261020.md`實作、實驗及比較成本／效果。本文件於新增位置的模型評估前凍結。

主方法J與A完全沿用提案：只對原custom已接受步，固定原位置、原接受距離、原模式及全部評分模型／alpha，拿掉一個錨點分量並正規化剩餘方向權重。剩餘norm<=1e-8以不動處理；有效替代loss=min(原位置loss,替代試走loss)；J=有效替代loss-原接受loss；A=sum J，signed原樣保留。走出box標未完成，不縮步／裁切／填0。A不再乘步長、不擬合、不正規化為總和1。

資料固定為`custom_direction_experiment_v2_20261020`保存的4pilot及20探索baseline，保留模型、所有原查詢、固定X0及原custom30。只從這些未含pooled標籤的baseline讀取方向及既有回覆。每迭代以模型可見float32位置作cache，先讀該迭代current／direction／trial的完整逐模型loss；相同新查詢可重用；缺回覆或cache矛盾則停止。新位置使用原scalar預測API逐模型評估，避免以batch改變數值或呼叫口徑；calls與samples分開列。全部24案A及原始回覆鎖定後才讀既有custom完整移除品質／上一版S作比較，不載入pooled或truth作J。

主效果：20舊探索團隊、10seed×兩情境、80人次；pilot16人次另列。每團隊Spearman(A,C_custom_full)及每情境平均、分布、10,000次seed bootstrap探索95%CI，RNG20261020。C極差<=1e-4或A全tie記未定義；未完成的團隊仍入覆蓋率分母，不刪個人救相關。固定比較為上一版S及alpha（uniform全tie記未定義）。次要列signed方向一致（delta=1e-4）、near-zero、負值計數、未校準MAE／RMSE，明示兩者尺與反事實不同。與S的相關改善配對區間屬探索，不能替代主門檻。

效果進階條件沿提案：兩情境有效團隊覆蓋>=90%、平均團隊內Spearman>=0.5且95%CI下界>0。成本條件為共同baseline加本方法輸入評估量至少比baseline加全LOO省50%，實測運算時間亦更低。不再沿用上一輪baseline只增10%成本門檻。任一未過即停止，不改分數、步長、閾值或追加seed。

成本實驗分三層：

1. 24案首次算分：逐案測模型載入、讀紀錄、核心方向／查詢／累積、磁碟保存及端到端時間；存輸入／輸出artifact bytes及process RSS高水位。核心計時不開tracemalloc。
2. 同來源冷啟動成本對照：以20案當時保存的baseline與80次LOO compute/calls/samples，加上本輪實測額外核心成本，另列歷史／本輪時間的比較限制，不把讀取既有LOO的0成本當其算法成本。
3. 4pilot同機配對實測：兩條pipeline均含模型反序列化／節點建立、正常求解；本方法另含只讀觀察器與單步評估，全LOO另含所有人移除求解並重用已載入模型。均不包含pooled、磁碟報告或訓練。每案每方法2次warmup、15組交替順序計時；首個warmup另量Python tracemalloc峰值，不納入中位數計時；所有warmup工作另記。每條pipeline內確認baseline與已鎖定原解／查詢相同，本方法A與首次算分相同，LOO解與原custom參考相同。不得靠偏換後端、batch或少記錄求成本優勢。

配對實測預定408個custom求解（兩clean案例各17×(1+4)，兩污染案例各17×(1+6)），只用4個舊模型案例，無訓練或新資料。首次算分及配對合計上限450個新custom求解、500,000模型輸入samples、300秒core compute、900秒執行時間；超限或技術不一致立即保存並停止。成本／效果共同在固定舊資料工作完成後判斷，未過不進新確認。

實作先測：加權消除／方向正規化、接受與拒絕、無方向、邊界不比較、skip anchor、float32 cache、signed累積與缺資料失敗。獨立驗收從保存回覆及原方向重建替代點、有效loss、J／A、排行及成本，pooled參考只作離線配對。原歷史演算法、正式結果均不改；不做alpha初始化／選擇／更新、pruning、付款、Shapley、Notion或新商模。
