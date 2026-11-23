# 非零目標的角色介入實驗

執行前固定：20個既有神經模型團隊（clean10隊各3人、controlled_poison10隊各5人），targets=[0,-0.5,0.5,2.7]，共80任務。重用歷史模型是配對新目標擴充，不是新模型獨立確認；不混入其他解析隊伍。

- D=2、box[-1,1]。T=-.5/.5可用(T,0)作解析可達見證；T=2.7可用(t,t)，t=(-2+sqrt(4+4T))/2，接近全域最大3（x=(1,1)），代表近上界輸出任務。見證只離線驗證，不提供給搜尋或反推。
- 固定歷史成員與均勻alpha，不重做Stage0／招募，污染成員為強制納入壓力測試。不能解讀為部署資格失效。
- 對每個T、每個成員重新執行1起點500步Adam .01的本地輸入反推；同一成員跨T共用隨機起點seed。初始及每步限制box，所有候選有限可行才進入；失敗保留，不另抽人。
- 重新以錨點平均作共同起點，原30輪custom求解。重用維護中的 frozen AST trace（不得改寫快照），server.target_T明確設定；Replies動態使用目標。
- CA：逐輪從full pre狀態減去owner錨點方向H_i，full模型目標與規則不變。CM：保留錨點、移除模型，重建pre的anchor/current losses後執行一輪。J：移除模型及其錨點，重建pre狀態後一輪。Full LOO：移除兩角色，從相同full錨點平均起點重跑30輪；此共同起點保留full錨點資訊，不是重新招募或反推的完整系統刪除。
- CA/CM/J按逐輪有號效果加總，F與Fminus重新評分同端點；真值只離線評估最終決策與whole-LOO，不參與求解。參照是指定whole-task LOO，不是公平真值。
- 錨點tau=max(1e-8,sum rho)，rho=max(1e-12,32 epsmach(1+absFalt+absFfull))；模型／J容差1e-10。兩評分器使用同一數值門檻。正分後normalize，無正分全額reserve；TV含reserve。沒有實際付款。
- 每scenario各10隊等權，200000次隊bootstrap、rng20261123。報有效n／null／zero/tie、sign flips、peer/full TV與各局部分數對truth whole-LOO的排名／符號一致。各T相對新T0作同隊配對描述95%區間，不宣稱多重推論或等效。
- 每次反推／完整與局部分支皆新計算，保存新端點與模型回覆、cache hits／misses。共用每任務獨立cache只作研究計量，不把其cost說成各方法獨立成本。來源hash、失敗與wall time保存；輸出拒覆寫。
