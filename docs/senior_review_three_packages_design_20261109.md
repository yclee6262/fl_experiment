# 學長審閱：三包既有資料驗證規約

2026-11-09。先登錄再分析；本輪是已使用歷史20團隊的補充診斷，不是獨立確認。

資料：10 clean＋10 controlled_poison，原30輪路徑。當前CA/CM採method5_cost_experiment_20261030_attempt_04的端點與逐模型loss；J採combined_contribution_layer2_experiment_v1_20261021的joint端點／common_losses；完整參與者LOO採同cost attempt的LOO端點。核對baseline、輪次、owner、模型與固定起點，不把不同CA版本直接合併。CA當前方向扣除不重配其餘錨點權重；J同時移除錨點／評分且重配剩餘權重，故交互殘差也包含規則差異。另列歷史對齊A2/M2/J的殘差，以分開語義。

第一包：固定端點後計算F、Fminus_i與E（凍結pooled絕對目標誤差）。Fminus排除自身、剩餘alpha正規化；self+peer=F，Fminus=peer/(1-alpha_i)。先保存端點與來源hash再載pooled；獨立重載pooled重評全部唯一點。檢查累積符號翻轉、每團隊排名、各池分配差距。alpha關係僅做可識別的描述；各隊內alpha常數時明列不可估，不以跨不同n的相關冒充權重影響。

第二包：CA、CM、CA+CM、J的三尺累積分數，各對完整LOO的同尺與pooled D，報每團隊Spearman、Kendall tau-b、三態符號一致及含同分top1/top2 Jaccard。團隊內近常數ptp<=1e-10時排名為null，coverage另列；符號容差1e-10，另以1e-4作歷史口徑敏感度。場景內團隊等權，無逐輪獨立樣本假設。交互殘差J−CA−CM另列符號、標準化絕對殘差及最大案例；歷史對齊角色單獨列。

第三包：當前無N版本影子分配，完整服務為分析假設，不聲稱真實履約事件已驗證。每人基本費100測試單位、B0=100n，BA=BM=500；各池獨立。CA可靠性tauA=max(1e-8,sum rho)，模型tauM=1e-10，累積後截正。基本費僅在SA<−tauA且SM<−tauM時取消；零／近零不取消。最大餘數法取整，按原owner字串排序決定同分，金額＋保留額需守恆。不發生真實付款。

主分配只用完整F的CA、CM；排除自身／pooled只作離線反事實診斷，不能回填部署付款。附模型tau=max(1e-8,T*1e-10)敏感度及原80/20 CM+N歷史對照。污染身份以模型artifact metadata.agent_data_profile.is_poisoned為準，不用分數反推標記。報三池金額、比例、基本費排除數與N領取條件；不預設污染者必為0，服務費與品質獎金解讀分開。

成本上限：不跑搜尋、不新增參與者評分、不訓練，pooled最多20,000次（含重評），10分鐘。缺資料或對齊／重評失敗停止，保留attempt；原檔hash結尾需不變。所有分析另存新目錄與報告，不覆蓋舊結果、不修改Notion。

驗收：1033角色輪與80名團隊成員的範圍以原資料核對，來源分數復原、baseline及端點對齊、self/peer重建、完整LOO復原、模型身份覆蓋、pooled重評零差、三池非負守恆；alpha不可識別不是實驗失敗。結果描述範圍及未建立有效性需明列，不因排名／付款可算就宣告公平或LOO替代性。
