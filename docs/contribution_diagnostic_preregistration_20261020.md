# 貢獻差距診斷：執行登錄

日期2026-10-20。使用者授權「依照新的診斷建議去做調整並比較」。沿用`contribution_diagnostic_next_steps_20261020.md`，凍結後完整執行兩階段，不依第一階段結果挑案例或改既有A。

固定20個探索團隊（10保存seed×Clean／污染）、80人次；原始來源為`custom_direction_experiment_v2_20261020`，A為`local_anchor_step_experiment_v1_20261020`鎖定分數。pilot不併入，不新增資料或訓練。五個量為A、C_anchor,L、C_anchor,E、C_full,L、C_full,E；全10種配對均固定報告，不挑最好的相關當方法成功。

第一階段：所有80個已產生的完整移除解，以原完整團隊所有模型與原alpha評L_S；不使用移除檔的剩餘團隊result.loss。baseline原L已保存，不額外重評。預算340模型samples，0solver，保存各原模型回覆與C_full,L；原C_full,E可重用。

第二階段：80次全程錨點移除，custom30、原X0、全部原模型／alpha／目標／box保留，只刪一個I_i。原方向權重在本批uniform，因此原solver在候選數不同時使用uniform剩餘方向，等於刪除後正規化；實作先assert本批權重uniform，不能默認泛化至其他alpha。使用原HostServer.optimize_candidate_subspace(strategy=custom)，不改solver。保存剩餘候選、全部評分者、原alpha、結果／歷程／停止訊息、API計數／查詢hash及最終既有回覆，無額外endpoint查詢。總上限80solves、200000samples、300秒solver；技術或預算失敗保存、不換seed補成功。

階段一輸出先hash鎖定；階段二全部解再hash鎖定，才離線載入pooled評已產生解，原pooled baseline／完整移除品質重用。pooled不進方向、接受、步長、alpha或刪人。truth不參與。原A、舊結果與歷史8來源均不改。

分析：每團隊Spearman、有效率、各分數正／近零／負（delta1e-4）及配對方向一致。A若全tie記未定義；四種C的極差<=1e-4記near-flat，凡涉及它的主配對相關為null，原始非定值rank只作診斷另存。每情境10seed，bootstrap10000次、RNG20261020、探索95%CI；未定義不補0，重抽無有效團隊另計。

固定附加差異對照：A對full,L與A對full,E；A對anchor,L與A對full,L；A對anchor,L與A對anchor,E。只在兩配對都有效的同團隊內相減，報common n、paired探索CI，不用不同有效子樣本的均值差解釋因素影響。

本輪交付完整矩陣及限制，不設定「必須找到主因」或相關過線條件，不調權重／正負／閾值／seed。診斷可定位差異，但不能證明原因唯一、可相加因果分解或參與者品質估計器已有效。不存在新確認或部署、alpha研究、pruning、付款、Shapley、Notion。

驗收：單元測試全模型固定評分、只刪候選仍保留全部模型／alpha、原X0／custom30不變、非uniform拒絕、delta近零／undefined處理。獨立驗收從保存逐模型回覆重算四種C，重載pooled驗證，不新問參與者模型；檢查角色／權重／起點與hash、矩陣／paired CI／成本。所有新輸出exclusive create、獨立目錄。
