# 自適應共享代理＋在地二次評分：獨立確認登錄

2026-10-20。開發唯一通過候選固定為adaptive共享探測、最終RBF代理、degree2_median_abs。開發10seed／20團隊，Clean0.60、污染0.56；只是從260個global-free開發候選（含代數重複）中選出的版本，不宣稱開發有效性。

## 不得更換的單一方法

1. 原custom30正常搜尋只記錄signed回覆；原X0、box、target0、uniform alpha不改。
2. 初始5×5共同格點；3輪以linear與RBF分歧、最多每輪12位置共同探測，精確選點規則沿用adaptive_shared_response_preregistration_20261020.md；全部原成員模型在共同位置回覆。
3. 最終同一庫，以RBF（thin_plate_spline、degree1、neighbors32、smoothing1e-10）逐人模擬刪模型、候選、alpha並正規化，custom30。這是便宜代理上模擬LOO，非嚴格非LOO。
4. 本次既有Stage1 mse最低3個提供者（同分agent_id排序），各以原正常X_full最近128個自己的observed資料，回傳degree2局部回歸摘要。基底intercept、線性、平方與兩兩交互；h、Gaussian weights、ridge=1e-6的規則沿用local_response_panel_preregistration_20261020.md。只讀X/observed_y/agent_ids，禁止clean_y、poison flags、truth、pooled作擬合；不改solver alpha。
5. Q(x)=median_j abs(local_quadratic_j(x))，A_i=Q(proxy_without_i)-Q(X_full)。原單位與正負保留，不做排名共識或翻號。3個提供者是新增client資料回饋假設，不是既有部署API完成；尚無隱私保證。

pooled只在全部A與解鎖定後評真實LOO和正常解，形成參考C。pooled不進A、求解、選點或刪人。truth不進方法。本次原模型不逐人從頭重求；真實LOO僅獨立驗證標準。

## 全新資料與結果界線

保留未用seeds2026102300–2026102319×Clean/controlled_poison共40團隊，新建資料、20個local models及pooled。舊開發20、上輪已揭露40全不算本次確認。資料生成與訓練沿用凍結原流程與epochs，沒有alpha初始化實驗。記錄語義資料hash，與既有全部已用seed不得重複。

資格或準備失敗留原seed失敗，不補樣；不在效果看到一半時停或增加seed。順序固定：完成全部資料→全部proxy解→全部A鎖定→全部真實LOO參考解→pooled離線揭露。參考原custom30，同X0、target0、box，刪除完整角色，剩餘alpha正規化。

每情境20團隊，各自平均團隊內Spearman>0.5、seed bootstrap10000/RNG20261020的95%下界>0、有效覆蓋>=90%才品質通過；下界>0.5另報。不能混情境平均掩蓋一邊失敗。原C範圍<=1e-4另標，完全同分留空不填0。方向deadzone1e-4，另報C差值、端點距離與所有失敗。

## 成本與驗收

40正常求解、最多160真實LOO、640代理模擬；額外原模型samples<=20000，全參與者模型samples<=500000、全部求解CPU<=600秒、client既有觀測處理40×3×128=15360筆；資料準備每案120秒、總1800秒。新增calls/samples、client資料掃描／回歸／摘要傳輸、proxy CPU、pooled離線評估分開。

相同正常求解＋本方法的模型samples需<=正常＋LOO的70%；端到端時間另以固定4案、2warmup/10次交替配對驗證，不將samples下降當成時間已省。若CPU較高如實列出，不能宣稱全部成本都下降。這個門檻只支持本次2D模型與資料，不外推複雜高維。

全部程式／依賴與文件先凍結。技術修復保留失敗版本、不可看品質後改方法；確認若不過保留負面結果，不換RBF設定、調鄰居數或追加seed追門檻。研究目標仍以有效性與成本界線為準，不因執行成功而標完成。
