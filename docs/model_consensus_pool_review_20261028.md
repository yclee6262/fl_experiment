# 模型池共識保守候選：主代理審查與實作交付

2026-10-28。規劃與判讀：主代理。實作：使用者指定GPT-6.1 Sol子代理。授權範圍是列設計並開始實作，本輪只進行技術測試，沒有啟動研究比較。

設計：`docs/model_consensus_pool_design_v1_20261028.md`。實作說明：`docs/model_consensus_pool_implementation_20261028.md`。程式：`contribution_analysis/model_consensus_pool_v1/`。技術證據：`contribution_analysis/model_consensus_pool_implementation_checks_20261028/`。

## 演算法

同輪保留錨點，對每受評owner的model-only full/minus接受端點，以所有模型的固定probe尺度換算loss差。每個評價owner內按原v合併為z。l=min各owner z，u=max各owner z；區間是內部混合評價範圍，非品質／統計信賴區間。各輪l有號累積S，S>max(1e−8,n*1e−10)才作分配權重；比例為正權重除以總和，固定整數預算採最大餘數及owner字典序同分規則。

原v加權中心C及own/others/peer只作診斷，不能事後替換主分數。完整矩陣缺漏或版本不符為held（比例／金額null）；無可靠正累積為reserved（比例null、金額0、預算保留）。基本保障及錨點池另計。

## 已完成

GPT-6.1 Sol交付JSON schema、手算範例、計分／結算核心、獨立reference、exclusive-create CLI、README、14項單元測試。本輪不載入solver或訓練模型。

主代理在交付前審查並要求修正：own/others加權分解與peer正規化、同endpoint ID的loss一致、非法owner/round型別、數值溢位、整數同分分配及精確deadzone邊界。門檻與核心採十進位輸入的Fraction精確運算；reference另行實作且不匯入core。兩者使用相同有理數類型，不能聲稱異數值函式庫交叉驗證；主代理另以手算輸入核對。

最終主代理重跑14項測試全部通過。獨立手算3owner×2輪、probe=[0,1]尺度=.8，累積S=.03/.03/−.005，比例.5/.5/0，501單位251/250/0；核對原始回覆差、未剪裁有號和、字典序取整。額外缺槽／重複／無minus／owner型別錯／同endpoint不同loss均held，零作用reserved，manifest加入truth欄位被拒絕。10項主代理檢查通過。

手算CLI試跑的模型與solver呼叫皆0，屬合成帳務驗收，不是研究效果。公開JSON schema核對代理範例與主代理輸入皆通過。詳細執行日誌、輸入、輸出及review腳本已保存，來源hash封存於technical_receipt.json。舊研究三組receipt未變。

## 未完成與未建立有效性

尚未實作全owner T3搜尋資料生成器及比較研究runner。因此目前只可讀完整既有slot JSON計分，尚不能一鍵從原搜尋生成完整實驗；不能用只含target0的歷史資料冒充全團隊。

offline hash／version／probe_id只是上游一致性宣告，尚未證明真實查詢、探測在搜尋前鎖定、solver接受端點或身分認證。未測品質、分配覆蓋、共識錯誤、錯誤者否決、正確少數漏獎、真實推論成本或實務部署。

新方法主張保守內部共識，不是解決上一輪的真實品質不可辨識。錯誤共識仍可能發獎，單一錯誤者可能讓所有人不分配；負S也不代表真實負貢獻。不得把技術通過改寫成有效分潤證據。

## 下一步提案

先接全owner model-only資料生成器，凍結新研究fixtures及覆蓋／漏獎驗收門檻，再依設計第7節執行比較。沿用原求解和接受規則；pooled僅在端點與分數確定後作離線驗證，truth僅診斷，不進模型池。未執行新比較、不修改歷史演算法、錨點池、alpha、舊結果或Notion。
