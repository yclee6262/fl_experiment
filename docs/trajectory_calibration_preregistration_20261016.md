# 過程統計校準樹v1：分階段實驗登錄

日期2026-10-16。使用者授權「設計相關實驗執行看看結果」。本文件於技術pilot與開發特徵產生前凍結。研究輸出：`contribution_analysis/trajectory_calibration_experiment_v1_20261016/`。

遵循`docs/low_cost_contribution_proposal_20261016.md`的5個個人＋3個情境特徵、深度2、每個子節點至少5個不同seed、seed平衡權重與固定切分規則。方法主張限定於Stage 3完整移除效果預測，不接α更新、pruning、付款或Shapley。LOO只作離線參考／標籤及獨立驗證。

## 一、凍結與技術pilot

凍結8個歷史來源與本輪程式、SHA256、套件／解譯器。所有輸出exclusive create，不覆蓋舊結果。使用先前預算研究pilot的2026101600／2026101601、Clean與controlled_poison共4個既有保存模型案例，不重訓。

記錄器使用原solver接受的history_metric介面觀察狀態、攔截原objective既有API回覆。舊P3保留其diagnostic歷程方式，新預算資料保留其objective歷程方式；前者的truth只維持舊診斷記錄，不進入特徵或決策。接受狀態以模型可見float32 X對應既有回覆。無法對應時失敗，不補查模型。

為可靠對應SLSQP已接受狀態，這版研究記錄器在記憶體暫存所有Q個已查詢objective點的回覆，再提取接受軌跡。因此額外cache成本是O(Qm)，不是只有O(Tm)；timing包含此成本。被拒試探點不作特徵。

每pilot案例先2次warmup，再15組交替順序的control／record配對（共4×32=128次baseline求解）。control／record都加相同calls計數器。record時間包含求解、記錄及特徵提取；不計原本也需要的模型載入與磁碟結果保存。另列此範圍限制，不宣稱全服務速度。

技術驗收：calls／input samples完全相同；X與原baseline最大差≤1e-7；objective一致；box≤1e-8；完整特徵有限。比對來源原始expected_calls，不只control與record互比。第一版記錄器測試5項通過後才進pilot。

成本閘門：將四案例各自15次record＋features時間的中位數加總，除以四個control中位數加總，≤1.10才進開發特徵提取。超過即停止此版本，保留時間樣本，不刪outlier或降低門檻。這是成本驗收，沒有獨立有效性主張；規則推論耗時尚未納入，故此閘門不等於最終線上10%目標已通過。

## 二、既有資料開發

技術／成本閘門通過後，重播P3正式3600–3629兩情境的60個baseline，以及新預算研究確認2026101610–2026101619兩情境A設定20個baseline。新研究A/B不得重複充當樣本。共80次baseline、320 effects、40 seed群組，均屬開發資料。

每筆只從完整團隊軌跡提取固定8個特徵，不輸入agent ID、seed、污染標記、pooled、truth、移除解。無pooled模型載入特徵程序。完整X必須重播一致才可配對既有C_full標籤；不一致停止，缺失不補0。

5折：將40個seed排序後以RNG20261016洗牌，再按序號mod5分折。每折8個validation seeds，兩情境及所有參與者不得跨折。每個人次訓練權重1/(2m)；每折獨立擬合，沒有確認／validation標籤回流擬合。

固定比較：0、按m分組的training平均C、模型特徵＋共同情境樹、候選特徵＋共同情境樹、完整8特徵樹。只有完整樹是主要方法，不用結果改選消融勝者。樹深度2、兩子節點各至少5個training seeds、切分最小化加權平方誤差、平手按特徵序號和門檻小者。

開發進階閘門：完整樹out-of-fold的MAE須在Clean及controlled_poison兩情境**各自低於**按m分組訓練平均C基準。此處只要求點估計改善，屬開發篩選；仍另外報方向、負值precision／recall及所有消融。任何情境未改善，停止獨立確認，不增加深度或重挑特徵。

開發資源：技術pilot＋開發合計最多208次baseline、500,000 API calls／input samples、300秒求解。超限保存失敗並停止。這是研究開發成本，不能藏入0線上查詢的說法。

## 三、條件式獨立確認

僅技術、成本與開發閘門全通過時，凍結模型與預測程式，查重並登錄20個全新seed（兩批各10、共40 scenarios／160 effects）。正式新資料產生前另存確認登錄及模型hash，使用原提案的70%方向、10% MAE改善、98.75% CI、負值識別與成本門檻。每個scenario先鎖定正常baseline的全部預測，再產生LOO參考。

此階段上限200次baseline／完整移除求解、500,000模型calls、1800秒solver、3600秒資料／模型準備。若前階段停止，本輪不啟動此階段，不消耗新seed。

## 四、保存與解讀

保存source、來源manifest／model hash、逐案技術狀態、完整接受軌跡、特徵、seed折別、每折模型、out-of-fold預測、MAE／方向／負值指標、閘門與成本。LOO標籤若用於訓練就只是開發答案，不再當新獨立確認。無效或不足結果如實報告，不推論所有過程統計方法都無效，不修改P3或先前預算研究。

已知本機OpenMP共享記憶體曾在沙箱內中斷，本輪自始使用相同本機解譯器、單執行緒與允許共享記憶體的執行環境，不連網。環境或程序失敗保留，不靜默換樣本。
