# 貢獻度分析研究：新對話交接文件

日期：2026-10-16。專案根目錄：`/Users/yclee/Documents/GitHub/fl_experiment`。
這是專案交接文件，需由新對話主動讀取，不是全域記憶設定。開始前核對當前程式與artifacts；本文未啟動新實驗。

## 1. 使用者最新決定與任務邊界

- α初始化與選擇策略尚未取得老師同意或comment，暫不啟動該方向的新實作與實驗。
- 現在優先研究「貢獻度分析」：先釐清定義、整理既有完整移除證據與負面結果，再決定下一輪可驗證問題。
- 先使用直接full-removal作參與者層級的離線分析基準；不要自動把影響函數、線性近似、支持度或其他低成本近似定為主方法。
- 老師期待的長期方向是合理、低成本的個別貢獻估計，但「直接完整移除作研究基準」和「部署近似」須分開。先分析清楚再提案，不預設必須立刻發明新估計器。
- 本啟動任務先做來源核對、既有結果分析與研究設計，不授權大型重跑、改動歷史方法、啟動pruning或付款。商模由另一條研究線處理。

## 2. 老師原始建議及其來源界線

2026-09-17音檔約36:55–40:43：最後一輪、最後K輪、從頭排除與Shapley是不同定義；過度近似可能離真實重要性很遠。約40:49–43:57：希望利用倒推過程的statistics，建立簡單、可推廣的個別貢獻估計，避免每個參與者都從頭重跑；只有指標合理時，負貢獻才有辨識拖累者的意義。

原音檔自動轉錄於2026-10-15從當時工具輸出復原，並非重新聽錄，含誤字且無可靠說話者分離：
- `docs/professor_audio_recovered_transcript_20260917.txt`
- `docs/professor_audio_progress_audit_20261015.md`

不能把後來的full-removal驗證說成老師指定的部署程序。`output/0917_法五研究報告_逐字稿.md`是簡報講稿，不是老師發言原始逐字稿。

## 3. 必須區分的三種量

令X_full為完整coalition的交付解；E_pooled(X)=abs(f_pooled(X)-target)。

| 量 | 做什麼排除 | 如何評分 | 可以說明什麼 |
|---|---|---|---|
| C_op | 只排除某人的候選點／方向，保留其模型、α與成員身分，重求X_without_proposal | L_eval(X_without_proposal)−L_eval(X_full) | 候選點排除下的內部影響 |
| C_candidate_pooled | 與上列同一個候選點排除解 | E_pooled(X_without_proposal)−E_pooled(X_full) | 候選點排除下的外部品質影響 |
| C_full | 同時移除候選點、local engine及α；剩餘α正規化後重求X_without_agent | E_pooled(X_without_agent)−E_pooled(X_full) | 固定本次倒推程序的完整參與者移除效果 |

差值為正：移除後誤差變大，原參與有幫助；差值為負：移除後誤差變小，原參與在此定義下不利。接近零須用事前定義的dead zone區分數值小差，不當成明確正／負。

例：完整解pooled誤差0.02，移除i後0.05，C_full=+0.03；移除j後0.01，C_full=−0.01。此為教學例，不是保存實驗資料。

### 完整移除基準的限制

- 這是倒推階段的反事實：local models固定，不重新訓練，也不補入新參與者。
- 同seed共用凍結evaluator、target、限制與solver設定；使用同一原始指定起點。原P3以完整候選點平均指定X-space起點；在縮小後子空間如何映射，必須查`host_server.py`，不要聲稱起始可行X必然完全相同。
- 移除會同時改變候選搜尋空間、loss模型群體與權重，因此C_full是整體程序效果，不等同某個純粹模型特性的因果分解。
- full-removal也受求解誤差影響，不是無誤差真值、全訓練歷程貢獻或Shapley。
- pooled只評解，不參與α、Stage3或移除決策。可以離線描述哪位移除後改善，不能據此宣稱已有可部署的global-free刪人機制。

## 4. 既有P3已完成，不必重做同一題

正式登錄與報告：
- `docs/p3_external_validity_formal_preregistration_20261006.md`
- `docs/p3_external_validity_formal_20261006.md`

正式資料：`alpha_weight_outputs/p3_external_validity_formal_v1_20261006/`。
- 正式seeds3600–3629；pilot0–4不納入。
- Clean：Stage1選出的三人coalition、靜態Uniform α、uniform內部evaluator。
- Controlled poison：固定五人含一名污染者、靜態Uniform α、trimmed20%內部evaluator。
- 2D、target0、box[-1,1]、tol1e-8、best_of/custom30；不更新α、不pruning、不付款。
- 60個seed-scenarios、240個參與者效果完成；720筆評分紀錄重載無差異，對應540個唯一解；成功解全部符合box。

| 主要指標 | Clean | Controlled poison |
|---|---:|---:|
| C_op與C_full平均Spearman | 0.083 | −0.047 |
| 三分類方向一致率 | 40.0% | 48.7% |
| 負C_op者完整移除後品質改善率 | 32.0% | 42.5% |
| C_candidate_pooled與C_full平均Spearman | 0.004 | 0.050 |
| pooled／truth方向衝突 | 23/90 | 42/150 |

主結論：`external_validity_not_established`。把候選點排除改用pooled評分，也未建立其對完整移除的有效性；不能只換evaluator就繼續宣稱完整參與者貢獻。

統計：seed為抽樣單位、10,000次bootstrap、固定RNG20261006，兩情境×兩主要指標的98.75% Bonferroni區間。原成功門檻包含Spearman點估計至少0.5及區間下界>0、方向一致率至少70%及區間下界>50%；均未達標。原dead zone1e-7，1e-6／1e-4為敏感度分析。新研究需重新登錄，不依既有正式結果重挑原門檻。

成本並非「候選排除一定便宜」：原每情境平均查詢，候選排除Clean1020.4／poison5061.7，完整移除573.3／3537.6。這只是原設定成本，不能泛化；新分析應核對各路徑是否包含相同工作與API輸入量。

## 5. 新對話優先讀取的程式與證據

以下皆相對於專案根目錄。
- `p3_contribution_validation.py`：baseline、候選排除、完整移除與保存。
- `host_server.py`：exclusion實作、候選子空間、起點映射、solver與可行性。
- `analyze_p3_external_validity.py`、`verify_p3_external_validity.py`、`reevaluate_p3_solutions.py`。
- `test_p3_external_validity.py`；必要時參考`analyze_p3_results.py`舊pilot。
- 正式輸出`analysis_v3/analysis_manifest.json`、`seed_metrics.csv`、`aggregate_metrics.csv`、`pooled_truth_conflicts.csv`。
- 正式輸出的`clean/`、`controlled_poison/`原始結果、manifest、artifacts與來源快照。

先檢查正式manifest、來源版本與欄位，再讀CSV；只重分析所需資料，避免掃描所有α歷史來源快照。引用新結論附有效n及失敗口徑。

## 6. 建議研究步驟：先分析，後決定是否需要新實驗

1. 以既有正式結果重新確認三種C的對象、正負號、近零比例、排序差異及成本口徑。先驗證欄位與程式，不要把本文件數字直接當當前程式的驗收結果。
2. 選代表案例：原估計判負但完整移除變差、兩者同向、近零、pooled/truth衝突。清楚展示移除了什麼、X變多少、各種品質差與solver狀態。不得把真實函數當裁判選對自己有利的結果。
3. 提出少量具體研究問題：完整移除效果對起點／求解精度是否穩定？候選方向效果與模型loss效果是否需要分開？哪些部署可得統計可能反映完整移除效果？這些是提案，不是已確認原因。
4. 先推薦一個最小研究設計。若研究低成本訊號，明列公式、可得資訊、成本與待驗證假說；不得自動採用支持度S、α大小或個人MSE作貢獻。
5. 既有正式資料若用於選公式，視為新研究開發資料；之後需全新未使用的確認案例，不將同批再分析稱為獨立驗證。

## 7. 驗收與協作原則

- 先固定貢獻對象、反事實範圍、主要指標、neutral閾值與成本預算，再跑pilot；不預設會找到有效方法。
- 品質參考由凍結pooled evaluator提供，truth只作診斷。α、原始signed C、正貢獻正規化q與付款比例分開。
- 全無正C時q無效，不以均勻q偽裝成有效貢獻或收斂。
- 保留資格、solver、無解、缺artifact等狀態；缺值不填零；無解不能送入模型。參與者不得當成跨seed獨立樣本。
- 相同條件配對；保留固定起點、剩餘IDs／α、每次X與solver狀態、可行性及查詢／耗時。重載容差1e-7，可行容差1e-8。
- 低成本需比較相同工作範圍的API calls、輸入樣本數與時間；外部評分另計。不以低成本犧牲語義後仍稱同一貢獻。
- 不覆寫舊結果、登錄或來源快照。不把負C等同惡意，也不從移除改善直接推論付款公平。
- 繁體中文；清楚區分已完成、負面結果與提案。方法說明包含公式、小例子與限制。
- 大型實驗依序排程；平行改程式需另行安排獨立worktree與artifacts，不能假設新對話隔離了程式。
- 新分析保存獨立目錄；本研究線維持固定設計／進度文件。Notion主文件由總控整合，本次不更新。

## 8. 新對話首次交付

在不修改舊演算法、不啟動大型實驗前提下，交付：既有證據核對表、貢獻三定義的可讀說明、2–3個可查證問題、推薦的最小下一輪設計及驗收標準。可做必要的既有資料唯讀統計並保存獨立報告；不得直接執行新正式實驗。若需近代文獻，另行查核原論文，不以未查證記憶聲稱已有學術支撐。
