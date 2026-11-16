# 依論文範圍精簡程式：保留建議

2026-11-16。依使用者指定：法五各Stage、各種分潤方法比較、α初始化、先保留α校正。本次僅盤點與提出清單，沒有刪檔或改動程式。前次是按用途搬移，本次才按論文範圍縮減內容。

## 最新範圍修訂（本節優先於下方較早建議）

使用者已指定：論文主線以0604及1112報告為主，0917的α校正保留程式碼；FedAvg等傳統方法比較明確保留。完整頁面與程式對照見 [目前論文範圍](thesis_scope_from_reports.md)。

- `experiments/baselines/` 不再列為整批封存候選：至少保留FedAvg訓練／聚合／反推、法五對照及評估成本；先抽出共用實作再淘汰重複展示入口。
- 1112主表是八法 U/Q/E/S/EU/EQ/SU/SQ；SE/QU、C/P及11/16比例研究保留為擴充候選，不自動納入主文，也未授權永久刪除。
- 1112已包含方向投影與抽樣CA，因此對應方法與結果必須保留；不能將所有A/E/B或抽樣工作整批封存。
- α校正仍完整保留程式／測試／重現依賴，但不因保留程式而列為論文主線。
- 下方廣義方法清單是依賴盤點與早期候選範圍；與本節不同者，以本節及新範圍對照為準。

## 保留原則

保留「方法＋正式比較入口＋分析／驗收＋必要資料與版本」，不保留所有探索過程在主要目錄。研究結果不支持假說不等於程式無用：若論文使用該負面結果，對應方法和重現證據仍應保留。使用者尚未指定初始化／分潤的最終比較名單，因此以下方法範圍是建議，不能當永久刪除白名單。

| 類別 | 處理 |
|---|---|
| 論文必要 | 留在主要程式與比較入口 |
| 技術依賴 | 目前先留，抽出需要的共用功能後再移除舊檔 |
| 歷史／候選 | 移出主要程式，封存，不列為推薦執行入口 |
| 可重建雜項 | 可清除快取與可重建暫存；不含唯一研究結果或原始資料 |

## 1. 法五核心各Stage：保留完整能力

- `core/host_server.py`：資格、選人、共識目標、候選子空間、BFGS/custom/best-of、邊界、排除、剪枝、原始分潤。即使部分機制尚未建立效度，仍屬本次指定範圍，不刪除。
- `core/agent_client.py`、`core/models.py`：Agent API、本地模型、反推。
- `core/pipeline.py`：目前只串資格→提案→求解；**不是完整所有Stage的入口**。後續可加明確選項串接校正、剪枝與分潤，保留各研究所用設定，不能默默啟用。
- `experiments/shared/dataset.py`、`run_experiments.py`、`research_evaluation.py`、`research_analysis.py`、`reevaluate_saved_solutions.py`、`smoke_main_flow.py`：資料、訓練、查詢記帳、離線評估及重現支援。
- `experiments/coalition/recruitment_selection.py` 與 P6：Stage1的選人／預算比較保留一條代表性入口（`p6_recruitment_experiment.py`＋分析／重評）；若論文使用枚舉的proxy有效性結果，連 `p6_coalition_enumeration.py`＋對應分析／重評保留。pilot與重複執行器可封存。
- 保留主流程、邊界、資格、預算與分潤守恆的相關測試；不是只留下能跑出X的最小demo。

## 2. α初始化：方法保留，批次腳本收斂

建議主比較先保留目前有成套報告的 U（舊A）、Q、E、EU、EQ、S、SU、SQ、SE、QU 十法。此處只是建議主表，未把其他初始化方法視為使用者已排除。

| 功能 | 必要程式 |
|---|---|
| U/Q/E與既有方法定義 | `initial_alpha_methods.py` |
| S支持訊號 | `peer_support_alpha.py` |
| 純E/S及四法／混合組合 | `pure_es_initialization.py`、`four_alpha_initialization.py`、`se_qu_initialization.py`、`support_quality_mix_experiment.py` |
| 方法註冊與對照 | `unified_alpha_methods.py`（要精簡內部依賴，不能直接刪） |
| 重現十法結果 | `run_four_alpha_batch.py`、`run_pure_es_batch.py`、`run_se_qu_batch.py`及其必要準備、分析、驗收；後續可合併為單一runner，而非永久保留每套pipeline |
| C/P初始化或解選擇策略 | `alpha_initialization_selection.py`、相關run/prepare/formal statistics/analysis/audit，作擴充比較；不可和固定初始化誤稱同一方法 |
| 11/16比例研究 | `alpha_mixture_exploration.py`、`four_way_bayes.py`及分析／核對，保留為初始化延伸或附錄；搜尋時用pooled的開發程序不能包裝為線上可用方法 |

早期 bootstrap、OOB、correlation、shrinkage、sparse、targeted-label、portfolio、transfer 等：若要放進「各種初始化方法比較」，保留方法定義與最終比較結果，但合併重複入口；若只寫上述主表與延伸，則將其獨立研究runner／報表封存。沒有在本次替使用者擅自刪掉這些方法。

### 已核對的依賴問題

只從 `initial_alpha_methods`、`peer_support_alpha`、`pure_es_initialization`、`se_qu_initialization`、`four_alpha_initialization` 與 `alpha_weight_experiments` 六個入口追蹤靜態import，就涉及25個模組。詳見 `retention_dependency_audit.json`。這不是完整執行／資料依賴，也不是25支即可交付。

其中包括 `next_alpha_initialization`、`conservative_shrinkage`、`adaptive_correlation_alpha`、`explore_cv_ab_switch`、`run_validation_calibrated_alpha`、`run_alpha_initialization_study`、`p5_scenario_suite`。有些只是被統一註冊器或共用helper間接拉進來。後續應把共用計算和資料準備抽出、只註冊需要的主方法，再刪掉非論文分支；不能先刪檔再補import。

## 3. α校正：保留完整研究，不只留一個更新函式

- `alpha_weight_experiments.py`：固定coalition、初始化、不同evaluator、fixed/damped/guarded更新、無效q、accepted state、停止、選配結算等。
- P2結果分析／繪圖與重評工具：保留動態對靜態的品質和成本比較。
- `p4_constrained_no_progress_pilot.py`及P4分析／繪圖：保留無進展停止機制和結果；區分停止與收斂。
- `p5_scenario_suite.py`、P5分析／重評與對應結果：保留跨情境比較；其中一部分也被其他保留程式依賴。
- 相關測試與登錄／最終結果保存。
- `main_alpha_calibration.py` 是舊入口候選，可在確認功能已被上述runner覆蓋後封存；`reanalyze_legacy_p2.py`屬歷史格式重分析，通常不必留在主要入口。

## 4. 分潤比較：分開貢獻分數與付款規則

「各種分潤方法」不能只留下付款公式。比較至少需要：產生分數的方法、評分尺度、正負／保留額處理、基本費與池額設定、映射付款、成本及驗收。

### 建議主比較保留

| 比較族群 | 保留目的／主要套件（`contribution_analysis/`下） |
|---|---|
| 原proposal分潤 | `HostServer`的原r＋q、最低報價與剩餘額規則；作歷史baseline，不等於已驗證公平 |
| 錨點／模型角色 | `local_anchor_step_v1`、`anchor_step_validation_v1`、`model_step_decision_v1`、`combined_contribution_layer1_v1`、`combined_contribution_layer2_v1`：保留需要的A/A2（CA）、M（CM）、聯合作用實作與比較 |
| 共同J與完整LOO | `contribution_four_way_v1`及實際載入的共用模組／快照；是後續方法的必要對照，不能當舊版刪掉 |
| 三池與模型池變體 | `three_pool_t0_t1_v1`、`three_pool_t2_anchor_v1`、`three_pool_t3_model_v1`、`anchor_pool_end_to_end_v1`、`model_pool_combinations_v1`及必要的V/CM/U/N分數模組；可明確區分主方法與比較臂 |
| 排除自評 | `model_pool_v_cm_validation_v1`、`peer_exclusion_cost_v1`、`senior_three_packages_v1`；保留F／Fminus／pooled對照，不能只留較新peer表而丟原始對照 |
| 成本 | `method5_cost_v1`、`anchor_ca_model_v_cost_v1`與上述成本runner的必要部分；後續合併共用計數，歷史結果維持原口徑 |
| 低成本近似 | `low_cost_loo_v1`：J最後接受輪、累積錨點位移及完整LOO對照，使用11/16 v2報告定義 |
| 分潤可信度檢查 | P3外部效度、`senior_remaining_v1`與11/10後續稽核。即使不列主表，仍保留必要證據和限制；不得沿用已撤回的「去重有效」說法 |

### 可作附錄或歷史候選

- `custom_realized_gain_v1`的A/E/B改善分帳，以及 `a_e_b_validation`／`a_e_b_capture`：若比較章會討論此類分帳，保留；若不討論，可封存研究入口，**但其中schema/runtime/observer等仍被三池與成本程式引用，必須先抽出helper**。
- 早期 process-gradient、geometric replay、rank consensus、reliability risk、validation judge、各種response proxy搜尋：未列入最終比較者可退出主要方法集，保留引用它的結果與必要實作。
- 模型分支候選搜尋（`model_branch_search_v1`）：屬搜尋改良，與分數／付款比較不同，依目前範圍建議封存。不要因為名字有model/contribution就全部留下。

### 舊快照仍是活躍依賴

`low_cost_loo_v1/run.py` 直接使用 `model_step_decision_experiment_v1_20261021/source_snapshot`、`contribution_four_way_v1`，且讀取 `custom_direction_experiment_v2_20261020`、`contribution_four_way_experiment_v2_20261022`、`combined_contribution_layer2_experiment_v1_20261021` 等資料。`method5_cost_v1`同樣把凍結快照放入import路徑。

因此不能以「所有source_snapshot都是重複碼」一律刪除。應先列動態import與檔案讀取依賴，確定哪些是唯一版本／必要模型／基準端點，再決定打包封存或抽出獨立版本。

## 5. 可以先移出主要程式的項目

1. **修訂：傳統FL比較保留。** `experiments/baselines/main.py`內FedAvg函式、`fl_comparison_numerical.py`的FedAvg／Krum與反推必須先留；`fl_comparison.py`／`fl_comparison_filtered.py`的醫療資料／Flower版本先保留候選，待對齊論文比較設計後才決定封存哪個重複入口。`ablation_main.py`依核心求解器消融用途判斷，不因舊檔名刪除。
2. 同研究的 `resume_*`、一次性 `finalize_*`、`supplement_*`、中途報告產生器及重複pipeline：在核對最終結果是否已整合後，留一個重現入口，其餘封存。檔名前綴本身不是判定依據。
3. 已被最終版取代且不再被引用的pilot／失敗attempt程式、舊格式轉換和簡報繪圖腳本：移出主要程式；保留論文使用的圖表來源和重要更正紀錄。
4. `.DS_Store`、`__pycache__`、`.pytest_cache`及確認可重建的LaTeX編譯暫存：可清除。模型、種子清單、原始數值、註冊規約、唯一快照及負面結果不是這一類。
5. 根目錄相容連結：待保留程式統一import／子程序入口後可大幅移除。它們本身幾乎不占空間，現階段刪除只會破壞舊匯入，不能解決實際重複程式問題。

## 6. 下一輪建議目標

保留四個清楚入口：法五全stage流程、分潤／貢獻比較、α初始化比較、α校正比較；其下共用模型、資料、solver、evaluator及成本記帳。舊方法只在比較註冊表中明確列出，探索歷史移到封存區。

順序：先固定主比較名單 → 抽出依賴helper → 合併重複runner → 比較保留方法的輸出與成本 → 轉移非主文歷史 → 清掉不再使用的相容連結和快取。這一輪不承諾最終剩幾支檔案，因為動態載入與資料依賴尚未完整掃描，也不把靜態25模組清單誤作精簡後成品。
