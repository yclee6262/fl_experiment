# 聯邦式倒推研究執行與驗收文件

建立日期：2026-09-19；最後更新：2026-09-23。依據：2026-09-17 討論影片前 45 分鐘、目前程式與 E14 輸出，以及本次確認的研究規劃。

用途：作為後續實作、實驗、結果判讀與老師討論的共同依據。P1 基礎設施、P2 正式比較、P3 pilot、P4 正式成本驗證，以及限制與配對修正後的 P5／P6 新版 pilot 已完成；各階段的功能證據、pilot 證據與正式研究證據仍須分開判讀。

## 本次實作紀錄（2026-09-19）

已完成三個基礎項目，但不代表整個 P0、P1 或 P2 完成：

1. **版本紀錄與防覆蓋**：每次執行寫入 `run_manifest.json`，包含 Git commit／工作目錄狀態、六個研究來源檔的 SHA-256、Python 與套件版本、有效參數和執行狀態。非空輸出目錄拒絕執行；每個完成條件立即保存 CSV。`complete` 表示所有條件正常完成；`running` 也可能代表中斷的執行，不能當成仍有背景工作。
2. **X 與貢獻保存**：summary 新增 `final_solution`、`final_candidate_points`、`final_raw_marginal_contributions`、`final_box_feasible`；rounds 新增 `candidate_solution`、`candidate_box_feasible`。最終 X 取 accepted state，不取最後被拒絕的候選。範圍檢查僅代表上下界，不代表已檢查任意額外限制。
3. **明確不更新基準**：`--no-alpha-update` 將有效 update rate 設為 0、policy 設為 fixed；可同時執行 Uniform／Stage 0。拒絕搭配 coalition stabilization。每個條件只有一個 calibration round，但仍包含既有貢獻診斷、必要的 solver recovery 與付款，因此不是保證只呼叫一次 solver。無有效 q 時保留既有停止原因，不假稱收斂。

驗證：26 項單元測試通過；小型端到端測試的 Uniform／Stage 0 各完成一輪、零權重更新，accepted X 與保存欄位一致。該測試僅使用 3 agents、20 samples、1 epoch，**不是 5-seed pilot 或正式品質比較**。

測試使用暫存 Python 環境 `/tmp/meeting-transcription-0918/bin/python`；此環境不是長期依賴配置。後續須使用具備 numpy、scipy、torch 的 Python 環境。

執行不更新基準的範例（執行前選擇新的空目錄）：

```sh
python alpha_weight_experiments.py --output-dir alpha_weight_outputs/no_update_reference_v1 --no-alpha-update --initializations uniform,stage0 --seeds 0 --dimensions 2 --poison-ratios 0 --evaluators uniform
```

2026-09-22 已完成 seeds 100–129 的 30-seed 正式比較，結果見 `docs/p2_formal_20260922.md`。來源雜湊提供版本核對，不等於已保存原始碼快照。

## P1 基礎設施實作紀錄（2026-09-22）

已完成：

1. `dataset.py` 同步保存每位 agent 的 `observed_y` 與合成環境可得的乾淨反事實 `clean_y`；污染 agent 的 local model 仍只看到 `observed_y`。
2. 每個 `(seed, dimension, poison_ratio, target)` 情境只建立一個凍結的 pooled-data global evaluator，所有 α 條件共用。它在 Stage 3A／3B、選人與付款完成後才評分，不進入演算法決策。
3. pooled dataset 使用固定 seed 分成 train／validation／test；validation 選 checkpoint，test 只報 evaluator fit。另報 test 中 `|y|` 前 25% 的 hard-region MSE。
4. 每個情境保存 `pooled_data.npz`、`local_models.pt`、`global_evaluator.pt` 與 `metadata.json`；summary 與 rounds 寫入外部 prediction/error 及 artifact 相對路徑。
5. `reevaluate_saved_solutions.py` 可從 summary 與凍結 evaluator 重評 accepted X，輸出外部誤差、真實函數誤差、box feasibility 與保存值差異，並拒絕覆蓋既有重評檔。

重要限制：

- pooled evaluator 使用合成環境中可取得的乾淨反事實標籤；真實應用通常無法取得污染 agent 的反事實乾淨標籤。
- pooled test split 用於檢查 evaluator 對合成資料分布的擬合品質，不是獨立的真實世界資料集。
- local models 仍使用各 agent 的整份 observed dataset 訓練；pooled evaluator 的 test split 沒有進入 evaluator 訓練，但其 X 可能曾被 local model 看過。該 split 衡量 evaluator fit，不應宣稱為整個系統的完全獨立 generalization test。
- 外部 evaluator 分數與真實函數誤差必須並列；兩者結論衝突時，依預先固定的 pooled evaluator 作主要判讀，並把衝突列為限制。

驗證證據：29 項單元測試通過；小型端到端 smoke 保存 60 筆 pooled samples、三個 local checkpoints、一個 global evaluator，Uniform／Stage 0 兩個 accepted X 經獨立重評後 `mismatches=0`。本 smoke 使用 3 agents、20 samples/agent、1 local epoch、5 evaluator epochs，不是研究結果。

重評範例：

```sh
python reevaluate_saved_solutions.py alpha_weight_outputs/no_update_reference_v1 --output alpha_weight_outputs/no_update_reference_v1_reevaluation.csv
```

## 0. 如何使用本文件

1. 依 P0 → P1 → P2 → P3 的順序建立證據；P4–P6 依前階段決策啟動。
2. 每次開始工作前核對版本、前置條件與驗收清單。
3. 只有附有可回查證據時才勾選驗收項目；實作完成、測試通過、研究假說成立是不同狀態。
4. 每階段完成後填寫文末的交付紀錄與決策紀錄。
5. 新結果不得覆蓋既有 E14 或其他原始實驗；規格改動須記錄原因與新版本。

| 階段 | 狀態 | 證據／結果 | 下一步 |
|---|---|---|---|
| P0 定義與版本 | 核心定義、來源快照與 `box_constrained` 交付規則已建立 | 本文件、run manifest、來源快照與 46 項回歸測試 | 固定正式版本與獨立 seeds |
| P1 保存與外部評估 | 已完成並通過正式重評 | 正式 240 個 accepted X 重評 `mismatches=0` | 維持 evaluator 凍結與保存規格 |
| P2 α 策略比較 | 正式實驗完成 | `docs/p2_formal_20260922.md`；60 個情境、240 個條件 | 以 A 為基準進入 P3 |
| P3 貢獻有效性 | 正式 30-seed 外部效度驗證完成 | `docs/p3_external_validity_formal_20261006.md`；60 scenarios、240 participant effects；未建立外部效度 | q 限定為候選點內部影響；pruning 與公平分潤主張維持停用 |
| P4 更新與 pruning | 無進展停止正式驗證與受限版再驗證通過 | `docs/p4_no_progress_formal_20260923.md`、`docs/validity_adjustment_pilot_20260923.md`；受限版 10 組 accepted state 零差異 | 保留 patience 2；pruning 與更新調參仍暫停 |
| P5 擴充與消融 | 受限、固定 participant IDs 的 8 情境 5-seed pilot 通過；正式實驗未開始 | `docs/validity_adjustment_pilot_20260923.md`；40 組有效 A/D 配對、80 個解重評零差異 | A 維持主方法；另行固定正式版本與 seeds |
| P6 招募與商業模式 | 三預算、兩 eta、三方法的 5-seed pilot 通過 | `docs/validity_adjustment_pilot_20260923.md`；90 個解重評零差異 | 招募代理目標有效性列為後續研究問題 |

## 一、研究目標與執行原則

### 1.1 主研究問題

在不共建 global engine 的實際倒推流程中，如何設定或調整參與者權重 α，使最終倒推解 X 具有較好的品質、可行性與穩定性，同時控制計算成本？

主要貢獻須建立在以下證據：

1. 最終 X 比合理基準更好，或在相近品質下成本更低。
2. 改善在多個隨機種子與研究情境下能重現。
3. 能解釋改善來源，以及方法失效的條件。
4. 貢獻估計與分潤規則具有明確定義和限制。

**α 與 q 一致是待驗證的機制假說，不是主要成功標準。**

### 1.2 必須分開判讀的結果

| 結果 | 能說明什麼 | 不能直接推論什麼 |
|---|---|---|
| α–q 差距縮小 | 權重更接近目前定義的貢獻分布 | X 一定更好 |
| α–q 達到收斂門檻 | 滿足指定的數值停止條件 | 找到全域最佳解 |
| 內部 L_eval 降低 | 目前評估群體對 X 的誤差較低 | 外部品質必然改善 |
| 外部誤差降低 | 外部評估模型判定 X 更接近目標 | 真實應用必然成功 |
| 真實函數誤差降低 | 合成環境中 X 更接近真實目標 | 能直接泛化到真實資料 |
| 成功付款 | 結算規則得到有效分配 | 演算法收斂或貢獻完全公平 |
| 貢獻為負 | 指定評估與排除方式下呈現負面影響 | 已證明是惡意參與者 |

### 1.3 執行順序與前置條件

| 順序 | 工作 | 前置條件 | 核心產出 |
|---|---|---|---|
| P0 | 固定研究定義與實驗版本 | 無 | 研究規格、版本清單 |
| P1 | 補齊結果保存與外部評估 | P0 | 可重評的實驗資料 |
| P2 | 比較 α 策略是否改善 X | P1 通過 | 配對比較與方向決策 |
| P3 | 驗證貢獻估計 q 的意義 | P1；參考 P2 | 貢獻有效性與成本報告 |
| P4 | 改善更新與 pruning | P2、P3 的決策 | 有證據支持的改良方法 |
| P5 | 擴大情境與消融 | P4 或保留基準方案 | 論文主要結果 |
| P6 | 招募、預算與商業模式 | 核心倒推主線穩定 | 系統延伸研究 |

## 二、共用定義與驗收規則

### 2.1 權重、貢獻與付款

| 符號 | 定義 |
|---|---|
| α | Stage 3 最佳化使用的權重 |
| C_i | 帶正負號的原始貢獻估計 |
| q_raw | 由正貢獻正規化後得到的分布 |
| q_update | 實際供 α 更新的分布，可能包含 exploration floor |
| q_pay | 結算使用的貢獻分布，可能來自 payment fallback |
| 最終付款 | 依既有基本報酬與剩餘預算分配規則計算的金額 |

不得把這些量統稱為 q 後混用。沒有可正規化的正貢獻時，記錄「無有效貢獻分布」及原因，不能直接解讀成均勻貢獻或已收斂。

### 2.2 品質指標

主要指標：`E_global(X) = |f_global(X) - Y*|`。2026-09-22 已確認正式實驗以凍結的 pooled evaluator 作主要品質判準。

合成環境的次要交叉檢查指標：`E_true(X) = |f_true(X) - Y*|`。它用於診斷 evaluator 偏差與解釋結果；與主要指標方向不同時必須並列揭露，但不阻擋正式實驗，也不取代主要結論。

另外記錄：

- 最終 X 是否滿足全部限制。
- 可行且誤差低於事先設定門檻的比例。
- 每個方法的中位數、平均數、四分位範圍與失敗率。
- Stage 3、貢獻估計、付款分別消耗的查詢量。
- 執行時間、接受／拒絕更新次數與停止原因。

**可行解品質與整體失敗率分開報告。**不得刪除失敗案例後，只展示成功案例的平均值。

### 2.3 配對比較規則

- 同一 seed 使用相同資料切分、local models、初始 coalition 與候選點。
- 凍結外部 evaluator，同一情境所有方法共用。
- evaluator 與真實函數的分數不回饋給測試案例的 α 更新、選人或 pruning。
- pilot 使用 seeds 0–4；正式比較使用獨立 seeds 100–129。
- 正式結果不得用於重新挑參數；若需改參數，視為新實驗版本。
- 品質配對差值：`ΔE = E_method - E_baseline`，負值表示方法較好。
- 使用 seed 為抽樣單位的配對 bootstrap，報告 95% 信賴區間；不得將同一 seed 的輪次當成獨立樣本。

### 2.4 成功判讀

| 判讀 | 條件 |
|---|---|
| 穩定改善 | 主要配對差值的 95% 信賴區間完全小於零，且未以更高失敗率掩蓋問題 |
| 成本改善 | 在相同外部品質要求下，查詢量或時間較少 |
| 證據不足 | 區間跨零；只能說未建立穩定改善 |
| 效果有限 | 只有部分情境改善，明確限定適用範圍 |
| 不支持假說 | α–q 更一致，但 X 沒改善或變差 |

達標門檻依應用單位與容忍度，在正式實驗前登錄；沒有合理業務門檻時，以連續誤差作主要結果，不任意宣稱達標率。

## 三、逐階段工作與完成標準

### P0：固定研究定義、版本與既有成果

**目的：**讓後續實驗可以追溯，避免新舊流程混比。

**執行項目**

1. 記錄 Git commit、未提交修改、Python／套件版本。
2. 保留 E14 與其他既有結果，不覆蓋原始目錄。
3. 整理 α 更新、evaluator、exclusion、pruning 與付款的實際資料流。
4. 寫清楚正式研究主張與探索性假說。
5. 確定 X 的允許範圍與額外限制；核對 Stage 3 是否真正保證限制。
6. 記錄每種貢獻估計的排除範圍，不只使用 LOO 名稱。

目前 exclusion 移除某個候選點，重新求解較小候選子空間；被排除者的 engine 仍參與最佳化與評估。它衡量特定條件下的候選點影響，不能直接命名為完整移除參與者的貢獻。實作變更後須重新核對此敘述。

**產出：**研究定義與流程表、實驗版本與既有結果索引、指標／符號／狀態字典。

**驗收**

- [ ] 分清 α、原始貢獻、更新 q 與付款 q。
- [ ] 說明 exclusion 排除與保留的內容。
- [ ] 指出最終採用哪個 accepted state。
- [ ] 說明可行條件由求解器保證或僅事後檢查。
- [ ] 不將新驗證回填成舊實驗已通過。

**決策關卡：**若最終 X 的可行性沒有保證，先補檢查與標示；正式品質比較前再決定是否需要最小幅度的限制處理修正。

### P1：保存實驗狀態與建立外部評估

**目的：**任何已交付的 X 都能用獨立基準重新評估。

#### A. 結果保存

| 類別 | 必要內容 |
|---|---|
| 識別 | run ID、版本、seed、完整參數 |
| 資料與模型 | 切分索引、模型 checkpoint、模型與資料識別 |
| 倒推輸入 | target、限制條件、初始 coalition、候選點 |
| 最終狀態 | accepted X、α、原始貢獻、各種 q、最終 coalition |
| 歷程 | 候選 X、接受與否、誤差、步長、停止原因 |
| Pruning | 移除對象、依據、移除前後狀態 |
| 成本與付款 | 各階段查詢量、耗時、付款來源與金額 |

沿用 CSV，新增欄位或伴隨檔案；不改寫舊欄位意義。

#### B. 外部 evaluator

1. 使用各方乾淨訓練資料的聯集建立 pooled-data global engine。
2. 驗證集選擇 checkpoint；測試集僅做最終品質檢查。
3. 污染實驗仍用乾淨外部評估資料，避免評估者被污染。
4. 記錄一般區域與較困難區域的預測誤差。
5. 合成真實函數與 global engine 同時評分。
6. 兩者對優劣判斷不同時獨立列出案例，不選擇性採用有利結果。

global engine 是研究用集中訓練評估模型，不代表已實作或部署聯邦訓練。

#### C. 舊結果處理

先檢查是否有足夠狀態可重評。若未保存 X 或模型，重新執行可重現條件；不得從單一誤差反推出 X。

**驗收**

- [x] 同一 accepted X 可由內部、global、真實函數三種來源評分（正式 240 個條件完成）。
- [x] 重載模型與 X 能重現分數（正式 clean／poison 各 120 個 accepted X，重評零差異）。
- [x] 拒絕的最後候選不覆蓋 accepted X（回歸測試涵蓋）。
- [x] evaluator 不參與演算法決策（僅在條件完成後評分）。
- [x] 所有條件使用相同 box feasibility 檢查。
- [x] 無解、無有效 q、solver failure 保留獨立狀態（既有狀態與測試保留）。

### P2：驗證 α 更新是否改善最終 X

**目的：**先回答值不值得更新，再研究如何收斂。

**2026-09-22 狀態：**A–D 的 seeds 0–4 pilot 與 seeds 100–129 正式比較均已完成。正式 60 個情境、240 個條件全部保存並可重評。`E_global` 的六個方法對 A 比較均未得到 95% CI 完全小於零；動態 C／D 成本約為 A 的 14.0–16.5 倍。完整正式證據、限制與決策見 `docs/p2_formal_20260922.md`。

| 方法 | 初始 α | 更新 |
|---|---|---|
| A | Uniform | 不更新 |
| B | Stage 0 品質權重 | 不更新 |
| C | Uniform | 既有固定步長更新 |
| D | Uniform | 既有 Guarded 更新 |

不更新須明確實作為單次倒推基準，不能用名為 fixed 的更新政策代替。

**控制條件**

- 固定 coalition，本階段不啟用 pruning。
- 沿用二維、target 0 設定。
- 乾淨與受控污染分開比較；受控污染固定 5 名參與者、1 名污染者。
- 同一情境所有方法使用相同內部 evaluator。
- 其他參數先沿用已檢視的新流程設定並完整保存。
- 最多 20 個 calibration rounds，同時報告實際查詢成本。

**依序執行**

1. 5-seed pilot：檢查保存、評分、耗時與錯誤狀態。
2. 修正純實作錯誤，固定正式版本。
3. 30-seed 正式配對比較。
4. 先看外部 X 品質，再看 α–q 與內部指標。
5. 檢查中位、改善最大、惡化最大與失敗案例。

**必要圖表：**配對外部誤差、真實函數誤差對照、α–q 差距與外部誤差關係、品質隨查詢量變化、停止原因與失敗率。

**驗收**

- [x] 正式實驗與 Uniform 不更新比較，而非只比較較弱初始值。
- [x] Clean／Poison 各保留全部 30 seeds，包括 1 個 box failure 與 1 個 invalid q。
- [x] 分開報告品質、成本、停止原因與失敗率。
- [x] 未將 α–q 收斂當成唯一成功；正式結果顯示一致性改善未轉化成穩定外部品質改善。
- [x] 未用測試 global evaluator 選最好的一輪作正式交付。
- [x] 必要圖表與配對 bootstrap 已產出。

P2 決策：目前以 A（Uniform、不更新）為主要品質／成本基準；C／D 保留為研究對照。下一步進入 P3，驗證 q 的語義與外部效度，暫不以正式結果重新調整 P2 參數。

**決策**

- D 優於 A／B：進入更新機制改良。
- C／D 只改善一致性：轉向分析 q 定義。
- A／B 同樣好且便宜：保留靜態權重，限縮動態更新適用情境。
- pooled evaluator 與真實函數判斷衝突：依 pooled evaluator 作主要結論，同時將真實函數結果與 evaluator／truth 分歧列為限制及診斷證據。

### P3：驗證貢獻估計與低成本近似

**目的：**查明 q 是否適合更新、pruning 或分潤。

**2026-10-06 正式狀態：**已用獨立 seeds 3600–3629 完成 Clean／Controlled poison 共 60 scenarios、240 participant effects。現行 operational `C_i` 對完整移除 pooled effect 的平均 Spearman 為 0.083／-0.047，方向一致率為 40.0%／48.7%，98.75% simultaneous CI 均未通過事前門檻。正式驗收與 720 筆保存評分重載通過，結論為「未建立外部效度」。詳見 `docs/p3_external_validity_formal_20261006.md`。

#### 第一層：候選點的影響

保留 engine 群體與評分標準，只移除候選點，檢查現有估計是否反映候選點影響。

#### 第二層：完整參與者的影響

移除該參與者的候選點、最佳化 engine 與 α，從固定倒推起點重跑，以凍結外部 evaluator 比較。這是倒推階段反事實實驗，不等同重新訓練所有 local models，也不等同重新招募其他參與者。

**小規模參考實驗**

- 最多 5 人 coalition 起步，控制資料、模型與隨機性。
- 比較現有估計與完整移除後的外部品質差。
- Shapley 在前述流程穩定後，才作少量離線參考。
- 先定義 coalition value 與空集合價值；未完成前不把 LOO 稱為 Shapley。

**指標：**正負號一致率、貢獻排序相關性、預測負貢獻者移除後的外部改善比例、全部無正貢獻發生率、查詢量與耗時。

比較以原始 C_i 為主、q 為輔，避免正值截斷掩蓋差異。

**驗收**

- [x] 區分候選點與完整參與者影響。
- [x] 排名不一致案例、每 seed 相關與原始 participant effect 均保存。
- [x] q 無效案例不被刪除（Clean／Poison 各 1/5）。
- [x] 報告方向／排名差異與兩種反事實的查詢成本。
- [x] 不把離線完整移除驗證當成部署方案。

**決策：**q 只能反映候選點影響時限制其主張；與外部改善關係弱時，不再以 α–q 對齊作主要方法理由。

### P4：依證據改善更新與 pruning

**目的：**每項修改對應已觀察到的問題。

**2026-09-23 狀態：**無進展停止已實作為 opt-in，patience 2 的 5-seed Clean／Poison pilot 分別節省 36.9%／54.8% 總查詢量，10 個配對的 accepted X、α、accepted round 與 `E_global` 均完全一致。機制維持 `converged=False` 並使用獨立停止原因。完整證據見 `docs/p4_no_progress_pilot_20260923.md`；正式 30-seed 驗證前預設仍為停用。

**正式更新：**seeds 100–129 的 patience 0／2 配對已完成。Clean／Poison 分別降低 48.8%／55.8% 總查詢量，60 個 accepted state 全部一致，且沒有新增 box failure、invalid q 或 solver failure。patience 2 已納入後續預設，指定 0 仍可重現舊行為。詳見 `docs/p4_no_progress_formal_20260923.md`。

**受限版再驗證：**`box_constrained`、seeds 0–4 的 Clean／Controlled poison 配對分別節省 36.9%／62.8% 查詢量，10 個 accepted state 全部一致，20 筆重評零差異。限制修正沒有破壞無進展停止的 accepted-state 保證。詳見 `docs/validity_adjustment_pilot_20260923.md`。

**修改順序**

1. 無進展停止：步長達下限、持續拒絕且 accepted state 無進展時停止；與收斂分開。
2. 更新穩定性：每次只比較一種改動，例如阻尼或單步 α 變化限制。
3. 貢獻計算成本：依 P3 評估降低頻率；重用前確認 coalition、α、候選點與評估設定相容。
4. Pruning：固定其他設定，比較開啟／關閉；記錄外部品質、正常者錯刪與污染者漏刪。
5. 付款：只讀最終 accepted、pruning 後狀態，記錄 eval、opt_fallback 或無有效來源。

**必要測試**

- [x] rejected proposal 不覆蓋 accepted state（無進展停止回歸與 10 組配對驗證）。
- [x] 無進展停止不被標示為收斂；無有效 q 仍保留獨立狀態。
- [x] pruning 後權重與 agent 順序對齊（Stage 3B 移除與 accepted snapshot 回歸測試）。
- [x] solver recovery 結果進入後續狀態（warm start 與 accepted-state 回歸測試）。
- [x] fallback 不回寫 α、eval reports 或 pruning 判斷（付款 fallback 回歸測試）。
- [x] 單一參與者、無正貢獻與求解失敗有明確結果（付款與不可行子空間測試）。

**驗收**

- [x] Pilot 改善成本：Clean／Poison 總查詢量降低 36.9%／54.8%。
- [x] Pilot accepted X 與 pooled `E_global` 逐值一致，證據不只來自內部 L_eval。
- [x] 與同環境、同版本 patience 0 control 作 seed 配對比較。
- [x] Stage 3、contribution 與 payment 查詢全部計入。
- [x] 無進展停止改善成本且不改 accepted state，保留；未改善的 α 調整與 pruning 不納入主要方法。

### P5：情境擴充與消融

**目的：**建立可寫入論文的適用範圍與限制。

**2026-09-23 新版 pilot：**已改用 `box_constrained`、固定 participant IDs、乾淨 pooled evaluator 與有效配對口徑。八情境共 80 筆成功交付皆可行、40 組 A/D 配對完整，重評零差異。D 在六個情境平均持平或惡化，查詢量為 A 的 6.40–12.98 倍；A 維持主方法。高目標附近平均僅 23.2 筆 evaluator 樣本，五個 seeds 全部標示 sparse。詳見 `docs/validity_adjustment_pilot_20260923.md`。

依序擴充資料異質性、樣本量不平衡、低品質參與者、刻意污染、更高維度與 coalition 大小、困難目標與外插情境。一次先改一個因素，首輪不跑完整笛卡兒積。

**消融至少包含：**不更新 α、更新但無 Guarded、有更新但無 pruning、完整方法、新貢獻近似對目前近似。

**驗收**

- [x] Pilot 的 A／D 有獨立效果比較；C 與 contribution/pruning 依 P2／P3 既有證據註明停止理由。
- [x] 舊版 hard-target box failure 保留為診斷；新版 80 筆成功交付均通過 solver 內 box 限制。
- [x] 高目標區域另報 evaluator truth MSE，並用真實函數交叉檢查。
- [x] 結論限於 seeds 0–4 的單因子 pilot。
- [x] 模型、資料、設定、來源快照、指令與 80 個解的重評結果足以重現新版 pilot。
- [x] 完成修正硬限制後的 P5 pilot；正式 30-seed 結果仍未執行。

### P6：參與者招募與商業模式

**目的：**核心品質結果成立後，補完整應用流程。

**2026-09-23 新版 pilot：**使用 budgets 2.5／3.5／8.0、diversity eta 0／0.5、10 名完整候選池與最多三人限制。90 筆結果均符合預算、人數與 box，重評零差異；exact 的代理目標逐 seed 不低於同目標 greedy，但六個條件的平均 pooled error 均未優於該條件中表現最好的 greedy。詳見 `docs/validity_adjustment_pilot_20260923.md`。

目前已有品質篩選、報價、預算與多樣性導向貪婪選擇；後續對齊老師流程並驗證，不重新建置整套系統。

**執行項目**

1. 分開資格測試目標與正式倒推目標。
2. 保存測試方案、品質分數與報價。
3. 定義品質與預算下的選人問題。
4. 小型實驗用精確求解比較貪婪法。
5. 若納入多樣性，明確定義目標函數，不直接假設仍是標準線性背包。
6. 分開討論招募成本、基本報酬與貢獻分潤。

**驗收**

- [x] 新版 5-seed pilot 的 90 個結果均未超出預算或三人上限。
- [x] 選人比較包含 final X 的 pooled／true 品質、可行性與查詢量。
- [x] Exact 與 objective greedy 使用相同集合目標與限制；legacy 另列為既有行為對照。
- [x] 招募承諾與基本報酬明列為同一筆負債；剩餘池與未執行的貢獻結算分開記錄，不提出完整公平性保證。
- [x] 完成多預算、eta 與實際 coalition-size 取捨 pilot；正式 seeds 尚未執行。

## 四、成果保存、討論節點與執行清單

### 4.1 階段交付紀錄模板

每階段複製一份，尚未完成的欄位明寫待驗證。

| 欄位 | 待填內容 |
|---|---|
| 階段／日期／狀態 | — |
| 問題 | 要回答什麼 |
| 假說 | 哪個機制帶來什麼效果 |
| 版本 | commit、設定、資料與模型識別 |
| 方法 | baseline、改動、控制條件 |
| 結果 | 品質、成本、失敗、不確定性 |
| 證據 | 執行指令、輸出位置、圖表、測試結果 |
| 結論 | 支持／不支持／證據不足 |
| 限制 | 未驗證情境與近似假設 |
| 下一步 | 繼續／修改／停止哪個方向 |

### 4.2 與老師同步

1. P0 後：確認主問題與成功標準。
2. P2 後：討論動態 α 是否值得繼續。
3. P3 後：確認貢獻定義與論文主張範圍。
4. P5 前：固定主方法，避免擴大後更換方向。

以上是研究溝通安排，不代表每個技術操作都需另行批准。

### 4.3 下一輪最小執行清單

- [x] 固定 P2 正式版本與參數。
- [x] 寫清 α、C_i、各種 q 與付款關係。
- [x] 核對 exclusion 語義，並使 warm start、recovery、exclusion 共用 solver 內 box 限制。
- [x] 補存 accepted X、模型與重現資訊。
- [x] 建立凍結的 pooled-data global evaluator。
- [x] 完成 evaluator 與真實函數交叉檢查。
- [x] 建立兩個不更新 α 基準。
- [x] 執行 5-seed pilot。
- [x] 通過資料流與保存驗收，固定正式研究口徑。
- [x] 執行 30-seed 配對比較，產出第一次方向決策。
- [x] P3 建立候選點排除與完整參與者移除的同 seed pilot 比較。
- [x] P4 實作 Guarded 無進展停止並完成同環境 5-seed 配對 pilot。
- [x] P4 執行 seeds 100–129 的 patience 0／2 正式配對驗證。
- [x] P5 執行 8 個單因子情境的 5-seed A／D pilot，並重評 80 個 accepted X。
- [x] P5 保留 hard-target box failure，確認正式實驗前須修正 solver 硬限制。
- [x] P6 建立共用集合目標的 exact／greedy 基準，完成 5-seed final-X 比較與 15 個解重評。
- [x] 實作 `legacy_unconstrained`／`box_constrained`，重現舊版 2 筆高目標違規並完成限制回歸測試。
- [x] 以固定 participant IDs 與乾淨 pooled evaluator 重跑 P5 新版 pilot；80 筆成功解皆可行且重評零差異。
- [x] 在受限版本重驗 P4 無進展停止；10 組 accepted state 零差異。
- [x] 以 budgets 2.5／3.5／8.0、eta 0／0.5 重跑 P6；90 筆結果通過預算、人數、可行性與重評驗收。

### 4.4 決策與規格變更紀錄

| 日期 | 階段 | 證據 | 決策／變更 | 影響版本與後續工作 |
|---|---|---|---|---|
| 2026-09-22 | P2 | 正式 30-seed `E_global` 比較均未建立穩定改善；C／D 成本為 A 的 14.0–16.5 倍 | 以 A 作目前基準，C／D 保留為研究對照，進入 P3 驗證 q | `docs/p2_formal_20260922.md`；P3 暫不啟用 pruning |
| 2026-09-22 | P3 | 5-seed pilot 的符號一致率 46.7%／60.0%，負貢獻移除改善率均為 50% | q 限定為候選點內部影響；暫停 pruning 與 Shapley，P4 先做無進展停止 | `docs/p3_contribution_pilot_20260922.md` |
| 2026-10-06 | P3 | 正式 Clean／Controlled poison 的平均 Spearman 0.083／-0.047，方向一致率 40.0%／48.7%；60 scenarios、240 effects 與重評驗收完整 | 未建立現行 `C_i` 的外部效度；保留作內部診斷，不用於自動 pruning 或完整公平分潤主張 | `docs/p3_external_validity_formal_20261006.md` |
| 2026-09-23 | P4 | 無進展停止 pilot 的 10 個 accepted state 完全一致；查詢量降低 36.9%／54.8% | 機制維持 opt-in，進入 30-seed patience 0／2 正式驗證；不改 P2 品質結論 | `docs/p4_no_progress_pilot_20260923.md` |
| 2026-09-23 | P4 | 正式 60 個 accepted state 完全一致；查詢量降低 48.8%／55.8%，無新增失敗 | patience 2 納入後續預設；P2 主方法仍為靜態 A，pruning 維持暫停 | `docs/p4_no_progress_formal_20260923.md` |
| 2026-09-23 | P5 | 8 情境、40 組 paired seeds；D 成本為 A 的 6.34–11.61 倍，且在低品質、五人 coalition、高 target 平均品質惡化；80 解重評零差異 | A 維持主方法；D 保留對照。hard target 暴露 box 不可行解，正式 P5 前先修正 solver 限制 | `docs/p5_one_factor_pilot_20260923.md` |
| 2026-09-23 | P6 | 15 個 coalition 無超預算／box failure；exact 的集合目標最高但 pooled error 並非最低；15 解重評零差異 | Exact 作小型離線基準；選人目標不可直接替代 final X 品質，正式版須以 pooled error 共同決策 | `docs/p6_recruitment_pilot_20260923.md` |
| 2026-09-23 | 有效性修正 | 舊版重現 2 筆 box 違規；新版 P5 80 解、P6 90 解皆可行且重評零差異；受限 P4 10 組 accepted state 一致 | 正式研究改用 `box_constrained`；A 維持主方法；招募代理目標有效性列為後續問題；正式 30-seed 另行固定版本 | `docs/validity_adjustment_pilot_20260923.md` |
| 2026-10-03 | 靜態初始 α | 五方法 × 五情境 × 30 seeds 共 750 條件；全部有效並通過 pooled 重評。三新方法於一般乾淨情境對 A 的校正區間皆跨 0；低品質情境對 A 改善，但未證明優於 B | 一般主方法維持 A；B 是低品質情境的後續候選。新方法不宣稱普遍優勢或付款公平性 | `docs/alpha_initial_formal_results_20261003.md` |
| 2026-10-08 | 重抽樣／指數初始 α | A、B、重抽樣品質、指數權重 × 三情境：60 條件 pilot 與獨立 seeds 4100–4129 的 360 條件正式比較，全數有效可行；權重與 pooled 重載通過，seed 4100 完整重播零差異；六項新法對 B 的 95% 與 Bonferroni 同時區間均跨零 | 未建立新初始化優勢，保留 A／B；重抽樣平均誤差略低但輸入點數較多，指數法未見一致收益。不依正式結果調參 | `docs/alpha_robust_initialization_results_20261008.md`；凍結原 Stage 3 目標、停用更新／pruning／結算 |

**第一個里程碑：建立可信、可重評的 X 品質比較。**依此決定後續優先研究 α 更新、貢獻估計或 pruning。
| 2026-10-08 | 重抽樣品質機制與獨立驗證 | 四種單次初始化 × 三情境：14,400探索、120樣本數pilot、600次數敏感性、獨立116seeds的4,176條件；全部有效可行，重載差0；seed4200三情境36條件完整重播差0；82項測試通過 | 探索中α盲測波動平均降低約7%–10%，但九項確認比較的Bonferroni同時區間均跨零；低品質對B有普通95%改善訊號，未通過校正；查詢重用省重複API但未建立普遍品質／成本優勢，保留A／B | `docs/bootstrap_quality_research_results_20261008.md`；停用更新／pruning／結算，固定原Stage3目標；±0.5 targets僅探索 |

## 留出驗證重抽樣＋收縮：2026-10-08 完成紀錄

| 日期 | 階段 | 證據 | 決策 | 結果文件 |
|---|---|---|---|---|
| 2026-10-08 | 留出驗證重抽樣＋收縮 | 五方法×三情境×獨立126seeds，1,890條件全部有效可行；六項A/B確認比較校正區間均跨零；378組重用核對、完整seed4400三情境15條件重播零差異；99測試通過 | 乾淨對A普通95%改善訊號未通過多重校正，對B未建立優勢；方法初始化＋Stage3耗時約A/B的5.5–5.8倍，不全面取代基準 | `docs/oob_quality_shrink_results_20261008.md`；原20盲測、停用更新／pruning／結算 |


## 2026-10-10：下一輪 α 初始化比較已交付

A、B、既有OOB品質收縮、最佳組合OOB收縮與bootstrap第90百分位保守品質，在三情境以新seeds4600–4629完成450條件。所有解可行，重評與完整重播差異0。兩新方法的12項確認比較均未建立校正後穩定優勢；既有OOB僅乾淨對A通過本輪18項Bonferroni判準，未優於B。最佳組合初始化＋Stage3時間約A的40–50倍，保守品質成本接近A/B但尚未證明品質優勢。

此輪樣本不與歷史輪次合併。完整方法、限制、結果與驗收見docs/next_alpha_results_20261010_notion.md及docs/next_alpha_requirement_audit_20261010.json。Notion：https://app.notion.com/p/3f5bf435d47481dd907bd5ce6449b6cb。


## 2026-10-10：保守品質＋留出收縮獨立比較已完成

重複五折20次，每折只用train16 bootstrap1000次估第90百分位風險q，以heldout4選Uniform收縮λ，平均100折。A、B、原保守品質、相同切分但不收縮的診斷、新方法，在三情境新seeds4800–4829完成450條件。重評與完整重播差異0；新方法12項Bonferroni配對比較皆未建立穩定優勢，初始化＋求解約原保守品質5倍。資料分離與診斷q對齊驗收通過。

維持基準，不把平均下降解讀成穩定改善。所有舊輪次不覆蓋、不合併。本地報告docs/conservative_shrink_results_20261010_notion.md；驗收docs/conservative_shrink_requirement_audit_20261010.json；Notion：https://app.notion.com/p/3f5bf435d47481a89cbedc0cd4c10669。

## 2026-10-10：R5 自適應誤差相關權重已完成並整合主文件

五折資格資料16／4切分，在λ0／0.25／0.5／0.75／1中選留出混合MSE最低的矩陣收縮強度，最後用全20筆重估一次α。A、B、原保守、固定Corr precision0.5與新方法，以正式新seeds5000–5033、三情境完成510條件。N34由事前研究解析度0.005和獨立於正式的開發批次配對SD計算，不保證檢定力；正式未用pooled調參。

全部510有效可行，α／pooled重載差0；29項回歸與失敗分析測試、102條件2550折內資料分離／矩陣／留出損失核對、102組舊固定法與B端點核對、六組实际fresh／cache倒推比較通過。完整重播seed5000三情境15條件，資料／模型／λ／α／X／分數差0。

新法乾淨對B通過六比較校正，平均Δ−0.00428587653、同時CI[−0.00872432993,−0.00021686039]；三情境均未勝固定0.5相關對照，其餘確認比較未通過。初始化＋Stage3時間約固定相關2.71–3.41倍、B3.73–3.82倍，不是整體流程成本。尚未建立自適應λ新增機制的收益；保留A／B，下一步限定乾淨情境三法獨立複驗，不增加變體。

本輪完整報告docs/adaptive_corr_results_20261010_notion.md；正式登錄docs/adaptive_corr_preregistration_20261010.json；逐項驗收docs/adaptive_corr_requirement_audit_20261010.json。後續唯一持續更新Notion為「α 初始化研究總紀錄」https://app.notion.com/p/3eebf435d47481f0998cdf5db86b68ce，R5已收入附錄F；舊單輪頁為歷史快照。舊研究不覆蓋、不併樣本。

## 2026-10-11：R6乾淨情境三法獨立複驗完成

沿用原B、固定相關0.5、自適應相關公式，正式新seeds6000–6039、N40、120條件全有效可行；三項Bonferroni同時區間皆跨零，R5乾淨對B的確認優勢未重現。不能宣稱三法等效或無效。自適應初始化＋Stage3约固定3.44倍、B3.55倍。

歷史15條件與R5一致，20回歸＋4分析失敗測試通過；40新方法條件1000折驗收通過；完整seed6000三條件重播，資料／模型／α／X／評分差0。維持A/B，暫停新增α變體，優先研究资格代理損失與交付品质關聯，尚未開始後續研究。

結果與R5分開，不合併样本；完整報告docs/corr_replication_results_20261011_notion.md，登錄docs/corr_replication_preregistration_20261011.json，驗收docs/corr_replication_requirement_audit_20261011.json。Notion仍為同一主文件，R6追加附錄G，無新頁。
