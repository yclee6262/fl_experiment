# Proposal 附近付費標記的初始 α：合成環境開發結果

## 目的與資料邊界

先前多個新初始 α 只重用 Stage 0 的 20 筆 blind labels，未能穩定超過既有 B 或五候選。這次檢查**真正不同的資訊來源**是否有幫助：在 Stage 2 proposals 已知、Stage 3 尚未開始時，於各 proposal 附近取得少量、獨立的可信標記，用來估計 local models 在倒推相關區域的誤差並設定初始 α。研究仍以凍結 pooled evaluator 的交付 X 誤差為主要品質判讀；pooled 分數不進權重或求解器。

本次只用先前已作開發的低品質 seeds 2600–2629；前 5 筆作流程 pilot，後 25 筆作另一開發批次。它們**不是新的獨立正式樣本**。合成環境直接呼叫真實函數產生額外標記；真實部署若無法在交付前取得相同類型的標記，此法不可照搬。標記的價格、等待時間與取得失敗風險尚未知，故本次只報數量，不推論貨幣效用。

## 事前固定的方法

對固定的三人 coalition，依 seed 固定 RNG，在每位參與者的 Stage 2 proposal 周圍各取 3 個二維點，共 **9 個新點**。擾動是標準差 0.1 的常態值，超出 box `[-1,1]` 則裁到邊界；不在最終 X 上查真值。合成真實函數為每點產生標記 `y_j`。各 local model 對全部 9 點批次預測一次，因此初始化新增 **3 次模型 API 呼叫、27 個模型輸入樣本**。

令 `m_i` 為既有 20 筆 Stage 0 blind-test MSE，`l_i=(1/9)Σ_j(f_i(z_j)−y_j)^2` 為 9 筆 proposal 附近標記上的 MSE。設 `r_i=(5m_i+9l_i)/14`，再取

`α_i = (r_i+10^-5)^-1 / Σ_k(r_k+10^-5)^-1`。

這是**新初始 α 公式**；5 筆 Stage 0 MSE 的偽計數抑制 9 筆局部標記帶來的過度變動。α 非負、總和為 1。新方法只執行**一次**新 Stage 3 集成目標 `|Σ_i α_i f_i(X)−T|`，不使用五候選選解。比較對照為同批保存模型下的單次既有 B、五候選新目標，以及舊五候選。兩批使用相同固定參與者、proposal、box 與 pooled evaluator；只有初始 α 所需的新增標記與權重不同。270 個新標記點與保存的 pooled 資料、Stage 0 blind 點在模型可見 `float32` 表示下均無完全相同座標。

## 結果與驗收

兩批 30／30 個 X 有效可行，沒有 solver 失敗。獨立驗證器從保存模型、proposal 與 seed 重算 270 個標記點、標記值、local predictions、α 及交付 pooled 評分，最大差皆為 0；Stage 0 MSE 另由原候選 CSV 核對。品質差定義為「新局部標記 α－對照」，負值代表新法較好。下表區間僅是**已用開發 seeds 的描述性**配對 bootstrap，不得當正式優勢檢定。

| 批次 | n | 相對單次 B 的 pooled 平均差 | 描述性 95% 區間 | 新較好／較差 | 相對新五候選平均差 |
|---|---:|---:|---:|---:|---:|
| pilot seeds 2600–2604 | 5 | +0.003443 | [−0.001121, +0.010689] | 1／4 | +0.003096 |
| 開發 seeds 2605–2629 | 25 | −0.001365 | [−0.006168, +0.003068] | 9／16 | +0.004161 |
| 合看 30 筆（只供描述） | 30 | −0.000564 | [−0.004784, +0.003209] | 10／20 | +0.003984 |

合看時相對 B 的差值中位數為 **+0.000553**，最差退步為 **+0.021360**（seed 2615）。平均略低主要不能代表多數 seed 受益：10 筆改善、20 筆惡化。新法平均 Stage 3 呼叫 **352.1**，既有 B 為 **328.2**；新法另多 3 次初始化批次呼叫和 9 筆新標記。五候選新目標平均 Stage 3 呼叫 1,723.8，品質平均優於這個新單次方法。因此目前沒有同時改善品質與成本的開發訊號，也沒有理由將標記成本視為可被品質收益抵銷。

## 決策與限制

**不把這個局部標記 α 升為主要方法，也不為目前公式啟動新正式 seeds。**先前用額外 80 筆一般區域標記的新候選亦未通過正式比較；本次改為 9 筆 proposal 附近標記，仍未看到跨兩開發批次一致的品質改善。不能再用這 30 筆 pooled 結果反覆調擾動標準差、標記數或偽計數後，把同一批資料稱為獨立驗證。

這只否定**此標記位置、數量和固定公式**已顯示出足夠訊號；不代表所有新增資訊均無價值。若真實應用可取得可靠且可計價的候選附近標記，較直接的研究問題可能是它們應用於**候選 X 的選解**，而非在同一批少量資料上繼續微調初始化。取得方式、每筆成本與可接受延遲需先明確定義，pooled evaluator 仍應凍結作交付後評估。

產物：`explore_targeted_label_alpha.py` 定義與執行方法，`verify_targeted_label_alpha.py` 獨立重評，`analyze_targeted_label_alpha.py` 統計。逐 seed X、α、標記與成本保存在 `alpha_weight_outputs/alpha_targeted_label_alpha_pilot_2600_2604.json`、`alpha_weight_outputs/alpha_targeted_label_alpha_development_2605_2629.json`；各自的 `.verification_v2.json` 通過。合併分析保存在 `alpha_weight_outputs/alpha_targeted_label_alpha_development_analysis_2600_2629.json`。

開發完成後的來源快照與輸入／輸出 SHA-256 在 `alpha_weight_outputs/alpha_targeted_label_alpha_source_snapshot_20261005/`、`alpha_weight_outputs/alpha_targeted_label_alpha_development_manifest_20261005.json`。這是**事後可重現封存**，不能改稱為事前登錄。
