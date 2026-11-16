# 初始 α 靜態倒推研究：正式結果與驗收（2026-10-03）

## 結論

本輪完成五種事前 α、五個情境、30 個獨立 seeds 的靜態倒推比較，合計 750 條件。**在一般乾淨情境，三個新方法均未建立相對 Uniform 的穩定 pooled 品質優勢；因此不取代 A（Uniform）作一般主方法。**在 `low_quality_25pct`，三個新方法相對 A 的配對 pooled 誤差均降低，且預先指定的三候選 Bonferroni 區間完全低於零；但既有 B（Stage 0 品質權重）的平均誤差更低，三新方法均未證明優於 B。此情境更支持繼續研究「何時切換到便宜的 Stage 0 品質權重」，而不是宣稱本輪找到新的普遍最佳 α。

α 是 Stage 3 的資訊權重，並非價格或付款份額。本輪不執行動態更新、貢獻估計、pruning、結算；不能從品質結果推論經濟公平性。

## 方法與有效性

五方法：A Uniform、B Stage 0 inverse-MSE、`quality_shrink=(A+B)/2`、`ensemble_shrink`（Stage 0 blind set 上的非負 simplex 集成，再與 A 對半混合）、`target_peer_shrink`（Stage 0 品質乘上其他模型對各 Stage 2 proposal 的 target 接近程度，再與 A 對半混合）。後兩者多出的模型查詢列入初始化成本。全部方法在同 seed／情境共用資料、local models、固定 coalition、proposals、box 限制與凍結 pooled evaluator。pooled 僅對有效交付 X 做事後評分；truth 僅作診斷。

先以 seeds 0–4 跑新版 pilot：125／125 條件完整，125 個解有效。單獨重播 seed 0，coalition、proposal、blind test、五種方法的 α、X 與 pooled 分數最大差均為 0。正式 seeds 300–329 事前查核未在既有 manifests 使用，來源與設定封存於 `alpha_initial_formal_preregistration_20261003.json`。正式 750／750 條件完整且有效，沒有 initialization fallback；所有 X 符合 `[-1,1]`（容差 `1e-8`），重載 pooled evaluator 的分數差不超過 `1e-7`，來源快照與 artifacts 雜湊通過驗證。

凍結 evaluator 的 target-near 測試樣本數中位數：二維相關情境約 261–266，五維 193；這五個情境均無 `<30` 的標記。pooled 模型對合成 truth 的測試 MSE 平均約 `0.00013`（二維）與 `0.00110`（五維）。這些是合成資料上的 evaluator coverage 與誤差，不能當成真實世界驗證。

## Pooled 主要品質

數值為 30 seeds 有效交付的 pooled 絕對 target 誤差平均；越低越好。五方法的每情境有效數均為 30，失敗數均為 0。

| 情境 | A Uniform | B Stage 0 | Quality shrink | Ensemble shrink | Target-peer shrink |
|---|---:|---:|---:|---:|---:|
| 參考乾淨 | 0.02042 | **0.01764** | 0.01876 | 0.01910 | 0.01896 |
| 樣本量不平衡 4× | 0.02187 | **0.01777** | 0.01978 | 0.01934 | 0.01962 |
| 低品質 25% | 0.03195 | **0.01795** | 0.02231 | 0.02244 | 0.02121 |
| 五維 | 0.04474 | 0.04197 | **0.03707** | 0.04844 | 0.03909 |
| 五人 coalition | 0.01371 | 0.01304 | 0.01262 | **0.01152** | 0.01223 |

粗體僅標記該列最低平均，不代表配對統計優勢。完整平均、中位數、Q1、Q3、失敗數、成本及每條件數值見 `analysis.json` 與各情境 `summary.csv`。

正式配對差定義為 `E_method − E_A`；負值有利新方法。三新方法在參考乾淨情境的平均差分別是 `−0.00166`、`−0.00132`、`−0.00146`，三個 98.33%（每情境三候選 Bonferroni）bootstrap 區間皆跨 0。因此核心情境**證據不足以宣稱改善**。樣本量不平衡、五維與五人 coalition 的三候選區間亦跨 0。

低品質情境的配對差與預先指定的 98.33% 區間：

| 方法對 A | 平均差 | 區間 |
|---|---:|---:|
| Quality shrink | −0.00964 | [−0.01872, −0.00158] |
| Ensemble shrink | −0.00951 | [−0.01719, −0.00328] |
| Target-peer shrink | −0.01074 | [−0.01756, −0.00518] |

即使用更保守、涵蓋五情境 × 三候選共 15 比較的 Bonferroni 區間作額外敏感度檢查，這三個上界仍為負（依序約 `−0.000075`、`−0.00203`、`−0.00423`）。然而 B 對 A 的低品質配對平均差是 `−0.01400`，95% 區間 `[−0.02062, −0.00759]`，且三新方法對 B 的差異區間均跨 0。**新方法在低品質情境改善 A，尚未改善 B。**

## 成本與 pooled／truth 衝突

參考乾淨情境每條件平均模型 API 查詢（初始化 + Stage 3）：A `305.7`、B `342.8`、Quality shrink `351.7`、Ensemble shrink `397.0`、Target-peer shrink `343.0`。Ensemble 另外平均有 3 次初始化模型 API 呼叫（每次 20 個 blind-test 輸入），Target-peer 有 6 次；A、B、Quality shrink 為 0。方法自身查詢量會受其 α 導致的求解器路徑影響，故不應把整體差異全歸因於初始化公式。共同訓練、qualification、proposal 與 pooled evaluator 的成本每 seed 僅記一次，沒有重複加進五方法。

參考乾淨情境中，與 A 配對的 30 seeds 裡，pooled 與 truth 對優劣方向衝突的次數：Quality shrink 11、Ensemble shrink 13、Target-peer shrink 9。低品質情境分別是 7、7、6。由於部分方法差值很小，即使 evaluator 的整體測試 MSE 低，排序也可能不同；這些衝突需隨結果揭露。truth 不回饋本輪方法選擇。

## 限制與下一步

本輪只含合成函數、固定 IDs、五個有完整配對的情境；異質性與污染情境因既有 P5 資格失敗問題未納入。Stage 0 blind test 僅 20 筆，集成權重可能對其過度擬合。這輪沒有真實招募者的策略性報價、支付激勵或公平性驗證，也沒有證明任何方法全域最佳。

維持 A 作一般靜態主方法，保留 B 作低品質情境的優先候選。下一個可驗證問題是：能否用**事前、無 pooled 洩漏**的品質診斷，可靠判斷何時採 B，而不增加乾淨情境的失敗或成本？應先固定切換規則，再用新的獨立 seeds 驗證；本輪正式 seeds 不可用來選門檻。三個新方法不應因低品質對 A 有利就宣稱超過既有 B。

## 產物索引

- 研究定義與驗收：`docs/alpha_initialization_study_20261003.md`
- 正式預登錄：`docs/alpha_initial_formal_preregistration_20261003.json`
- 新版 pilot：`alpha_weight_outputs/alpha_initial_pilot_v2_20261003/`
- 隔離重播：`alpha_weight_outputs/alpha_initial_replay_seed0_v1_20261003/`
- 正式結果、`analysis.json`、`verification.json`、圖與逐條件 artifacts：`alpha_weight_outputs/alpha_initial_formal_v1_20261003/`
- 程式：`initial_alpha_methods.py`、`run_alpha_initialization_study.py`、`analyze_alpha_initialization_study.py`、`verify_alpha_initialization_study.py`

此結果是新實驗版本，不與過去 P2／P5 的動態 α 結果拼成同版本樣本。沒有覆蓋歷史輸出。
