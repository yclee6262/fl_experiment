# 四種單次 α 初始化：執行規格與驗收

日期：2026-10-08。本輪先完成 5-seed pilot，再完成独立 30-seed 比較；不預設新方法較好。登錄檔：`alpha_robust_initialization_preregistration_20261008.json`。

## 方法定義

- A：三人等權，每人 1/3。
- B：原 20 筆 Stage 0 盲測 MSE 取倒數，`α_i = norm(1/(MSE_i+1e-5))`。
- 重抽樣品質：保存三個模型對原 20 筆的預測，每次對資料列共同有放回抽取 20 筆，重算個別 MSE 與 B 權重，重複 1,000 次後平均。重抽樣 RNG 為 `20261008 + seed`；保存權重標準差與 2.5%/97.5% 分位數，這些是重抽樣波動診斷，非正式品質 CI。不重訓模型、不新增標記。本版重新取得三次批次預測，按方法獨立成本計入 3 requests/60 input samples。
- 指數權重：Uniform prior，`α_i = norm(exp(-(MSE_i-min(MSE))/τ))`；`τ=max(median(MSE),1e-5)`，溫度比例固定 1，不調參。扣掉最小值只是數值穩定，不改變原 softmax 權重。

四方法均只執行一次 Stage 3，固定原目標 `Σα_i|f_i(X)-target|`、`best_of`、Custom 30 iterations、box [-1,1]。不更新 α、不剪枝、不結算、不開 Stage 3 快取。pooled evaluator 在交付之後才評分，不影響初始化或 solver；truth 為事後診斷。

## 三情境

所有情境固定 IDs [1,2,20]、三人 coalition、二維、target 0、20 筆資格測試、local models 30 epochs、pooled evaluator 100 epochs。每 seed 四方法共用資料、切分、模型、proposal、evaluator。

- 乾淨：三人各 500 樣本，observed label noise SD=0.05。
- 低品質 25%：20 名訓練參與者中最後 5 名的 observed label noise SD 改為 0.3；固定 coalition 中只有 ID 20 受影響。外部評估仍用基準乾淨標記設定。
- 樣本量不平衡 4×：20 人樣本數按既有規則分配，固定三人為 229、247、917；保持相同模型訓練 epoch，因此樣本數也影響訓練步數。標記 noise SD 均 0.05。

## 統計與验收

Pilot seeds 0–4 共 60 條件，只驗收流程。正式 seeds 4100–4129 共 360 條件；啟動前查核 294 份 manifests/登錄檔，未發現重用。以 seed 配對平均差 `新法 pooled error − B pooled error` 為主要比較，負值有利新法。兩新法 × 三情境共六项確認比較，使用 10,000 次 bootstrap、固定 RNG 20260924，報一般 95% 及 99.1667% Bonferroni 同時區間。對 A 的比較與 B−A 為描述性參考。

只有來源/資料/權重/分數重載驗收通過、完整 30 有效配對、同時區間完全低於零且未增加觀察失敗数，才支持情境限定品質優勢。品質與 requests、input samples、time 分開，不預設誤差與金錢的換算。

每條件即時保存 CSV；資格失敗、求解失敗與不可行解保留，品質留空。外部評分成本不混入 Stage 3。模型訓練/資格/proposal/evaluator 共用成本每 seed 一次，獨立部署成本則各方法需承担一次共用準備。

## 重現

使用 `/Users/yclee/anaconda3/bin/python`，OMP/MKL/OpenBLAS threads 均固定 1。runner `run_alpha_initialization_study.py`；驗收與統計 `analyze_robust_alpha_study.py`；必要參數、環境、來源快照、模型、pooled 資料、切分、proposal、α 與 X 全部保存。既有輸出不覆蓋。
