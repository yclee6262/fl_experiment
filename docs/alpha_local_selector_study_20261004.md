# 局部校正選擇初始 α：研究紀錄

## 問題與方法

先前把 Best-agent 和 Local-bias-risk 加為第六／第七個初始 α 候選，獨立 seeds 700–759 沒有建立相對原五候選的穩定 pooled 品質收益，且增加模型請求。本輪測試另一種資訊配置決策：**保留原五個初始 α 與五次 Stage 3，只改從五個 X 中選哪個交付**。被選中的 X 對應哪組 α，該組就是交付時的初始 α。

原方法在 20 筆 Stage 0 盲測資料上，把 B 權重集成的有號殘差對 `[1,x]` 做一次全域 ridge 校正，再以校正後的絕對目標差選 X。新方法對每個候選 X 以 `K_j(X)=exp(-||x_j-X||²/(2×0.25²))` 加權同一批殘差；各 X 各自求 ridge 係數，截距懲罰 0.01、各座標係數懲罰 1。它選 `abs(sum_i B_i f_i(X)-local_bias(X)-target)` 最小的可行候選。

資料來源僅有事前 Stage 0 標記、local model 預測與 Stage 3 候選 X。pooled evaluator、合成真實函數皆只於交付後評分。其經濟意義是：固定資訊取得預算，用與候選交易位置更相關的歷史預測誤差來決定採用哪個初始權重方案；沒有實際貨幣收益、誘因相容或付款公平性的主張。新規則不多做 Stage 3，但局部校正需要本地計算，時間另計。

## 開發訊號與 pilot

在已使用過的 30、60、60 seeds 上離線比較多種風險與校正規則後，局部 bandwidth 0.25 在三批的 pooled 配對平均差（新減原五候選）分別為 -0.000423、-0.000617、-0.000495。合計 150 seeds 的描述性平均為 -0.000529，95% 區間仍跨零。此規則是從多個試法中選出的，這些**開發數字不能作正式優勢**。

5-seed pilot（seeds 0–4）共 20／20 條摘要條件有效；原全域與新局部選擇的 10／10 次重算通過。A、B、原五候選的 X 與 pooled 分數逐項等於舊 pilot，確保新流程只改選解規則。pilot 中原五候選平均 pooled 誤差 0.014757，新局部規則 0.014407；模型請求均為 2031.0。pilot 只驗收流程與成本計數。

## 正式登錄與門檻

`docs/alpha_local_selector_formal_preregistration_20261004.json` 在正式啟動前固定全新 seeds 800–859、同一基本乾淨情境、五組 α、bandwidth 0.25 及主要判準。要求 60 個有效配對，`E_local-E_global` 的固定 RNG 10,000 次 seed-bootstrap 95% 區間完全小於零、無較高失敗數，並分開報告模型請求及時間。只有基本情境通過後，才擴到複雜情境。若未通過，保留負面結果，不用正式 seeds 修改 bandwidth 後仍宣稱獨立驗證。

正式執行目錄：`alpha_weight_outputs/alpha_local_selector_formal_v1_20261004/`。

## 獨立正式結果與決策

60／60 個 seeds 均有雙方有效交付解，120／120 次選擇可從保存模型、盲測資料及候選 X 重算；13／13 個事前登錄來源雜湊相符。共同分析確認 240／240 條摘要條件完整、無重複或缺漏。79 項回歸測試通過。

| 方法 | 平均 pooled 絕對誤差 | 平均 truth 絕對誤差 | 平均方法模型請求 | 平均方法內本機秒數 |
|---|---:|---:|---:|---:|
| 原五候選全域校正 | 0.014729 | 0.012572 | 1901.5 | 0.153112 |
| 新五候選局部校正 | 0.015467 | 0.012415 | 1901.5 | 0.153147 |

新法減舊法的 seed 配對 pooled 差平均 **+0.000738384**；10,000 次 seed-bootstrap 的 95% 區間 **[-0.000607652, +0.002373968]**，跨零。60 seeds 中新法較好 5、相同 43、較差 12；觀察失敗數均為零。新法 truth 均值略低，但 pooled 是事前主要品質判準；逐 seed 的 pooled／truth 優劣方向衝突 8／60。由於 pooled 沒建立改善，**正式門檻未通過**，不擴到複雜情境，也不把 bandwidth 0.25 納入主要方案。

同批模型請求完全相同，局部 ridge 的本機計時只多約 0.000035 秒／條件；這是本地工作負載，不代表實際網路延遲。缺少品質優勢時，不能宣稱它更具經濟效率。

另在三批已使用 seeds 上做開發篩選：盲測 bootstrap 最佳專家機率、Stage 0 leave-one-out 邊際資訊增益、前兩名專家、Quality shrink、Ensemble shrink、Target-peer shrink 作為第六候選。它們的 pooled 差跨批次方向不穩定，尚無足夠理由消耗新的正式 seeds。這些結果保存於 `alpha_weight_outputs/alpha_information_portfolio_development_20261004.csv`；不把開發平均寫成正式效果。

可重評檔案：`alpha_weight_outputs/alpha_local_selector_formal_v1_20261004/local_selector_decision.json`、`selection_verification.json`、`analysis.json`、`reference_clean/summary.csv`、`reference_clean/candidate_rows.csv`、`reference_clean/source_snapshot/`。
