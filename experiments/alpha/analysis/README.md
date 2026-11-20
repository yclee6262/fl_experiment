# 初始化結果分析

```sh
python run.py alpha-summary --output outputs/alpha-summary-new
```

`summarize_ten_initializers.py` 讀取 `alpha_weight_outputs/se_qu_extension_v1_20261107/registration.json` 指定的原六法、E/S 及 SE/QU 結果。核對 2700 筆配對覆蓋、原結果雜湊、權重組成、保存決策與 18 筆重播，重新計算均值和查詢數。既有校正區間取自凍結分析，不重新執行模型或 bootstrap。

輸出 `means_and_queries.csv`、`paired_comparisons.csv`、`audit.json` 及資料摘要。`--figure <path.pdf>` 可由相同數據重建初始化圖。

`analyze_four_alpha.py` 對應原六法的 12 個預先指定比較；`analyze_pure_es_extension.py` 對應純 E/S 補充分析。SE/QU 的 17 個比較是另一定義的 family，不能將分別校正的區間說成十法所有配對的共同校正。
