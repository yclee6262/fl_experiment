# FedAvg 比較

```sh
python run.py baseline --output-dir outputs/fedavg-new \
  --dimensions 2 --agents 3 --samples 32 --rounds 1 \
  --mse-threshold 1000000 --disable-inverse-check
```

這是小型流程檢查設定。正式比較應明確指定節點數、樣本數、訓練輪次、資格門檻與隨機種子。

`compare_fedavg.py` 使用合成資料比較 FedAvg 後反推與協作反推，輸出 manifest、execution.log、report.json。`fl_comparison_numerical.py` 提供原數值基準。誤差由已知合成函數評估。資料生成相同，但 admission、模型初始狀態與搜尋預算存在差異，不能視為已完全匹配的效能 benchmark。

不需要醫療資料集下載或 Flower；相關舊展示已移入本機封存。

[paired_benchmark](paired_benchmark/README.md)提供30seed、三維度與兩污染條件的配對基準、硬查詢預算、同成員FedAvg控制及資源計量；正式入口 `python run.py paired-benchmark --help`。
