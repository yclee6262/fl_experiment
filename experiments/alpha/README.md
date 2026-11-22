# 十種固定 alpha 初始化

四個基礎分布為 U（均勻）、Q（盲測 inverse-MSE）、E（simplex ensemble fitting）、S（其他模型對候選的支持度）。六個混合為 EU、EQ、SU、SQ、SE、QU，皆取兩個完整 alpha 向量各一半，再執行一次 Stage 3 best-of。

```sh
python run.py initialization \
  --source alpha_weight_outputs/four_alpha_v1_20261014/formal_batch_1/source \
  --seeds 20000 --scenarios reference_clean \
  --output-dir outputs/initialization-new
```

省略 `--scenarios` 時重播三情境。seed 必須存在於來源。來源須包含各情境的 `shared_costs.jsonl`、候選、盲測資料、`local_models.pt` 與 pooled evaluator；這些資料不在 Git。

每個方法使用同一案例、固定聯盟及提案，停用剪枝與結算。先封存全部方法的 `decisions.jsonl`，再載入 pooled evaluator 評分至 `results.jsonl`。有效狀態和 fallback 要和平均誤差一起檢查。模型查詢成本的共同準備與逐方法求解分開記錄。

- [methods](methods/README.md)：方法定義與介面。
- `replay.py`：載入既有模型與獨立求解。
- [analysis](analysis/README.md)：統計與十法彙整。
- [results](results/README.md)：論文使用的十法精簡數據。
- [solver_diagnostics](solver_diagnostics/README.md)：固定原模型與錨點，比較差分步長及推論精度；屬數值診斷，不修改正式求解器設定。

修訂實驗以 `--solver-policy float32_slsqp_v1` 明確啟用固定差分步長；省略時仍為歷史模式。設定、成本口徑及後續硬預算研究見[求解規約](../shared/SOLVER_POLICY.md)。
