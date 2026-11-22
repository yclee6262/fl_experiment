# 初始化結果分析

```sh
python run.py alpha-summary --output outputs/alpha-summary-new
```

`summarize_ten_initializers.py` 讀取 `alpha_weight_outputs/se_qu_extension_v1_20261107/registration.json` 指定的原六法、E/S 及 SE/QU 結果。核對 2700 筆配對覆蓋、原結果雜湊、權重組成、保存決策與 18 筆重播，重新計算均值和查詢數。既有校正區間取自凍結分析，不重新執行模型或 bootstrap。

輸出 `means_and_queries.csv`、`paired_comparisons.csv`、`audit.json` 及資料摘要。`--figure <path.pdf>` 可由相同數據重建初始化圖。

`analyze_four_alpha.py` 對應原六法的 12 個預先指定比較；`analyze_pure_es_extension.py` 對應純 E/S 補充分析。SE/QU 的 17 個比較是另一定義的 family，不能將分別校正的區間說成十法所有配對的共同校正。

## 既有引擎紀錄查核

```sh
python experiments/alpha/analysis/audit_saved_engines.py \
  --output-dir outputs/alpha_saved_engine_audit/new-run
```

讀取相同十法來源，保留來源雜湊，彙整 `chosen_engine` 與兩個 `engine_losses`，不新增模型查詢；輸出目錄已存在時拒絕覆寫。歷史 `bfgs` 欄位在受限案例代表 SLSQP 係數引擎。引擎選中率不能單獨證明數值失效；原模型的配對診斷見上層 `solver_diagnostics/` 文件。


## 修訂求解器後的十法重分析

設計在 [revised_analysis_design.md](revised_analysis_design.md) 固定；執行：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.alpha.analysis.analyze_revised_initializers \
  --input-root outputs/alpha_initialization_revised/task5_20261123 \
  --output-dir outputs/alpha_initialization_revised/task5_20261123/analysis-new
```

必須有三批完成的 `float32_slsqp_v1` 結果。程式拒絕既有輸出資料夾；由保存 X 衍生真值至 `derived_records.jsonl`，不改寫原資料。輸出含80列分方法／情境／指標摘要、原12對與SE/QU17對及45對探索family的完整配對區間、pooled/truth方向一致率與Spearman、歷史差異及270個評估器的MSE/RMSE與樣本數。所有bootstrap以90個seed為單位共同重抽200000次；不稱獨立確認或等效。

`truth_paired_intervals.pdf` 呈現九法相對U的真值誤差差值。程式可核對完整覆蓋、alpha及來源hash；原模型端點重算由上層 `revised_initialization/validate.py` 驗收。
