# 修訂版十法初始化重分析

沿用原90 seeds × 三情境的模型、盲測、錨點與四基礎／六混合權重，明確啟用 `float32_slsqp_v1`；共2700個決策。屬歷史案例重分析，沒有新增獨立確認樣本。數值規約見 [shared/SOLVER_POLICY.md](../../shared/SOLVER_POLICY.md)。

```sh
OMP_NUM_THREADS=1 python -m experiments.alpha.revised_initialization.run \
  --output-dir outputs/alpha_initialization_revised/new-run
```

輸出目錄拒絕覆寫，registration保存模型、評估器與輸入hash及code hash。每批呼叫已實作的 `run.py initialization`，保存batch日誌與manifest。執行中斷保留各批已寫入資料；不得直接覆寫或把不完整紀錄當完成。

完整結果寫完後，依 [分析設計](../analysis/revised_analysis_design.md) 重算真值、配對區間、指標一致性與成本；不回寫原始結果，不沿用舊決策的信賴區間。原helper的 `truth_error`可能為null，須由保存X衍生並記載，不視為零誤差。

每個案例先封存十法決策，再做pooled離線評估。共同盲測3個batch API查詢（60個模型輸入樣本）、支持6個非自身查詢與來源資料準備，另列而不混入stage3；各方法的stage3與初始化成本個別保留。分析不設定事後等效界限。

端點獨立驗算：

```sh
OMP_NUM_THREADS=1 python -m experiments.alpha.revised_initialization.validate \
  --run-dir outputs/alpha_initialization_revised/new-run
```

重新載入270組原模型及評估器，核對保存X的Lopt與pooled、檔案hash及決策先封存後評估的紀錄，輸出endpoint_acceptance.json；拒絕覆寫該驗收檔。此驗算新增的8100參與者及2700評估器查詢另列，不加到方法的stage3成本。
