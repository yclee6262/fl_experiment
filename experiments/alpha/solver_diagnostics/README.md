# 係數引擎精度診斷

本套件執行論文修訂任務 3。它是舊模型的配對數值診斷，不是獨立確認，也不修改正式引擎設定。

## 執行前固定的設計

- 來源：`four_alpha_v1_20261014/formal_batch_1/source`，三情境，每情境依 seed 排序的首 30 個案例（20000–20029）。冒煙測試用首 1 個。
- 原模型權重與錨點不變，T=0、均勻 alpha、box [-1,1]、係數初值 1/K、自訂引擎初值為錨點平均、自訂 30 輪。
- 五種數值設定：float32 預設差分；float32 eps=1e-6、1e-5、1e-4；原 float32 權重升為 float64 推論且用預設差分。後者沒有重訓。
- 每設定分別呼叫原係數及原自訂引擎，再按有效解的同設定 Lopt 選出 best-of（平手優先係數），並記錄 solver success 與 valid delivery 的差別。
- 原 float32 預設另與正式 best-of 入口比對 X、loss、chosen_engine，並與保存的 A_uniform 結果核對 X、loss、引擎與查詢數；要求差值 <=1e-12、引擎與查詢數完全相同。來源各情境 host_server 快照須與目前核心雜湊一致。
- 記錄初末 Lopt、真值誤差、Linf 移動、停滯（<=1e-12）、無改善（改善<=1e-10）、box 違反、API 呼叫／輸入樣本、時間、例外。初值／診斷評分成本另列，不混入求解成本。
- 對初始 beta 的每維作正差分，量測 x 轉 float32 完全相同及 Lopt 完全相同的比例；此診斷不等同整條軌跡的梯度品質。
- 輸出只表示本批 T=0、D=2、K=3、均勻權重結果，不推及十種權重或正式新情境；不在此選定正式步長。

## 使用

從 repo 根目錄執行：

```sh
OMP_NUM_THREADS=1 python -m experiments.alpha.solver_diagnostics.run \
  --source alpha_weight_outputs/four_alpha_v1_20261014/formal_batch_1/source \
  --limit 30 --output-dir outputs/alpha_solver_diagnostics/run-name
```

輸出資料夾不得已存在。`manifest.json` 記錄來源模型／紀錄與程式雜湊、版本及環境；`results.jsonl` 保存所有案例設定，含失敗；`summary.json` 提供聚合與配對差值。程式使用暫時 scoped patch 注入 eps，離開區塊必定恢復原 minimize。任何 core 檔案均不變。

測試：`OMP_NUM_THREADS=1 python -m pytest -q tests/test_alpha_solver_diagnostics.py`，涵蓋暫時 patch 的例外還原、精度介面等價與拒絕覆寫。
