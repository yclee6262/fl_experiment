# Alpha 校正迴圈研究

本資料夾保留迭代 alpha 校正的完整研究實作。固定初始化比較位於相鄰的 `alpha/`；兩個入口分開執行。

## 方法與程式

`experiment.py` 提供固定聯盟的 Stage 3A 校正，以及可選的 Stage 3B 剪枝後重啟。保留原研究的校正、回復及結算邏輯。`run.py` 負責參數檢查、新輸出目錄保護與失敗狀態；也可透過 `python -m experiments.alpha_calibration` 執行。

資格分數 r、搜尋權重 alpha、任務後貢獻分布 q 分開處理。每一校正輪：

1. 使用目前 alpha 執行 Stage 3 搜尋。
2. 移除各 owner 的候選重新搜尋，保留模型評分面板，據此計算 operational exclusion scores。
3. 正分正規化得到 q_raw；若設定 exploration mass，另保存加入均勻底量後的 q。
4. 產生 alpha_next = normalize((1 − eta) alpha + eta q)，下一輪重新求解並評估。
5. 摘要與結算使用最後已接受且已評估的 alpha、X、q；不使用尚未評估的下一輪提案或最後被拒絕的狀態。

這裡的候選排除是整段搜尋的 operational score，與單輪 CA／CM／J、同時移除模型和候選的 whole-task LOO 有不同定義。pooled evaluator 與 synthetic truth 用於離線評估，不作 alpha 更新的輸入。校正中的 L_eval 由所選本地模型評分規則決定。

## 執行方式

從 repo 根目錄執行；只需要 `requirements.txt` 的依賴，這個入口會自行生成合成資料及訓練模型。

```sh
python run.py calibration --help
```

小型流程檢查（均勻／資格初始化，三種更新策略）：

```sh
python run.py calibration \
  --output-dir outputs/calibration-new \
  --seeds 42 --dimensions 2 --poison-ratios 0 \
  --initializations uniform,stage0 --evaluators uniform \
  --update-rates 0.5 --update-policies fixed,adaptive,guarded \
  --max-calibration-rounds 2 --custom-iterations 3 \
  --num-agents 3 --samples-per-agent 32 --epochs 1 \
  --external-evaluator-epochs 3 \
  --mse-threshold 1000000 --disable-inverse-check \
  --constraint-mode box_constrained --no-settle-payment
```

此設定放寬資格並縮短訓練、搜尋與校正，用於確認流程。研究比較需另外固定各項設定及評估標準。輸出目錄必須不存在，避免覆寫完成的實驗。

固定 alpha 的對照：

```sh
python run.py calibration \
  --output-dir outputs/calibration-fixed-new \
  --seeds 42 --dimensions 2 --poison-ratios 0 \
  --initializations uniform,stage0 --evaluators uniform \
  --no-alpha-update --no-settle-payment \
  --constraint-mode box_constrained
```

上述第二個命令使用預設研究規模（20 節點、每節點 500 筆、30 epochs）。若需快速檢查，可加上第一個範例的節點、樣本、訓練與搜尋設定。

## 主要設定

| 設定 | 含義 |
|---|---|
| `--initializations` | `uniform`、`stage0`、`dirichlet`、`reversed`；這是原校正研究的起始分布，不是十法固定初始化比較 |
| `--update-policies fixed` | 固定更新步長 eta；eta 大於零時 alpha 仍會更新 |
| `--update-policies adaptive` | alpha–q 差距增加時縮小步長，受 minimum update rate 限制 |
| `--update-policies guarded` | 重新求解後，L_eval 比已接受狀態差超過容差時拒絕該提案，縮小步長並從已接受狀態再提案 |
| `--no-alpha-update` | 固定 alpha 對照，只執行一輪；或將 update rate 設為零 |
| `--evaluators` | `trimmed`、`uniform`、`median`、`reputation`、`optimization`；最後一種讓評分跟隨 alpha，屬循環評分對照 |
| `--max-calibration-rounds` | 校正輪數上限；另有固定點容差、無有效正分及 guarded 無進展等停止原因 |
| `--guarded-no-progress-patience` | 預設 2；在最小步長連續拒絕達門檻即停止；0 表示關閉此停止條件 |
| `--enable-coalition-stabilization` | 開啟可選 Stage 3B，單次移除一個節點後重啟 Stage 3A；預設關閉 |
| `--no-settle-payment` | 停用結算；原研究預設開啟，付款以 r 與貢獻分布計算 |
| `--constraint-mode` | 建議明列 `box_constrained`；歷史預設為 `legacy_unconstrained` |
| `--comparison-suite` | `custom` 接受自訂組合；`p2`／`p5` 保留原研究的指定條件組 |

## 輸出與重現

- `run_manifest.json`：參數、版本、狀態、來源與產物雜湊。
- `summary.csv`：每條件最終接受狀態、停止原因、誤差與查詢成本。
- `rounds.csv`：各輪 alpha／q、接受與拒絕、步長、候選及評分。
- `agents.csv`：逐節點資格、權重與貢獻資料。
- `payments.csv`：啟用結算時的付款資料。
- `pruning.csv`：啟用 Stage 3B 時的紀錄。
- `artifacts/`：本地模型、資料與離線 evaluator。
- `source_snapshot/`：該次執行的來源；其中 `alpha_weight_experiments.py` 對應本資料夾的完整校正程式。

輸出寫入忽略的 `outputs/`，程式與本資料夾說明納入 Git。過去的校正資料仍可保留於本機 `alpha_calibration_outputs/` 或 `alpha_weight_outputs/`，不隨 repo 提交。

## 驗證

```sh
python -m pytest -q tests/test_alpha_calibration.py
```

回歸案例保留自原研究測試，涵蓋 r/alpha/q 分離、無資訊貢獻、exploration floor、guarded 拒絕後恢復、無進展停止、固定權重、結算分離及來源快照。這些檢查確認程式行為，研究效果由各自保存的實驗結果評估。
