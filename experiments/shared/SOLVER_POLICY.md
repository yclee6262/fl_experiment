# 修訂版求解規約 v1

機器可讀設定：[solver_policy_v1.json](solver_policy_v1.json)。介面：[solver_policy.py](solver_policy.py)。本規約是任務三歷史案例診斷後的開發決策，不是獨立確認結果。

## 已採用的數值設定

- 具名規約 `float32_slsqp_v1`：float32原模型、不重訓；SLSQP絕對差分步長1e-5、ftol1e-9、maxiter200；輸入box[-1,1]、交付可行容差1e-8。
- beta不是simplex；只透過A beta受box限制。係數起點1/K，方向引擎從錨點平均出發，自訂30輪、方向權重alpha、保持原退火與切線設定。
- 固定初始化重分析保留best_of，以同一Lopt選有限且可行的端點；精確平手先係數。solver success、有效交付與停止原因分開報告。
- 採用1e-5的依據：已測步長中最小且在本批270個起點擾動均有可辨別目標變化；90案例無起點停滯。沒有以真值誤差挑選步長；1e-6雖平均真值更低，仍有零差分及無改善案例。不能宣稱1e-5是通用最佳值。
- 主引擎消融固定為係數單獨、自訂單獨、best_of；這些是同數值設定下的比較，不會先選贏家再抹除其他結果。
- 舊呼叫預設 `coefficient_eps=None`、歷史模式不變。新規約必須明確啟用；歷史輸出及凍結來源不可改寫。

## 成本與預算

1. 初始化重分析沿用上述迭代上限，報告實際模型查詢，**沒有硬查詢上限**，不能稱同查詢成本比較。雙引擎查詢須兩者相加。
2. 任務8的同成本搜尋研究固定panel評估預算100、200、400，對K個模型為K倍單模型查詢。三人團隊對應300、600、1200。這是預先指定的資源網格，不是從真值選最佳預算。
3. 同成本研究須另實作硬預算runner：關閉快取、在呼叫前檢查剩餘預算、初點／候選挑選／終點評估全計入；best_of兩引擎各半、不轉移未用額度；返回預算內已評估的最佳有限可行點，無可交付點要保留失敗。模型查詢以每個輸入樣本計，不得以batch呼叫偷換成本。
4. 上述硬預算runner尚屬任務8，**現在的initialization入口不執行該上限**。訓練、反推、初始化與貢獻評分成本分項並報總計；FedAvg訓練通訊不等同搜尋query。
5. 只使用可取得的Lopt作線上選擇；合成真值為離線主評估，pooled為輔助，不可餵回搜尋或評分規則調參。

## 目前可執行的入口

```sh
python run.py initialization --source alpha_weight_outputs/four_alpha_v1_20261014/formal_batch_1/source \
  --seeds 20000 --scenarios reference_clean --solver-policy float32_slsqp_v1 \
  --output-dir outputs/initialization/revised-new-run
```

省略 `--solver-policy` 等於historical。每列結果保存具名規約與雜湊；manifest保存完整規約。其他stages/coalition/baseline入口尚未自動切換，必須在其重跑任務中明確接入並驗證，不能只改核心預設造成隱性研究變更。

## 回歸驗證

```sh
OMP_NUM_THREADS=1 python -m experiments.alpha.solver_diagnostics.validate_policy \
  --source alpha_weight_outputs/four_alpha_v1_20261014/formal_batch_1/source \
  --reference outputs/alpha_solver_diagnostics/task3_verified_20261123 \
  --output-dir outputs/alpha_solver_diagnostics/policy-regression-new
```

對任務三90案例檢查兩個具名規約×係數/custom/best_of：共540檢查，逐一比較X、Lopt、引擎及query。此驗證測試正式設定介面是否等價於診斷的暫時步長注入，並非新增有效性證據。輸出保留來源hash、失敗與新增查詢成本。
