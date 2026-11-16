# 加權集成 Stage 3 目標：乾淨與樣本不平衡情境 pilot

## 問題與方法

低品質情境的獨立 180-seed 正式比較已支持：把 Stage 3 搜尋目標從 `Σ_i α_i |f_i(X)−T|` 改為 `|Σ_i α_i f_i(X)−T|`，在該情境平均降低 pooled 誤差與 local-model 請求。這裡只問該效果在**乾淨**及**4 倍樣本量不平衡**情境是否有初步轉移訊號，不宣稱正式跨情境優勢。

兩情境都新產生 seeds 2900–2904。每 seed 先用原流程保存資料切分、local-model checkpoints、固定 coalition `[1,2,20]`、Stage 2 proposals、五個初始 α、box `[-1,1]` 及凍結 pooled evaluator。原 Stage 3 目標產生舊五候選與原 B20 ridge 選解；新目標在**同一批保存模型與 α**上重新求解五候選，仍用相同 B20 選解。pooled evaluator 只在求解與選解後評分，合成 truth 只作診斷。兩個 pilot 均**不開 Stage 3 快取**，隔離目標函數的效果。

原流程輸出各保存於 `alpha_weight_outputs/alpha_ensemble_target_transfer_clean_pilot_2900_2904/` 與 `alpha_weight_outputs/alpha_ensemble_target_transfer_imbalance_pilot_2900_2904/`；新目標逐候選重播、評分及成本存於同名 `.json`。兩個原流程各 5／5 seeds 完整，選解重播 5／5 通過。獨立的 `verify_ensemble_target_replay.py` 重新載入全部 50 個新候選 X、α、pooled evaluator 與 local models，並重算 10 次選解；兩情境的評分與選解最大差均為 0，候選失敗 0。

## Pilot 結果

差值定義為「新目標－舊目標」；pooled 誤差差為負才代表新目標較好。模型呼叫只計 Stage 3 的 local-model API 請求，非真實網路價格。

| 情境 | 有效配對 | pooled 平均差 | 新目標較好／較差 | truth 平均差（診斷） | Stage 3 平均請求差 |
|---|---:|---:|---:|---:|---:|
| 乾淨基本 | 5／5 | −0.016426 | 4／1 | −0.010742 | −93.6 |
| 樣本量不平衡 4× | 5／5 | +0.002975 | 2／3 | +0.000310 | +14.4 |

乾淨情境的五筆 pooled 差依 seed 2900–2904 為 `−0.016574、−0.005316、+0.001562、−0.034480、−0.027323`；不平衡情境為 `−0.001921、+0.001643、+0.019618、+0.000084、−0.004549`。兩情境均有退步案例。乾淨 seed 2900 的 pooled 改善與 truth 變差方向相反，說明平均結果不能取代逐案例 evaluator 診斷。

**判讀：**乾淨情境有值得獨立驗證的初步訊號；不平衡情境呈現相反方向，與先前已用資料的負面開發訊號一致。每情境僅五個 seeds，不計算或宣稱正式品質優勢，也不根據本批修改目標公式、選解或 α。下一輪若要確認跨情境適用範圍，須先固定正式比較範圍、獨立 seeds、配對區間與失敗判準；不平衡情境應完整保留作限制檢查，不能只報乾淨情境。

## 重現與驗收指令

```bash
python run_validation_calibrated_alpha.py --output-dir alpha_weight_outputs/alpha_ensemble_target_transfer_clean_pilot_2900_2904 --scenario reference_clean --seeds 2900,2901,2902,2903,2904
python explore_ensemble_target_objective.py alpha_weight_outputs/alpha_ensemble_target_transfer_clean_pilot_2900_2904 --seeds 2900,2901,2902,2903,2904 --output alpha_weight_outputs/alpha_ensemble_target_transfer_clean_pilot_2900_2904.json
python verify_validation_calibrated_alpha.py alpha_weight_outputs/alpha_ensemble_target_transfer_clean_pilot_2900_2904
python verify_ensemble_target_replay.py alpha_weight_outputs/alpha_ensemble_target_transfer_clean_pilot_2900_2904 alpha_weight_outputs/alpha_ensemble_target_transfer_clean_pilot_2900_2904.json
python run_validation_calibrated_alpha.py --output-dir alpha_weight_outputs/alpha_ensemble_target_transfer_imbalance_pilot_2900_2904 --scenario sample_imbalance_4x --seeds 2900,2901,2902,2903,2904
python explore_ensemble_target_objective.py alpha_weight_outputs/alpha_ensemble_target_transfer_imbalance_pilot_2900_2904 --seeds 2900,2901,2902,2903,2904 --output alpha_weight_outputs/alpha_ensemble_target_transfer_imbalance_pilot_2900_2904.json
python verify_validation_calibrated_alpha.py alpha_weight_outputs/alpha_ensemble_target_transfer_imbalance_pilot_2900_2904
python verify_ensemble_target_replay.py alpha_weight_outputs/alpha_ensemble_target_transfer_imbalance_pilot_2900_2904 alpha_weight_outputs/alpha_ensemble_target_transfer_imbalance_pilot_2900_2904.json
```

Runner 會拒絕寫入已有內容的輸出目錄；重播請使用全新目錄，不覆蓋以上原始 pilot。
