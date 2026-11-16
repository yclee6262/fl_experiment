# P4 Guarded 無進展停止正式結果（2026-09-23）

## 正式問題與判準

本實驗驗證 patience 2 的 Guarded 無進展停止，能否在不改變 accepted state 與 pooled evaluator 品質的前提下降低查詢成本。

正式成功需同時滿足：

1. seeds 100–129 每一組 control／treatment 的 accepted X、α、accepted round 與 `E_global` 一致。
2. 不增加 box failure、invalid q 或 solver failure。
3. Stage 3、contribution 與 payment 總查詢量降低。
4. 提前停止保持 `converged=False`，停止原因獨立記為 `guarded_no_progress`。

## 配對設計

- Control：`guarded_no_progress_patience=0`。
- Treatment：`guarded_no_progress_patience=2`。
- seeds 100–129，2D，target 0，Uniform 初始 α，Guarded eta 0.5，最多 20 rounds。
- Clean：Stage 1 coalition、uniform evaluator。
- Poison：固定 5 人含 1 名污染者、trimmed evaluator。
- Control 與 treatment 使用相同 Anaconda Python、程式版本、資料、local models、候選點與 frozen pooled evaluator 設定。
- evaluator 只在演算法完成後評分，不參與提前停止。

## 正式結果

| 指標 | Clean | Controlled poison |
|---|---:|---:|
| Paired seeds | 30 | 30 |
| 觸發無進展停止 | 25/30 | 28/30 |
| Accepted-state mismatch | 0/30 | 0/30 |
| 平均 rounds：control → treatment | 18.37 → 8.93 | 19.50 → 8.60 |
| 平均總查詢：control → treatment | 32,143.1 → 16,453.3 | 198,527.2 → 87,837.8 |
| 全 seeds 總查詢節省 | 48.8% | 55.8% |
| 僅觸發 seeds 的查詢節省 | 54.5% | 58.3% |
| Treatment box failure | 0/30 | 0/30 |
| Treatment invalid q | 0/30 | 0/30 |
| Treatment solver failure | 0/30 | 0/30 |

60 個配對 seed 的 final accepted X、final α 與 `E_global` 最大絕對差均為 0，final accepted round 亦全部相同。未觸發停止的 7 個 seeds，其 rounds、停止原因與查詢量完全不變。

四組輸出各自從 frozen evaluator checkpoint 重載 accepted X；120 個保存解全部 `mismatches=0`。

## 決策

正式結果支持將 patience 2 納入後續程式版本的 Guarded 預設值：

- `--guarded-no-progress-patience 2` 現為預設。
- 指定 `0` 可重現 P2 時期跑到 max rounds 的舊行為。
- manifest 保存有效 patience，舊結果不會被回填或改寫。
- `guarded_no_progress` 只代表 accepted state 在目前 Guarded proposal 規則下無進展，不代表 α–q 固定點收斂或全域最佳。

這項結果只證明計算成本改善，不改變 P2 的方法選擇：目前仍以 Uniform、不更新 α 作主要品質／成本基準；動態 Guarded 保留為研究對照與特定情境方法。

## 輸出

- `alpha_weight_outputs/p4_no_progress_formal_control_clean_20260923`
- `alpha_weight_outputs/p4_no_progress_formal_clean_20260923`
- `alpha_weight_outputs/p4_no_progress_formal_control_poison_20260923`
- `alpha_weight_outputs/p4_no_progress_formal_poison_20260923`
- `alpha_weight_outputs/p4_no_progress_formal_analysis_20260923`

正式圖使用 `p4_no_progress_formal_v2.png`。同目錄較早的 `p4_no_progress_formal.png` 保留但標題仍寫 pilot，僅屬產圖過程紀錄，不作正式引用。

## 限制與下一步

本次只驗證 2D、target 0、目前兩種 coalition/evaluator 情境。若未來更改 Guarded proposal、最低步長、接受門檻或 solver，需重新驗證重複 proposal 假設。

依 P2／P3 決策，下一個 P4 工作不啟用 pruning，也不繼續調整 α 更新。可進入 P5 的單因素情境擴充，並以靜態 A 為主要基準、Guarded patience 2 為成本已改善的研究對照。
