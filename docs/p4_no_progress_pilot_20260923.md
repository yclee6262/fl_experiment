# P4 Guarded 無進展停止 Pilot（2026-09-23）

## 目的與完成條件

P2 顯示部分 Guarded runs 在步長下限持續拒絕相同 proposal，最後只因 `max_rounds` 停止。P4 第一項修改的目的，是在 accepted state 已無進展時停止重複 Stage 3 與 contribution 計算，同時維持 delivered X、α、外部品質與付款所讀取的 accepted state。

本機制不改變收斂定義。觸發後的停止原因是 `guarded_no_progress`，`converged=False`。

## 實作規則

新增 `--guarded-no-progress-patience`：

- `0`：停用，完整保留原本行為。
- 正整數：當 Guarded proposal 被拒絕、更新步長已達 `minimum_update_rate`，且連續拒絕次數達 patience 時停止。
- 本 pilot 使用 patience 2。
- 到達門檻的最後一輪不再產生下一個未測 proposal。
- 最終摘要與付款仍從最後 accepted state 讀取；被拒絕候選只保留在 rounds 歷程。
- `guarded_no_progress` 與 `fixed_point_tolerance`、`audited_uninformative`、`max_rounds` 分開保存。

回歸測試建立一個會在最小步長重複拒絕的案例：原流程會跑滿 20 rounds，新流程在第 6 round 停止，accepted X 與 α 不變。

## 配對設計

使用同一個 Anaconda Python、相同程式版本、seeds 0–4 與相同資料／模型／evaluator 設定，配對比較：

- Control：patience 0。
- Treatment：patience 2。
- Clean：Stage 1 coalition、uniform evaluator。
- Poison：固定 5 人含 1 名污染者、trimmed evaluator。
- 兩者皆為 Uniform 初始 α、Guarded、eta 0.5、最多 20 rounds。

不得直接與較早 P2 pilot 的數值比較，因較早結果使用目前已失效的暫存 Python 環境，solver 與套件環境不同。本次因果比較只使用同環境重跑的 control／treatment。

## 結果

| 指標 | Clean | Controlled poison |
|---|---:|---:|
| Paired seeds | 5 | 5 |
| 觸發無進展停止 | 2/5 | 4/5 |
| Accepted-state mismatch | 0/5 | 0/5 |
| 平均 rounds：control → treatment | 16.0 → 10.8 | 16.2 → 7.2 |
| 平均總查詢：control → treatment | 30,640.2 → 19,330.2 | 161,894 → 73,177 |
| 全 seeds 總查詢節省 | 36.9% | 54.8% |
| 僅觸發 seeds 的查詢節省 | 63.8% | 55.7% |

10 個配對 seed 的 final accepted X、final α、accepted round 與 `E_global` 全部逐值一致，最大差值為 0。未觸發的 seeds 其 rounds、停止原因與查詢量也完全不變。

四個 control／treatment 輸出各自從 evaluator checkpoint 重載 accepted X，合計 20 個解皆為 `mismatches=0`。

輸出：

- `alpha_weight_outputs/p4_no_progress_control_clean_20260923`
- `alpha_weight_outputs/p4_no_progress_clean_20260923`
- `alpha_weight_outputs/p4_no_progress_control_poison_20260923`
- `alpha_weight_outputs/p4_no_progress_poison_20260923`
- `alpha_weight_outputs/p4_no_progress_analysis_20260923`

## 判讀與下一步

本 5-seed pilot 通過 P4 的最小技術驗收：改善查詢成本、沒有只依賴內部 `L_eval`、使用同環境配對比較，且新增成本已計入。它不證明動態 α 本身值得採用；P2 對動態更新的品質結論維持不變。

目前保留此機制為 opt-in，預設 patience 0，以免用 5-seed pilot 改寫既有方法。若要納入主要方法，下一步使用獨立 seeds 100–129 作 patience 0／2 正式配對，成功條件為：

1. 所有觸發案例的 accepted X、α、accepted round 與 `E_global` 皆一致。
2. 不增加 box failure、invalid q 或 solver failure。
3. 查詢量有實質降低，停止原因維持獨立。
4. 正式結果不用於重新挑 patience；若修改 patience，建立新實驗版本。
