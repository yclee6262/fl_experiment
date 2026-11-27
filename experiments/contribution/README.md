# 角色、分潤與成本

```sh
python run.py roles --output-dir outputs/roles-new
python run.py sampling --output-dir outputs/sampling-new
python run.py contribution --output-dir outputs/cost-new --limit 1
```

- `compare_roles.py`：重算保存端點的錨點 CA、模型 CM、聯合 J 與 full-task LOO 評分，並比較 full、peer、pooled 評分及影子分潤。
- `sample_rounds.py`：預設 20 團隊、每比例 2000 次抽樣，保留原 threshold；輸出每隊及整體 TV。`--draws`、`--limit` 可供小型檢查。只重播保存分數，無新增模型查詢。
- `replay_cost_comparison.py`：預設 24 解析模型團隊 × CA+V／CA+CM／J／LOO；`--limit 1` 選第一隊。每隊每臂獨立冷快取，核對歷史端點及計算實際 cache-miss calls。

必要本機資料：

| 入口 | 來源 |
|---|---|
| roles | `contribution_analysis/combined_contribution_layer2_experiment_v1_20261021/`、`method5_cost_experiment_20261030_attempt_04/`、`model_pool_combinations_20261105_attempt_01/`，及 cases 指向的 models/evaluator |
| sampling | `contribution_analysis/method5_cost_experiment_20261030_attempt_04/registration.json` 與 `records/*_New.json` |
| contribution | `contribution_analysis/model_pool_v_cm_validation_20261107_attempt_02/registration.json` 與 records |

原始資料與新輸出不隨程式提交。各入口必須使用不存在的輸出資料夾；保存端點的評分不會回饋搜尋。模型／錨點移除、評分器與時間範圍是不同選擇；query cost 不是網路時間或付款金額。

## 低成本近似完整 LOO 分配

[驗證設計](loo_allocation_validation/design.md)與[固定設定](loo_allocation_validation/protocol.json)：最後接受輪共同移除錨點及模型（J單輪）、累積錨點位移，與完整 LOO 的 bonus 分配比較。F₋ᵢ為主、完整F作對照，不使用pooled；主要終點是比例誤差、分配狀態與成本。

實作與分析程式放在 `experiments/contribution/loo_allocation_validation/`，測試放在 `tests/`，輸出放在 `outputs/loo_allocation_validation/<run-id>/`。`loo-allocation` 的用法與舊資料探索重播流程見[實驗說明](loo_allocation_validation/README.md)。

新增人數與問題複雜度探索：`python run.py loo-scaling --output-dir outputs/loo_allocation_scaling/<new-run-id>`，規約見[scaling_design.md](loo_allocation_validation/scaling_design.md)，使用合格乾淨團隊及F₋ᵢ分配參考。

## 固定端點角色統計

[role_statistics](role_statistics/README.md)補齊論文角色分析的有效隊數、隊層級bootstrap區間、評分器配對差異與衍生影子付款；使用保存端點，不重新執行搜尋。

## 代理準確度與成本查驗

[proxy_diagnostics](proxy_diagnostics/README.md)分開查驗V／CM及論文有號方向投影；[cost_diagnostics](cost_diagnostics/README.md)查核24解析隊的四方案冷快取實測成本與抽樣重播界線。

## 非零目標的角色介入

[nonzero_targets](nonzero_targets/README.md)固定20個歷史團隊，重新反推四個目標並計算CA/CM/J與whole-task LOO；入口 `python run.py nonzero-roles --help`。保存逐筆模型回覆驗算與隊層級統計，與不同起點穩定度研究分開。

## 反推與求解起點穩定度

[start_stability](start_stability/README.md)固定模型交叉5組反推×3組中央求解起點，核對角色符號、付款資格、金額及reserve；另有完全匹配的40個Shapley賽局及獨立歷史600三人賽局參照。入口 `python run.py start-stability --help`，主分潤政策的帳本重算見下節。

## 結算政策重算

[settlement_policy](settlement_policy/README.md)使用任務 11 的保存分數，比較原影子帳本、固定服務費的影子配置與獎金保留政策。入口 `python run.py settlement-replay --help`。
