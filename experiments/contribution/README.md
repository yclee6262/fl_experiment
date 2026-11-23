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

## 固定端點角色統計

[role_statistics](role_statistics/README.md)補齊論文角色分析的有效隊數、隊層級bootstrap區間、評分器配對差異與衍生影子付款；使用保存端點，不重新執行搜尋。
