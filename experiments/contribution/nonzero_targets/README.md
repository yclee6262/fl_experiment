# 固定團隊的非零目標角色實驗

依[固定設計](design.md)，20個歷史神經模型團隊，T=0/−0.5/0.5/2.7，共80新反推任務。每個目標重新生成錨點、full軌跡、局部CA/CM/J及共同起點whole-task LOO；不挪用舊端點。

```sh
OMP_NUM_THREADS=1 python -m experiments.contribution.nonzero_targets.run \
  --output-dir outputs/nonzero_role_targets/new-run
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.contribution.nonzero_targets.analyze \
  --source outputs/nonzero_role_targets/new-run --output-dir outputs/nonzero_role_targets/new-run/analysis_final
OMP_NUM_THREADS=1 python -m experiments.contribution.nonzero_targets.validate \
  --source outputs/nonzero_role_targets/new-run
OMP_NUM_THREADS=1 python -m experiments.contribution.nonzero_targets.validate_events \
  --source outputs/nonzero_role_targets/new-run
```

輸出拒絕覆寫。`--smoke`只用一個歷史team、T=.5、短反推及3輪求解，不能作正式結果。

- `run.py`：維護中的frozen custom trace、branch_state與one-round decide，不改歷史snapshot。目標透過server.target_T動態傳入；模型與cases來源hash保存。
- `analyze.py`：每情境10隊bootstrap，report原始真值誤差、inverse高誤差、角色符號／排名、peer/full TV、reserve與target−0配對差。
- `validate.py`：重新載入原模型，獨立核對full／LOO端點、逐輪角色分數、加總分數與來源hash。`validate_events.py`再逐筆重算cache miss回覆、核對cache hit、跨目標起點與本地反推殘差。
- 所有既有成員固定，受控污染者強制納入；不執行Stage0或品質gate。逆推偏差高仍保留，不能宣稱這是篩選後的部署結果。
- 每個owner跨target共享相同隨機起點seed；這不是跨起點穩定度驗證（後者屬任務11）。
- 真值只用於離線決策與whole-LOO參照；局部分數與分配只使用F/Fminus。whole-LOO沿用full錨點平均起點，並非重新招募／反推的系統整體刪除。
- 每個任務使用獨立cache，報cache miss／requests；對observer與原custom的獨立核對成本另列，不能冒充各種方法獨立成本。
- 200000次隊bootstrap僅描述此歷史模型擴充，非獨立確認、等效性或公平證明。失敗、無正分reserve、undefined rank保留。

`sign_flip_summary.csv`以每隊分子／分母做cluster ratio區間並保留零分母draw；零事件退化[0,0]不是族群零風險證明。

正式manifest在執行前另外鎖定實際函式來源（HostServer、load_server、trace、方向重建、one-round decide、反推）及其hash，避免只記錄別名模組。正式任務9最終run為`outputs/nonzero_role_targets/task9_final_v1/`；第一版缺部分helper的執行前鎖，已保留並另重跑，不能將其成本併入單次方法成本。

最終run的`analysis_final/`保存統計表；`acceptance.json`與`event_acceptance.json`保存兩層驗證結果。`replay_acceptance.json`逐任務核對第一次與最終重跑，僅排除wall seconds；`prior_development_costs.json`另記第一次與smoke的額外計算，不加入最終方法查詢數。

`python -m experiments.contribution.nonzero_targets.validate_statistics --source outputs/nonzero_role_targets/new-run`從原始80任務獨立重建240摘要與144配對區間；它讀取`analysis_final/`，新run分析時請將`--output-dir`設為該目錄。正式結果與驗收摘要見[results](results/README.md)。
