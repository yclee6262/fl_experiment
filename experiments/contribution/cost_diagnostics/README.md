# 四方案冷快取成本查核

先由既有入口實際重播24個解析模型團隊，每隊每方案使用獨立空快取：

```sh
OMP_NUM_THREADS=1 python run.py contribution --output-dir outputs/contribution_cost/new-run/replay
python -m experiments.contribution.cost_diagnostics.analyze \
  --source outputs/contribution_cost/new-run/replay --output-dir outputs/contribution_cost/new-run/analysis
```

摘要檢查96個輸出的端點核對、cache事件數、calls=requests−hits、各stage合計及歷史基準端點。區分原搜尋、錨點分支、模型／共同／LOO分支與最後評分；驗證成本不混入方法查詢。mean/SD為24隊描述統計，不推論至真實網路延遲或訓練成本。

AV、ACM、J、LOO是不同counterfactual，成本不能當作相同數量的無損近似。解析custom來源不涉及本次float32係數引擎修正。原來源與新輸出分開保存，拒絕覆寫。
