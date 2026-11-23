# 觀測代理V與錨點投影查驗

依[固定設計](design.md)，使用保存端點與方向紀錄重算，不重新搜尋、不新增模型查詢。這是歷史資料重分析。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.contribution.proxy_diagnostics.analyze \
  --output-dir outputs/contribution_proxy/new-run
```

新目錄拒絕覆寫。需要本機 `model_pool_v_cm_validation_20261107_attempt_02/` 的registration、124個30輪紀錄、24個cost紀錄，以及20個 `method5_cost_experiment_20261030_attempt_04/records/*_New.json`。輸出保存來源hash、有效隊數、零分數、無正分保留額與200000次按隊重抽的描述性95%區間。

- `V_CM_summary.csv`：20神經、80既有解析、24新增解析來源各自的V/CM比較，full F和peer分開；不可合併成124隊同質母體。另有scenario/n分層CSV。
- `projection_summary.csv`：論文signed projection公式與同軌跡full-F anchor removal比較。由H、g與已觀察改善重算，核對各輪加總等於改善。這不是模型方向／路徑長投影，也不是剪正後重分配版本。
- `*_teams.csv`、`*_owners.csv`及projection_rounds：原始分母、簽名及會計量；含reserve分配以累積有號分數取正後計算。
- `audit.json`：來源／設計／分析hash、36492個V/CM分數核對、1938個fresh cost端點核對與投影守恆誤差。

V對CM是不同介入的差異，不預設CM為公平真值。Spearman可能為null（恆定分數），不得補0。符號以1e-10三分類，top-level sign_agreement是各隊同權平均；opposite_ratio是隊cluster重抽的總相反數／總owners，兩者分母與權重不同。

成本來源相容性：24 fresh的原cost AV使用full-F V，其ACM保留相同V/CM端點。本分析同時重算peer；11/12的peer成本重播才是CA+V／CA+CM正式peer查詢口徑。`baseline_only_V_new_calls=0`僅指既有baseline試探的V增量，CA分支及固定評分仍需成本，不將V的零增量擴張成CA+V全流程零成本。

此次不使用無法追溯的45%摘要。投影重算使用20神經團隊251輪、80owners；可檢查會計量與介入量差異，不能據此宣稱投影可替代CA或具有跨起點穩定性。
