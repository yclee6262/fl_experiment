# 固定端點角色統計與隊層級區間

依[固定設計](design.md)，重算20團隊（80 owner appearances）的有號分數對pooled整任務LOO的排名／符號指標，並補齊有效隊數、隊bootstrap區間及分配結果。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.contribution.role_statistics.analyze \
  --source outputs/role_statistics/task6_20261123/replay \
  --output-dir outputs/role_statistics/task6_20261123/analysis
```

來源由現有`python run.py roles --output-dir <new>`產生。輸出目錄拒絕覆寫。新分析無新增模型或solver查詢；來源重播的離線pooled成本由其completion記錄。不是新起點穩定度或獨立確認。

- `sign_flips.csv`：表6.10的總分子／分母、有效分母隊數及cluster ratio CI。
- `allocation_summary.csv`：表6.11含reserve的TV均值／區間及無正分池數。
- `rank_summary.csv`：表6.12及peer/pooled補充，逐格有效n、null、ties、Spearman/Kendall/符號/topk。
- `rank_changes.csv`：同隊peer−F變化；8個Spearman變化另作探索family校正，其他描述性。
- `payment_summary.csv`、`payment_changes.csv`：支付、保留額、標記者金額與基本費取消。F核對歷史影子政策；peer/pooled為本次衍生影子政策，不是假稱原付款。
- `team_*.csv`、`owner_payments.csv`：可核對逐隊／逐人資料。
- `audit.json`：來源與設計hash、重算及預算守恆驗收。

95%區間來自每情境10隊，不能證明付款公平、抗策略行為或跨起點穩定；owner與round不當作獨立樣本。缺失排名保持null，不補0。

`basic_cancellations.csv`按情境、評分器及標記分組列基本費取消分子／人數。零事件bootstrap區間可退化為[0,0]，不能當作族群零風險保證。
