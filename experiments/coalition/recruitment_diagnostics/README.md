# 實際招募規則與固定人數診斷

以[固定設計](design.md)分析原875個歷史聯盟結果；不修改原求解器，也不新增模型訓練或搜尋。來源為 `alpha_weight_outputs/p6_enum_v1_20260925/`。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.coalition.recruitment_diagnostics.analyze \
  --source alpha_weight_outputs/p6_enum_v1_20260925 \
  --output-dir outputs/coalition_recruitment/new-run
```

輸出拒絕既有目錄。`policy_seed_results.csv`提供5 seeds×預算/eta×政策×指標：實際core、品質排序、原集合目標、均勻可負擔聯盟及與core同人數的隨機期望。`policy_summary.csv`以seed同權平均，列SD與小樣本描述性bootstrap區間；不將875聯盟當獨立實驗。`policy_changes.csv`另列core相對各對照的同seed誤差／百分位差與描述性區間。

`fixed_size_correlations.csv`與`correlation_summary.csv`同時報混合人數及固定K1/2/3的Heta與負誤差Spearman，保留少樣本/恆定值null。percentile以低誤差為佳、精確同分取中點。`audit.json`保存來源hash、模型artifact hash、30條件真實HostServer選擇核對及5個原legacy設定核對。

原core為quality-per-bid搭配多樣性，不等於集合加總目標；品質only也不是quality-per-bid。D2預設Kred1，所以最多3人與原枚舉相容。原legacy_selected僅在Bselect8、eta.5核對。

修訂規約的正式招募決策重跑仍屬任務8。五seed不足以建立普遍招募優勢，bootstrap區間只描述這批來源；任何跨人數混合相關都須與固定人數結果並列解讀。
