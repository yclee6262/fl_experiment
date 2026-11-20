# 預算招募與聯盟枚舉

```sh
python run.py coalition --help
```

`p6_coalition_enumeration.py` 枚舉合格候選的一至三人聯盟，讀取保存結果或執行新搜尋。`p6_recruitment_experiment.py` 提供資料準備及候選資訊。`recruitment_selection.py` 實作 quality-share sum 加 pairwise residual diversity 的集合目標。`analyze_p6_enumeration.py` 彙整預算內相關性與最佳可負擔聯盟的誤差差距；`reevaluate_p6_coalitions.py` 重載模型檢查保存結果。

枚舉目標與 core 中逐次 quality-per-bid 招募是不同規則。pooled 最佳聯盟是離線評估參照，不提供給招募規則。

訓練、資格設定與搜尋參數由命令列記錄；使用新輸出資料夾，避免覆寫完成的枚舉結果。歷史結果位於未追蹤的 `alpha_weight_outputs/p6_enum_v1_20260925/`。
