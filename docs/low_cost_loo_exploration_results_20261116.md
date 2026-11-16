# 兩種低成本貢獻近似與完整LOO：探索完成

2026-11-16。已核對原資料、完成20個既有團隊/80人的探索與獨立程式核對。最後輪主定義依本次使用者回覆為重做最後接受輪（同時移除錨點與模型，J單輪）；原錨點單輪降為角色對照。

- [完整研究報告](../contribution_analysis/low_cost_loo_exploration_20261116/report_v2_joint_last.md)
- [計算前規約與使用者修訂](../contribution_analysis/low_cost_loo_exploration_20261116/design.md)
- [新案例獨立確認提案（未執行）](../contribution_analysis/low_cost_loo_exploration_20261116/independent_confirmation_plan_v2.md)
- [核對結果](../contribution_analysis/low_cost_loo_exploration_20261116/verification.json)
- [研究總成本](../contribution_analysis/low_cost_loo_exploration_20261116/final_accounting.json)

兩主方法相對完整LOO節省總模型評分67.10%/69.20%，但固定pooled下平均分配TV49.76%/71.07%，有效分配團隊19/20與18/20。沒有建立bonus分配替代性；目前僅適合末輪敏感度或原路徑位移診斷。本批不是獨立資料確認，完整LOO也不是Shapley或公平保證。

保留原有研究、模型、solver、alpha及結果；沒有啟動新訓練、付款或pruning。11/12解析模型成本與本次神經網路批次分開，原簡報F/F_-i不一致已在完整報告釐清。

使用者最新釐清及更新後結果見v2報告；舊報告、規約與原始結果不覆寫。J單輪尚無同規格重複總時間，不能沿用錨點單輪計時。
