# 修訂規約十法精簡結果

原90 seeds、三情境、270模型組重播，共2700有效決策、零fallback；求解規約float32_slsqp_v1。這是歷史案例的修訂重分析，不是獨立確認。

- groups.csv：真值／pooled整體與情境均值、seed SD、95%區間與查詢成本。
- comparisons.csv：完整原12、SE/QU17、十法45對探索family的配對差與區間；兩指標、各情境的校正分開，不當作跨全部分析的共同保證。
- direction_agreement.csv、endpoint_correlations.csv：指標方向一致率（ties分開）及相關。
- historical_changes.csv：新舊決策／指標／成本差異；配對普通95%區間屬描述性分析。
- evaluator_quality.csv：270評估器MSE、RMSE與評估樣本數。
- audit.json：來源hash、分析版本與覆蓋；詳細原始結果、端點驗算與驗收留在outputs/alpha_initialization_revised/task5_20261123/。
- truth_paired_intervals.pdf：九法對U真值差值圖。

統計先平均每seed的三情境，90 seeds共同重抽200000次。整體真值45對探索校正區間14項排除零，主要差異涉及S/SU較高誤差；EU平均最低，但相對U/Q的校正區間均包含零，未建立等效性。pool與truth逐案例的非平手配對方向一致8956/12106=73.98%，不能把這12106項重疊比較視為獨立樣本。

Stage3平均查詢由372.44增至658.38；沒有硬查詢預算，不稱同成本優勢。修訂結果不取代上層保存的歷史數據，論文正文尚待任務14整合。

重建說明見[分析入口](../../analysis/README.md)、[批次重播](../../revised_initialization/README.md)。
