# R6：乾淨情境相關權重獨立複驗—完成

## 固定範圍

三法B／固定相關0.5／自適應相關，沿用R5原公式與solver。二維、乾淨target0、IDs1/2/20、20筆资格資料、box容差1e-8。pooled只對交付X事後評分，truth診斷；無α更新、pruning、結算或新增標記。

## 執行與結果

- 歷史pilot5000–5004共15條件，與R5 α／X／分數最大差0，狀態及Stage3查詢一致。
- N由R5乾淨34個配對的三組SD最大值0.00971768752、事前研究解析度0.005、名義80%常態近似及三比較Bonferroni規劃。N40；不保證power，也不是業務門檻。舊資料只規劃N，不併正式。
- 正式新seeds6000–6039，三方法共120／120有效可行。來源／資料／模型／快照雜湊與重載通過；α／pooled／資格預測差0。
- 自適應−B平均Δ−0.00038750479，98.3333%同時CI[−0.00191389816,0.00178529533]。
- 固定−B平均Δ0.00046587368，同時CI[−0.00174611517,0.00327202971]。
- 自適應−固定平均Δ−0.00085337847，同時CI[−0.00256726237,0.00045989992]。
- 三項都未通過；R5的確認優勢未在本輪重現。不證明方法等效或無效。
- 自適應初始化＋Stage3約固定法3.44倍，B3.55倍；只計這兩階段，不稱完整流程倍數或貨幣淨收益。
- 20項回歸＋4項分析失敗測試通過；40新方法條件1000折內矩陣／留出損失／切分核對，舊固定0.5與B端點、兩組實際fresh／cache交付比較通過。
- 完整重播seed6000，重新產生資料、模型、資格、proposal及三方法解；離散決策一致，α／X／分數最大差0。

## 可重現入口

首次pipeline使用獨立新輸出，禁止覆蓋既有結果。重新分析不需要重啟模型訓練。

```text
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /Users/yclee/anaconda3/bin/python correlation_replication_pipeline.py
/Users/yclee/anaconda3/bin/python analyze_correlation_replication.py alpha_weight_outputs/corr_replication_formal_v1_20261011 --registration docs/corr_replication_preregistration_20261011.json
/Users/yclee/anaconda3/bin/python alpha_weight_outputs/corr_replication_extra_acceptance_20261011.py
/Users/yclee/anaconda3/bin/python alpha_weight_outputs/corr_replication_full_replay_20261011.py --verify-existing
/Users/yclee/anaconda3/bin/python alpha_weight_outputs/corr_replication_report_20261011.py
/Users/yclee/anaconda3/bin/python alpha_weight_outputs/corr_replication_completion_audit_20261011.py
```

## 成本及檔案

共同準備成本每seed一次，在來源shared_costs.jsonl；資格快取重建在calibration.jsonl。額外來源Uniform求解、舊OOB重用核對、驗收和重播另作研究成本，不能消失或加成某方法的部署成本。API呼叫與模型輸入點分開；未知成本不填0。

完整結果含results.jsonl、fold audit、analysis、verification、extra_acceptance、replay_verification及兩份精簡CSV。來源環境封存、正式登錄、回歸logs及逐項完成驗收均保留。CSV不包含完整折內矩陣，該細節在JSONL。

Notion只更新原主頁：https://app.notion.com/p/3eebf435d47481f0998cdf5db86b68ce，R6在附錄G。已回讀驗證82張原生表格、10份下載與完整歷史附錄；本地同一主文件鏡像同步。維持A/B，不升格自適應；下一步先研究資格代理損失與倒推品質的關係，尚未開始。
