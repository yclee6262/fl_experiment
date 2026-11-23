# 四方案實測成本及抽樣重播

summary.csv與audit.json：24解析隊×四方案的實測冷快取查詢，96端點驗證通過。AV/ACM/J/LOO平均1580.9583/2211.4583/1187.5833/1888.75；各方案counterfactual不同，不是等值近似。

sampling_summary.json：20神經隊的固定軌跡回合抽樣，2000 draws×五比例；TV數字重現原表。此重播無新增模型查詢，不是實測線上省查詢，與24解析隊成本不能混成同一條成本準確度曲線。

重建入口見[上層README](../README.md)，完整新輸出位於outputs/contribution_cost/task7_20261123/。原資料不覆寫。
