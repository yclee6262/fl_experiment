# 十法精簡數據

- `means_and_queries.csv`：十法的整體與三情境 pooled 誤差、有效／失敗／fallback 數、Stage 3 查詢成本。
- `paired_comparisons.csv`：SE/QU 相關 17 個配對差及其校正區間。
- `audit.json`：來源雜湊、方法 key 與資料覆蓋。

90 seeds × 三情境 × 十方法，共 2700 筆。每 seed 先等權平均三情境，再平均 90 seeds。EU 整體平均最低；SE、QU、E 分別在乾淨、低品質及不平衡情境平均最低。17 個新增比較的校正區間都跨零。

來源為同一批既有模型的補充配對研究。重建方式見 [analysis](../analysis/README.md)；模型與逐案記錄另行提供。

[revised_v1](revised_v1/README.md)為修訂求解規約後的新版本，含真值主指標與重新計算區間；本層原歷史檔案保留。
