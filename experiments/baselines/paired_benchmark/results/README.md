# 配對決策基準精簡結果

30新seed×D2/5/10×乾淨／30%不相容標籤，180案例、3600有效結果。設定與分析規約見[上層README](../README.md)與[protocol](../protocol.json)。

groups.csv含120方法／預算／條件摘要；primary_comparisons.csv為400panel的30項配對主比較，200000次seed-bootstrap及Bonferroni30區間；resources.csv與case_resources.csv分列model queries、梯度計算、訓練與數值payload。statuses.csv保留預算停止與solver狀態。有限可行交付不是全域最優保證。

主要結果：best-of在6條件皆優於錨點平均；相對最佳錨點3/6、FedAvg全員4/6、相同入選成員FedAvg1/6的主比較有較低誤差之校正區間支持；相對Nelder–Mead六項均跨零。不能從未顯著推論等效。起點、權重、同模型與同查詢上限比較符合宣告規約；端到端訓練、反推與通訊資源並非完全相同。

T=0的已知可行原點有真值誤差0；本結果沒有證明搜尋在一般反推任務的必要性或降維優勢。nonzero目標仍屬任務9。

完整模型、data、trace（264MB級）與來源hash留在outputs/paired_benchmark/task8_formal_v1/；精簡驗收見acceptance_summary.json。analysis_final為最後資料架構版本；初版analysis保留，groups與主比較數值完全一致。
