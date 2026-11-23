# 固定端點角色統計精簡數據

來源為20歷史團隊，兩情境各10隊，80 owner appearances。每個情境按隊重抽200000次；普通95%區間是描述性，rank_changes的8個Spearman差另有探索family校正。完整設計及重建入口見[上層說明](../README.md)。

rank_summary與rank_changes補齊各格有效隊數、Spearman/Kendall/符號及topk；sign_flips列事件分子分母與cluster ratio區間；allocation_summary包含reserve；payment及basic_cancellations列資金與人數分母。來源hash與程式版本見audit.json。

原full F支付已與保存結果核對；peer/pooled支付是本次衍生影子政策。標記者full模型池3509/5000，衍生peer為0；不代表正式支付、誘因相容或一般抗攻擊保證。零事件bootstrap退化區間不是族群零風險證明。

完整逐隊逐人輸出、原端點與主代理驗收在outputs/role_statistics/task6_20261123/，不納入Git。論文尚未更新：『排除自評降低LOO一致性』需依跨零差值區間修正；『12位未標記被取消基本費』應限定污染組，另有乾淨組1位。
