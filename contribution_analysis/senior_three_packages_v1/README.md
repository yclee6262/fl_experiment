# 固定端點角色與分潤比較

輸入為 20 個既有團隊的 full、anchor-removal、model-removal、joint-removal 與 whole-task removal 端點。先鎖定端點，再以 full F、排除自身的 F_-i 與獨立 pooled evaluator 分別評分；不重新挑端點。

各 owner 的局部有號分數先逐輪加總。anchor 門檻 max(1e-8, sum(rho))，model 門檻 1e-10。影子基本費為 100，兩個 role pool 各 500；兩角色均可靠為負時取消基本費。正分池按比例及 largest remainder 分配，無正分則保留。

使用 `python run.py roles --output-dir <new-directory>`。輸出包括 endpoints_locked、participant_metrics、rank_metrics、judge_comparisons、interactions、payments、analysis 與 audit JSON。cases 指向的模型、資料及三組原始端點需另行提供。這些分數是聲明 intervention 與 evaluator 下的效果，不等同 Shapley 值或一般公平性保證。
