# 核心流程

- `agent_client.py`：本地預測模型與輸入反推；模型權重及梯度留在本地。
- `host_server.py`：資格／聯盟選擇、候選收集、兩個搜尋引擎、候選排除、剪枝與原始結算。
- `models.py`：本地預測網路與反推用的輸入層。
- `pipeline.py`：對已訓練節點執行資格、提案、搜尋的簡單介面。

`solve_decision(server, agents, qualification_options=..., custom_iterations=30)` 回傳 `solution`、`history`、`states`。完整剪枝與結算由 `python run.py stages` 執行。

模型權重 alpha、資格分數及貢獻份額有不同用途。係數引擎搜尋候選的 linear span；方向引擎由候選平均出發。受限設定為 box_constrained，有效交付須有限且符合 box 容差。搜尋有效不代表已證明全域最優。
