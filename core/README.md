# 主流程

本目錄集中可共用的決策流程。合成資料、模型訓練批次、FedAvg 比較及研究分析都在 `experiments/`。

| 檔案 | 職責 |
|---|---|
| `pipeline.py` | `solve_decision` 串接資格／選人、蒐集候選解、既有 best-of 求解器 |
| `host_server.py` | Host 資格檢查、聯盟選擇、共識目標、受限求解器，以及歷史貢獻／剪枝／結算實作 |
| `agent_client.py` | Agent 本地訓練介面、預測 API 與局部反推 |
| `models.py` | 本地神經網路模型結構 |

`solve_decision(server, agents, qualification_options=..., custom_iterations=...)` 接收已建立的 Host 與已訓練 Agent；Host 的權重、約束與參數由呼叫者明確設定。此函式只封裝原有三次階段呼叫，沒有更改求解器或預設研究方法。

新入口不自動執行剪枝與分潤。這兩項歷史能力仍原樣留在 Host，可由研究程式明確呼叫；它們的科學有效性尚不能由程式可執行推得。α 初始化方法也保留在研究目錄，未把探索結果升級成部署預設。

原 `main.py` 實際包含訓練、剪枝、付款、FedAvg 與繪圖，因此歸到 `experiments/baselines/main.py` 作歷史展示，不能視為當前推薦的完整研究設定。原 `dataset.py` 是合成資料產生器，歸到 `experiments/shared/dataset.py`。

實際小型接線範例見 `experiments/shared/smoke_main_flow.py`；該範例刻意放寬資格門檻以測通流程，不得作正式成果。
