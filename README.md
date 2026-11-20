# 協作決策反推與貢獻分潤

本專案保留論文使用的決策流程、FedAvg 比較、十種固定 alpha 初始化、預算招募、角色分潤與評估成本實驗，另保留獨立的 alpha 校正迴圈研究。論文原稿另行管理。

## 目錄

| 資料夾 | 內容與操作說明 |
|---|---|
| [core](core/README.md) | 節點、模型、資格審查、反推、協作搜尋與原始結算 |
| [experiments](experiments/README.md) | 各實驗的執行入口 |
| [alpha 校正](experiments/alpha_calibration/README.md) | 固定／自適應步長及 guarded alpha 校正研究 |
| [contribution_analysis](contribution_analysis/README.md) | 角色與成本重播必需的歷史實作及凍結求解器 |
| [project_support](project_support/README.md) | 歷史匯入名稱的相容支援 |
| [tests](tests/README.md) | 保留方法的測試 |

## 開始使用

使用 Python 3.11，安裝 `requirements.txt` 中的依賴，從專案根目錄執行：

```sh
python run.py list
python run.py stages --help
python run.py initialization --help
python -m pytest -q
```

新實驗必須指定不存在的輸出資料夾。各類實驗的資料需求、完整命令與輸出格式見各資料夾 README。

模型、原始逐案紀錄與輸出不納入 Git。歷史重播需另外提供來源資料；只有程式 checkout 不含訓練好的模型。過去探索程式與工作紀錄集中於忽略的 `.local_archive/`，Git 歷史也保留整理前版本。請勿將本機封存區一起分享。
