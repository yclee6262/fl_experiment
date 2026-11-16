> 本文件記錄第一次搬移；目前位置與最新驗證請見 [第二次整理](../reorganization_20261116/README.md)。

# 執行與重現指南

## 環境

本次使用既有 Python 3.11 環境。`requirements.txt` 記錄核心與通用測試使用的套件版本；完整平台／版本見 `environment.json`。未在全新環境重新安裝，不能保證不同OS、GPU或套件版本逐位重現。

如需另建環境，可於根目錄執行：

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

歷史 FL 醫療資料範例另外引用 `flwr`、`medmnist`、`torchvision`；它們不屬於本次最小主流程驗證，未安裝或重跑這些完整資料實驗。請先查對應程式與舊環境，不要以最小依賴清單宣稱所有歷史範例已可重現。

## 驗證程式整理

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q
python run.py list
python run.py experiments/shared/smoke_main_flow.py --output-dir /tmp/fl-smoke-new
python run.py experiments/alpha/methods/alpha_weight_experiments.py --help
```

預設 pytest 只收集 `tests/`，避免相容連結與快照重複收集。貢獻度版本測試需另指定原套件，例如 `python -m pytest contribution_analysis/local_anchor_step_v1/test_kernel.py`；不可直接對所有結果快照遞迴收集。

`run.py` 接受移轉清單中的分類路徑，將工作目錄設為專案根目錄，傳遞其餘參數給原腳本。相對輸出路徑因此相對專案根目錄，不是呼叫者原目錄。跨資料夾呼叫時使用啟動器的絕對路徑與絕對輸出位置。

## 主流程與研究實驗的區別

- `core/pipeline.py` 提供已訓練 Agent → 資格選人 → proposals → best-of 解的共用函式。
- `experiments/shared/smoke_main_flow.py` 用3個合成模型、1訓練epoch及放寬門檻測通流程。它停用 inverse gate，不能用該結果宣稱資格機制或品質有效。
- 原 `main.py`、`run_experiments.py` 仍維持歷史預設，包含歷史剪枝／分潤行為。只為相容保留，不表示推薦現在研究採用。

## 歷史研究重現

1. 由 `experiments/README.md` 找到所需報告，讀該批規約、來源快照與實驗參數。
2. 使用該批封存程式及資料，不把最新程式冒充當時版本。部分舊規約只存hash沒有完整snapshot，須如實標記無法僅靠目前版本精確重建。
3. 新跑資料寫到新的空目錄。已有固定輸出目錄且不支援參數的歷史 runner，需在隔離工作副本設定新輸出位置及新規約；不要對完成研究直接再執行。
4. 確認 cohort、失敗列、checkpoint、evaluator、source hash、完整性及驗收器一致，再作結果比較。

本次9支程式只調整來源根目錄定位，科學計算本體未改。但來源hash確實會改，所以舊規約拒絕目前版本是合理行為，不可取消hash檢查。定位函式放在各檔內，沒有引入快照可能遺漏的新helper依賴；它可辨識包含 `host_server.py`、`run_experiments.py`、`dataset.py` 的平面快照。

## 交給學長時要包含什麼

建議提供整個可讀工作目錄中的 `core/`、`experiments/`、`tests/`、根目錄相容連結、`run.py`、`pytest.ini`、依賴清單、`docs/`，並依研究選擇所需的 `contribution_analysis/` 套件及資料。只傳 `core/` 或只傳新的資料夾會缺少舊匯入相容層。

相容連結全部是相對路徑；使用保留符號連結的封裝／傳輸方式。Windows或不保留連結的雲端解壓工具可能破壞它們，需在目標機確認。本次沒有製作跨平台獨立發行包，也沒有執行Git提交或對外傳送。

論文 `paper/proposal/` 原本即被Git忽略，現仍忽略。模型checkpoint與部分結果同樣可能未追蹤；不能只用Git commit當作完整實驗交接包。交接前依指定研究的manifest附必要資料即可，不必把所有重複快照都寄出。
