# 法五協作決策研究

論文主線以0604與1112報告為範圍：法五完整Stage、傳統FedAvg比較、八種α初始化、分潤／貢獻及成本比較。0917的α校正保留程式與實驗支援。

Git收錄範圍與需另外提供的研究資料見[repo內容說明](docs/reorganization_20261116/repository_contents.md)。

## 現在從哪裡開始

- [執行與重現指南](docs/reorganization_20261116/README.md)：五個執行入口、參數和輸出。
- [整理與驗證結果](docs/reorganization_20261116/verification.md)：封存清單、實跑範圍與限制。
- [程式位置清單](docs/reorganization_20261116/layout.json)：新位置、原位置及保留原因類別。
- [論文範圍對照](docs/handoff_20261116/thesis_scope_from_reports.md)、[逐章落差](docs/handoff_20261116/paper_gap.md)。

```text
core/                         Host、Agent、模型與共用求解流程
experiments/
  baselines/                  FedAvg／法五比較及保留的傳統FL版本
  alpha/compare_initializations.py  1112八種初始化比較
  alpha/calibration/          0917 α校正（保留程式）
  alpha/methods/              主方法及必要依賴
  alpha/runners/              保留的歷史批次入口
  alpha/analysis/             保留的分析
  contribution/              四方案成本重播、主題套件索引與P3
  coalition/                  選人、預算、枚舉
  protocols/                  P2／P4／P5重現支援
  shared/                     資料、訓練、評估、共用分析
archive/legacy/               112支非主線腳本的實體封存
archive/contribution_packages/  非主線貢獻研究索引
contribution_analysis/        凍結研究套件與結果，保留原路徑
alpha_weight_outputs/        歷史α研究資料與模型
tests/                       通用測試
paper/proposal/               論文原稿
```

原214筆移轉清單（213支原有程式＋1支前次新增smoke）中，92筆保留為主要功能／測試，10筆為技術依賴，112筆移到封存區。根目錄同名檔仍是相對相容連結，連到目前實體位置；舊匯入與歷史子程序仍使用它們。貢獻度主索引縮至20個套件；其餘原套件與結果仍可在歷史區找到。

```sh
python run.py list
python run.py initialization --help
python run.py contribution --help
python run.py baseline --help
python run.py calibration --help
python run.py stages --help
python -m pytest -q
```

執行器固定以專案根目錄作工作目錄。新實驗使用新輸出資料夾。程式尚依賴相容連結及既有資料，分享時須提供完整必要目錄並保留連結；不是只複製 `core/` 就能重現歷史實驗。

255項通用測試及60項保留貢獻套件測試通過；24團隊×四方案成本重播與歷史一致；八初始化×三情境的抽樣重播與歷史解一致。完整範圍見驗證紀錄。歷史MedMNIST／Flower比較另需其專用依賴，未在本次重跑。
