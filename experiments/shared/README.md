# 資料與實驗共用支援

`dataset.py` 生成合成資料；`research_evaluation.py` 訓練／保存／讀取離線 pooled evaluator；`research_analysis.py` 提供 seed 配對統計與覆蓋檢查。`runtime.py` 只保留計數器、資格資料準備、參數及 Stage 3 呼叫，動態 alpha 校正迴圈獨立放在 [alpha_calibration](../alpha_calibration/README.md)。

完整流程的小型檢查：

```sh
python run.py stages --output-dir outputs/stages-new --seeds 42 \
  --dimensions 2 --rhos 0 --num-agents 3 --samples-per-agent 32 --epochs 1 \
  --mse-threshold 1000000 --disable-inverse-check --custom-iterations 3
```

此命令放寬資格門檻且關閉反推 gate，只驗證接線。正常研究須固定資格、預算、模型訓練與搜尋設定。輸出包含 selection、pruning、payment、搜尋及離線評估資料。傳統完整流程與固定聯盟 box-constrained 初始化重播應分開解讀。

[修訂版求解規約](SOLVER_POLICY.md)固定具名數值設定與後續預算比較；`solver_policy.py`供新實驗明確啟用，既有預設保持歷史行為。
