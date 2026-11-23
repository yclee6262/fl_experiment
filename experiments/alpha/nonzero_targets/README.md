# 非零目標下的十法初始化

固定設計見[protocol.json](protocol.json)。沿用任務五同一90 seeds×三情境模型及盲測，針對T0/−.5/.5/2.7重新反推。D2函數上界為3，T2.7為上界90%，見證點t=−1+sqrt(3.7)各座標相同；不假設所有解都靠邊界。

固定原三人成員、不重新招募或依目標品質剔除。所有目標使用相同起點seed、500步反推；高反推誤差保留。十法沿正式方法實作，只改目標、提案與顯式求解規約。使用400 panel硬上限的best_of各200，與任務五未限查詢版本分開；目標效果使用本研究新T0對照，不混用舊T0。

主指標真值，Lopt為次指標；不載入pooled、不用真值挑alpha。先保存全部方法決策才衍生真值。每seed先平均三情境；九法對U×三非零目標共27主比較，200000次配對bootstrap／Bonferroni27；場景與T0為描述性。失敗留status、n、不補零。資料重用是目標擴充，不稱獨立確認。

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run.py nonzero-initialization \
  --output-dir outputs/nonzero_initialization/new-run
OMP_NUM_THREADS=1 python -m experiments.alpha.nonzero_targets.validate \
  --source outputs/nonzero_initialization/new-run
OPENBLAS_NUM_THREADS=1 python -m experiments.alpha.nonzero_targets.analyze \
  --source outputs/nonzero_initialization/new-run --output-dir outputs/nonzero_initialization/new-run/analysis
OPENBLAS_NUM_THREADS=1 python -m experiments.alpha.nonzero_targets.validate_statistics \
  --source outputs/nonzero_initialization/new-run
```

輸出拒絕覆寫；`--smoke`只執行一個模型案例的四目標，不能代替正式統計。驗算分為原模型端點／來源／預算與統計獨立重算兩層。模型訓練為既有來源成本，本次新增反推、共同盲測／支持矩陣、初始化及搜尋成本分列，不宣稱端到端同成本。

正式四目標結果與完整27主比較見[results](results/README.md)。
