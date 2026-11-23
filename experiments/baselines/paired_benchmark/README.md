# 配對決策與資源基準

[protocol.json](protocol.json)在正式執行前固定。30個新seed×D2/5/10×乾淨/30%不相容標籤，共180案例。這是新seed合成benchmark，方法及數值設定已受歷史資料開發影響，不包裝為廣泛獨立有效性確認。

所有方法使用同一案例資料與共同初始網路權重。獨立本地訓練30epochs、FedAvg30rounds各1epoch，兩側Adam都逐epoch/round重設，樣本順序按agent與round固定。保留FedAvg全部20人，以及與協同搜尋所選聯盟相同成員的控制，區分資格與模型聚合效應。

模型與反推保存在本機outputs；不共享模型參數的介面只針對協同決策模式。計算量、資料通訊與搜尋成本分列，不能將查詢上限相同寫成端到端資源完全相同。通訊是已宣告數值payload估計，不是網路封包實測。

## 搜尋與反推

- 共同盲測20點；原core資格／招募；目標0。反推gate與proposal重用相同本地點（各模型1起點500步），避免對同一目標重算。此重用與先前分開重新隨機反推的歷史流程有別，明確記錄。
- FedAvg兩種成員版本各5起點，每起點500步，選各自模型下最小目標誤差的最終點；起點在首次評估前限制box。
- 錨點平均、最佳錨點及原點各自實際計量；最佳錨點依可取得的Lopt選，不依真值選。原點用於揭露T0特殊性，不視為一般可用優化演算法。
- coefficient/custom/best_of/Nelder–Mead/直接SLSQP使用100/200/400個panel評估硬上限。每panel為K次單模型單樣本查詢；所有初點、history、試探及終點計入。best_of各半、不轉移；無快取。
- 預算內交付已評估最佳有限可行點；這是硬預算下的交付規則，與原engine僅回傳accepted endpoint的未限預算流程分開標示。可能因此交付搜尋時評過的錨點。
- 真值只做離線評估，不回饋選擇；solver成功、有限可行交付、預算停止與無解分開保存。禁止丟棄失敗後補抽案例。

## 成本

模型訓練樣本、backward steps、局部反推、Host每模型輸入樣本、盲測、資格反推檢查、各搜尋與FedAvg反推分開列。model留本地不代表既有訓練免費。

通訊採float32：每次模型查詢傳D個輸入與1個回應，重複查詢不扣快取；FedAvg每round每client上傳及下載所有參數。提案及bid／MSE等metadata另列。另報同步盲測／gate／proposal批次、搜尋panel序列及FedAvg輪次；不把多provider API呼叫相加稱為同步通訊輪數。不含headers/加密/壓縮，未測網路延遲。

## 分析規約

每D×rho各30seed，配對重抽200000次，完整保留可行／失敗與有效pair n。主比較取400 panel的best_of對最佳錨點、錨點平均、Nelder–Mead、FedAvg全體與同成員控制，共6條件×5=30比較作Bonferroni family；其他預算、引擎及直接SLSQP為描述性。不事後因顯著性挑預算，失敗不補0。

## 執行與驗收

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run.py paired-benchmark --output-dir outputs/paired_benchmark/new-run
OMP_NUM_THREADS=1 python -m experiments.baselines.paired_benchmark.validate --run-dir outputs/paired_benchmark/new-run
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.baselines.paired_benchmark.analyze --source outputs/paired_benchmark/new-run --output-dir outputs/paired_benchmark/new-run/analysis
```

主入口拒絕既有輸出；`--resume`僅在protocol與runtime hash一致時，核對已完成case／results hash後跳過。未完成案例保留並要求檢查，不自動補抽或覆寫。最終驗算另外檢查模型／data／trace。

`--smoke`使用seed39999、小型資料驗證無聯盟失敗路徑；`--validation-case`使用seed39998與正式訓練參數驗證成功路徑。兩者不可作正式180案例證據。正式原始資料、模型、逐次搜尋trace與驗收留在outputs；[results](results/README.md)只保留精簡統計。

validate會重建全部seed資料、獨立載入模型驗算交付端點、檢查完整條件覆蓋、模型payload、硬預算與預算內best-observed交付。分析規約見[ANALYSIS.md](ANALYSIS.md)；95%描述區間與30項主比較校正區間分開。

數值通訊payload中的資格metadata估計為每人8bytes（bid/status等純量）；另計4D bytes的反推提案，以及實際模型查詢的每點4(D+1)bytes。它不是抓包量測。`qualification_parallel_stages_estimate`是假設可並行provider的流程階段估計，不能當作實測往返時間。FedAvg反推在Host本地進行，遠端model queries為0不等於無梯度計算。
