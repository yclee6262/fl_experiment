# 非零目標與受限搜尋空間

[protocol.json](protocol.json)在讀取新結果前固定。沿用任務八已訓練模型，固定其保存招募順序的前K=1或3人，D=5或10，兩污染情境各30seed；不重新依新目標挑選參與者。原T0、T=-.5、T=.5及近上界T=.9(2D−1)各自重新反推錨點，共960個目標／子聯盟任務。模型沿用不代表新獨立確認；新計算成本及失敗全保留。

K<D以外，逐任務保存錨點linear rank與affine rank。係數引擎限linear span；custom的方向及起點限affine hull；兩者不同，不能一律寫成凸包。以相同初始錨點平均與100/200/400 panel硬預算，和全空間Nelder–Mead、直接SLSQP比較；每panel為K單模型query。若原招募不足K，保留所有方法失敗，不另抽人。

合成h(x)=sum(x)+sum adjacent products，box[-1,1]的上界2D−1由全1向量達到。等座標解析見證 t1 滿足 D t+(D−1)t²=T，證明四目標在原box可達。見證到候選linear span／affine hull的距離只診斷該點幾何位置，不能證明整個目標解集合不可達。linear與affine box可行性以線性規劃核對；可行起點並不代表能達成目標。

主要離線指標為abs(h(X)−T)，另列Lopt、actual query、loss與truth差距、rank及見證距離。真值不回饋搜尋。主family為D×rho×K×三個非零T的24項best_of−NM@400；30seed配對bootstrap200000次及Bonferroni24。T0及其他引擎／預算為描述性。缺失保留n、不得補0。

執行：

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.baselines.nonzero_subspace.run \
  --source outputs/paired_benchmark/task8_formal_v1 --output-dir outputs/nonzero_subspace/new-run
OPENBLAS_NUM_THREADS=1 python -m experiments.baselines.nonzero_subspace.analyze \
  --source outputs/nonzero_subspace/new-run --output-dir outputs/nonzero_subspace/new-run/analysis_final
```

`--smoke`僅一個原模型案例、兩K、四T，屬程式接線檢查；正式輸出拒絕覆寫，`--resume`只接受相同runtime/protocol與已完成任務hash。中斷任務保留，不靜默重抽或改寫歷史來源。原模型、盲測與招募的成本屬來源，不計為0；本次僅新增target-specific inversion及搜尋，不含模型訓練。

## 次要解析診斷

執行期間、尚未讀取正式主比較前，另加入不改搜尋或主family的K1解析診斷：h(beta a)=beta sum(a)+beta² sum adjacent products，beta在±1/maxabs(a)之內。端點及區間內頂點給出此linear span在box內的精確真值range與最小真值誤差；custom K1的affine hull只有原anchor，故其真值誤差固定。這些只由保存錨點在分析階段衍生，不餵回模型搜尋，不對K3推論精確可達範圍。輸出single_anchor_truth_bounds.csv，將幾何下界與交付誤差分開。

K=1與K=3同時改變模型panel及候選空間，所以跨K差異不能純粹歸因於幾何維度。識別搜尋空間效果的直接比較是在固定K、同panel、同錨點、同Lopt與同預算下，受限係數／方向引擎對全空間方法。即使如此，演算法路徑也不同，不能把所有差異唯一歸因於空間。

原模型／trace／預算驗算與統計獨立重算：

```sh
OMP_NUM_THREADS=1 python -m experiments.baselines.nonzero_subspace.validate --source outputs/nonzero_subspace/new-run
OPENBLAS_NUM_THREADS=1 python -m experiments.baselines.nonzero_subspace.validate_statistics --source outputs/nonzero_subspace/new-run
```

`validate_statistics`讀取上述`analysis_final/`。正式960任務的結果、完整24主比較與K1解析下界見[results](results/README.md)。
