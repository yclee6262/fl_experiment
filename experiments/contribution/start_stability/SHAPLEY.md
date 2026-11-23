# 歷史三人賽局的精確Shapley參照

固定日期：2026-11-24。這是任務十一的歷史資料子研究；與20個角色團隊的跨起點實驗及新配對Shapley參照分開。

## 執行前固定設計

來源為`alpha_weight_outputs/p6_enum_v1_20260925/`：5個seed(0–4)、每seed10個候選、所有1/2/3人聯盟175個，共875列。每seed的120個三人集合各定義一個賽局，共600個；每賽局的全部7個非空子聯盟使用同seed的保存結果。

執行前核對原manifest及71份已鎖artifact、模型／評分器／metadata來源、正式target=.5與qualification target=0、D2、box、固定候選與ID對應。來源snapshot以每個子聯盟均勻alpha、該聯盟自身候選平均起點及historical best_of執行；chosen solver可以在custom與SLSQP間改變，這是相同best_of規則的一部分。原SLSQP差分設定維持歷史值，不將此批改稱新數值規約的重跑。所有來源與新程式／設計hash執行前鎖定，結束後再核對。

只使用已保存端點及已知合成函數h(x)=x₁+x₂+x₁x₂計算真值。固定無合作決策x₀=(0,0)，T=.5，因此基準誤差b=.5。效用定義為v(empty)=0及v(S)=b−|h(X_S)−.5|；真值僅離線評估，不新增模型query或搜尋，不載入pooled模型計分。

## 精確參照與配置診斷

三人賽局逐人計算全部子集的標準階乘加權邊際值；另以全部6個permutation獨立重算。同一賽局的LOO為v(N)−v(N\{i})。保留有號分數、正／零／負三分類（tol=1e−10）、無法定義的排名、失敗及缺值。

比較指標為Spearman排名、符號一致率、絕對分數差，以及各自正分normalize後的allocation TV；最後一格為reserve，沒有正分時全額reserve，不補零付款。此配置是診斷性影子比例，不是正式付款或公平真值。保留每賽局及每owner數據。

空聯盟基準的解析檢查：若b增加delta而所有非空聯盟端點不變，每個Shapley平移delta/3，LOO不變；固定delta=.3驗算。這說明Shapley的符號與正分配置取決於無合作基準。不得事後依喜好改baseline。

## 統計與界線

600個賽局共享5組模型，不能視為600個獨立樣本。每個指標先在seed內平均120個賽局，再以最多5個seed等權計算均值、SD及200000次描述性bootstrap，rng20261124。無法定義的排名另列有效game數、有效seed數；不宣稱多重校正、等效、獨立確認或公平性。

Fminus對不同owner使用不同panel，不是一個共同特徵函數，本子研究不將其拼成Shapley向量。此歷史best_of／各聯盟自身起點賽局也不同於20角色團隊的共同起點LOO；兩批不得直接配對局部CA/CM/J分數。

## 執行與驗收

```sh
OPENBLAS_NUM_THREADS=1 python -m experiments.contribution.start_stability.historical_shapley \
  --source alpha_weight_outputs/p6_enum_v1_20260925 \
  --output-dir outputs/contribution_start_stability/task11_historical_v1
```

輸出拒絕覆寫。`games.csv`包含全部600賽局、`owners.csv`1800人次、`seed_summary.csv`五個seed及有效分母、`summary.csv`描述區間；`compatibility.csv`逐seed列來源與共同規則。manifest／audit／independent_acceptance保存來源鎖、875列真值／box核對、600個完整子集覆蓋、效率恆等式、階乘與permutation相符、baseline平移、seed聚合與bootstrap獨立驗算。所有新產物留於忽略的outputs。
