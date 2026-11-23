# Alpha的目標與方向作用

固定設計見[protocol.json](protocol.json)。沿原90個clean D2模型，固定原三人成員與盲測，T=±.5。模型副本的輸出加固定+.5，條件為control、單一成員偏差及全體共同偏差；偏差成員按全域seed序號mod3輪替，各30次，不按誤差挑人。這是訓練後的校正偏差壓力測試，不模擬策略性訓練攻擊。原模型檔案保留。

每條件重新反推、重算十法alpha，跨條件使用相同反推起點。同一條件內所有方法及2×2共用錨點與起點；跨偏差條件同時改變候選，不能純歸因於alpha。全體共同偏差在相同輸入上保留模型相對分歧（float32容差2e-6）；另報共同誤差與分歧，避免將二者混同。

十法coupled best_of使用400 panel（兩引擎各200）。主2×2固定Q及EU，custom單引擎200 panel：UU、AU、UA、AA的第一個字母為目標權重、第二個為方向權重，A為該條件已算好的alpha。coefficient沒有方向權重，best_of勝者也不拿來代替custom消融。

UU與AA直接重用coupled搜尋的custom分支；AU／UA各獨立求解。每條件實際執行14個search：10 coupled best_of及4 custom。兩份factorial表中的UU是同一觀測，不重複計成本。每個branch均依自己的目標Lopt選預算內最佳觀測點；跨權重比較用離線真值或共同均勻評分，不能直接比較不同Lopt。

主family：兩規則×兩目標×三條件×objective/direction/interaction，共36項；次family為十法九個對U×六條件，共54項。兩family分開作90seed、200000次配對bootstrap及Bonferroni區間。偏差成員權重變化另報描述性區間，權重下降與品質改善是兩個終點。有限box交付、高反推殘差、失敗與成本均保留；不宣稱公平、穩定優勢或新模型獨立確認。

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.alpha.weight_roles.run \
  --output-dir outputs/alpha_weight_roles/new-run
OMP_NUM_THREADS=1 python -m experiments.alpha.weight_roles.validate \
  --source outputs/alpha_weight_roles/new-run
OPENBLAS_NUM_THREADS=1 python -m experiments.alpha.weight_roles.analyze \
  --source outputs/alpha_weight_roles/new-run --output-dir outputs/alpha_weight_roles/new-run/analysis
OPENBLAS_NUM_THREADS=1 python -m experiments.alpha.weight_roles.validate_statistics \
  --source outputs/alpha_weight_roles/new-run
```

來源須提供原三批`reference_clean/shared_costs.jsonl`與90個`local_models.pt`；路徑由原SE/QU registration前三批明確解析。manifest在執行前鎖來源、設定與實際runtime hash，結束時再核對；原模型不覆寫。輸出拒絕既有資料夾，`--smoke`執行第一個案例的六個條件，仍使用正式500步反推及搜尋預算；不作90seed統計。

正式入口亦可用`python run.py alpha-roles --output-dir outputs/alpha_weight_roles/new-run`。

`searches.jsonl`是14次實際搜尋，`factorial.jsonl`是由其中custom分支衍生的8格；成本以實際搜尋表計算。`decisions.jsonl`先保存14次決策且不含truth，完成後再衍生真值／共同評分。初始化、共享盲測、固定輸入偏差診斷、本地梯度反推、搜尋與驗收各自計量。查詢上限相同不代表實際成本相同。

另可用`python -m experiments.alpha.weight_roles.validate_replay --source outputs/alpha_weight_roles/new-run --reference outputs/nonzero_initialization/task9_v1`核對全部未加偏差控制組與任務九。`--smoke-reference`可另外指定已驗證的本研究smoke；排除seconds與gzip容器hash後比較原始決策、模型回覆與查詢。這是回歸檢查，不新增模型query。

正式540條件與驗收摘要見[results](results/README.md)。`plot --analysis RUN/analysis`可在本機輸出全部48格的描述圖；完整逐次回覆、圖與原始紀錄保留在outputs。
