# 近似完整 LOO 分配的驗證

以F₋ᵢ為主、完整F作對照，比較LAST_J（最後接受輪共同移除錨點與模型）及CUM_A（累積錨點位移）對固定完整LOO的bonus比例誤差、分配／保留池狀態、模型評分與本機時間。程式不載入pooled。

從repo根目錄執行：

```sh
python run.py loo-allocation \
  --source-dir .local_archive/contribution_analysis/low_cost_loo_exploration_20261116 \
  --data-root . \
  --output-dir outputs/loo_allocation_validation/<new-run-id>
python -m pytest -q tests/test_loo_allocation_validation.py
```

輸出必須為新目錄。單次執行完成20個既有團隊、每方法1暖機及3次計時；先核對每情境首個seed，再處理其餘案例。模型載入及檔案I/O不包含在核心時間。相同輸出父目錄下先前各attempt的completion.json推論量會累加到400000研究上限，不可搬資料夾來重置預算。程式不支援原目錄resume；失敗輸出保留，再次執行須指定新run-id。

必要資料：source-dir中的scored.json與source_lock.json，以及data-root內正式保留的`contribution_analysis/custom_direction_experiment_v2_20261020/baselines/`、`model_step_decision_experiment_v1_20261021/source_snapshot/`和案例指向的local_models.pt。模型hash與來源記錄比對。封存目錄只讀JSON資料，不匯入封存腳本。缺來源時停止，不改用解析模型補隊。

- `allocation.py`：正分、留池、缺失、TV與保留池擴展TV。
- `methods.py`：凍結solver的只讀串流觀測、LAST_J/CUM_A及獨立cache。
- `run.py`：參數、輸入鎖、模型載入、門檻核對、重播與預算。
- `verify.py`：獨立逐筆cache審計與從原始回覆重算peer/full。
- `analyze.py`：團隊等權統計、門檻／分母、診斷與報告。

設計見[design.md](design.md)，固定規則見[protocol.json](protocol.json)。設計文件保留執行前狀態；實際執行狀態以每個run的completion.json為準。實作沒有更改方法／分配／誤差門檻。當全負或近零時不均分；符號及位置只作診斷。

每個run輸出原始端點和signed分數、逐筆模型事件、peer主成本和完整F補查、三次時間、團隊／參與者統計、門檻覆蓋率、來源hash及驗證。各種分配狀態保留於分母；不拿僅有效隊伍平均冒充全體表現。核心陣列bytes不含Python物件、推論cache及完整稽核輸出，O(nd)僅描述額外方法狀態。

2026-11-20的舊資料探索輸出為`outputs/loo_allocation_validation/20261120_peer_02/`；首個數值加總順序失配attempt也保留。新案例獨立確認尚未啟動，需另估訓練與確認預算。20隊不能兼作確認資料。


## 增加人數與問題複雜度

執行[新探索規約](scaling_design.md)：3/5/10/20個乾淨合格agent、2/5/10維及5維高交互作用，各5個配對新seed，共80隊。使用同一個F₋ᵢ分配比較與獨立冷cache，每cell首seed重複計時並檢查60輪參考敏感度。

```sh
python run.py loo-scaling --output-dir outputs/loo_allocation_scaling/<new-run-id>
```

`scaling_data.py`負責乾淨資料、local模型訓練與固定合成資格篩選；`scaling.py`負責新seed排重、巢狀團隊、預算及重播；`scaling_analysis.py`彙整人數／維度／交互作用配對差、尾端分配風險與參考穩定性。它們沿用本套件的methods/allocation/verify，不改共用核心演算法。

此入口會新訓練local模型，最多800個候選、2500萬次模型query與1800秒wall；不訓練或載入pooled。它是新分布探索，不能因使用新seed便自稱先前方案的正式獨立確認。新輸出目錄已存在即停止；不要用舊模型批次取代本規約未完成的組別。
