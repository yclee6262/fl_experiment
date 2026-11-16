# 四方法比例的有限貝葉斯最佳化（事前規則 v1）

本輪接續使用者目標，擴大上一輪只測四法等權的不足；新結果獨立存放，不改上一輪門檻或結果。

研究產物是離線開發所得、部署時固定的w，並非部署時每任務以pooled尋優。w順序U,Q,E,S，α=Σw_k A_k。四者均使用，搜尋域w_k≥0.05、總和1；0.05是避免退化成兩方法的設計限制，非經驗最佳門檻。域外及邊界方法沿用既有對照，不聲稱本域找到全域最佳。

資料：先前已看過的30 seeds（每批前10，三情境完整）用於搜尋；其餘60歷史seeds只在候選鎖定後一次性檢查，不再回饋調參，亦非新確認，因其舊方法和整體結果早已看過。C/P逐案結果不使用。全新資料只在後續確認生成。

目標：每seed先平均三情境，d_s(w)=pooled_error_s(w)−EU_s。最小化J(w)=mean(d)+SE(d)，SE是完整seed樣本標準差/√30。此一標準誤懲罰為預設穩定性偏好，非信賴界保證。所有評估用相同資料／模型／anchor／20盲測／best_of custom30；只改initial α。E/Q/S回退沿用上一輪，失敗不補抽，遇異常停止而保留。

模型：用現有27點及新增點的J擬合GaussianProcessRegressor，Constant×Matérn ν=.5，長度尺度[.05,2]，normalize_y，nugget1e-6，固定RNG；ν=.5容許較粗糙曲面，但求解路徑跳躍仍可能使代理失準。每輪保存預測、std、EI與實際觀測。代理std僅對有限開發曲面的模型不確定性，不是新任務泛化CI。

先以四個(.4,.2,.2,.2)的置換形成中心周圍的3D simplex設計，避免只有單一內點。再順序12次最大化Expected Improvement：I=max(J_best−J(w),0)，EI=(best−μ)Φ(z)+σφ(z)。SLSQP在12個固定Dirichlet起點＋當前最佳內點做受限連續最佳化；這些僅算便宜代理，不呼叫Stage3。距已有點L2<.01排除，避免無限重複。固定16個新比例預算，不因看見優勢追加。選擇所有已測內點中J最低者（含等權），精確w鎖定。

鄰域：鎖定點周圍沿sum-zero空間的三個正交Helmert方向±0.05；若近域邊界，縮短該方向步長至最大可行的90%，不改中心。共6點，僅診斷，不再用鄰點取代中心。至少4/6鄰點平均對EU負，且鄰點平均中位差負，才稱有限方向有支持；不能證明完整穩定區域。

歷史轉移檢查：只評鎖定中心、等權、EU於其餘60案例（三情境），EU/等權同機重跑便於成本描述，保存對舊EU重播差。這是180×3=540求解；搜尋16×90=1440，鄰域540，上限2520（另驗收不當新品質樣本）。無新訓練或標籤。預留數分鐘，實測完整成本。

推薦帶入新確認的門檻：60歷史案例中心對EU平均≤−.001、三批和三情境均負、seed勝率≥55%、完整有效，方法requests/samples各≤EU1.10，並通過鄰域診斷。普通95%完整seed bootstrap僅描述（20000次），不能消除開發選擇偏差或歷史資料再利用。另報中心對等權，防止最佳化本身沒有收益。若失敗仍交付找到的比例、演算法與反例，維持EU，不續搜到顯著。

成本：所有部署四方法只需一次E擬合＋Q/U＋六個排除自身的S預測＋一次Stage3。重用盲測快取；搜尋總成本為離線研究成本，不能混入單次部署或省略。保存初始化／支持／搜尋requests,samples,seconds及shared準備，truth僅診斷。

資料完整性：source snapshot、舊檔hash、每任務決策先落盤後pooled、無evaluator參數進initializer、α simplex／box、全部重載重評、重複/缺漏/失敗、GP提案和選點重算、最後完整重播至少一seed三情境。最佳化無全域或泛化保證。

方法參考：https://botorch.org/docs/acquisition ; https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html ; https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html 。本環境sklearn1.3.0/scipy1.13.1，以實際可用介面為準。

## 完成進度（不更改上方封存規則）

已完成16個學習比例、六個可行鄰點、60歷史案例轉移檢查與精確EU對照。U落在下界造成原三個零半徑probe，另補三個固定切線方向；泛化EU在一案例有浮點路徑差，另跑原版EU180任務，修正明細保留verification_amendment.md。共2970主結果＋3完整重播，全部有效／無fallback，來源與舊結果雜湊未改，驗收通過。
鎖定w=(.05,.22838596640363512,.46186232017738726,.2597517134189776)，30案例對EU差−.00200623；60案例對EU差+.00116515、勝29/60，六鄰點僅3平均改善。未通過事前推薦門檻，維持EU，不用60案例再改w。完整研究報告：alpha_weight_outputs/four_way_bayes_v1_20261116/report.md。
