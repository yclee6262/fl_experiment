# 任務七：觀測代理與投影查驗設計

本設計先於新彙總固定。V/CM來源為model_pool_v_cm_validation_20261107_attempt_02的30輪紀錄：historical20神經團隊、known_analytic80既有80解析團隊、new_confirmation24解析團隊分開，不以phase名稱將本次重分析稱獨立確認。

- V與CM逐輪有號加總後比較；total(full F)與peer_normalized(F_-i)分開。容差1e-10三分類正／零／負，報符號相同、嚴格相反、zero/tie、Spearman有效n與含reserve正分分配TV。
- 每phase以team為單位200000次bootstrap，rng20261123；普通95%為描述性，不把owners/rounds當獨立、不做跨所有指標共同推論。分母、no-positive及來源hash完整保存。
- 同24fresh來源的cost紀錄要核對baseline、逐輪與V/CM端點；成本重播若改用peer需以相同端點重新評分。V在原baseline既有試探點重評分的增量查詢0，並不代表CA+V全流程零成本。
- 投影使用論文Eq5.14 signed A_it=DeltaF_t (H_i·g)/||g||²，從method5_cost實驗20神經團隊New紀錄重算；非custom_direction_v2的模型位移／路徑長分數，非剪正規化版本。
- 每輪核對sum H_i=g及sum A_it=DeltaF_t；zero direction時gain為0才給零會計額，標示zero_direction，非除0。逐owner有號加總後與同軌跡full-F C_A比較，符號容差1e-10；分配兩側均用max(1e-8,sum rho)後取正分含reserve，避免門檻差異混淆。
- 投影結果按clean10/controlled_poison10隊分組，teambootstrap；為會計量對介入量的描述差異，不是把C_A當公平真值。未用無法追溯的45%摘要。
- 不重跑solver、不修改歷史來源。新分析新增model/solver calls=0，拒覆寫新輸出。
