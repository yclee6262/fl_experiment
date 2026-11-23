# 招募診斷固定設計

沿用歷史 p6_enum_v1_20260925：5 seeds(0–4)、每seed10位通過資格者、875個一至三人聯盟、T=.5/D2。此為保存決策重分析，保留原係數引擎，不是修訂求解器的新實驗或獨立確認。

- 預算2.5/3.5/8、eta0/.5；Kmax=3，與core D=2、預設Kred=ceil(.3D)=1、Kapi=10一致。
- 實際core逐次r/bid*(1+eta*diversity)，diversity為與既選者最大絕對cos相似度之補數；空集合diversity1。預算依core嚴格不超出（不額外放寬1e-12），分數平手依原候選順序。min score0。離線可負擔聯盟全集沿原枚舉容差<=budget+1e-12，與core原規則嚴格比較分開記錄；原set objective在seed1/B3.5/eta.5選擇的浮點bid為3.5000000000000004，在原容差內。
- quality_only：每輪選可負擔的最高r，忽略bid作排序，但支付仍須可負擔。core eta0則是quality-per-bid，兩者不是同一規則。
- set_objective：原Heta精確枚舉（品質加總+正規化pairwise多樣性），同score低bid優先，再ID排序。
- random_feasible：均勻抽取所有非空可負擔、最多3人聯盟；用已枚舉結果計算精確期望，不新增Monte Carlo。它不是逐次均勻抽人，也不是全選3人；另報與core同人數的均勻隨機基準。
- 各政策均不看pooled/truth作選擇。離線比較pooled誤差及由X重算的truth誤差、相對最佳可負擔誤差gap、百分位（小為佳）：(嚴格較小數+.5*相等數)/可負擔總數。相等使用精確保存浮點數，不另設事後容差；隨機期望百分位為.5。
- 比較Heta與負誤差的Spearman，先報混合人數，再固定K1/2/3與預算各自重算；恆定或少於2個有效聯盟為null。報各格聯盟數及seed有效數。
- 每個seed是統計單位；5seed均值、SD與200000次bootstrap描述區間，rng20261123。聯盟共享模型，不當作875個獨立實驗；區間不是確認性檢定。core對各對照另以同seed差值報描述性配對區間（誤差與百分位）。
- legacy_selected只用於來源原設定Bselect8、eta.5之核對；不得貼到其餘預算條件。
- 保存輸入/分析/core來源hash、狀態、覆蓋與例外。不得改寫歷史模型或結果；修訂規約下的正式決策重跑屬任務8。本分析不新增模型/求解呼叫。
