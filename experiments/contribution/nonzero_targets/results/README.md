# 四目標角色介入結果

正式來源：`outputs/nonzero_role_targets/task9_final_v1/analysis_final/`。20個固定歷史模型團隊×T0/−.5/.5/2.7共80任務；clean每T10隊各3人、受控污染每T10隊各5人。全部有限box可行交付，未重做資格gate。每隊重新反推及重新執行full、CA/CM/J與共同起點whole-task LOO。

| T | clean平均真值誤差 | 受控污染平均真值誤差 | clean模型角色peer/full TV | 受控污染模型角色peer/full TV |
|---|---|---|---|---|
| 0 | 0.006850 | 0.011060 | 0.686716 | 0.637604 |
| −0.5 | 0.008950 | 0.070584 | 0.726446 | 0.769385 |
| 0.5 | 0.010055 | 0.056159 | 0.749696 | 0.597802 |
| 2.7 | 0.052336 | 0.067666 | 0.934204 | 0.434545 |

非零目標保留明顯的評分panel敏感性，其方向依情境與目標改變。例：clean模型角色TV在T2.7相對T0增加0.247488，隊配對95%區間[0.042160,0.473909]；這是描述區間，不宣稱全表多重校正。T2.7受控污染的單一owner反推模型殘差>.1完整保留。

- `summary.csv`：240筆隊層級均值／區間，含truth、角色與whole-task truth LOO的排名／符號、reserve及TV；有效n與undefined保留。
- `paired_target_changes.csv`：144筆同隊target−新T0差異。
- `sign_flip_summary.csv`：24筆F正分轉peer負分的team-cluster比例區間，零分母draw明列。
- `tasks.csv`：80任務交付、反推、查詢及observer對照成本。
- `acceptance_summary.json`：41來源執行前鎖、27330分數／400truth／1680模型端點檢查、224356個cache-miss模型回覆重算、435162 cache hits，以及240摘要與144配對區間獨立重算。

新研究cache misses224356、反推forward samples160320；observer等價檢查額外27666。方法共用各任務cache，不冒稱這是各種角色方法的獨立冷快取成本。原首次run與smoke另列成本；為補齊helper的執行前來源鎖而重跑80任務，數值除seconds相同、七張統計CSV逐位元相同。完整校驗保留在正式run。

truth whole-task LOO是指定反事實參照：移除模型與錨點後，從full錨點平均的同一起點重跑，保留該起點資訊；不是重新招募／反推的系統整體刪除。真值不參與角色評分或付款，未執行實際付款；本研究未改變起點seed，起點穩定度屬後續任務11。
