# 模型共識池 v1 實作交付

2026-10-28。**技術實作完成；研究有效性未建立。** 設計來源為 `docs/model_consensus_pool_design_v1_20261028.md`，未更動設計文件、歷史算法、結果、固定研究入口或 Notion。

## 已交付

所有程式均新增於 `contribution_analysis/model_consensus_pool_v1/`：

- `core.py`：跨欄位 validate、固定 probe 線性分位尺度、每 owner 原 v 正規化、逐輪 min/max、有號累積、固定門檻、精確最大餘數整數分配。
- `reference.py`：不匯入核心函式的獨立 Fraction 重算及 audit；核對 C/own/others/peer、逐輪上下界、有號和、比例、整數金額。
- `runner.py`：離線 JSON CLI、拒絕重複 JSON key／非法外層、exclusive-create 輸出、輸入與 source hashes、零 solver/API 呼叫成本紀錄。
- `input.schema.json`、`examples/hand_calculation.json`、`README.md`、`tests/test_core.py`。

精確十進位輸入轉 Fraction 計算，含 1e-12／1e-10／1e-8 門檻與取餘同分；只在 JSON 報告時轉有限浮點數。相同端點 ID 的完整 loss 字典必須一致，同輪所有 owner 的 full ID 與 loss 亦必須一致。own 與 others 是原權重加權的 C 分解，peer 才是其他 owner 原 v 正規化平均；皆僅診斷。

## 本輪實測

實際指令：

```sh
python -m unittest contribution_analysis.model_consensus_pool_v1.tests.test_core -v
python -m contribution_analysis.model_consensus_pool_v1.runner --input contribution_analysis/model_consensus_pool_v1/examples/hand_calculation.json --output /tmp/model-consensus-smoke-20261028-final.json
```

14 項單元測試通過，最近紀錄 `/tmp/model_consensus_pool_unit.log`：手算分配、正負相抵、全非正及零作用保留、缺漏／重複／空 slot、未知及不可 hash owner/round、壞 shape／非法 loss、無 minus、單 owner／零 owner 權重、零／近零／精確門檻尺度、版本與 hash、full 與端點 ID 一致性、0/1/2/500/501/10^30 預算、字典序 tie、名冊／事件重排、同 owner 等價模型拆分、固定端點 loss+probe 同比例、C/q/金額篡改偵測、診斷分解、exclusive-create 與非法 JSON CLI。

微型 smoke 結果 `allocated`，a/b/c=250/250/0，reserve=0，輸入3718 bytes；計時約0.001086秒，solver_calls=0、model_api_calls=0。這是合成 fixture 技術 smoke，非研究比較。公開 settle 參數防護加入後已重跑14項全部通過；最終紀錄以 `/tmp/model_consensus_pool_unit.log` 為準。

## 尚未完成與限制

沒有 full-owner T3 分支生成器，沒有新比較研究 runner；完整可執行入口目前是離線 JSON scoring runner。下一輪必須另行凍結 fixtures、source、probe 和驗收後，再由原輪狀態生成每個 owner×round 的完整 model-only minus，不得將僅 target0 的歷史資料當完整矩陣。本輪未執行30設定、真實模型比較、神經搜尋或付款。

版本／hash／probe_id 僅驗證宣告一致，不證明身份、真實查詢、先於搜尋鎖定或接受端點來源；上游生成器仍須建立此證據。共識錯誤與正確少數否決仍是方法限制。程式通過不等於準確、公平或可部署。
