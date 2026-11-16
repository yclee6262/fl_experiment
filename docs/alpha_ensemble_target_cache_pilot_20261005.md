# 加權集成目標的 Stage 3 查詢快取：等解 pilot

## 目的與比較口徑

原五候選流程的快取已在另一組 pilot 中確認可以重用相同 `AgentNode` 輸入。本次檢查同一快取用在新的 Stage 3 加權集成目標 `|Σ α_i f_i(X)−T|` 時，是否仍保持候選解、最終選解與 pooled 評分相同，以及實際節省多少模型呼叫。這是成本工程檢查，**不是新初始 α 的品質實驗**。

沿用已完成的低品質正式研究中 seeds 2700–2704 的 local models、proposal、初始 α 與凍結 pooled evaluator。原無快取參考來自 `alpha_weight_outputs/alpha_ensemble_target_formal_assessment_2700_2879/seed_rows.jsonl`；快取重播保存於 `alpha_weight_outputs/alpha_ensemble_target_cache_pilot_2700_2704.json`。每個 α 候選的 Stage 3 單獨建立快取，輸入鍵為模型實際接收的 `float32` 形狀與位元內容；模型 API 計數器位於快取內層，因此只計真正呼叫模型的 miss。每次候選結束即移除快取，選解器與 pooled evaluator 不使用它。

## 結果與驗收

| 項目 | 結果 |
|---|---:|
| 重播 seeds／Stage 3 候選 | 5／25 |
| 候選 X、α、solver、內部分數、最終選解及 pooled／truth 分數最大差 | 0 |
| 無快取 Stage 3 模型呼叫 | 9,618 |
| 快取後 Stage 3 模型呼叫 | 8,709 |
| 快取命中／節省模型呼叫 | 909（9.45%） |
| 每候選 `misses = 實際呼叫` 且 `hits + misses = 無快取呼叫` | 25／25 通過 |

這個百分比只以 **Stage 3 local-model API 呼叫**為分母；共同資料與模型準備、初始化、選解、外部評分和網路延遲未包含。五個 seeds 是先前正式品質研究的子集，重用它們只驗證等解與成本記帳；不得將本 pilot 當作新目標的另一批獨立品質證據。快取只適用於目前固定且確定性輸出的 `AgentNode` 模型，模型權重變更、非確定性遠端 API 或不同輸入轉換都需重新驗證。

重播命令：

```bash
python explore_ensemble_target_objective.py alpha_weight_outputs/alpha_ensemble_target_baseline_formal_2700_2879 --seeds 2700,2701,2702,2703,2704 --cache-stage3 --output alpha_weight_outputs/alpha_ensemble_target_cache_pilot_2700_2704.json
python verify_ensemble_target_cache.py alpha_weight_outputs/alpha_ensemble_target_formal_assessment_2700_2879/seed_rows.jsonl alpha_weight_outputs/alpha_ensemble_target_cache_pilot_2700_2704.json
```

下一步可在更多保存模型上重播以估計呼叫節省的變異，並在新情境與實際供應方計價單位下重新計算整體成本。無快取的正式品質比較維持原版解讀，不因本快取 pilot 重寫。

## 全 180-seed 保存模型等解稽核

已進一步重播同一低品質正式批次的全部 seeds 2700–2879，保存於 `alpha_weight_outputs/alpha_ensemble_target_cache_replay_2700_2879.json`。這仍是**同一批既有模型的工程重播**，不增加獨立品質樣本。`verify_ensemble_target_cache.py` 逐一比較 900 次候選求解及 180 次交付選解：X、α、solver、內部分數、pooled／truth 評分最大差均為 **0**；每候選的快取命中加實際模型呼叫，與無快取呼叫數完全相等。

| 全批 Stage 3 模型呼叫 | 每 seed 平均 | 180 seeds 合計 |
|---|---:|---:|
| 新集成目標，無快取 | 1,685.83 | 303,450 |
| 新集成目標，有快取 | 1,505.82 | 271,047 |
| 節省 | 180.02（10.68%） | 32,403 |

相對舊目標正式批次的平均 1,817.35 次 Stage 3 呼叫，「新目標加快取」在**相同 seeds 的呼叫計數**平均少約 17.14%。這是目標函數與等解快取的合計觀察差，並非兩個百分比相加；不包含 Stage 3 外成本。正式品質改善仍由先前**未開快取**的事前登錄比較支持，快取只透過全 900 候選等解檢查繼承其 X 與 pooled 分數。

全批等解驗證命令：

```bash
python verify_ensemble_target_cache.py alpha_weight_outputs/alpha_ensemble_target_formal_assessment_2700_2879/seed_rows.jsonl alpha_weight_outputs/alpha_ensemble_target_cache_replay_2700_2879.json
```
