# Alpha 初始化與參與者貢獻方法調整

> 更新日期：2026-10-06  
> 本頁目的：整理目前已完成的研究調整、實驗結論，以及需要與老師確認的下一階段方向。

## 一、目前完成的工作

本階段主要完成三件事：

1. 建立一致的外部品質評估標準，用來比較不同 Alpha 方法最後產生的倒推解。
2. 比較靜態 Alpha、動態更新與多種 Alpha 初始化方法，確認目前哪些方法有證據支持。
3. 重新檢查原本 leave-one-out（LOO）貢獻的定義，並建立「完整移除參與者」的反事實比較方法。

目前研究重點不是繼續增加候選方法，而是先確認：

- Alpha 初始化使用了什麼資訊；
- 最終倒推解是否真的改善；
- 額外品質是否值得增加的模型查詢成本；
- 所定義的參與者貢獻，是否真的對應到「沒有這位參與者時會發生什麼事」。

---

## 二、建立外部品質評估標準

### 2.1 為什麼需要外部評估標準

原本的 Stage 3 最佳化與 Alpha 更新，使用參與者 local models 所形成的內部目標函數。若仍用相同的內部函數評估結果，可能只能證明演算法更符合自己的最佳化目標，無法證明最後交付的倒推解 `X` 真的更好。

因此，本研究為每個實驗情境建立一個凍結的 pooled-data global evaluator，並以

`E_global(X) = |f_global(X) - Y*|`

衡量最終解與目標值的距離。數值越低，代表交付的 `X` 經外部模型判斷後越接近指定目標。

### 2.2 使用邊界

global evaluator 僅作為研究階段的事後評分標準：

- 不參與 Alpha 初始化；
- 不參與 Alpha 更新；
- 不參與 coalition 選擇；
- 不參與 Stage 3 求解或候選解選擇；
- 不假設實際部署時存在可共同使用的 global model。

因此，它的角色是「外部研究評審」，不是實際系統中的決策模型。

---

## 三、動態 Alpha 更新與 Guarded 早停

正式配對實驗比較以下四種設定：

- A：Uniform Alpha，不更新；
- B：Stage 0 品質 Alpha，不更新；
- C：Uniform 起始，固定步長更新；
- D：Uniform 起始，Guarded 更新。

實驗結果顯示，動態更新雖然能讓 Alpha 更接近演算法所計算的貢獻分布 `q`，但沒有在 global evaluator 上建立穩定的最終解品質改善；Guarded 的模型查詢成本則明顯高於靜態方法。因此，「Alpha 更接近 q」不能直接當成「最終 X 品質更好」的證據。

### 無進展早停

部分 Guarded 實驗在最小更新步長下會持續拒絕幾乎相同的 proposal，直到用完最大 rounds。為避免重複計算，本研究加入無進展早停：當更新步長已達下限，且連續多次 proposal 都被拒絕時，提前停止後續重複求解。

正式驗證結果如下：

- Clean 情境的總查詢量降低 48.8%；
- Controlled poison 情境的總查詢量降低 55.8%；
- 60 組配對實驗的 final accepted X、Alpha、accepted round 與外部品質均完全一致；
- 沒有增加 solver failure、box failure 或 invalid q。

早停改善的是不必要的計算成本，不代表 Guarded 已經收斂，也不改變「動態更新尚未證明優於靜態基準」的結論。

---

## 四、已嘗試的 Alpha 初始化方法

Alpha 在本研究中代表 Stage 3 使用各 local model 資訊的權重，不是參與者的價格、付款比例或真實貢獻。

### 4.1 基準方法

| 方法 | 核心概念 | 目前判讀 |
|---|---|---|
| Uniform（A） | 所有參與者等權，`Alpha_i = 1/n` | 不需額外初始化資訊，作為一般低成本基準。 |
| Stage 0 品質（B） | 將 20 筆 Stage 0 盲測的個別 MSE 取倒數後正規化 | 在已測低品質情境中優於 Uniform，是目前最重要的低成本單次初始化候選；但尚不能宣稱在所有情境下最佳。 |

Stage 0 品質權重的計算為：

`Alpha_i = normalize(1 / (MSE_i + epsilon))`

其意義是：Stage 0 盲測誤差較低的 local model，在倒推時取得較高資訊權重。它衡量的是盲測預測品質，不是付款或公平貢獻。

### 4.2 控制權重過度集中的方法

| 方法 | 核心概念 | 目前判讀 |
|---|---|---|
| Quality shrink | 將 Uniform 與 Stage 0 品質權重各取一半 | 低品質情境優於 Uniform，但尚未證明優於 Stage 0 品質權重。 |
| MSE power 1.5 | 將 inverse-MSE 提高到 1.5 次方，更集中相信盲測表現最佳者 | 已完成測試，但尚無單獨優於 A／B 的正式證據；權重過度集中可能放大盲測誤差。 |
| Best-agent | 將全部權重給 Stage 0 MSE 最低者 | 沒有建立穩定品質改善，且可能忽略模型互補性。 |
| Sparse top-two | 只保留 Stage 0 表現最佳的兩位參與者權重 | 有降低部分模型查詢的可能，但目前只有開發性證據，品質改善未重現。 |

### 4.3 使用模型互補或殘差結構的方法

| 方法 | 核心概念 | 目前判讀 |
|---|---|---|
| Ensemble shrink | 在 Stage 0 盲測集上尋找最佳非負模型組合，再向 Uniform 收縮 | 低品質情境優於 Uniform，但未證明優於 B；有小樣本過擬合與額外查詢風險。 |
| Corr precision | 同時考慮個別 MSE 與不同模型殘差的共同變化 | 可避免重複相信具有相似錯誤的模型，但尚無單獨優於 A／B 的正式證據。 |
| Marginal information | 暫時拿掉某個模型的預測，觀察既有 Stage 0 集成 MSE 增加多少 | 部分實驗曾改善結果，但未在不同情境與複驗中建立穩定優勢。此處只移除模型預測，不等於完整移除參與者。 |

### 4.4 使用目標區域或 proposal 局部資訊的方法

| 方法 | 核心概念 | 目前判讀 |
|---|---|---|
| Target-local precision | 加重接近本次 target 的 Stage 0 樣本，判斷誰在目標附近較準 | 正式比較未證明優於 Stage 0 品質權重。 |
| Target-peer shrink | 由其他模型判斷各參與者 proposal 是否接近 target，再向 Uniform 收縮 | 低品質情境優於 Uniform，但未證明優於 B；若多個模型有共同偏差，可能一起做出錯誤判斷。 |
| Local-bias-risk | 估計 proposal 區域的模型偏差平方與剩餘誤差，再依風險反比分配權重 | 20 筆與 100 筆標記版本均已測試；100 筆版本需要額外可信標記，但正式比較未通過品質門檻。 |
| Proposal 附近標記 | 在各參與者 proposal 附近取得新的可信標記，重新估計局部誤差 | 需要額外標記成本，目前開發測試未顯示穩定收益。 |

### 4.5 本階段結論

目前沒有證據支持某一種 Alpha 初始化在所有情境下普遍最佳。較穩健的說法是：

- Uniform 是一般、低成本且容易解釋的基準；
- Stage 0 品質權重在已測低品質情境最有競爭力，是目前優先保留的低成本單次初始化方法；
- 其他方法提供不同的風險控制、模型互補或局部資訊假說，但目前尚未穩定超越 Stage 0 品質權重；
- 部分複雜方法需要更多模型查詢或可信標記，必須把資訊取得成本一起納入評估。

---

## 五、為什麼需要修改原本的 LOO 方法

### 5.1 原本 LOO 實際衡量什麼

原本的 operational LOO 並沒有完整移除參與者。對參與者 `i`，它只從 Stage 2 所形成的候選子空間中移除該參與者的 candidate point，再重新求解 restricted solution。

但是：

- 該參與者的 local model 仍在 Stage 3 evaluator 中；
- 該參與者的 Alpha 仍保留；
- 該參與者仍然是 coalition 的成員。

因此原方法回答的是：

> 如果不能使用這位參與者提供的候選方向，但仍可使用他的模型與權重，最終解會受到多少影響？

這比較接近「候選方向／子空間的重要性」，而不是「參與者本身的貢獻」。

### 5.2 新的完整參與者移除

新的 full-removal counterfactual 會同時移除：

1. 該參與者的 candidate point；
2. 該參與者的 local model engine；
3. 該參與者的 Alpha。

接著將剩餘參與者的 Alpha 重新正規化，並從相同固定起點重新執行 Stage 3。為保持比較範圍清楚，此實驗不重新訓練其他 local models，也不重新招募替代者。

它回答的是：

> 如果這位參與者沒有加入目前的 coalition，在其他條件固定下，最後交付的倒推解會如何改變？

### 5.3 兩種方法的差別

| 比較項目 | 原本 candidate-only LOO | 新的 full participant removal |
|---|---|---|
| 移除 candidate point | 是 | 是 |
| 移除 local model engine | 否 | 是 |
| 移除該參與者 Alpha | 否 | 是 |
| 重新正規化剩餘 Alpha | 否 | 是 |
| 重新執行 Stage 3 | 是 | 是 |
| 代表候選方向影響 | 是 | 部分包含 |
| 代表「沒有此參與者」的反事實 | 否 | 較接近 |
| 等同 Shapley value | 否 | 否 |

### 5.4 為什麼不能繼續把原 LOO 當成參與者貢獻

正式實驗使用獨立 seeds，比較原 operational contribution 與 full-removal 後的 pooled 品質變化。結果顯示：

- Clean 情境的平均 Spearman correlation 為 0.083；
- Controlled poison 情境為 -0.047；
- 方向一致率分別只有 40.0% 與 48.7%；
- 即使使用 pooled evaluator 重新評分 candidate-only 解，與 full-removal 的排名關係仍接近零。

這表示主要問題不只是內部 evaluator 與外部 evaluator 的尺度不同，而是兩種方法本來就在回答不同的反事實問題。

例如，原方法將某人判定為負貢獻，不代表完整移除他之後一定會改善最終品質。正式結果中，原方法判定的負貢獻者，在完整移除後實際改善 pooled 品質的比例只有：

- Clean：32.0%；
- Controlled poison：42.5%。

因此，如果研究問題是 participant-level contribution，原本的 candidate-only LOO 定義不足，需要改用完整參與者移除，或研究能近似 full-removal 的低成本方法。

### 5.5 目前研究決策

- 原 operational LOO 可以保留為「候選點排除下的內部影響」診斷；
- 不再把它稱為完整參與者貢獻；
- 不以原方法得到的負貢獻直接進行自動 pruning；
- 不以原 q 支持公平分潤或 Shapley value 的主張；
- 若付款或參與者貢獻需要 participant-level 解釋，應改以 full-removal 為概念標準；
- full-removal 仍不是 Shapley value，因為它沒有枚舉所有 coalition 與加入順序。

---

## 六、下一階段方向：待老師確認後再執行

### 方向一：是否繼續研究多候選初始化

多候選方法的想法是使用數組不同初始 Alpha，各自執行一次 Stage 3，再從多個可行解中選擇一個交付。它可能提高找到較好解的機會，但也會增加模型查詢與求解成本。

本階段只將它保留為後續研究方向，不把它列為目前主要方法。若老師同意繼續，下一階段才固定：

- 候選 Alpha 的組合；
- 內部選解規則；
- 最大可接受查詢成本；
- 相對單次 Stage 0 方法必須達成的品質改善門檻；
- 使用全新 seeds 的正式驗證方式。

### 方向二：使用 full-removal 計算參與者貢獻

若論文需要 participant-level contribution，下一階段預計直接使用 full-removal：依序完整移除每位參與者的 candidate、local model engine 與 α，將剩餘 α 重新正規化，再重新執行 Stage 3，比較移除前後的最終解品質。

此方法回答的是「如果這位參與者未加入目前 coalition，最終交付解會如何改變」，比原本只排除 candidate direction 的 LOO 更符合參與者層級的貢獻定義。

---

## 七、目前可向老師確認的決策

1. 是否同意目前以 Uniform 作一般基準，並將 Stage 0 品質權重保留為低成本、低品質情境的優先初始化方法？
2. 是否同意暫停擴充更多 Alpha 初始化公式，先不把多候選法納入主要流程？
3. 是否同意將原 LOO 重新命名為「candidate-only internal influence」，不再作 participant contribution 解釋？
4. 論文是否需要 participant-level contribution？若需要，下一階段直接使用 full-removal 計算；若不需要，則將原 LOO 限定為內部診斷即可。

## 八、一句話總結

目前證據支持使用外部 evaluator 檢查最終解品質、保留 Stage 0 品質權重作低成本初始化候選，並將參與者貢獻的定義由「只移除候選方向」修正為「完整移除 candidate、model 與 Alpha」的 full-removal；多候選初始化與是否正式採用 full-removal，待老師確認後再進入下一階段。
