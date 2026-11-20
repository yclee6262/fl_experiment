# 初始化方法實作

| 檔案 | 用途 |
|---|---|
| `initial_alpha_methods.py` | U、Q、QU 與 SLSQP ensemble fit；保留原數值與 fallback 規則 |
| `unified_alpha_methods.py` | E 與 EQ；可重用相同盲測預測快取 |
| `peer_support_alpha.py` | 非自身模型支持度與 SU |
| `support_quality_mix_experiment.py` | SQ |
| `four_alpha_initialization.py` | 原 U/Q/EU/EQ/SU/SQ 的固定比較及預先指定 contrasts |
| `pure_es_initialization.py` | 純 E、S |
| `se_qu_initialization.py` | SE、QU |

initializer 接收節點順序、盲測 MSE／預測／標籤、候選與 target，回傳同順序的 alpha 及 audit。支持矩陣排除對角線。缺失資料、單節點與 fitting failure 依各方法明列，不以 pooled 分數選權重。

歷史方法 key 為相容資料格式保留；U 對應 `A_uniform`，Q 對應 `B_stage0`。混合的是 alpha，不是不同求解的 X。
