# Custom 實現改善歸屬：實作設計 v1

日期：2026-10-27。狀態：**設計完成，尚未實作、尚未執行本版驗證。**

## 1. 決策、範圍與交付

使用者希望從 custom 本來就產生的資訊建立有合理經濟解釋的貢獻帳，暫不比較 LOO。本文件供後續實作者使用。本次只新增設計文件。

主問題：在固定團隊、模型、錨點、權重、起點及實際搜尋路徑下，如何將已接受步驟的內部改善，按實際方向支持程度分帳，並保留模型評價與服務成本？

保留 J 的既有定義及所有歷史結果。本版是獨立方法 `custom_realized_gain_v1`，不得覆寫歷史 A、B、M、J 欄位或回填正式確認結果。模型與錨點分帳不相加；不產生支付金額、不更新 alpha、不 pruning、不使用 Shapley、不重訓。pooled 不進入搜尋或主分數；本版必要驗收不需要 pooled 或 truth。

實作者應交付：唯讀觀察器、事件格式、純離線計分器、獨立重建驗證器、單元及整合測試、成本量測工具、案例報告、來源及輸出雜湊清單。

## 2. 與既有研究的關係

已核對 `host_server.py:_solve_subspace_custom`、`_consensus_loss`，並對照 `j_loo_start_stability_experiment_v1_20261027/frozen/host_server.py` 的方向計算。既有程式可參考 `contribution_analysis/dual_role_v2/direction_observer.py` 與 `core.py`，但不得直接把其輸出改名成本版。

`docs/contribution_dual_role_economic_design_v2_20261020.md` 已提出「方向投影 × 實際改善」的有號 A/B 分帳；`docs/custom_direction_results_20261020.md` 記錄過方向分解及其他估計目標的驗證。新的矩陣分解不是首次提出，本版也不能以守恆宣稱新的因果發現。

| 項目 | 舊雙角色設計 | 本版 |
|---|---|---|
| 錨點主分數 | 有號投影乘改善，可負或大於總改善 | 正向投影先在歸屬單位內聚合，再正規化分攤改善 |
| 模型主輸出 | 方向有號作用 B，及品質折扣 M=qB | 更新端點的個別加權 loss 改善 E；只稱改善評價 |
| 品質折扣 | 既有驗證 MSE 轉成 q | 不使用 |
| 方向來源矩陣 | 模型 × 錨點 | 固定採錨點 × 模型；仍保留有號，不裁切每個格子 |
| 評估問題 | 包含對移除效果的比較 | 運算忠實、帳目守恆、成本、尺度與歸屬敏感性 |

先前相關結果不能當成本版的驗收，也不因本次改變研究問題而撤回舊結論。

## 3. 經濟解釋與不能推出的結論

1. 內部 loss 的下降是本次團隊目標的已實現改善；它不是貨幣或外部福利。
2. 錨點帳是「依已採用方向的正向支持程度分攤成果」的明示規約，不是唯一公平分配。
3. 模型 E 帳是各評分觀點如何評價同一個更新，不是模型正確性或造成改善的因果效果。
4. 服務帳是呼叫及資源成本，不是價值。重複呼叫不增加成果分數。
5. 抵銷訊號、被拒絕的探索、停止判斷可能有價值；本版沒有替這些價值定價。
6. 錨點 A 與模型 E 各自描述同一份 G，不能 A+E，也不能據此分兩次同一筆收益。
7. 既有權重是本次契約／程序條件。分數受權重影響，不是權重獨立的能力排名。
8. 不宣稱激勵相容、抗虛假身分、抗操弄、模型品質或跨任務公平已成立。

## 4. 支援條件及符號

- K 個錨點、M 個模型、d 維輸入；允許 K≠M，不以陣列下標隱含同一所有人。
- 錨點 I_k；方向權重 a_k；模型權重 v_m。兩種權重分開保存原值與正規化值，均須有限、非負且總和正。
- 固定模型 f_m、目標 T、box、模型順序、錨點順序、起點及求解設定。不得由計分器改動。
- ell_m(x)=abs(f_m(x)-T)，L(x)=Σ_m v_m ell_m(x)，Σv=Σa=1。
- 本版只支援固定加權和目標。median、trimmed、動態權重、隨機評分與模型版本中途變動標記 unsupported；不偷偷改成均勻權重。
- 原 solver 接受判斷一律使用原 objective，不使用 history_metric。history_metric 可不同，不能拿來算 G。
- 固定 `anchor_id -> owner_id` 與 `model_id -> owner_id`，開跑前登錄，不根據結果重組。

## 5. 從原演算法讀取的狀態

原 solver 初始化會計算 history metric、各錨點 objective、目前 objective；eta 初始 0.1，mode=secant，delta=0.0001。實作者讀取實際值，不另寫第二套設定強行覆蓋。

每輪先計算方向：距離 <1e-8 的錨點跳過；dynamic 的實際有界探測步 h<=1e-12 時跳過。原總方向 norm<=1e-8 就停止。其餘情況正規化後試走；退火最多10次，以原嚴格 candidate_loss<best_loss 判斷接受，縮步與成功後 eta 更新照原程式。secant 失敗可能切換 dynamic；dynamic 失敗或其他停止分支須保存。

所有輪都要有結束事件，包括在 history append 前 break 的輪。不能只從 history 長度推算輪數。試走縮步不重新生成方向，不可對每個 attempt 重複分配同一輪成果。

## 6. 方向來源的精確定義

輪 t 的起點 x_t。r_kt=||I_k-x_t||，u_kt=(I_k-x_t)/r_kt。

對非跳過的錨點：

```
secant:
  z_kt = I_k
  h_kt = r_kt
dynamic:
  h_kt = solver 實際使用的 _bounded_step 結果
  z_kt = x_t + h_kt*u_kt

b_kmt = (ell_m(z_kt)-ell_m(x_t))/h_kt
V_kmt = a_k*v_m*b_kmt*u_kt
H_kt = Σ_m V_kmt
F_mt = Σ_k V_kmt
g_rebuilt = Σ_k H_kt = Σ_m F_mt
```

跳過的錨點所有 V/H 設0，保留原因。不要補查模型、不要用固定 delta 代替被 box 截短的實際 h。

`g_solver` 必須是原程式正規化前的 gradient。重建值只核對，不回寫。原程式先聚合 scalar loss 再作差，與先作逐模型差再聚合可能有浮點差異，須用數值容差比較，不宣稱必然 bitwise 相等。

令 n_t=||g_solver||。若 n_t<=原停止閾值，只核對未正規化方向並保存停止事件，q/P/投影份額為null，A/E增量0，不執行除法。其餘情況 q_t=g_solver/n_t。q 是上坡訊號方向，實際移動方向為 -q；使用 dot(H,q) 正值代表支持實際移動，不能誤將負號再翻一次。

可保存有號純量矩陣 P_kmt=dot(V_kmt,q_t)，其列和是錨點投影，欄和是模型方向投影；總和約等於 n_t。P 不正值化、不變成付款表。模型 E 與 P 欄和不是同一件事。

## 7. 錨點主帳 A_support

實作同時提供錨點明細與固定所有人聚合。成果分配的歸屬單位須在 manifest 指定：v1 主輸出為 `owner`；每人一錨點時與前述錨點公式一致。

```
p_kt = dot(H_kt,q_t)
p_ot = Σ_{k: owner(k)=o} p_kt
positive_ot = max(p_ot,0)
support_sum_t = Σ_o positive_ot
share_ot = positive_ot/support_sum_t
G_t = L_before - L_accepted_after
A_increment_ot = G_t*share_ot
A_support_o = Σ_{accepted t} A_increment_ot
```

必須先在 owner 內相加，再取正部。不能逐錨點裁負後才加總到 owner，否則同一所有人的正負訊號拆分可能增加份額。錨點級投影全部保留；不額外產生一份可與 owner 帳相加的分配總額。

非接受輪 G=0、A_increment=0；可保留方向診斷，`allocation_share=null`，避免把沒有成果的方向份額誤作報酬。接受輪若 n<=原停止閾值、support_sum<=0、位移為0或 G<=0，技術驗收失敗，不均分、不加 epsilon 救分母。

理想算術下：A>=0、每接受輪 Σshare=1、Σ_o A=Σ_t G。負投影記為 opposing_projection，正投影記為 supporting_projection；不是負品質或懲罰。

不再乘實際步長：G 已包含實際被接受端點的改善。縮步成功只分配最終被接受那一步的 G。

## 8. 模型改善評價帳 E

```
E_increment_mt = v_m*(ell_m(x_t)-ell_m(x_(t+1)))
E_model_m = Σ_{accepted t} E_increment_mt
E_owner_o = Σ_{m:owner(m)=o} E_model_m
```

保留正負，不裁切、不乘品質權重、不乘呼叫數、不正規化為支付比例。非接受輪增量0。

Σ_m E_increment=G_t，Σ_m E_model=G_total；固定條件下，每個模型也應滿足 E_model_m=v_m*(ell_m(x0)-ell_m(x_final))。這個端點恆等式顯示 E 不包含「評分如何改變中間決策」的完整資訊，因此不能宣稱 E 是模型決策貢獻。

模型 E 可大於總 G，也可為負，因模型意見抵銷；不能將 E/G 解讀為安全的分配比例。G=0 時所有主分配比例為 null；不得使用均分補值。

## 9. 全程改善、數值敏感性與結果狀態

G_total=Σ接受輪G_t；核對 G_endpoint=L_initial-L_delivered。delivered 應對應原 solver 的 global_best_S。若沒有接受步驟，A/E皆0且份額null，狀態 no_realized_gain，不是程式失敗。

主計算使用 float64。基本核對採 abs(a-b)<=1e-10+1e-10*max(abs(a),abs(b))；向量按最大元素誤差及尺度核對。配置與數值環境保存，不可看到結果後放寬門檻。

原 norm/距離/步長/接受閾值不變。歸因容差只做檢查，不修改 solver 接受規則。非常小但真正接受的 G 仍計入；另標記 `gain_below_reporting_tolerance`，不刪步。

抵銷診斷：

```
rho_anchor = ||g_solver|| / Σ_k ||H_kt||
rho_owner  = ||g_solver|| / Σ_o ||Σ_{k in o}H_kt||
rho_model  = ||g_solver|| / Σ_m ||F_mt||
```

分母0時 null；任一有效 rho<1e-6 記 numerically_sensitive。保持原分數，不調權或刪除案例。報敏感輪數及其 G 占總 G 比例。正向份額有界並不保證方向對微小擾動穩定。

狀態分離：execution_status、attribution_status、verification_status。至少區分 complete/failed，available/no_realized_gain/numerically_sensitive/unsupported/missing_trace，passed/failed/not_run。unsupported 或缺資料不填零。

## 10. 唯讀紀錄架構

建議獨立模組 `contribution_analysis/custom_realized_gain_v1/`：

| 檔案 | 責任 |
|---|---|
| schema.py | typed records、版本、欄位及形狀驗證 |
| observer.py | 只記錄原 solver 已算出的值及事件 |
| objective_trace.py | 攔截既有模型回覆與 objective 聚合，不補查 |
| accounting.py | 從完整 trace 離線計算 A/E/P/診斷 |
| verify.py | 獨立公式重建及查詢／搜尋一致性核對，不匯入 accounting |
| run_pilot.py | manifest、control/record 執行、成本與輸出 |
| report.py | 以固定模板輸出案例與技術結果 |
| tests/ | 合成、分支、整合及機制性測試 |
| README.md | 命令、輸出、限制與驗收順序 |

優先用經來源 hash 鎖定、注入前後 AST 結構可檢查的唯讀 adapter；可參考舊 observer。若改為正式 callback，須保證 default None 完全走原流程，並在新變更中獨立審查。不能改歷史凍結 solver。

舊 Collector 依座標保存「最近一次回覆」不足以作本版完整證據。應保存每次 objective 的 eval_id 及每次 API 的 query_id，並明確連結 solver 實際使用的回覆。secant 要連到初始化時的 anchor eval；目前點連到建立 best_loss 的那一次 eval；dynamic 連當輪 probe eval；接受端點連被接受的 trial eval。

允許以座標建立索引方便查找，但不能覆蓋事件、不能新增快取而跳過原本呼叫。history_metric、final_return 等額外原生呼叫也保存並單獨分類。相同輸入但回覆不一致時標 unsupported_stochastic，保留證據，不任選最近值。

observer 持有陣列副本，不保留會被 solver 改寫的 view。觀察器錯誤應中止研究記錄並保留失敗，不能輸出完整成功的歸因檔。部署行為不在本版範圍。

建議介面（僅設計，不是本次實作）：

```python
record_custom_run(server, initial_solution, solver_config, output_dir) -> RunArtifact
compute_accounts(manifest, evaluations, iterations) -> Accounts
verify_run(control, recorded, accounts) -> Verification
```

compute_accounts 不能存取模型、server、pooled 或網路；純函式重跑必須產生相同數值。verify 讀原事件，以獨立求和及公式重建，不把新計分器輸出當真值。

## 11. 事件與輸出格式

輸出目錄 exclusive-create；存在即拒絕覆寫。建議 `contribution_analysis/custom_realized_gain_pilot_v1_<date>/`。

### manifest.json

schema_version、method_version、run_id、created_at、source_hashes、Python/NumPy/模型套件版本、dtype、device、thread設定、model hashes、anchor座標、anchor/model/owner ID及映射、原始與正規化權重、target、box、X0、solver參數、attribution_unit、容差、truth_loaded=false、pooled_loaded=false。起點來源亦記錄（explicit或anchor_mean）。

### evaluations.jsonl

每個 objective 事件包含 eval_id、event_index、purpose、iteration nullable、anchor_id nullable、attempt nullable、x、model_query_ids、individual_losses[M]、objective_value、weights_version。purpose 分 initialization_anchor/current、dynamic_probe、trial、history_metric、final_return 等。

### queries.jsonl

query_id、eval_id nullable、model_id、model_version、exact input及shape/dtype、output、call_seconds、input_samples、success/error。唯一查詢鍵以模型版本及精確 float64 bytes 建立，僅用統計不節省呼叫。記錄API呼叫與輸入樣本數，兩者不能混淆。

### iterations.jsonl

每輪一筆：t、mode_before/after、x_before、current_eval_id、L_before、eta_before/after、delta、directions[K]（skip_reason、distance、u、actual_h、source_eval_id、original_derivative、direction_weight）、g_solver、g_norm、trials[]（eval_id、requested_eta、actual_bounded_step、accepted）、accepted_eval_id nullable、x_after、L_after、stop_reason、mode_switch。

零方向輪也有紀錄。若因異常無法結束，run_status=incomplete，不能默認零改善。

### accounts.json / diagnostics.jsonl

每輪：G、p_anchor、p_owner、allocation_share或null、A_increment_owner、E_increment_model、P[K,M]、rho及數值核對誤差。

全程：A_support_owner、E_model、E_owner、G_total、G_endpoint、owner_share=A/G或null、accepted/rejected counts、status、缺失及敏感覆蓋率。未分配量 residual=G_total-sum(A)原樣保存，不補給最大份額者。

V[K,M,d] 可在除錯模式輸出；正常模式從原回覆重建，不必長期存整個張量。P是有號方向訊號表，不是聯合收益分配表。

### costs.json、verification.json、report.md、receipt.json

costs 分模型呼叫、輸入樣本、唯一輸入數、各模型耗時、搜尋含觀察耗時、離線計分時間、序列化時間、輸出bytes、Python peak；若無RSS/native記憶體量測則明說。receipt 包含來源、配置、所有結果 hash、驗證狀態，不將 hash 通過當成經濟有效性。

## 12. 離線計分流程

```text
validate manifest / fixed weighted-sum objective / trace completeness
initialize A(owner)=0, E(model)=0
for iteration in execution order:
    resolve actual evaluation event references
    rebuild per-model slopes, H/F and signed P
    compare rebuilt direction with original g_solver
    record skips, cancellations, trials and transition
    if no accepted trial: record zero increments; continue
    assert exactly one accepted trial, positive actual gain and motion
    verify actual motion is collinear with -g_solver
    aggregate anchor projections by fixed owner, THEN take positive parts
    compute shares and A increments
    compute endpoint model E increments
    verify per-step conservation and accumulate
verify A/E/G endpoint identities and return explicit statuses
```

模型 E 的單輪負值要保留，即使團隊 G>0。初始錨點評分及失敗探索只進成本／診斷帳。不能用 elapsed time 或縮步次數把 G 放大。

## 13. 手算案例與預期輸出

一維、三錨點各屬不同owner、二模型，a=[1/3,1/3,1/3]，v=[0.5,0.5]；取 u_k=+1，方向差商為：

```
b = [[4,14], [8,-2], [-4,-2]]
V = [[2/3,7/3], [4/3,-1/3], [-2/3,-1/3]]
H = [3,1,-1], F = [4/3,5/3], g=3, q=+1
```

假設這一步被接受，模型loss由 [0.6,0.4] 變成 [0.44,0.36]：L從0.50變0.40，G=0.10。A=[0.075,0.025,0]；E=[0.08,0.02]；ΣA=ΣE=0.10。P=V（本例q=1）。這是純計分合成fixture，不是實驗結果；整合測試另用可執行函數產生一致的實際查詢。

若錨點1與3同屬甲，錨點2屬乙，owner投影變成[2,1]，A_owner=[0.066666...,0.033333...]；不能先把負錨點裁零而得到甲75%。owner映射必須事前固定，不能事後選有利分組。

另一個模型端點fixture：加權改善[0.08,0.04,-0.02]合計0.10，負的第三項照存，不轉成0，不稱第三個模型有害。

## 14. 測試與技術驗收

### A. 純計分必測

1. 上述手算矩陣、份額及E逐項吻合；K≠M及非均勻權重。
2. 無接受輪：A/E=0、share=null；缺trace不可偽裝成此狀態。
3. 正交與負投影：符合點積定義；正交不自動宣稱無價值。
4. 強抵銷：例如H=[100,-99]，份額有界，負項不獲分；保留rho警示測試及微擾敏感性。
5. 先owner聚合再取正部，與先裁負的錯誤實作可區分。
6. 模型E含負數／大於G仍守恆，固定模型端點恆等式成立。
7. 0權重、錨點重合、dynamic探測被box截短或跳過。
8. 非有限／負權重／錯ID／缺回覆／多個accepted／不支援objective：明確失敗。
9. 改變展示分類閾值不改原始A/E；不偷偷把小G設0。
10. 調換ID與同步陣列排列只改標籤，分數於容差內一致；浮點和順序不同可有微差。

### B. 求解無干擾必測

同一確定性案例以未觀察原solver與觀察版執行。逐項比較：API模型順序、輸入bytes、回覆、呼叫與樣本數、accepted endpoint、final solution/loss、history/states、模式與停止流程。確定性同環境下要求原輸出與查詢序列完全相同；新增離線歸因的重建式才使用數值容差。

至少覆蓋：首試成功、多次縮步成功、secant全失敗切dynamic、dynamic失敗停止、零總方向早停、邊界限制、無退火、不同history_metric、原預算用盡。同一位置被不同用途重複詢問時，eval引用須正確且不能省略呼叫。

### C. 機制性驗證

- 固定已保存trace，重複讀取查詢或記帳事件不能增加A/E（重複event ID應拒絕，而不是重算）。
- 相同owner的一個錨點拆成兩個相同錨點，將原a等分、總權重不變，固定起點與其他條件：聚合訊號及owner份額於容差內應不變。搜尋若受近閾值浮點誤差影響須報告，不能把差異藏起來。
- 改用兩個新owner或增加總權重不在上述不變性保證內；必須明列漏洞，不宣稱抗Sybil。
- 在同一固定trace上所有loss乘正c：方向投影同比縮放，份額不變，A/E/G乘c。實際重新搜尋可能因絕對停止閾值而改變，不要求端到端尺度不變。
- 在已接受步上微擾模型loss，報最大份額變化及rho關係；這是敏感性診斷，不事後挑擾動或改公式。

以上只證明規則特性與忠實實作。缺少反事實或獨立價值證據時，不宣稱因果公平、模型準確性或經濟最適。

## 15. 最小執行順序、資料與成本

階段1：合成fixtures與分支覆蓋。階段2：既有pilot案例（可沿用layer2的pilot，登錄明確case IDs與hash）；clean/poison各選事前固定首案，兩案各使用原固定起點、custom30及原模型。先以1次control/record核對正確，再3組交替順序計時；不新增訓練、LOO或刪人分支。不得把同案例重複當獨立樣本。

開發資料只用上述pilot及合成fixtures。獨立驗證器可使用相同原始事件但不同程式路徑，這是程式獨立核對，不是獨立資料泛化。若使用過去已探索的20隊測覆蓋，明列舊資料探索。未來要作有效性主張，需另行登錄未看過的團隊／任務及明確效用定義；本版不預先把測試集當開發集。

成本量級：原本每次評分的模型回覆僅保存一次事件；新增模型推論必須為0。離線矩陣計算每輪約O(KMd)，彙總約O(KM+K+M)。若保存原始查詢輸入，記錄空間也受原查詢數及d影響；不能只報矩陣空間。避免常駐保存所有V；串流計分的工作記憶體可控制在單輪大小。

建議工程目標（待實作驗收，非已達成）：兩pilot各自觀察版搜尋時間中位數<=control的1.15倍，計分另報；若原測時太短，先報絕對ms與不確定性，不武斷判定比例。輸出<=100MB/案、單案<=60秒或20,000模型樣本，超限即停。這些是有限pilot資源限制，不能用來改solver參數掩飾超限。

研究控制執行與時間重複會實際花費額外推論，另列research total；「部署新增推論0」只指同一次原搜尋加觀察器，不包含為驗證而重跑的control。報總搜尋時間、模型耗時、離線計分及落盤成本，不能把紀錄成本排除後宣稱零成本。

## 16. 停止條件、報告與完成定義

任一查詢／求解流程改變、方向／改善不能重建、資料遺失、隨機回覆、資源超限：停止擴大案例，保留失敗資料，修正後另開run。unsupported 不用替代算法補值。

強抵銷只觸發敏感標記，數學重建通過仍可交付描述性分數，但不得宣稱穩健。複製測試或排序測試失敗先定位浮點／映射／公式問題；不能自行修改正部規則追求好看結果。

報告必分：

- 已完成：事件完整性、控制一致性、各項守恆、成本實測、例子。
- 未建立有效性：外部品質、模型評分品質、公平至與激勵、制衡價值、跨任務泛化。
- 下一步提案：根據實際失敗模式決定需補哪個測試，不自動引入LOO、付款或新模型。

完成本版實作的條件：A/B/C測試有逐項結果、兩pilot技術驗收通過或明確停止、獨立驗證器通過、成本完整、可由原事件重算、所有輸出及來源封存。技術完成不等於經濟有效性建立。

## 17. 實作者工作清單

1. 凍結本設計及來源hash，建立新模組與schema；先完成手算fixtures。
2. 實作帶eval/query ID的唯讀observer；先驗證不增加任何模型呼叫。
3. 實作純離線A/E/P計分與狀態；加入owner先聚合規則。
4. 實作不依賴計分器的獨立重建驗證。
5. 完成求解分支、矩陣／端點守恆、缺失及數值測試。
6. 執行有限pilot與成本量測；失敗保留、不擴跑。
7. 輸出繁體中文報告、receipt及可重現指令；請使用者檢視後再決定下一輪研究。

目前本清單全部屬下一步工作；本次沒有建立新程式、執行新實驗或修改Notion。
