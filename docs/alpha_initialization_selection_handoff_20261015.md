# α 初始化與選擇策略：新對話交接文件

> 2026-10-16狀態更新：使用者確認此方向尚未取得老師同意或comment，暫不啟動新實作與實驗。現階段優先轉入貢獻度分析；見`docs/contribution_analysis_handoff_20261016.md`。以下保留為未執行的研究交接草案，不代表已批准啟動。

建立日期：2026-10-15。專案：`/Users/yclee/Documents/GitHub/fl_experiment`。
本文件是專案內的研究交接資料，不是Codex全域記憶；新對話須主動讀取。內容是截至本日期的快照，開始工作前核對實際程式、manifest與結果。尚未實作新的α選擇策略，也尚未執行下一輪實驗。

## 1. 本對話的研究問題與範圍

面對一個倒推任務，如果已有多組候選α，能否利用實際部署可取得的資訊選擇較有機會產生良好X的權重，並在獨立資料上優於固定使用某一初始化方法？

- 主要品質判準：`E_pooled(X) = abs(f_pooled(X) - target)`，越低越好。
- 部署沒有global model。pooled evaluator只在交付X固定後作離線評分，不能進入α初始化、候選α選擇、pruning或交付X選擇。
- 真實函數只作診斷，不能回饋演算法。
- 先研究選擇策略，不同時新增動態α更新、pruning、Shapley或付款公平性。
- 商模與低成本貢獻估計屬其他研究線；不要自行擴大本任務。
- 先產出研究設計；本輪啟動prompt沒有授權直接跑正式比較。

## 2. 目前固定的方法名稱與公式

符號：N為入選參與者數，U_i=1/N；z_k、t_k為共同有標籤盲測輸入與答案；f_i為local model。目前主要設定N=3、盲測n=20。

| 代號 | 名稱 | 初始化α | 保留的機器識別碼 |
|---|---|---|---|
| A | 均勻基準 | U | A_uniform |
| Q | 品質基準 | norm(1/(MSE_i+1e-5)) | B_stage0 |
| E | 純組合法 | w | stacked_full |
| EU | 組合均勻法 | 0.5w+0.5U | ensemble_shrink |
| EQ | 組合品質法 | 0.5w+0.5Q | B_stacked_half |
| S | 純支持法 | s | pure_support |
| SU | 支持均勻法 | 0.5s+0.5U | peer_support_shrink |
| SQ | 支持品質法 | 0.5s+0.5Q | support_quality_mix50 |

### E：模型組合

所有模型預測相同20筆盲測。令P[k,i]=f_i(z_k)，求解：

`min_w mean((P @ w - t)**2), subject to w_i>=0 and sum(w)=1`。

現行用SciPy SLSQP、uniform起點、各權重界限[0,1]、ftol=1e-12、maxiter=500；沒有提供解析jac，因此使用數值差分。固定模型預測，不重訓模型。純E讀取擬合所得raw_scores；不能因legacy helper名稱是ensemble_shrink，就錯把純E也混入uniform。純E擬合失敗目前回退品質基準Q，保存fallback原因。

EU／EQ的50/50是固定設計比例，未證明最優；此次不得自行調整。

### S：候選點支持

每位參與者i提出X_i；其他模型j預測p[j,i]=f_j(X_i)，排除j=i。

- d[j,i]=abs(p[j,i]-target)。
- e_i=median_j!=i(d[j,i])。
- tau=median(所有非自身d)+1e-5。
- s_i=exp(-e_i/tau)/sum_l(exp(-e_l/tau))。

實作使用減去最大logit的穩定計算，與上述正規化公式等價。N=3時各候選點收到兩筆評分，e_i中位數等於兩數平均；tau由全部六筆計算。
純S不混均勻或品質。SQ是支持與品質各半，不是相乘。支持度不是正確機率、完整參與者貢獻或付款比例。

### 命名注意

- 使用`docs/alpha_method_display_names_current.json`作目前名稱依據。
- 舊B已改稱Q，舊EB改EQ、SB改SQ；不要改寫封存結果ID與數值來源。
- 歷史Q可能指quality_shrink（均勻與舊B各半），不是現在的品質基準Q。
- 歷史E／S可能指EU／SU；本輪E／S明確指純方法。
- 舊QS「品質乘支持後再收縮」不在本輪八方法內。

## 3. 最新可確認的研究結果

最新同資料八方法比較：90個資料案例，三批各30，每案例三情境。先對三情境等權平均，再跨90案例平均；不是270個獨立樣本。

| 方法 | 整體平均pooled誤差 |
|---|---:|
| EU | 0.016653 |
| E | 0.017227 |
| SQ | 0.017276 |
| EQ | 0.017668 |
| Q | 0.017815 |
| A | 0.017967 |
| S | 0.018559 |
| SU | 0.019075 |

全部2160條件有效且可行。EU在乾淨／低品質平均最低，樣本量不平衡純E略低。平均排名不是穩定優勢。

- 原六方法1620條件是事前登錄的三批獨立確認，未建立混合候選穩定優於A/Q。
- 純E/S追加540條件，沿用同一批模型／資料；屬同批追加對照，不是新的獨立確認。
- EU−E、EQ−E、SU−S、SQ−S四個收縮對照，其一般與四項校正區間均跨零；未建立收縮穩定收益，不代表無效或等效。
- 不可說「所有歷史比較CI都跨零」：原六方法SU−EU校正區間為正，但未達原登錄的所有穩定判定條件。
- 不混合歷史不同cohort的均值，也不把任何目前方法稱為通用最佳。
- 原確認採200,000次seed配對bootstrap與多重比較校正，另要求跨批一致等門檻；新策略應重新事前登錄比較規則，不能事後挑漂亮結果。

本輪控制：IDs1/2/20、三人、2D、target0、box[-1,1]、tol1e-8、best_of/custom30，不更新α、不pruning、不結算、不新增標籤。三情境為reference_clean、low_quality_25pct、sample_imbalance_4x。低品質只污染local observed labels，pooled維持乾淨設定。

## 4. 必讀證據與程式入口

以下路徑相對於專案根目錄；新對話先讀摘要與驗收，不必掃描所有歷史輸出。

- `docs/alpha_method_display_names_current.json`：目前命名。
- `docs/eight_alpha_teacher_conclusions_next_20261014_notion.md`：教授版方法與最新結論。
- `docs/four_alpha_results_20261014.md`：六方法原確認，內文仍有歷史B/EB/SB名稱，讀時套用alias。
- `alpha_weight_outputs/four_alpha_v1_20261014/formal/formal_analysis.json`：原正式分析。
- `alpha_weight_outputs/four_alpha_v1_20261014/final_acceptance.json`：原驗收。
- `alpha_weight_outputs/pure_es_extension_v1_20261014/analysis.json`：追加純方法分析。
- `alpha_weight_outputs/pure_es_extension_v1_20261014/acceptance.json`：2160條件完整性與重載核對。
- `docs/pure_es_extension_registration_20261014.json`：追加比較與來源索引。
- `initial_alpha_methods.py`、`unified_alpha_methods.py`：品質／組合初始化。
- `peer_support_alpha.py`、`pure_es_initialization.py`：支持訊號與純方法。
- `four_alpha_initialization.py`、`run_four_alpha_batch.py`、`analyze_four_alpha.py`：六方法批次與分析。
- `run_pure_es_batch.py`、`analyze_pure_es_extension.py`：純方法追加。
- `host_server.py`：Stage3、候選空間與限制。
- `docs/professor_audio_progress_audit_20261015.md`：老師原意、已完成與缺漏。

Notion目前教授版：https://app.notion.com/p/3f9bf435d474819cbdf6ffd8b651d7a3 。本任務先不修改；總控對話統一整合，避免多任務同時編輯。

## 5. 下一階段要設計的選擇實驗

尚無定案選擇公式。先提出少量可解釋的候選，明確區分兩種成本不同的流程：

1. 求解前：只讀盲測、模型預測、候選點資訊，先選一組α，再倒推一次。
2. 求解後：每組候選α都倒推，以部署可得的local評分選交付X；這是多次求解策略，必須計入完整額外成本，不能冒稱單次初始化。

必要設計要求：

- 各策略同案例共用模型、盲測資料、候選點與evaluator。
- 原八方法保留作固定策略對照，優先解釋A/Q/EU/E/SQ；不能只與較弱方法比。
- 任何經驗規則／學習式selector只能在開發資料上設定，確認資料不回饋調參。若使用歷史pooled品質訓練selector，須明示離線監督來源並嚴格按完整案例分開開發／確認，與完全無pooled訓練版本區分；不預設此次採用。
- pooled逐案挑最佳α只能作離線oracle參考，不可作實際selector；若保存此參考須單獨標示。
- 說清評分者與α是否解耦、有無用自身模型自評、代理指標為何可能對應X品質。
- 保存選中方法、α、交付X、選擇訊號、理由、fallback、失敗與所有成本。
- 保留資格失敗，不默默強制入選或刪除失敗案例。
- 先設計、再實作與smoke、再pilot、再封存正式設定及全新未用seeds。

## 6. 工作與驗收約定

- 回覆使用繁體中文，先說研究問題、做法與解讀，再提供必要細節。方法講清楚公式與例子。
- 正式研究先固定規則；pilot只查流程。所有已登錄候選保留跨批比較，弱證據標示「證據不足」，不直接判定無效。
- 成功交付X必須box可行；重載容差1e-7，可行容差1e-8。
- 按manifest核對所有預期條件、重複鍵、缺漏、失敗與artifact；品質與失敗率分開，bootstrap以完整案例／seed為單位。
- 每輪獨立新輸出目錄；保留舊manifest、數值程式快照與結果。
- 不同對話同目錄不隔離程式；如需平行修改，先由使用者選擇worktree並核對來源版本、未提交檔案與模型artifacts可用性。
- 同一台機器的大型訓練與計時實驗依序執行；檢查既有工作，避免重複啟動。
- 更新本研究線固定設計／進度文件，結果由總控整合；不要每個小實驗建立新Notion頁面。

## 7. 新對話第一次應交付

先核對本文件與當前來源，交付：2–3種可行選擇策略、具體訊號／計算規則／小例子、可部署性、成本與偏差風險、比較基準、開發／pilot／確認資料分離、驗收與停止條件，並推薦最小可行pilot。

此時不要直接修改既有演算法、啟動長時間實驗或發布Notion。設計確認後再進入實作。
