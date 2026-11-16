import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1]/'low_cost_loo_exploration_20261116'
def read(n):return json.loads((P/n).read_text())
s=read('summary.json');people=read('participant_metrics.json');teams=read('team_metrics.json');records=read('scored.json');timing=read('timing_summary.json');paths=read('path_diagnostics.json')
names={'LAST_ANCHOR_REPLAY':'最後接受輪重做（主方法）','CUM':'累積錨點位移（主方法）','LAST_JOINT_REPLAY':'最後接受輪共同移除','LAST_TRY_REPLAY':'最後嘗試輪重做','LAST':'最後步直接扣分量','LAST_RENORM':'最後步扣除並重正規化','LAST_TRY':'最後嘗試步直接扣分量','CUM_MODEL':'累積模型位移'}
def get(name,sc='all',ev='pooled'):return next(x for x in s if x['method']==name and x['scenario']==sc and x['evaluator']==ev)
def pct(x):return f'{x*100:.2f}%'
def f(x):return '未定義' if x is None else f'{x:.6f}'
# Scientific static figure with exported PNG, no browser required.
fig,ax=plt.subplots(1,3,figsize=(14,4),layout='constrained')
for name,label,color in [('LAST_ANCHOR_REPLAY','Last-round replay','#0072B2'),('CUM','Cumulative vector','#D55E00')]:
 v=get(name);ts=[x['tv'] for x in teams if x['method']==name and x['evaluator']=='pooled' and x['tv'] is not None]
 ax[0].plot(np.sort(ts),np.arange(1,len(ts)+1)/len(ts),label=f'{label} (n={len(ts)})',color=color)
 ps=[x for x in people if x['method']==name];ax[1].scatter([x['loo_pooled_score'] for x in ps],[x['pooled_score'] for x in ps],label=label,alpha=.65,s=18,color=color)
 ax[2].scatter(v['samples']/20,100*v['tv']['mean'],s=65,label=label,color=color)
ax[0].set(xlabel='Bonus redistribution fraction (TV)',ylabel='Fraction of defined teams',xlim=(0,1));ax[0].legend(fontsize=8)
ax[1].plot([-.06,.27],[-.06,.27],'k--',lw=.8);ax[1].set(xlabel='Full LOO signed contribution',ylabel='Approximate signed contribution')
ax[2].scatter(28537/20,0,color='black',label='Full LOO reference');ax[2].set(xlabel='Local model evaluations / team',ylabel='Mean TV (%)');ax[2].legend(fontsize=8)
fig.suptitle('Exploration on 20 reused teams; fixed offline pooled evaluator')
fig.savefig(P/'comparison.png',dpi=180);plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
for ax,d in zip(axs,paths):
 a=np.array(d['base_path']);b=np.array(d['loo_path']);ax.plot(a[:,0],a[:,1],'-o',ms=2,label='Full path');ax.plot(b[:,0],b[:,1],'-o',ms=2,label='LOO path')
 for key,marker in [('cum','x'),('last','+'),('full','s'),('loo','D')]:ax.scatter(*d[key],marker=marker,s=70,label=key)
 ax.set_title(d['key'].replace('budget_','')+' / '+str(d['agent_id']),fontsize=9);ax.set_aspect('equal',adjustable='datalim');ax.set(xlabel='x1',ylabel='x2')
axs[0].legend(fontsize=7);fig.savefig(P/'path_cases.png',dpi=180);plt.close(fig)
text=['''# 兩種低成本近似與完整 LOO：小規模探索報告

2026-11-16。**兩種主方法都節省計算，但本批沒有建立能替代完整 LOO 的 bonus 分配近似。**最後一輪重做平均省65.06%模型評分；累積向量省69.20%。共同 pooled 評分下，平均分配TV約69.10%與71.07%，且存在未定義或接近零分配。不能因成本低就推薦付款或刪人使用。

本次使用20個舊開發團隊、80人：seed2026101610–1619 × clean（3人）/controlled_poison（5人）。沒有訓練、沒有新增確認案例。模型和搜尋紀錄來自既有研究；本次補算近似點及技術重播。所有歷史檔案保留。

## 1. 原始資料與目前研究核對

- 已讀 `docs/contribution_analysis_handoff_20261016.md`、`docs/contribution_analysis_start_prompt_20261016.md`、`docs/contribution_four_way_comparison_plan_20261022.md`、`docs/contribution_four_way_comparison_results_20261022.md` 與 v2 原始結果、凍結 solver、observer、engine、成本程式。交接文件當時只授權計畫；本次使用者已明確授權小規模實作與探索。
- 10/22 是20個神經網路團隊；S1是原A+M，S2是可重縮步A2+M，S3是J，S4是完整LOO。其分數共同用完整 F；不存在把舊S1直接改稱 Cᴬ+V 的依據。原完整LOO總樣本28,537，本次同快取口徑重現。
- 11/12 的24個解析合成團隊是另一批，`peer_exclusion_cost_20261112_attempt_02/report.md` 最終評分改為F_-i；表格1,581.0 / 2,211.5 / 1,187.6 / 1,888.8不可混作本輪分母。簡報2,211.4是四捨五入欄位相加，正確總數為2,211.5。
- 閱讀會議逐字稿PDF頁19–23（00:49–01:22，略過暫離）。老師確實要求與完整LOO比較比例、累積向量與實際LOO解差；這些是建議及待驗證近似。自動逐字稿未聽校，含辨識不清，不據此推定唯一數學實作。
- 視覺核對這次提供的26頁 `1112 法五研究報告.pdf`：第20頁Cᴹ、第22頁V文字寫完整F，公式卻是F_-i。第21頁例子正確為8−6=2；第23頁這一版本已是9−6=3，不能套用較早11/11檔案的例子錯誤。本報告以實作、明示評估目的與固定尺區分版本。
- 可直接沿用：全20隊模型、原搜尋方向/實際接受點、80個完整LOO端點、A2/M/J逐輪反事實端點與固定pooled。174個與歷史四方法輸入共同的來源hash全部吻合。

## 2. 精確定義

依使用者本輪回覆，主「最後一輪」採**最後接受步驟的輪前狀態**，移除參與者i的錨點，剩餘方向權重重新正規化；保留全部模型及原完整F。從該狀態重新估方向、採原eta、原secant/dynamic模式，最多10次二分縮步、box限步與接受／拒絕，只做一輪，不繼續後續搜尋。重算分支的anchor losses與當前loss；無接受則留輪前點。這是原A2的一輪版本，不是從頭移除。原狀態保留前面所有人的作用。

另列最後嘗試輪（含拒絕/停止）重做及最後接受輪共同移除錨點+模型的對照。20隊中18隊最後嘗試輪未接受，因此不能把這兩個「最後」視為同義。

累積法主定義：H_it=alpha_i a_it u_it、g_t=sum H_it，d_it=−ell_t H_it/||g_t||，D_i=sum_t d_it，x_tilde_-i=x_final−D_i。其中a_it是原全員模型評分產生的方向導數，ell_t是經縮步與邊界限步後的**實際接受距離**。拒絕／零方向輪d=0。不重正規化、不重算路徑、不移除已含在方向導數中的模型影響。固定起點不歸入D_i。記錄向量不是分配當輪scalar改善量，過去投影分數的負面結果不能直接用來否定它。

本 solver 沿搜尋方向縮短距離，沒有對已接受點逐座標截斷，所以此分解可重建原位移。原H分解與歷史模型M分解都能重建g；**守恆不代表歸屬唯一或因果可加**。主CUM使用H，CUM_MODEL僅敏感度，不能相加。所有逐輪重建最大誤差4.32e−14、全程2.83e−16；另用內積解倍率的獨立重算誤差5.72e−17。

構造點先保存raw，再對box[-1,1]做clip，修復不查loss、不繼續最佳化。本批兩主方法0/80需修復；CUM_MODEL有2/80需修復，相關結果屬已修復版本，raw座標、距離誤差及修復距離皆保留。原始可行性未以缺值補零。

原求解器沒有保存最後beta凸組合：它直接按方向與步長更新x。因此本次未測「刪除最終beta加權組合」版本，不能把本報告說成涵蓋老師可能指涉的所有最後組合做法。

## 3. 完整LOO與共同評估器

完整LOO沿用固定local models、原指定X-space起點、target=0、2D box、custom最多30輪。初始eta=.1，方向norm<=1e−8停止；secant縮步失敗切dynamic，dynamic再失敗停止；移除整位參與者的錨點、模型及alpha，再正規化剩餘alpha。不重新訓練、無新的搜尋隨機性。原始起點在這個custom直接使用，不是BFGS的子空間映射。

此20隊custom比較的內部objective是alpha加權絕對模型誤差，包括controlled_poison；不混用更早P3的trimmed objective。完整LOO端點、歷程及原D_L、D_E均核對通過。這是固定程序離線反事實，不是Shapley、全域最佳解或公平保證。

原生F分數=F(xminus)−F(xfinal)，可用部署時local models算；F_-i另存診斷，跨i其評分模型不同，不把它冒充固定跨人品質尺。主比較改用每個scenario原凍結pooled E(x)=|f_pooled(x)−Y*|，同一隊所有反事實使用同一個E。先保存端點鎖，再載入E評分；不參與搜尋與接受。

**pooled來源限制：**沿用的evaluator是原scenario全部20名來源者的clean pooled資料所訓練，並非僅本次入選3/5人的重新pooled模型。這是既有離線參考尺，包含合成實驗可取得的乾淨標籤，不是部署可得真值。本次不重訓以免換尺，也不載入truth函數評解。後續確認若要改成任務coalition pooled，必須另立版本，不能與本批直接合併。

C_i=E(xminus_i)−E(xfinal)。共同baseline下，近似與LOO的品質差恰等於貢獻差：C_hat−C_LOO=E(x_hat)−E(x_LOO)；兩欄是同一誤差，不是兩個獨立成功證據。

## 4. 分配規則與探索門檻

保留有號C；p_i=max(C_i,0)、P=sum p_i，P>1e−8才令q_i=p_i/P。P<=1e−8不分配、q缺失，不補均分；另標示P<=1e−4的比例脆弱情況。符號deadzone1e−4，參考非零才進主要符號分母，預測近零算未命中；另存1e−6/1e−3敏感度，不改原始分數。

TV=0.5 sum|q_hat−q_LOO|，可解讀為要重新挪動多少比例的bonus才能對齊。TV0表示一致、TV1表示完全錯置。任一q未定義不算TV，但覆蓋率記失敗。另列最大單人比例誤差。排名用含tie平均名次；常數不定義，近零下的非零微差排名也不能當穩健訊號。

結果計算前列TV<=5%/10%/20%探索門檻，**不是使用者接受標準**。嚴格覆蓋另要求最大單人差<=10百分點、符號一致>=80%、雙方非脆弱。兩種主方法在三個探索門檻都是0/20；即使只看TV、不加其他條件，最後輪仍0/20，CUM僅20%門檻有1/20。

## 5. 主結果：固定pooled評分

團隊統計先各隊計算再等權；TV只平均可定義隊，其餘必須與無效數一起讀。位置／逐人品質分位數是80人的描述統計，不把80人當獨立樣本。

| 指標 | 最後接受輪重做 | 累積錨點位移 |
|---|---:|---:|''']
a=get('LAST_ANCHOR_REPLAY');b=get('CUM')
for label,vals in [('平均TV（有效隊數）',[f"{pct(v['tv']['mean'])}（{v['tv']['n']}/20）" for v in [a,b]]),('TV中位數 / P90 / 最大值',[' / '.join(pct(v['tv'][k]) for k in ['p50','p90','max']) for v in [a,b]]),('q未定義團隊',[str(v['invalid_allocations']) for v in [a,b]]),('任一方總正分<=1e−4團隊',[str(v['fragile_teams']) for v in [a,b]]),('平均最大單人比例差',[pct(v['max_share_error']['mean']) for v in [a,b]]),('平均Spearman',[f"{v['rho']['mean']:.3f}" for v in [a,b]]),('平均符號一致率',[pct(v['sign_match']['mean']) for v in [a,b]]),('團隊等權品質／貢獻MAE',[f(v['quality_contribution_mae']['mean']) for v in [a,b]]),('逐人位置L2：平均 / P50 / P90 / max',[' / '.join(f(v['position'][k]) for k in ['mean','p50','p90','max']) for v in [a,b]]),('逐人品質差：平均 / P50 / P90 / max',[' / '.join(f(v['quality_error'][k]) for k in ['mean','p50','p90','max']) for v in [a,b]])]:text.append('| '+label+' | '+' | '.join(vals)+' |')
text.append('''
位置單位為原2D決策變數，box對角線為sqrt(8)；正規化位置差另存CSV。最後輪符號一致13%**不表示87%全反號**：70個參考非零者中，7同號、59近零、4異號；CUM則40同號、1近零、29異號。這是逐人計數，與團隊等權13%/55.33%分母不同。

| 情境 | 方法 | 平均TV / 有效n | 排名rho | 符號一致 | 總模型評分節省 |
|---|---|---:|---:|---:|---:|''')
for sc in ['clean','controlled_poison']:
 for name in ['LAST_ANCHOR_REPLAY','CUM']:
  v=get(name,sc);text.append(f"| {sc} | {names[name]} | {pct(v['tv']['mean'])} / {v['tv']['n']} | {v['rho']['mean']:.3f} | {pct(v['sign_match']['mean'])} | {pct(v['savings'])} |")
text.append('''
![效果與成本](comparison.png)

## 6. 原生分數與版本敏感度

以下全部使用相同近似端點；F、F_-i、pooled不是同一評分目的。F_-i跨人換尺，僅診斷，不作主品質結論。

| 方法 | 評估尺 | 平均TV / 有效n | 排名rho | 符號一致 |
|---|---|---:|---:|---:|''')
for name in ['LAST_ANCHOR_REPLAY','CUM']:
 for ev in ['full','peer','pooled']:
  v=get(name,ev=ev);text.append(f"| {names[name]} | {ev} | {pct(v['tv']['mean'])} / {v['tv']['n']} | {v['rho']['mean']:.3f} | {pct(v['sign_match']['mean'])} |")
text.append('''
| 其他預列版本（pooled） | 平均TV / 有效n | 排名rho | 符號一致 | 總評分節省 |
|---|---:|---:|---:|---:|''')
for name in names:
 if name in ['LAST_ANCHOR_REPLAY','CUM']:continue
 v=get(name);text.append(f"| {names[name]} | {pct(v['tv']['mean'])} / {v['tv']['n']} | {v['rho']['mean']:.3f} | {pct(v['sign_match']['mean'])} | {pct(v['savings'])} |")
text.append('''
同時移除模型讓最後輪對照平均rho升至約.346、TV降至約49.76%，但仍不可靠且沒有通過任何嚴格探索門檻。改為模型分量的CUM_MODEL平均TV48.67%，但逐人品質MAE升至.08155，最大1.28426，另有2點修復；不能只挑分配指標宣稱改善。最後步重正規化僅個別團隊碰巧達門檻，沒有建立可部署的辨識規則。

## 7. 成本：模型、向量、時間與快取

每方法獨立快取，僅重用該次baseline已查過的相同模型及float64輸入；不同方法不分享新增查詢。baseline依原solver每次實際呼叫全額計費。下表是產生解加原生完整F貢獻評分，pooled離線驗證另計。

| 方法 | 全20隊baseline | 新增評分 | 總評分 | 每隊平均總數 | 相對LOO節省 |
|---|---:|---:|---:|---:|---:|''')
for name in ['LAST_ANCHOR_REPLAY','CUM']:
 v=get(name);text.append(f"| {names[name]} | 8,448 | {v['extra']:,} | {v['samples']:,} | {v['samples']/20:.2f} | {pct(v['savings'])} |")
text.append('''| 完整LOO | 8,448 | 20,089 | 28,537 | 1,426.85 | 參考 |

逐隊節省率也不相同：最後輪的團隊平均62.65%，最小37.39%、中位61.32%、P90 73.84%、最大76.14%；CUM平均66.72%，最小42.86%、中位67.89%、P90 76.77%、最大83.40%。主表65.06%/69.20%是總量比，不能與逐隊平均混稱。最後輪cache命中3,540筆，CUM命中80筆。

若baseline已完成，最後輪只需新增1,523次、CUM340次，對LOO新增20,089次分別省92.42%與98.31%。**CUM的向量構造本身零新增模型查詢，但要把新點轉成貢獻分數仍須評分**，不能說整個方法零成本。本批340=10×3²+10×5²；全員F每個反事實點查n個模型，因此最終評分成本仍可達O(n²)，不是僅O(n)。

CUM新增向量工作O(Tnd)，可串流維持O(nd)累積量，不需保存全部歷程；最後輪重做需每人方向／縮步計算及額外模型評分。已另實作最小串流觀測器，凍結solver所有數值步驟不變，端點、分數、呼叫量與紀錄後處理版本一致。

每隊一暖機、三次交錯計時，以下為20隊三次中位數之和：

| 方法 | 核心秒數 | 相對LOO |
|---|---:|---:|''')
lt=next(x['total_team_median_seconds'] for x in timing if x['method']=='LOO')
for t in timing:text.append(f"| {names.get(t['method'],'完整LOO')} | {t['total_team_median_seconds']:.4f} | {t['total_team_median_seconds']/lt:.3f}倍 |")
stream=read('streaming_timing.json');vt=sum(np.median([r['stream_vector_seconds'] for r in stream if r['key']==x['case']['key'] and r['method']=='CUM']) for x in records)
text.append(f'''
CUM串流callback向量計算時間合計約{vt:.6f}秒（逐隊中位數加總，含計時器的小量擾動），另計O(nd)扣除及clip。記錄保留原始含大量驗證的後處理計時，但主表採最小串流CUM時間，避免把研究核對算成方法成本。

時間僅同機微型模型的描述量測；含推論、快取、分支與評分，不含模型載入、檔案I/O、離線pooled。CUM包含串流觀測；最後輪目前讀已保存輪前狀態，**未計新增線上狀態擷取成本**，不能稱三方法完全端到端或外推遠端服務延遲。兩組計時在同一工作階段分批執行，未作延遲效益顯著性主張。

pooled主評分可重用xfinal，獨立部署式離線評估上限每法每隊n+1個點；本次跨研究方法共享純離線cache後實際585個唯一評分。這種跨法共享只節省研究成本，沒有免費送進部署成本帳。

## 8. 失敗案例與假說檢查

1. **早期作用、最後一步小：**80/80人的最後分量norm占個人全程分量norm總和不到10%，因此本資料沒有「最後一步大」對照組，不能估計該條件的因果效果；但18/20隊最後輪分配總正分處於脆弱區，與末步分數衰減相符。污染seed2026101611、id3：完整pooled貢獻+0.050333，最後輪僅+0.00000255，位置誤差.370711。忽略前期影響是結構性限制，不是加小數精度就可解決。
2. **累積方向不等於移除位移：**污染seed2026101613、id15，D與實際完整LOO解差cosine=−.998735，幾乎反方向；向量長度比.733881。CUM評分+0.263661，完整LOO−0.003652，品質差.267313，位置差1.237213（box對角線的43.74%）。沒有越界修復，所以不能歸咎clip。沿全員原路徑的加總守恆，並不約束刪人後的路徑。
3. **分配完全錯置：**乾淨seed2026101616，最後輪q=(.15305,0,.84695)，完整LOO=(0,1,0)，TV=1；近似總正分僅7.78e−5，比例不穩。污染seed2026101615，CUM的TV也為1，但總正分.084109，並非近零造成，顯示光加near-zero防護不夠。
4. **方向抵銷：**事前norm(g)/sum norm(H)<.1只涵蓋2隊/8人；該組CUM平均位置差.02964，反而低於其他72人的.18192。沒有支持「本批強抵銷造成更大誤差」；樣本少且其他條件混雜，不推論抵銷有益。
5. **模型作用：**最後接受輪的模型移除端點距離大於錨點移除者62/80人。其最後輪近似品質MAE .01072，其餘18人.00478；但這是局部分支診斷，非全程角色因果分解。共同移除末輪雖有改善仍不足，不能把全部失敗只歸因漏掉模型。
6. **路徑改變：**72/80人的完整LOO歷程長度與全員不同；歷程長度只算線索，長度相同不代表路徑相同。另對上述固定規則選出的3例重播並保存完整輪前座標，與舊LOO端點精確一致；3例都從第1次更新後就已分歧。這支持存在實際路徑差異，但不證明所有誤差都由同一機制造成。

![代表案例路徑](path_cases.png)

目前沒有可用來事先挑出「可靠隊伍」的已驗證條件。CUM的一個TV<=20%成功案例只是事後個案，未通過符號／最大份額聯合要求，不能包裝成適用族群。

## 9. 驗證、限制與研究成本

- 核對20隊原解、80個完整LOO端點、歷程、原D_L/D_E；所有最後輪反事實匹配已保存A2/J端點。
- 50,675筆查詢事件獨立逐筆重算cache hit及新增計費，通過。分配用1−overlap獨立核對TV，排名用SciPy獨立核對。全負、總正分極小、常數排名、deadzone邊界測試通過。
- 主探索與四次技術計時195,760個local評分；串流成本核對75,036；3例路徑核對688次；合計271,484個local模型評分，低於300,000上限。主計算6.13秒、串流量測2.49秒；均不含整體人工閱讀、套件載入與報告時間。研究重播不當作部署一次的成本。0訓練、585個pooled唯一評分。
- 舊資料20隊不提供新資料效度確認；不對80人做獨立樣本推論。分位數與情境分開報告，不以平均遮蔽最大誤差。未跑新大型研究。
- 報告完整公開預列變體，不依結果調符號、權重或挑選較好evaluator。方法不產生有效分配時保留缺失，不補0或均分。

## 10. 現在能用在哪裡

**最後一輪重做：**若問題本來就是「這個末輪的建議有何局部影響」，可用作局部敏感度診斷，成本低且語義匹配；若要整個任務貢獻或bonus，尤其最後步已很小，這批不適用。沒有證據說加進模型移除就足夠。

**累積位移：**適合低成本記錄原搜尋過程的方向來源、守恆及路徑診斷。能節省更多查詢，但不是可信的完整刪人解；在本批甚至可與完整LOO向量反向。不能據其正負做付款或淘汰。

**要回答指定程序的完整參與者移除：**目前保留完整LOO。若只要求較鬆的探索分配誤差，仍沒有任何主方法在預列5%/10%/20%門檻建立可靠替代。完整LOO本身也不保證公平、Shapley或外部真實品質。

獨立確認方案另存 `independent_confirmation_plan.md`，本次沒有啟動。全部原始signed分數、q、端點、成本、分位數、各方法／各尺結果、來源hash及驗證檔在此目錄；程式位於 `../low_cost_loo_v1/`。
''')
(P/'report.md').write_text('\n'.join(text))
print('report',P/'report.md')
