import json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from four_way_bayes import ROOT,read
import oob_shrinkage_research as shared

a=json.loads((ROOT/'analysis.json').read_text());s=a['methods'];lock=a['lock'];rows=read(ROOT/'results.jsonl');obs=read(ROOT/'observations.jsonl')
fig,axes=plt.subplots(1,2,figsize=(12,4))
axes[0].plot(range(16),[o['mean_delta'] for o in obs],'o-',label='Mean paired delta');axes[0].plot(range(16),[o['J'] for o in obs],'.-',label='Mean + SE');axes[0].axhline(0,color='gray',ls='--');axes[0].set_xlabel('Sequential evaluation');axes[0].set_ylabel('Difference versus EU');axes[0].legend()
names=['train_equal','train_selected','audit_equal','audit_candidate']
for i,m in enumerate(names):
 q=s[m];lo,hi=q['ci95'];axes[1].errorbar(q['delta'],i,xerr=[[q['delta']-lo],[hi-q['delta']]],fmt='o',capsize=3)
axes[1].set_yticks(range(4),names);axes[1].axvline(0,color='gray',ls='--');axes[1].set_xlabel('Paired difference versus EU');fig.suptitle('Offline development: exploratory intervals, not new confirmation');fig.tight_layout();fig.savefig(ROOT/'optimization_and_transfer.png',dpi=160);plt.close(fig)
lines=['# 四方法比例：有依據的搜尋與結果（2026-11-16）','',
'本輪已實作有限貝葉斯最佳化，確實尋找四種方法同時使用的比例。結果是固定比例的離線開發，部署時僅算一次initial α及一次Stage3。',
'',f"鎖定w（順序U,Q,E,S）＝{lock['w']}。精確機器值保存在locked_candidate.json，顯示百分比僅供閱讀，不把四捨五入後比例冒充已驗證比例。選擇依據J＝30案例對EU平均差＋完整seed標準誤，鎖定於{lock['method']}。",'',
'## 為何這不是任意湊比例','',
'1. 問題只有三個自由度：四個非負方法權重總和1。預定每項≥5%，確保研究四法；不是宣稱5%有理論最佳性。',
'2. 重用27個舊測點，先以四個中心周圍的simplex設計點辨識往哪個方法加權的方向。',
'3. 用Gaussian process擬合觀測到的品質曲面；每輪連續最大化Expected Improvement，從可能較佳或尚不確定的區域選一個實際測點。共12輪，總新增16比例，不依結果增加輪數。大量便宜的代理運算不是大量實際法五求解。',
'4. 目標加入案例間標準誤，避免只看最小平均；此懲罰不保證新任務穩定。每一步預測/EI/實測J均封存，驗收重新產生12個建議比例，最大差0。',
'5. 找到比例後鎖定，六個可行鄰域方向只診斷；其餘60舊案例一次性轉移檢查，沒有用它們再修改比例。',
'', '方法依據：[BoTorch acquisition](https://botorch.org/docs/acquisition)、[GaussianProcessRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html)、[Matérn kernel](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html)。本輪是自訂有限研究程序，不聲稱EI或代理假設在本任務有最佳化／泛化保證。',
'', '## 配對品質與轉移檢查','',
'30搜尋案例與60歷史檢查案例不得合併成獨立確認。後者雖未用於此次比例調整，其舊方法與整體報告已被看過。下表CI為20,000次完整seed配對bootstrap，非選擇校正後的正式推論。負差有利候選。',
'', '|資料／方法|pooled均值|對EU差|探索95%區間|勝/平/負|三批平均差|', '|---|---:|---:|---|---|---|']
for m in ['train_equal','train_selected','audit_EU_canonical','audit_equal','audit_candidate']:
 q=s[m];lines.append(f"|{m} (n={q['n']})|{q['mean']:.8f}|{q['delta']:+.8f}|[{q['ci95'][0]:+.6f},{q['ci95'][1]:+.6f}]|{'/'.join(map(str,q['win_tie_loss']))}|"+'/'.join(f'{x:+.6f}' for x in q['batch_delta'])+'|')
lines+=['',f"60案例鎖定候選−等權的平均差＝{a['audit_candidate_minus_equal']:+.8f}，用於判斷搜尋是否帶來超過等權的收益。",'',
'|方法|乾淨對EU差|低品質差|不平衡差|配對差中位|p10/p90|', '|---|---:|---:|---:|---:|---|']
for m in ['train_selected','audit_equal','audit_candidate']:
 q=s[m];lines.append('|'+m+'|'+'|'.join(f'{x:+.6f}' for x in q['scenario_delta'])+f"|{q['median']:+.6f}|{q['p10_p90'][0]:+.6f}/{q['p10_p90'][1]:+.6f}|")
lines+=['','## 鄰域與數值敏感性','',
f"六個不同可行鄰點中{a['neighbors_improving']}/6平均優於EU；預定鄰域門檻通過={a['neighbor_gate']}。U位在5%域下界，向外不能走；原三個零半徑probe無法當穩定證據，保留為退化診斷，補三個固定的Q/E/S切線方向。這只檢查有限方向，無法聲稱整片穩定區域或域外更差。",'',
'原版EU與數學等價的通用公式在一個舊案例出現約1e−16權重差，導致pooled差0.004257；可見既有求解器路徑對數值敏感。主分析改用原版EU精確初始化，對180個舊結果全部吻合；原通用EU結果保留。修正未改候選。見verification_amendment.md及verification_initial.json，沒有隱藏首次驗收失敗。',
'', '|鄰點|w(U,Q,E,S)|對EU差|', '|---|---|---:|']
for m,q in s.items():
 if (m.startswith('neighbor') or m.startswith('tangent')) and np.linalg.norm(np.array(q['w'])-np.array(lock['w']))>1e-6:lines.append(f"|{m}|{np.round(q['w'],5).tolist()}|{q['delta']:+.6f}|")
lines+=['','## 成本、集中與尾端','',
'|方法|初始化含支持ms|Stage3 ms|方法requests|requests/EU|samples/EU|max α均值|有效模型數|truth診斷|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for m in ['audit_EU_canonical','audit_equal','audit_candidate']:
 q=s[m];lines.append(f"|{m}|{q['init_ms']:.3f}|{q['search_ms']:.3f}|{q['method_requests']:.2f}|{q['requests_ratio']:.3f}|{q['samples_ratio']:.3f}|{q['max_alpha_mean']:.3f}|{q['effective_models_mean']:.3f}|{q['truth_mean']:.6f}|")
lines+=['',f"實際2970次新求解合計{a['research_requests']:,}模型請求，搜尋{a['research_search_seconds']:.2f}s；另3次完整重播。新增450次是邊界鄰域與EU精確對照修正，不是用結果續搜比例。初始化包含一次E擬合、Q/U與S六次非自身預測；qualification共同盲測3 requests/60 samples可重用，不新增標籤。舊模型、anchor與共享準備沿用來源artifacts，不重訓。研究成本與部署單次成本分開。",'',
'同機計時對照仍是各方法分批順序執行，非逐任務交錯的正式時間測試，只作描述；共享訓練／資格／提案成本見上一輪shared.jsonl（約4.94秒/任務），不是本輪新測時間，也不以共享大成本掩蓋方法增量。所有成分均為正，所以四法的成分開銷結構相同，比例影響主要在後續搜尋路徑。',
'', '集中度與truth僅診斷，不能推出集中必定好／壞，也不使用truth調比例。最差三個轉移任務：']
for r in s['audit_candidate']['worst_tasks']:lines.append(f"- seed{r['seed']}／{r['scenario']}：pooled={r['pooled']:.6f}，對EU差{r['delta']:+.6f}，α={np.round(r['alpha'],4).tolist()}。")
lines+=['','## 結論與下一步','',f"通過事前轉移與鄰域推薦門檻：{a['recommend_new_confirmation']}。"+('可帶入全新確認，但未足以部署替換EU。' if a['recommend_new_confirmation'] else '維持EU；交付此比例作探索所得研究假說，不宣稱穩定改善，也不再擴張搜到顯著。'),
'', '已測：舊27點、4個中心設計點、12個EI建議點、6個非退化鄰點；另保留3個退化probe。未測：整個連續simplex、5%以下、其他目標／維度／Agent數及新訓練cohort。不證明任何比例全域最佳。',
'', '下一步獨立確認可鎖定本次精確w、等權與原EU三臂。兩個主要比較為鎖定w−EU及鎖定w−等權，事前封存Bonferroni家族與效果門檻、功效與樣本量；全新seeds/data/models/anchors/blind/evaluator，三批全跑完一次分析，不改比例、不依效果加樣。先做流程pilot，不能用pilot品質調規則。若本次門檻未過，這只能是弱證據假說的後續研究，非已取得的成功候選。完整獨立確認流程延續docs/alpha_mixture_independent_confirmation_plan_20261116.md，候選與比較家族以本輪更新為準。',
'', '## 驗收與重現','',
'2970條件完整、缺漏0、重複0、失敗0、fallback0。全量α／pooled重載差0，12個EI提案重建差0，鎖定候選重建一致；原EU180任務與鎖定候選3任務完整重播皆一致。4項測試通過（含資料驅動提案、simplex/新點、seed分組與成分回退）。',
'', '來源快照：source_snapshot；分析與修正快照：analysis_source_snapshot；事前設計：registered_design.md；主結果：results.jsonl；決策：decisions.jsonl；建議理由：proposals.jsonl；實測目標：observations.jsonl。重跑須新ROOT，禁止覆寫。',
'', '![最佳化過程與歷史轉移](optimization_and_transfer.png)']
(ROOT/'report.md').write_text('\n'.join(lines)+'\n')
snap=ROOT/'analysis_source_snapshot';snap.mkdir(exist_ok=True)
for n in ['analyze_four_way_bayes.py','supplement_four_way_bayes.py','report_four_way_bayes.py']:(snap/n).write_bytes(Path(n).read_bytes())
shared.dump(ROOT/'analysis_source_hashes.json',{p.name:shared.digest(p) for p in snap.iterdir()})
print(json.dumps({'w':lock['w'],'audit_delta':s['audit_candidate']['delta'],'audit_equal_delta':a['audit_candidate_minus_equal'],'gate':a['recommend_new_confirmation']}))
