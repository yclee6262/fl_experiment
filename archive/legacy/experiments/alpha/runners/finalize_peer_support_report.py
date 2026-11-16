"""Format audited results for reading; no experiment parameters are changed."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from peer_support_alpha import METHODS,LABELS,SCENARIOS
from explain_peer_support_results import table,num

root=Path('alpha_weight_outputs/peer_support_v2_20261013');out=root/'posthoc';analysis=json.loads((root/'formal_analysis.json').read_text());audit=json.loads((root/'completion_audit.json').read_text());assert audit['passed']
primary=[c for c in analysis['comparisons'] if c['endpoint']=='equal_scenario_mean']
codes=dict(zip(METHODS,['A','B','Q','S','QS','E']));fig,ax=plt.subplots(figsize=(8,4))
for j,c in enumerate(primary):
 color='#c44e52' if c['method']=='peer_support_shrink' else '#4c72b0';ax.plot(c['mean_delta'],j,'o',color=color);ci=c['CI_simultaneous']
 if ci is not None:ax.plot(ci,[j,j],color=color,linewidth=2)
ax.axvline(0,color='gray',linewidth=1);ax.set_yticks(range(len(primary)),[codes[c['method']]+' - '+codes[c['baseline']] for c in primary]);ax.set_xlabel('Pooled error difference: negative = improvement');ax.set_title('Seven registered comparisons (adjusted intervals)');fig.tight_layout();fig.savefig(out/'paired_pooled_differences.png',dpi=180);plt.close(fig)
# Export compact summaries rather than putting source paths alone in Notion.
fields=['method','method_label','endpoint','eligible_n','valid_n','method_failure_n','mean','median','Q1','Q3']
with (out/'quality_summary.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
 for g in analysis['groups']:writer.writerow({**g,'method_label':LABELS[g['method']]})
fields=['method','baseline','endpoint','eligible_n','n','mean_delta','CI95','CI_simultaneous','stable_advantage','method_failure_exclusions']
with (out/'paired_comparisons.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
 for c in analysis['comparisons']:writer.writerow({k:json.dumps(c[k],ensure_ascii=False) if isinstance(c[k],(list,dict)) else c[k] for k in fields})
report=Path('docs/peer_support_results_20261013_notion.md');body=report.read_text()
explanation='本輪 6,534 個條件全部有效可行，沒有資格或求解失敗，三批的重載分數、α 與矩陣差均為 0；独立稽核與 29 項測試通過。未達優勢門檻的原因是品質比較或跨批重現不足。\n'.replace('独立','獨立')
explanation+='**E 的整體 pooled 平均最低，這是本輪六方法的平均排名，不代表本輪已正式證明 E 穩定優於所有方法。**主要確認家族是 S／QS 的七組比較，沒有新增事後的 E 優勢檢定。\n'
explanation+='S 對 B、E 的校正區間全部大於零，支持 S 的平均誤差較高。QS 對 A 的合併校正區間全部小於零，但三批中只有一批的一般 95% 區間全負，未通過事前要求的至少兩批重現門檻。QS 對 Q 的平均差為正且三批皆為正，校正區間跨零；目前未建立加入支持的額外收益，也未證明兩者等效。\n'
body=body.replace('## 這輪回答了什麼\n','## 這輪回答了什麼\n'+explanation,1)
head=body.index('<table header-row="true" fit-page-width="true">');body=body[:head]+'整體平均為三情境誤差等權平均後，再跨獨立案例平均；越低越好。中位數表示一半案例不超過此誤差；Q1、Q3 分別為第 25、75 百分位，呈現中間一半案例的範圍。有效／合格數表示成功可評分的案例數與共同資格通過數。\n'+body[head:]
case_data=json.loads((out/'explanation_cases.json').read_text());diag=json.loads((out/'candidate_diagnostics.json').read_text());per_seed=[]
from scipy.stats import spearmanr
for scenario in SCENARIOS:
 grouped={}
 for r in diag:
  if r['scenario']==scenario:grouped.setdefault(r['seed'],[]).append(r)
 for seed,rows in grouped.items():
  s=[r['support'] for r in rows];e=[r['candidate_pooled_error'] for r in rows]
  rho=spearmanr(s,e).statistic if np.ptp(s)>0 and np.ptp(e)>0 else None
  per_seed.append({'scenario':scenario,'seed':seed,'rho':float(rho) if rho is not None and np.isfinite(rho) else None})
(out/'support_correlation_by_seed.json').write_text(json.dumps({'kind':'posthoc_descriptive_only','rows':per_seed,'no_candidate_independence_claim':True},ensure_ascii=False,indent=2)+'\n')
counts={}
for c in case_data['pooled_truth_conflicts']:counts[c['method']]=counts.get(c['method'],0)+1
text='## 結果如何解讀與下一步\n候選點支持度與候選點 pooled 誤差的描述性排序相關約為乾淨 −0.325、低品質 −0.381、不平衡 −0.283：較高支持有一定品質線索，但把這個訊號配給提出者的模型，沒有得到更好的整體倒推品質。候選點嵌套於同一案例，這些相關不作獨立樣本的顯著性或因果證明。\n'
text+='在 S−A 與 QS−Q 的合計 2,178 個配對診斷中，有 '+str(len(case_data['pooled_truth_conflicts']))+' 個 pooled／truth 方向相反案例；分別為 S−A '+str(counts.get('peer_support_shrink',0))+' 個、QS−Q '+str(counts.get('target_peer_shrink',0))+' 個。這表示兩種評估對該案例的改善方向不同，不是執行失敗。主要結論仍按事前固定的 pooled 判準，所有衝突案例保存於附件。\n'
text+='下一步建議保留 E 與 Q 作初始化候選，另行規劃 E 相對 A／B／Q 的獨立確認；不因本輪結果調參後回填。本輪 S／QS 不列為較佳主方案。若繼續研究支持訊號，可另設「用支持度選候選點／起點、α 固定」的實驗，因為它直接對應訊號的候選點品質語義；這只是後續建議，尚未執行。\n'
body=body.replace('<details>\n<summary>支持度實驗附錄',text+'\n<details>\n<summary>支持度實驗附錄',1)
cost=json.loads((out/'cost_report.json').read_text());costrows=[]
for m in cost['method_independent_execution']:
 n=m['planned_conditions'];total=m['independent_application_total_aggregate'];specific=m['method_specific_aggregate']
 mean=lambda scope,f:None if scope[f]['sum'] is None else scope[f]['sum']/n
 costrows.append([LABELS[m['method']],num(mean(specific,'requests')),num(mean(specific,'input_samples')),num(mean(specific,'seconds')),num(mean(total,'requests')),num(mean(total,'seconds'))])
costtable=table(['方法','初始化＋倒推查詢平均','輸入點數平均','初始化＋倒推秒數平均','含共用準備查詢平均','含共用準備秒數平均'],costrows)
addition='\t## 成本欄位與兩種計算口徑\n'+'\n'.join('\t'+line for line in costtable.splitlines())+'\n\t查詢為本地模型 API 介面呼叫，輸入點數另計；不是實際付費網路請求。方法獨立執行口徑加入一次共用的本地訓練、資格與 proposal 準備；整批共享口徑只計一次共用準備與一次支持矩陣。外部 evaluator 訓練、重載、對角線、source A 快取生成與外部評分另列研究成本。失敗不從成本彙總刪除，未知成本不填零。\n'
addition+='\t整批共享應用查詢總數：'+str(cost['batch_shared_execution']['requests']['application_total'])+'；方法各自獨立的總數不能直接相加作共享成本。耗時只描述本次本地執行，沒有宣稱實際部署金額。\n'
body=body.replace('\t## 成本與案例\n',addition+'\t## 成本與案例\n',1)
# Put image placeholders in an appendix; upload sources replace them before publication.
visuals='\t## 圖表與下載\n\t配對差圖：橫軸負值較好，橫線為七組比較校正區間；QS−A 的區間全負仍未達跨批門檻。\n\tPEER_PRIMARY_IMAGE\n\t支持與候選品質散點圖：每個點是候選，僅作事後描述，沒有把同一案例三個候選當三個獨立實驗。\n\tPEER_SUPPORT_IMAGE\n\tα 變化與交付誤差圖：負的縱軸代表改善，零線上下的案例都保留；不能從圖中挑交付輪次。\n\tPEER_ALPHA_IMAGE\n\tPEER_DOWNLOADS\n'
body=body.replace('\t## 正式範圍\n',visuals+'\t## 正式範圍\n',1);report.write_text(body)
print(json.dumps({'report':str(report),'quality_csv_rows':len(analysis['groups']),'comparison_csv_rows':len(analysis['comparisons']),'correlation_seed_rows':len(per_seed),'conflicts_by_comparison':counts},ensure_ascii=False))
