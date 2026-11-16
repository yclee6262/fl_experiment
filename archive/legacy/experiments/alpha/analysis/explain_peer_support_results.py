"""Post-delivery diagnostics and readable report; never changes alpha or delivery."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import oob_shrinkage_research as shared
from peer_support_alpha import METHODS,SCENARIOS,LABELS
from peer_support_costs import summarize_costs
from research_evaluation import load_evaluator_artifact,score_solution,synthetic_truth

def table(heads,rows):
 return '<table header-row="true" fit-page-width="true">\n'+'\n'.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in [heads]+rows)+'\n</table>'
def num(x):return '不適用' if x is None else f'{x:.6f}'
def interval(x):return '不可計算' if x is None or x[0] is None else '['+', '.join(num(v) for v in x)+']'

def explain(registration):
 reg=json.loads(Path(registration).read_text());root=Path(reg['output']);analysis=json.loads((root/'formal_analysis.json').read_text());diagnostics=[];allrows=[];costs=[];allpeer={}
 audit=json.loads((root/'completion_audit.json').read_text())
 if not audit['passed'] or audit['all_conditions']!=reg['expected_conditions'] or analysis['status']!='formal':raise RuntimeError('Complete independent audit required before final report')
 out=root/'posthoc';out.mkdir(exist_ok=False)
 for batch in reg['batches']:
  directory=Path(batch['results']);rows=list(map(json.loads,(directory/'results.jsonl').read_text().splitlines()));allrows.extend(rows)
  peer={(r['scenario'],r['seed']):r for r in map(json.loads,(directory/'peer_evidence.jsonl').read_text().splitlines())}
  allpeer.update(peer)
  for r in map(json.loads,(directory/'shared.jsonl').read_text().splitlines()):
   costs.append(r)
   if r['eligibility']!='eligible':continue
   server,item,artifact=shared.load_base(batch['source'],r['scenario'],r['seed']);evaluator,_=load_evaluator_artifact(artifact);evidence=peer[r['scenario'],r['seed']]
   for i,x in enumerate(server.I_list):
    pred,error=score_solution(evaluator,x,0);diagnostics.append({'scenario':r['scenario'],'seed':r['seed'],'agent_id':server.trusted_agents[i].agent_id,'candidate':np.asarray(x).tolist(),'support':evidence['support_audit']['support'][i],'peer_error':evidence['support_audit']['peer_target_errors'][i],'candidate_pooled_error':error,'candidate_truth_error':abs(float(synthetic_truth(np.asarray(x)[None,:])[0])),'diagnostic_only':True})
 shared.dump(out/'candidate_diagnostics.json',diagnostics)
 # This correlation is descriptive; candidates nested within a seed are not independent inference units.
 rho={}
 fig,axes=plt.subplots(1,3,figsize=(12,3.5))
 for ax,scenario in zip(axes,SCENARIOS):
  records=[d for d in diagnostics if d['scenario']==scenario];ss=[r['support'] for r in records];ee=[r['candidate_pooled_error'] for r in records]
  correlation=spearmanr(ss,ee).statistic if len(records)>1 and np.ptp(ss)>0 and np.ptp(ee)>0 else None;rho[scenario]=None if correlation is None or not np.isfinite(correlation) else float(correlation)
  ax.scatter(ss,ee,s=8,alpha=.35);ax.set_title(scenario);ax.set_xlabel('Candidate support');ax.set_ylabel('Candidate pooled error')
 fig.tight_layout();fig.savefig(out/'support_vs_candidate_quality.png',dpi=160);plt.close(fig)
 primary=[c for c in analysis['comparisons'] if c['endpoint']=='equal_scenario_mean'];fig,ax=plt.subplots(figsize=(9,4))
 for j,c in enumerate(primary):
  ci=c['CI_simultaneous'];delta=c['mean_delta']
  if delta is not None:
   ax.plot(delta,j,'o');
   if ci is not None:ax.plot(ci,[j,j],'-')
 ax.axvline(0,color='gray');ax.set_yticks(range(len(primary)),[c['method']+' - '+c['baseline'] for c in primary]);ax.set_xlabel('Paired pooled error difference (negative is better)');fig.tight_layout();fig.savefig(out/'paired_pooled_differences.png',dpi=160);plt.close(fig)
 index={(r['scenario'],r['seed'],r['method']):r for r in allrows};cases=[];conflicts=[]
 for scenario in SCENARIOS:
  for method,base in [('peer_support_shrink','A_uniform'),('target_peer_shrink','quality_shrink')]:
   pairs=[]
   for seed in reg['seeds']:
    a=index[scenario,seed,method];b=index[scenario,seed,base]
    if a['status']==b['status']=='valid':
     delta=a['pooled_error']-b['pooled_error'];truth=a['truth_error']-b['truth_error'];pairs.append((delta,seed));
     if delta*truth<0:conflicts.append({'scenario':scenario,'seed':seed,'method':method,'baseline':base,'pooled_delta':delta,'truth_delta':truth})
   pairs.sort()
   if pairs:
    for role,pos in [('largest_improvement',0),('median',len(pairs)//2),('largest_worsening',len(pairs)-1)]:
     delta,seed=pairs[pos];cases.append({'scenario':scenario,'method':method,'baseline':base,'case':role,'seed':seed,'pooled_delta':delta,'method_state':index[scenario,seed,method],'baseline_state':index[scenario,seed,base]})
 shared.dump(out/'explanation_cases.json',{'cases':cases,'all_failure_rows':[r for r in allrows if r['status']!='valid'],'pooled_truth_conflicts':conflicts,'descriptive_support_spearman':rho,'no_candidate_independence_claim':True})
 cost_report=summarize_costs(allrows,costs)
 # Source generation additionally executed A to construct and verify the cache.
 # Keep this research overhead out of the six-method application comparison.
 source_baseline=[]
 for b in reg['batches']:
  for scenario in SCENARIOS:
   for r in csv.DictReader((Path(b['source'])/scenario/'summary.csv').open()):
    source_baseline.append({'scenario':scenario,'seed':int(r['seed']),'status':r['status'],
      'stage3_requests':int(r['stage3_requests']) if r.get('stage3_requests') else None,
      'stage3_input_samples':int(r['stage3_input_samples']) if r.get('stage3_input_samples') else None,
      'stage3_seconds':float(r['stage3_seconds']) if r.get('stage3_seconds') else None,
      'research_only':True})
 cost_report['source_baseline_cache_research_overhead']=source_baseline
 shared.dump(out/'cost_report.json',cost_report)
 # Delivered-state alpha changes are explanatory only, never a tuning rule.
 changes=[]
 for scenario in SCENARIOS:
  for seed in reg['seeds']:
   for method,base in [('peer_support_shrink','A_uniform'),('target_peer_shrink','quality_shrink')]:
    a=index[scenario,seed,method];b=index[scenario,seed,base]
    if a['status']==b['status']=='valid':
     changes.append({'scenario':scenario,'seed':seed,'method':method,'baseline':base,
       'alpha_L1_change':float(np.abs(np.asarray(a['alpha'])-np.asarray(b['alpha'])).sum()),
       'pooled_delta':a['pooled_error']-b['pooled_error']})
 shared.dump(out/'alpha_changes.json',changes)
 fig,axes=plt.subplots(1,2,figsize=(10,4))
 for ax,(method,base) in zip(axes,[('peer_support_shrink','A_uniform'),('target_peer_shrink','quality_shrink')]):
  for scenario in SCENARIOS:
   rec=[c for c in changes if c['method']==method and c['scenario']==scenario]
   ax.scatter([c['alpha_L1_change'] for c in rec],[c['pooled_delta'] for c in rec],s=8,alpha=.4,label=scenario)
  ax.axhline(0,color='gray');ax.set_title(method+' - '+base);ax.set_xlabel('Alpha L1 change');ax.set_ylabel('Delivered pooled error difference')
 axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'alpha_change_vs_pooled_difference.png',dpi=160);plt.close(fig)
 text='<mention-date start="2026-10-13"/>\n# 候選點支持度：六方法正式比較\n本輪測試讓其他模型支持的候選點提出者取得較高 α；不新增標記，不動態更新、不剪枝、不付款。三種情境為乾淨、低品質、樣本量不平衡，所有方法共用資料、模型、候選點與 solver。pooled 僅作交付後評分。\n'
 text+='主要比較是共同前置資格通過案例的三情境等權平均 pooled 誤差；共同資格失敗與方法求解失敗分開保存。\n'
 pc={(c['method'],c['baseline']):c for c in primary}
 def judgement(method,base):
  c=pc[method,base]
  if c['stable_advantage']:return '通過事前登錄的穩定改善條件'
  if c['n']!=c['eligible_n']:return '有方法失敗造成配對缺失；僅能報條件品質診斷'
  if c['mean_delta'] is None:return '沒有可計算的有效配對'
  if c['CI_simultaneous'] is not None and c['CI_simultaneous'][1]<0:return '合併區間支持改善，但跨批重現條件未通過'
  return '平均較好，但尚未建立穩定優勢' if c['mean_delta']<0 else '目前未建立改善，不能宣稱永遠無效'
 text+='## 這輪回答了什麼\n'
 text+='- **支持度本身（S 對 A）**：'+judgement('peer_support_shrink','A_uniform')+'。\n'
 text+='- **品質之外的支持增益（QS 對 Q）**：'+judgement('target_peer_shrink','quality_shrink')+'。\n'
 winners=[m for m in ['peer_support_shrink','target_peer_shrink'] if all(pc[m,b]['stable_advantage'] for b in ['A_uniform','B_stage0','ensemble_shrink'])]
 text+='- **是否優於本輪三個品質基準 A／B／E**：'+('、'.join(LABELS[m] for m in winners)+'皆通過三組基準比較。' if winners else '沒有支持度方案同時通過三組基準比較。')+'\n'
 text+='以上主張僅適用於本輪三人、二維、目標 0、共同資格通過的三情境平均；不是完整招募流程或其他目標的普遍保證。\n'
 text+=table(['方法','整體 pooled 平均','中位數','Q1','Q3','有效／合格數'],[[LABELS[g['method']],num(g['mean']),num(g['median']),num(g['Q1']),num(g['Q3']),str(g['valid_n'])+'／'+str(g['eligible_n'])] for g in sorted([g for g in analysis['groups'] if g['endpoint']=='equal_scenario_mean'],key=lambda g:float('inf') if g['mean'] is None else g['mean'])])+'\n'
 text+='## 支持度是否帶來改善\n差值為方法誤差減基準誤差，負值較好；穩定通過要求校正區間全負與三批重現，平均較好不直接當成穩定優勢。\n'+table(['比較','平均差','校正區間','正式穩定門檻'],[[LABELS[c['method']]+' − '+LABELS[c['baseline']],num(c['mean_delta']),interval(c['CI_simultaneous']),'通過' if c['stable_advantage'] else '未通過'] for c in primary])+'\n'
 text+='S−A 檢驗支持訊號本身；QS−Q 檢驗支持在相同品質與收縮之外的增益。要稱比目前候選更好，還須勝過 B、E。QS 若僅勝過 B、未勝過 Q，不能把改善歸因於支持。支持不是參與者貢獻或分潤公平性。\n'
 text+='## 方法定義\n'+table(['方法','做法'],[[LABELS[m],description] for m,description in zip(METHODS,['等權','原 20 筆平均平方誤差取倒數正規化','均勻與 B 各一半','其他模型對提出者候選點的目標差距轉成支持，與均勻各一半','B 乘支持度正規化，再與均勻各一半','原 20 筆混合預測最佳組合與均勻各一半'])])+'\n'
 text+='共同支持尺度 τ 使用所有非自身目標差距的中位數加 0.00001；每個候選點的 e 為其他模型目標差距的中位數。支持分數為 exp(−e/τ)，再正規化成 s；S 為均勻與 s 各一半。QS 則先將 B 乘 s 正規化，再與均勻各一半。三人時只有兩個評估者；共識可能一起錯，不能取代正確答案。\n'
 appendix=['## 正式範圍','三批共 '+str(len(reg['seeds']))+' 次資料重複，保存 '+str(analysis['condition_count'])+' 條件；共同合格 '+str(analysis['common_eligible_n'])+' 組。seeds：'+str([b['seeds'] for b in reg['batches']]),'登錄：'+registration,'執行狀態：'+json.dumps(analysis['execution_statuses'],ensure_ascii=False),'7 組主要比較，200,000 次配對 bootstrap，Bonferroni 校正；各情境 95% 區間僅探索。','## 各情境品質',table(['情境','方法','平均','中位數','有效／合格'],[[g['endpoint'],LABELS[g['method']],num(g['mean']),num(g['median']),str(g['valid_n'])+'／'+str(g['eligible_n'])] for g in analysis['groups'] if g['endpoint']!='equal_scenario_mean']),'## 跨批比較',table(['方法','基準','三批平均差'],[[LABELS[c['method']],LABELS[c['baseline']],str([b['mean_delta'] for b in c['batches']])] for c in primary]),'## 成本與案例','成本紀錄：'+str(out/'cost_report.json')+'；候選支持事後診斷：'+str(out/'candidate_diagnostics.json')+'；改善／惡化／中位／失敗與 pooled-truth 衝突：'+str(out/'explanation_cases.json')+'。','## 來源與驗收','登錄、來源快照、每批完整性與重載驗收保留於輸出目錄。v1 因空區間測試斷言格式錯誤停止；v2 修正測試並重做 pilot，公式未調整，v1 未完成正式資料不併入結果。']
 examples=[c for c in cases if c['case']=='median' and c['scenario']=='reference_clean' and c['method']=='peer_support_shrink']
 if examples:
  chosen=examples[0];key=(chosen['scenario'],chosen['seed']);evidence=allpeer[key];ids=evidence['agent_ids'];p=np.asarray(evidence['prediction_matrix']);audit=evidence['support_audit'];s=np.asarray(audit['support']);mse=np.asarray(evidence['stage0_mse']);b=1/(mse+1e-5);b/=b.sum();sw=np.asarray(index[key+('peer_support_shrink',)]['alpha']);qsw=np.asarray(index[key+('target_peer_shrink',)]['alpha'])
  appendix.extend(['## 實際矩陣與權重示例（事後解釋）',
   '選用乾淨情境 S−A 配對誤差差值的中位案例，seed '+str(chosen['seed'])+'。例子僅解釋公式，不用它挑參數、交付輪次或宣稱所有案例相同。目標為 0；每個欄位是提出者的候選點，每列是評估模型，對角線不納入支持計算。數值顯示六位小數，計算使用保存的完整精度。',
   table(['評估模型']+['Agent '+str(i)+' 的候選點' for i in ids],[['Agent '+str(ids[j])]+[num(p[j,i])+('（自己，不計入）' if i==j else '') for i in range(3)] for j in range(3)]),
   '共同尺度 τ＝全部非自身預測目標差距的中位數＋0.00001＝'+num(audit['peer_error_scale'])+'。每個提出者的 e 由其他兩個模型計算；支持先取 exp(−e/τ)，再正規化。',
   table(['提出者','其他模型差距 e','支持權重 s','原品質 B','S 的 α','QS 的 α'],[['Agent '+str(ids[i]),num(audit['peer_target_errors'][i]),num(s[i]),num(b[i]),num(sw[i]),num(qsw[i])] for i in range(3)]),
   'S 的 α＝0.5×1/3＋0.5×s。QS 的 α＝0.5×1/3＋0.5×正規化(B×s)，所以 QS 同時受品質影響，不保證只依支持度排序。矩陣是 local models 的預測，不是正確答案；實際交付品質另由 pooled 模型事後檢查。'])
 text+='<details>\n<summary>支持度實驗附錄｜統計、成本、情境與失敗</summary>\n'+'\n'.join('\t'+line for entry in appendix for line in entry.splitlines())+'\n</details>\n';Path('docs/peer_support_results_20261013_notion.md').write_text(text)
 return {'report':'docs/peer_support_results_20261013_notion.md','cases':len(cases),'conflicts':len(conflicts),'diagnostic_candidates':len(diagnostics)}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('registration');a=p.parse_args();print(json.dumps(explain(a.registration)))
