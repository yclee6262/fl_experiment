"""Independent arithmetic audit and method-focused six-method report."""
import torch
import json,itertools
from pathlib import Path
import numpy as np
from research_analysis import audit_coverage,paired_bootstrap_mean_ci
from four_alpha_initialization import METHODS,SCENARIOS,LABELS,CONTRASTS
from oob_shrinkage_research import dump,digest
from analyze_four_alpha import valid
ROOT=Path('alpha_weight_outputs/four_alpha_v1_20261014')
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def f(x):return '不可計算' if x is None else f'{x:.6f}'
def interval(x):return '不可計算' if x is None or any(y is None for y in x) else f'{x[0]:+.6f} 至 {x[1]:+.6f}'
def audit(kind):
 reg=json.load(open(ROOT/(kind+'_registration.json')));output=Path(reg['output']);a=json.load(open(output/('formal_analysis.json' if kind=='formal' else 'pilot_analysis.json')));index={};errors=[];checks=[]
 for b in reg['batches']:
  r=Path(b['results']);v=json.load(open(r/'verification.json'));rows=lines(r/'results.jsonl');coverage=audit_coverage(rows,list(itertools.product(SCENARIOS,b['seeds'],METHODS)),['scenario','seed','method'])
  if not v['passed'] or not coverage['complete']:errors.append('batch verification/coverage failed')
  for name,h in reg['source_sha256'].items():
   if digest(name)!=h or digest(r/'source_snapshot'/name)!=h:errors.append('source mismatch '+name)
  for row in rows:
   key=(row['scenario'],row['seed'],row['method'])
   if key in index:errors.append('duplicate across batches')
   index[key]=row
  checks.append(dict(results=str(r),expected=len(b['seeds'])*18,coverage=coverage,reload_max=v['max_reload_difference']))
 allseeds=[s for b in reg['batches'] for s in b['seeds']];differences=[]
 fit_diff=0.
 for sc,k in itertools.product(SCENARIOS,allseeds):
  eu=index[sc,k,'ensemble_shrink'];eb=index[sc,k,'B_stacked_half']
  if valid(eu) and valid(eb) and not eu['alpha_audit'].get('fallback_reason') and not eb['alpha_audit'].get('fallback_reason'):
   fit_diff=max(fit_diff,float(np.max(abs(np.array(eu['alpha_audit']['raw_scores'])-eb['alpha_audit']['ensemble_fit']['raw_scores']))))
 if fit_diff>1e-7:errors.append('EU and EB fitted different w')
 for g in a['groups']:
  sc=SCENARIOS if g['endpoint']=='equal_scenario_mean' else [g['endpoint']]
  values=[float(np.mean([index[s,k,g['method']]['pooled_error'] for s in sc])) for k in allseeds if all(valid(index[s,k,g['method']]) for s in sc)]
  if g['valid_n']!=len(values):errors.append('group valid n mismatch')
  if values:
   actual=[np.mean(values),np.median(values),*np.quantile(values,[.25,.75])];saved=[g['mean'],g['median'],g['Q1'],g['Q3']]
   differences.extend(abs(float(x)-y) for x,y in zip(actual,saved))
 for c in a['comparisons']:
  sc=SCENARIOS if c['endpoint']=='equal_scenario_mean' else [c['endpoint']]
  ds=[np.mean([index[s,k,c['method']]['pooled_error']-index[s,k,c['baseline']]['pooled_error'] for s in sc]) for k in allseeds if all(valid(index[s,k,c['method']]) and valid(index[s,k,c['baseline']]) for s in sc)]
  if c['n']!=len(ds):errors.append('paired n mismatch')
  if ds:
   differences.append(abs(float(np.mean(ds))-c['mean_delta']))
   differences.extend(abs(x-y) for x,y in zip(paired_bootstrap_mean_ci(ds),c['CI95']))
 family=[c for c in a['comparisons'] if c['endpoint']=='equal_scenario_mean'];quantile_diff=0.
 if kind=='formal':
  usable=[c for c in family if c['n']==c['eligible_n'] and c['n']>0 and not c['method_failure_exclusions']]
  if usable:
   samples=np.load(output/'bootstrap_means.npy',mmap_mode='r');tail=.05/(2*len(CONTRASTS));ci=np.quantile(samples,[tail,1-tail],axis=0).T
   if samples.shape!=(200000,len(usable)):errors.append('bootstrap array shape')
   for c,q in zip(usable,ci):
    quantile_diff=max(quantile_diff,float(np.max(abs(q-c['CI_simultaneous']))))
    stable=bool(q[1]<0 and all(b['mean_delta']<0 for b in c['batches']) and sum(b['CI95'][1]<0 for b in c['batches'])>=2)
    if c['stable_advantage']!=stable:errors.append('stability flag differs')
 if max(differences or [0])>1e-12 or quantile_diff>1e-12:errors.append('numeric summary mismatch')
 conflicts=[];costs=[]
 for method in METHODS:
  rows=[r for r in index.values() if r['method']==method]
  costs.append(dict(method=method,n=len(rows),valid=sum(valid(r) for r in rows),mean_stage3_requests=float(np.mean([r['stage3_requests'] for r in rows])) if all('stage3_requests' in r for r in rows) else None,mean_stage3_seconds=float(np.mean([r['stage3_seconds'] for r in rows])) if all('stage3_seconds' in r for r in rows) else None,mean_initialization_requests=float(np.mean([r['initialization_requests'] for r in rows])) if all('initialization_requests' in r for r in rows) else None,mean_initialization_seconds=float(np.mean([r['initialization_seconds'] for r in rows])) if all('initialization_seconds' in r for r in rows) else None,support_calls_per_independent_run=6 if method in ['peer_support_shrink','support_quality_mix50'] else 0))
 for m,b in CONTRASTS:
  count=0;total=0
  for sc,k in itertools.product(SCENARIOS,allseeds):
   r,t=index[sc,k,m],index[sc,k,b]
   if valid(r) and valid(t):
    total+=1;count+=int((r['pooled_error']-t['pooled_error'])*(r['truth_error']-t['truth_error'])<0)
  conflicts.append(dict(method=m,baseline=b,n=total,opposite_direction=count))
 report=dict(passed=not errors,errors=errors,batches=checks,condition_count=len(index),saved_mean_quartile_CI_max_difference=max(differences or [0]),family_quantile_max_difference=quantile_diff,EU_EB_fitted_weight_max_difference=fit_diff,execution_statuses=a['execution_statuses'],costs=costs,pooled_truth_conflicts=conflicts)
 dump(output/'completion_audit.json',report)
 if errors:raise RuntimeError(str(errors))
 return a,report

def main():
 state=json.load(open(ROOT/'pipeline_status.json'))
 if state['status']!='complete':raise RuntimeError('Pipeline incomplete')
 outputs={k:audit(k) for k in ['pilot','development','formal']};formal,check=outputs['formal'];dev,dc=outputs['development']
 text=['# α 初始化聚焦比較：A、B、EU、EB、SU、SB','', '**主要品質判準是交付 X 的 pooled 絕對目標誤差；越低越好。探索資料與獨立確認分開，不合併平均。**','',
 '## 方法與目的','', '|代號|名稱|做法|','|---|---|---|',
 '|A|均勻基準|每人 1/N|','|B|品質基準|盲測個人 MSE 加小量取倒數正規化|','|EU|組合均勻法|0.5w + 0.5U|','|EB|組合品質法|0.5w + 0.5B|','|SU|支持均勻法|0.5s + 0.5U|','|SB|支持品質法|0.5s + 0.5B|','',
 'w 用原 20 筆盲測找非負、總和 1 的組合，使混合預測 MSE 最小；U 是均勻；B 是個人品質反比。s 是排除自身的候選點交叉評分：目標絕對差中位數 e、全部非自身差距中位數加 1e-5 的 tau、exp(-e/tau) 正規化。SB 不相乘、不加均勻。舊 QS 不在本輪。','',
 '目的：EU/EB 與 SU/SB 比較混合對象；EU/SU 與 EB/SB 比較訊號來源。四候選各與 A、B 比較，避免只在較弱候選中挑勝者。','',
 '## 控制與完成狀態','',
 '三情境為乾淨、低品質 25%、樣本量不平衡 4 倍。固定 IDs 1/2/20、三人、2D、target0、box[-1,1]、tol1e-8、best_of/custom30。50/50 固定、不更新 alpha、不剪枝、不結算、不新增標記。pooled 僅在交付後評分，truth 僅診斷。','',
 f'Pilot 保存 {outputs["pilot"][1]["condition_count"]} 條件；舊三批探索 {dc["condition_count"]} 條件；獨立確認 {check["condition_count"]} 條件、90 個抽樣案例（三批各30）。所有條件及狀態核對、重載與來源雜湊通過。','',
 '独立確認使用全新資料、local models、proposal 與 pooled evaluator；開始前查核未用過的 seeds。90 案例是固定規模，沒有保證檢定力，區間跨零不能說等效。'.replace('独','獨'),'',
 '## 獨立確認：主要 pooled 結果','',
 '每個案例先將三情境誤差等權平均，再跨案例平均。中位數是50百分位，Q1/Q3是25/75百分位；有效數以三情境都有效為條件，失敗另列。','',
 '|方法|平均|中位數|Q1|Q3|有效／共同合格|','|---|---:|---:|---:|---:|---:|']
 for g in sorted([g for g in formal['groups'] if g['endpoint']=='equal_scenario_mean'],key=lambda g:float('inf') if g['mean'] is None else g['mean']):text.append(f'|{LABELS[g["method"]]}|{f(g["mean"])}|{f(g["median"])}|{f(g["Q1"])}|{f(g["Q3"])}|{g["valid_n"]}/{g["eligible_n"]}|')
 text+=['','## 十二組主要配對','', '差值為前者誤差減後者；負值有利前者。200,000 次 seed 配對 bootstrap，十二組 Bonferroni 校正。穩定門檻另要求三批平均全負、至少兩批一般95%區間全負、完整有效配對且觀察失敗不增加。情境結果仅探索。'.replace('仅','僅'),'', '|比較|平均差|一般95%區間|十二組同時區間|穩定改善|','|---|---:|---|---|---|']
 for c in formal['comparisons']:
  if c['endpoint']=='equal_scenario_mean':text.append(f'|{LABELS[c["method"]]} − {LABELS[c["baseline"]]}|{f(c["mean_delta"])}|{interval(c["CI95"])}|{interval(c["CI_simultaneous"])}|{"通過" if c["stable_advantage"] else "未通過"}|')
 stable=[(LABELS[c['method']],LABELS[c['baseline']]) for c in formal['comparisons'] if c['stable_advantage']]
 text+=['',f'通過的主要比較：{stable if stable else "無"}。平均第一不代表穩定勝過所有方法；未通過不表示無效或等效。','', '## 各情境（探索性）','', '|情境|方法|pooled 平均|有效／合格|','|---|---|---:|---:|']
 for g in formal['groups']:
  if g['endpoint']!='equal_scenario_mean':text.append(f'|{g["endpoint"]}|{LABELS[g["method"]]}|{f(g["mean"])}|{g["valid_n"]}/{g["eligible_n"]}|')
 text+=['','## 舊資料探索：不與新資料合併','', '|方法|探索 pooled 平均|有效／合格|','|---|---:|---:|']
 for g in dev['groups']:
  if g['endpoint']=='equal_scenario_mean':text.append(f'|{LABELS[g["method"]]}|{f(g["mean"])}|{g["valid_n"]}/{g["eligible_n"]}|')
 text+=['','## 失敗、成本與驗收','',f'獨立確認全部執行狀態：{formal["execution_statuses"]}。共同資格失敗保留且不補抽；品質推論限定共同合格人口，方法失敗不刪除。','', '|方法|Stage3 平均查詢|Stage3 平均秒|初始化平均秒|初始化實際 calls|支持度獨立準備 calls|','|---|---:|---:|---:|---:|---:|']
 for c in check['costs']:text.append(f'|{LABELS[c["method"]]}|{f(c["mean_stage3_requests"])}|{f(c["mean_stage3_seconds"])}|{f(c["mean_initialization_seconds"])}|{f(c["mean_initialization_requests"])}|{c["support_calls_per_independent_run"]}|')
 text+=['','表中是本機方法階段成本，模型訓練/資格/proposal 的共享成本每 seed 保存一次，沒有重複加總；研究矩陣重載、diagonal 診斷及 pooled 評分另列，不加入 Stage3。API介面呼叫數不是貨幣价格。EU 舊入口重新預測相同20筆，初始化3calls；EB入口重用共同資格快取，為0calls。這是實作快取差異，不是資訊量差異；EU/EB實際擬合的w另作一致性核對，不用成本推論品質優勢。'.replace('价格','價格'),'',
 'pooled/truth 衝突表示同一配對在兩評估來源得到相反改善方向，不是求解失敗；主要判讀仍用 pooled，完整十二組計數在 completion_audit.json。','',
 f'獨立算術稽核：保存平均/四分位/一般CI最大差 {check["saved_mean_quartile_CI_max_difference"]}；200,000抽樣同時區間最大差 {check["family_quantile_max_difference"]}。每批完整重載結果在 verification.json。','',
 '最初 sandbox 子程序因 OpenMP 共享記憶體啟動錯誤中斷，無 pilot 數值結果；改於正常本機環境執行，原錯誤及恢復記錄保留，沒有更改方法或容差。','',
 '## 後續','', '依獨立確認決定是否有候選值得採用；證據不足時保留方法、明確限制情境，不調參後回填本輪。沒有執行新的比例搜尋、pruning、Shapley、分潤或經濟公平性實驗。','',
 '完整資料：alpha_weight_outputs/four_alpha_v1_20261014；研究規格：docs/four_alpha_comparison_plan_20261014.md；登錄：docs/four_alpha_registration_20261014.json。']
 report='\n'.join(text)+'\n';(ROOT/'report.md').write_text(report);Path('docs/four_alpha_results_20261014.md').write_text(report)
 dump(ROOT/'final_acceptance.json',dict(passed=True,independent_audits=[str(Path(json.load(open(ROOT/(k+'_registration.json')))['output'])/'completion_audit.json') for k in outputs],stable_primary=stable,formal_conditions=check['condition_count']))
 print(json.dumps(dict(passed=True,stable=stable,means={g['method']:g['mean'] for g in formal['groups'] if g['endpoint']=='equal_scenario_mean'})))
if __name__=='__main__':main()
