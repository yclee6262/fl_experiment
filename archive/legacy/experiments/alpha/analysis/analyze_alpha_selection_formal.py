"""Preregistered three-batch analysis, quality/failure/cost gates kept separate."""
import torch
import json,itertools
from collections import Counter
from pathlib import Path
import numpy as np
import oob_shrinkage_research as shared
from run_alpha_initialization_selection import lines
from alpha_initialization_selection import DISPLAY
from alpha_selection_formal_statistics import intervals,stable_advantage
from research_analysis import audit_coverage

def analyze(regpath):
 reg=json.load(open(regpath));root=Path(reg['output']);out=root/'formal_analysis';out.mkdir(exist_ok=False)
 idx={};elig={};batch_for={};all_rows=[];costs=[];evidence=[];errors=[]
 for bi,name in enumerate(reg['batches']):
  conf=reg[name];p=Path(conf['results']);manifest=json.load(open(p/'manifest.json'));v=json.load(open(p/'verification.json'))
  if manifest['status']!='complete' or not v['passed']:raise RuntimeError('Formal batch not verified '+name)
  rows=lines(p/'results.jsonl');expected=list(itertools.product(reg['scenarios'],conf['seeds'],reg['methods']+reg['strategies']))
  if not audit_coverage(rows,expected,['scenario','seed','method'])['complete']:raise RuntimeError('Coverage mismatch '+name)
  for r in rows:
   key=r['scenario'],r['seed'],r['method']
   if key in idx:raise RuntimeError('Duplicate cross batch key')
   idx[key]=r
  for common in lines(p/'shared.jsonl'):elig[common['scenario'],common['seed']]=common['eligibility']=='eligible'
  for seed in conf['seeds']:batch_for[seed]=bi
  all_rows.extend(rows)
  if (p/'deployment_costs.jsonl').exists():costs.extend(lines(p/'deployment_costs.jsonl'))
  if (p/'selector_evidence.jsonl').exists():evidence.extend(lines(p/'selector_evidence.jsonl'))
 seeds=[s for name in reg['batches'] for s in reg[name]['seeds']];eligible=[s for s in seeds if all(elig.get((sc,s),False) for sc in reg['scenarios'])]
 methods=reg['methods']+reg['strategies'];groups=[]
 def valid(sc,s,m):
  row=idx[sc,s,m];x=row.get('X')
  return row['status']=='valid' and x is not None and np.asarray(x).shape==(2,) and np.isfinite(x).all() and np.max(np.abs(x))<=1+1e-8 and row.get('pooled_error') is not None and np.isfinite(row['pooled_error'])
 def seed_error(s,m,scenarios):return float(np.mean([idx[sc,s,m]['pooled_error'] for sc in scenarios]))
 for endpoint in ['equal_scenario_mean']+reg['scenarios']:
  scs=reg['scenarios'] if endpoint=='equal_scenario_mean' else [endpoint]
  pop=eligible if endpoint=='equal_scenario_mean' else [s for s in seeds if elig.get((endpoint,s),False)]
  for m in methods:
   good=[s for s in pop if all(valid(sc,s,m) for sc in scs)];a=np.array([seed_error(s,m,scs) for s in good])
   groups.append({'endpoint':endpoint,'method':m,'display':DISPLAY.get(m,m),'eligible_n':len(pop),'valid_n':len(a),'method_failure_n':len(pop)-len(a),
                  'mean':float(a.mean()) if len(a) else None,'median':float(np.median(a)) if len(a) else None,'Q1':float(np.quantile(a,.25)) if len(a) else None,'Q3':float(np.quantile(a,.75)) if len(a) else None})
 comp=[];complete_columns=[];usable=[]
 for j,(a,b) in enumerate(reg['contrasts']):
  good=[s for s in eligible if all(valid(sc,s,a) and valid(sc,s,b) for sc in reg['scenarios'])]
  delta={s:seed_error(s,a,reg['scenarios'])-seed_error(s,b,reg['scenarios']) for s in good}
  failures={m:sum(not valid(sc,s,m) for s in eligible for sc in reg['scenarios']) for m in [a,b]}
  batches=[]
  for bi,name in enumerate(reg['batches']):
   pop=[s for s in eligible if batch_for[s]==bi];ds=[delta[s] for s in pop if s in delta]
   ci=intervals(np.array(ds)[:,None],reg['bootstrap_replicates'],reg['analysis_rng']+bi+1,reg['comparison_family'])['ordinary95'][0] if ds else None
   batches.append({'batch':bi+1,'eligible_n':len(pop),'n':len(ds),'mean_delta':float(np.mean(ds)) if ds else None,'CI95':ci})
  entry={'method':a,'baseline':b,'eligible_n':len(eligible),'n':len(good),'incomplete_seeds':[s for s in eligible if s not in delta],
         'mean_delta':float(np.mean(list(delta.values()))) if delta else None,'CI95':None,'CI_simultaneous':None,'batches':batches,
         'delivery_failure_counts':failures,'stable_advantage':False,'seed_deltas':[{'seed':s,'delta':delta[s]} for s in good]}
  if len(good)==len(eligible) and good:
   usable.append(j);complete_columns.append([delta[s] for s in eligible])
  elif good:
   entry['complete_pair_diagnostic_CI95']=intervals(np.array(list(delta.values()))[:,None],reg['bootstrap_replicates'],reg['analysis_rng'],reg['comparison_family'])['ordinary95'][0]
  comp.append(entry)
 if usable:
  matrix=np.array(complete_columns).T;np.save(out/'complete_seed_contrast_matrix.npy',matrix)
  ci=intervals(matrix,reg['bootstrap_replicates'],reg['analysis_rng'],reg['comparison_family'])
  for col,j in enumerate(usable):
   e=comp[j];e['CI95']=ci['ordinary95'][col];e['CI_simultaneous']=ci['simultaneous'][col]
   e['stable_advantage']=stable_advantage(e['CI_simultaneous'],[b['mean_delta'] for b in e['batches']],[b['CI95'] for b in e['batches']],not e['incomplete_seeds'],e['delivery_failure_counts'][e['method']]<=e['delivery_failure_counts'][e['baseline']])
 # Cost gates are independent of inferential quality gates.
 cost_summary={}
 for m in reg['strategies']:
  ratios={k:[c['method_costs'][m][k]/c['method_costs']['ensemble_shrink'][k] for c in costs] for k in ['seconds','requests','input_samples']}
  medians={k:float(np.median(v)) if v else None for k,v in ratios.items()}
  perbatch=[]
  for bi,name in enumerate(reg['batches']):
   cs=[c for c in costs if batch_for[c['seed']]==bi]
   perbatch.append({'batch':bi+1,**{k:float(np.median([c['method_costs'][m][k]/c['method_costs']['ensemble_shrink'][k] for c in cs])) if cs else None for k in ratios}})
  cost_summary[m]={'eligible_scenario_n':len(costs),'median_method_cost_ratios_to_EU':medians,'batches':perbatch,
                   'budget':reg['cost_limits'][m],'budget_passed':bool(costs and all(v is not None and v<=reg['cost_limits'][m] for v in medians.values())),
                   'all_batches_within_time_budget':bool(costs and all(b['seconds'] is not None and b['seconds']<=reg['cost_limits'][m] for b in perbatch))}
  # Full deployment cost includes preparation once; source A/evaluator are research overhead, not deployment.
  full=[]
  for c in costs:
   common=sum(c['shared_preparation'].get(k,0) for k in ['training_seconds','qualification_seconds','proposal_seconds'])+(c.get('fixed_proposal_seconds') or 0)+c['shared_signal_cost']['blind_seconds']
   full.append((common+c['method_costs'][m]['seconds'])/(common+c['method_costs']['ensemble_shrink']['seconds']))
  cost_summary[m]['median_total_preparation_plus_method_time_ratio_to_EU']=float(np.median(full)) if full else None
 result={'status':'independent_confirmation','planned_seed_n':len(seeds),'common_eligible_n':len(eligible),'common_qualification_failure_seeds':[s for s in seeds if s not in eligible],
         'condition_count':len(all_rows),'groups':groups,'comparisons':comp,'formal_family':reg['comparison_family'],'cost_summary':cost_summary,
         'execution_statuses':{s:{m:sum(r['status']==s and r['method']==m for r in all_rows) for m in methods} for s in sorted({r['status'] for r in all_rows})},
         'selected_frequency':{m:dict(Counter(DISPLAY.get(e[m]['selected_method'],e[m]['selected_method']) for e in evidence)) for m in reg['strategies']},
         'stable_primary':[(c['method'],c['baseline']) for c in comp if c['stable_advantage']],
         'scope':'conditional shared qualification; scenario intervals/descriptive ranks exploratory; no selector changes',
         'per_seed_method_errors':[{'seed':s,**{m:seed_error(s,m,reg['scenarios']) if all(valid(sc,s,m) for sc in reg['scenarios']) else None for m in methods}} for s in eligible]}
 shared.dump(out/'analysis.json',result)
 print(json.dumps({'conditions':len(all_rows),'eligible':len(eligible),'stable':result['stable_primary']}),flush=True)
 return result

if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('registration');x=a.parse_args();analyze(x.registration)
