"""Independent raw-row arithmetic/bootstrap, coverage, source and replay audit."""
import torch
import json,itertools,hashlib
from pathlib import Path
from collections import Counter
import numpy as np


def records(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main(regpath):
 reg=json.load(open(regpath));root=Path(reg['output']);analysis=json.load(open(root/'formal_analysis/analysis.json'));errors=[];rows=[];index={};eligible=[];batchmap={};max_mean=0.;max_ci=0.;source_records=[]
 if digest(regpath)!=json.load(open(root/'registration_digest.json'))['sha256']:errors.append('registration hash')
 for filename,h in reg['source_sha256'].items():
  if digest(filename)!=h or digest(root/'source_snapshot'/filename)!=h:errors.append('source hash '+filename)
 for filename,h in reg['prior_results_sha256'].items():
  if digest(filename)!=h:errors.append('prior result hash '+filename)
 for bi,name in enumerate(reg['batches']):
  conf=reg[name];p=Path(conf['results']);saved=records(p/'results.jsonl');coverage=Counter((r['scenario'],r['seed'],r['method']) for r in saved)
  expected=set(itertools.product(reg['scenarios'],conf['seeds'],reg['methods']+reg['strategies']))
  if set(coverage)!=expected or any(v!=1 for v in coverage.values()):errors.append('independent coverage '+name)
  if not json.load(open(p/'verification.json'))['passed']:errors.append('reload '+name)
  common={(r['scenario'],r['seed']):r for r in records(p/'shared.jsonl')}
  for seed in conf['seeds']:
   batchmap[seed]=bi
   if all(common.get((sc,seed),{}).get('eligibility')=='eligible' for sc in reg['scenarios']):eligible.append(seed)
  for r in saved:
   key=r['scenario'],r['seed'],r['method'];index[key]=r
   if r['status']=='valid':
    x=np.asarray(r['X']);alpha=np.asarray(r['alpha'])
    if not np.isfinite(x).all() or np.max(np.abs(x))>1+1e-8 or alpha.min()<0 or abs(alpha.sum()-1)>1e-8:errors.append('delivery validity')
  rows.extend(saved)
  for sc in reg['scenarios']:
   for r in records(Path(conf['source'])/sc/'shared_costs.jsonl'):
    if 'blind_test_X' not in r:continue
    h=hashlib.sha256(json.dumps([r[k] for k in ['blind_test_X','blind_test_y','proposals']],sort_keys=True).encode()).hexdigest()
    source_records.append({'batch':bi+1,'scenario':sc,'seed':r['seed'],'combined_input_sha256':h})
 # Independent denominator checks.
 if len(rows)!=reg['expected_conditions'] or len(eligible)!=analysis['common_eligible_n']:errors.append('counts')
 def good(sc,s,m):
  r=index[sc,s,m]
  return r['status']=='valid' and r.get('pooled_error') is not None and np.isfinite(r['pooled_error'])
 def seedvalue(s,m,scs):return sum(index[sc,s,m]['pooled_error'] for sc in scs)/len(scs)
 for g in analysis['groups']:
  scs=reg['scenarios'] if g['endpoint']=='equal_scenario_mean' else [g['endpoint']]
  pool=eligible if g['endpoint']=='equal_scenario_mean' else [s for name in reg['batches'] for s in reg[name]['seeds'] if (g['endpoint'],s) in {(r['scenario'],r['seed']) for r in rows if r['status']!='qualification_failure'}]
  vals=[seedvalue(s,g['method'],scs) for s in pool if all(good(sc,s,g['method']) for sc in scs)]
  if len(vals)!=g['valid_n']:errors.append('group denominator')
  if vals:max_mean=max(max_mean,abs(float(np.mean(vals))-g['mean']))
 usable=[];columns=[]
 for j,c in enumerate(analysis['comparisons']):
  a,b=c['method'],c['baseline'];goodseeds=[s for s in eligible if all(good(sc,s,a) and good(sc,s,b) for sc in reg['scenarios'])]
  vals=[seedvalue(s,a,reg['scenarios'])-seedvalue(s,b,reg['scenarios']) for s in goodseeds]
  if len(vals)!=c['n']:errors.append('contrast denominator')
  if vals:max_mean=max(max_mean,abs(float(np.mean(vals))-c['mean_delta']))
  if len(vals)==len(eligible) and vals:usable.append(j);columns.append(vals)
 def ci_from_weights(matrix,rngseed):
  n,k=matrix.shape;rng=np.random.default_rng(rngseed);rep=reg['bootstrap_replicates'];boot=np.empty((rep,k))
  # Different chunk size from production; same registered random stream.
  for start in range(0,rep,2000):
   size=min(2000,rep-start);counts=rng.multinomial(n,np.full(n,1/n),size=size);boot[start:start+size]=counts.dot(matrix)/n
  return np.quantile(boot,[.025,.975],axis=0).T,np.quantile(boot,[.05/(2*11),1-.05/(2*11)],axis=0).T
 if usable:
  matrix=np.array(columns).T;ordinary,corrected=ci_from_weights(matrix,reg['analysis_rng'])
  for col,j in enumerate(usable):
   c=analysis['comparisons'][j];max_ci=max(max_ci,float(np.max(np.abs(ordinary[col]-c['CI95']))),float(np.max(np.abs(corrected[col]-c['CI_simultaneous']))))
  for bi in range(3):
   sub=matrix[[batchmap[s]==bi for s in eligible]]
   if not len(sub):continue
   ordinary,_=ci_from_weights(sub,reg['analysis_rng']+bi+1)
   for col,j in enumerate(usable):
    b=analysis['comparisons'][j]['batches'][bi];max_ci=max(max_ci,float(np.max(np.abs(ordinary[col]-b['CI95']))));max_mean=max(max_mean,abs(float(np.mean(sub[:,col]))-b['mean_delta']))
 for c in analysis['comparisons']:
  expected=bool(c['CI_simultaneous'] is not None and c['CI_simultaneous'][1]<0 and c['n']==len(eligible) and len(eligible)>0 and all(b['mean_delta'] is not None and b['mean_delta']<0 for b in c['batches']) and sum(b['CI95'] is not None and b['CI95'][1]<0 for b in c['batches'])>=2 and c['delivery_failure_counts'][c['method']]<=c['delivery_failure_counts'][c['baseline']])
  if expected!=c['stable_advantage']:errors.append('stability gate mismatch')
 if max_mean>1e-12 or max_ci>1e-12:errors.append('independent arithmetic/interval mismatch')
 # Fresh source inputs must not duplicate the historical 90 or initial pilot.
 oldreg=json.load(open('alpha_weight_outputs/alpha_initialization_selection_v1/validation_v1_1/registration.json'));historical=set()
 sourcepaths=[Path(b['source']) for b in oldreg['historical_batches']]+[Path(oldreg[s]['source']) for s in ['smoke','pilot']]
 for source in sourcepaths:
  for sc in reg['scenarios']:
   for r in records(source/sc/'shared_costs.jsonl'):
    if 'blind_test_X' in r:historical.add(hashlib.sha256(json.dumps([r[k] for k in ['blind_test_X','blind_test_y','proposals']],sort_keys=True).encode()).hexdigest())
 seen={}
 for r in source_records:
  h=r['combined_input_sha256']
  if h in historical:errors.append('formal/historical input duplicate')
  if h in seen and seen[h]!=r['seed']:errors.append('input duplicate between distinct seeds')
  seen[h]=r['seed']
 replays=[json.load(open(root/f'replay_verification_batch_{i}.json')) for i in range(1,4)]
 if not all(r['passed'] for r in replays):errors.append('full replay failure')
 audit={'passed':not errors,'errors':errors,'saved_conditions':len(rows),'planned_seed_n':reg['planned_seed_n'],'common_eligible_n':len(eligible),'max_mean_difference':max_mean,'max_CI_difference':max_ci,
        'source_checks':source_records,'replays':replays,'prior_results_unchanged':True,'source_and_registration_unchanged':True,'seed_scenarios_never_independent_units':True}
 (root/'independent_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
 acceptance={'flow_and_reproducibility_passed':not errors,'formal_conditions':len(rows),'planned_seed_n':reg['planned_seed_n'],'common_eligible_n':len(eligible),'stable_primary':analysis['stable_primary'],
             'cost_budget_passed':{m:v['budget_passed'] for m,v in analysis['cost_summary'].items()},'quality_and_cost_separate':True,'replays_passed':all(r['passed'] for r in replays),
             'independent_audit_passed':not errors,'prior_results_unchanged':True,'historical_pooled_selector_training':False}
 (root/'final_acceptance.json').write_text(json.dumps(acceptance,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v for k,v in audit.items() if k not in ['source_checks','replays']}),flush=True)
 if errors:raise RuntimeError(str(errors))
if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('registration');x=a.parse_args();main(x.registration)
