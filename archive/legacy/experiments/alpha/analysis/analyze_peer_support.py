"""Preregistered conditional-eligibility paired pooled comparison."""
import argparse,itertools,json
from pathlib import Path
import numpy as np
from scipy.stats import norm
from peer_support_alpha import METHODS,SCENARIOS,CONTRASTS,LABELS
from research_analysis import paired_bootstrap_mean_ci,audit_coverage
from oob_shrinkage_research import dump

def indexed_rows(reg):
 index={};eligibility={};errors=[]
 for bi,b in enumerate(reg['batches']):
  root=Path(b['results']);m=json.loads((root/'manifest.json').read_text());v=json.loads((root/'verification.json').read_text())
  if m['status']!='complete' or not v['passed']:errors.append('batch not verified')
  rows=list(map(json.loads,(root/'results.jsonl').read_text().splitlines()));expected=list(itertools.product(SCENARIOS,b['seeds'],METHODS));cover=audit_coverage(rows,expected,['scenario','seed','method'])
  if not cover['complete']:errors.append('missing/duplicate conditions')
  for r in rows:
   k=(r['scenario'],r['seed'],r['method'])
   if k in index:errors.append('duplicate cross-batch key')
   index[k]=r
  shared=list(map(json.loads,(root/'shared.jsonl').read_text().splitlines()))
  keys=[(s['scenario'],s['seed']) for s in shared]
  if set(keys)!=set(itertools.product(SCENARIOS,b['seeds'])) or len(set(keys))!=len(keys):errors.append('shared eligibility coverage')
  for s in shared:eligibility[s['scenario'],s['seed']]=s['eligibility']=='eligible'
 if errors:raise ValueError(str(errors))
 return index,eligibility

def valid(row):
 return row['status']=='valid' and row.get('X') is not None and np.asarray(row['X']).shape==(2,) and np.isfinite(row['X']).all() and max(abs(np.array(row['X'])))<=1+1e-8 and row.get('pooled_error') is not None and np.isfinite(row['pooled_error'])

def eligible_seeds(reg,eligibility,endpoint):
 scenarios=SCENARIOS if endpoint=='equal_scenario_mean' else [endpoint]
 return [s for b in reg['batches'] for s in b['seeds'] if all(eligibility[k,s] for k in scenarios)]

def statistics(reg,index,eligibility):
 groups=[];comparisons=[]
 for endpoint in ['equal_scenario_mean']+SCENARIOS:
  sc=SCENARIOS if endpoint=='equal_scenario_mean' else [endpoint];seeds=eligible_seeds(reg,eligibility,endpoint)
  for m in METHODS:
   values=[float(np.mean([index[s,k,m]['pooled_error'] for s in sc])) for k in seeds if all(valid(index[s,k,m]) for s in sc)]
   q=np.quantile(values,[.25,.75]).tolist() if values else [None,None]
   groups.append({'endpoint':endpoint,'method':m,'eligible_n':len(seeds),'valid_n':len(values),'method_failure_n':len(seeds)-len(values),'mean':float(np.mean(values)) if values else None,'median':float(np.median(values)) if values else None,'Q1':q[0],'Q3':q[1]})
  for m,base in CONTRASTS:
   ds=[];batches=[];excluded=[]
   for bi,b in enumerate(reg['batches']):
    bd=[]
    for seed in b['seeds']:
     if seed not in seeds:continue
     if not all(valid(index[s,seed,m]) and valid(index[s,seed,base]) for s in sc):excluded.append(seed);continue
     d=float(np.mean([index[s,seed,m]['pooled_error']-index[s,seed,base]['pooled_error'] for s in sc]));bd.append(d)
    ds.extend(bd);batches.append({'batch':bi,'n':len(bd),'mean_delta':float(np.mean(bd)) if bd else None,'CI95':paired_bootstrap_mean_ci(bd)})
   comparisons.append({'endpoint':endpoint,'method':m,'baseline':base,'eligible_n':len(seeds),'n':len(ds),'mean_delta':float(np.mean(ds)) if ds else None,'CI95':paired_bootstrap_mean_ci(ds),'batches':batches,'method_failure_exclusions':excluded,'CI_simultaneous':None,'stable_advantage':False,'deltas':ds})
 return groups,comparisons

def analyze(registration,formal=False):
 reg=json.loads(Path(registration).read_text());root=Path(reg['output']);index,eligible=indexed_rows(reg);groups,comp=statistics(reg,index,eligible)
 if formal:
  family=[c for c in comp if c['endpoint']=='equal_scenario_mean'];usable=[c for c in family if c['n']==c['eligible_n'] and c['n']>0 and not c['method_failure_exclusions']]
  # Shared resampling preserves all methods and all scenarios within a seed.
  if usable:
   matrix=np.array([c['deltas'] for c in usable]).T;rng=np.random.default_rng(reg['analysis_rng']);rep=reg['bootstrap_replicates'];n=matrix.shape[0];samples=np.lib.format.open_memmap(root/'bootstrap_means.npy',mode='w+',dtype='float64',shape=(rep,len(usable)))
   for start in range(0,rep,1000):
    size=min(1000,rep-start);w=rng.multinomial(n,np.full(n,1/n),size=size);samples[start:start+size]=w@matrix/n
   samples.flush();tail=reg['family_error']/(2*len(CONTRASTS));intervals=np.quantile(samples,[tail,1-tail],axis=0).T
   for c,ci in zip(usable,intervals):
    c['CI_simultaneous']=ci.tolist();c['stable_advantage']=bool(ci[1]<0 and len(c['batches'])==3 and all(b['mean_delta'] is not None and b['mean_delta']<0 for b in c['batches']) and sum(b['CI95'][1] is not None and b['CI95'][1]<0 for b in c['batches'])>=2)
 for c in comp:del c['deltas']
 status={s:{m:sum(r['status']==s and r['method']==m for r in index.values()) for m in METHODS} for s in sorted(set(r['status'] for r in index.values()))}
 result={'status':'formal' if formal else 'pilot','groups':groups,'comparisons':comp,'execution_statuses':status,'planned_n':sum(len(b['seeds']) for b in reg['batches']),'common_eligible_n':len(eligible_seeds(reg,eligible,'equal_scenario_mean')),'condition_count':len(index),'qualification_population':'common three-scenario pre-method eligibility','formal_family':7,'exploratory_scenario_intervals':'ordinary 95%; no confirmatory scenario claims'}
 dump(root/('formal_analysis.json' if formal else 'pilot_analysis.json'),result)
 print(json.dumps({'status':result['status'],'conditions':len(index),'eligible_n':result['common_eligible_n'],'primary_means':{g['method']:g['mean'] for g in groups if g['endpoint']=='equal_scenario_mean'},'stable':[(c['method'],c['baseline']) for c in comp if c['stable_advantage']]}));return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('registration');p.add_argument('--formal',action='store_true');a=p.parse_args();analyze(a.registration,a.formal)
