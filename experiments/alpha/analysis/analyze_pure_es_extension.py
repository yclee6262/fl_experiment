"""Four newly defined shrinkage contrasts, with reused-data limits."""
import torch
import json,itertools
from pathlib import Path
import numpy as np
from four_alpha_initialization import METHODS as SIX,LABELS as SIX_LABELS
from analyze_four_alpha import valid
from research_analysis import audit_coverage,paired_bootstrap_mean_ci
from oob_shrinkage_research import dump,digest
ROOT=Path('alpha_weight_outputs/pure_es_extension_v1_20261014')
LABELS=dict(SIX_LABELS,stacked_full='E：純組合法',pure_support='S：純支持法')
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def main():
 reg=json.load(open(ROOT/'registration.json'));methods=SIX+reg['methods'];idx={};errors=[];maxima=dict(alpha=0.,pooled=0.,matrix=0.,raw_signal=0.);coverage=[]
 for b in reg['batches']:
  for directory,ms in [(b['results'],reg['methods']),(b['six_method_results'],SIX)]:
   p=Path(directory);rows=lines(p/'results.jsonl');c=audit_coverage(rows,list(itertools.product(reg['scenarios'],b['seeds'],ms)),['scenario','seed','method']);coverage.append(c)
   v=json.load(open(p/'verification.json'))
   if not c['complete'] or not v['passed']:errors.append('coverage/reload failed')
   for key,x in v['max_reload_difference'].items():maxima[key]=max(maxima[key],x)
   for r in rows:
    key=r['scenario'],r['seed'],r['method']
    if key in idx:errors.append('duplicate key')
    idx[key]=r
  if digest(Path(b['six_method_results'])/'results.jsonl')!=reg['original_six_results_sha256'][b['six_method_results']]:errors.append('old results changed')
  for file,h in reg['source_sha256'].items():
   if digest(file)!=h or digest(Path(b['results'])/'source_snapshot'/file)!=h:errors.append('source hash mismatch')
 seeds=[k for b in reg['batches'] for k in b['seeds']];groups=[];comparisons=[]
 for sc,k in itertools.product(reg['scenarios'],seeds):
  for raw,mix in [('stacked_full','ensemble_shrink'),('pure_support','peer_support_shrink')]:
   p=idx[sc,k,raw];m=idx[sc,k,mix]
   if valid(p) and valid(m):
    diff=float(np.max(abs(.5*np.array(p['alpha'])+.5/3-np.array(m['alpha']))));maxima['raw_signal']=max(maxima['raw_signal'],diff)
 if maxima['raw_signal']>1e-7:errors.append('raw signal differs from prior mixed methods')
 if errors:dump(ROOT/'acceptance.json',dict(passed=False,errors=errors));raise RuntimeError(str(errors))
 for endpoint in ['equal_scenario_mean']+reg['scenarios']:
  scenarios=reg['scenarios'] if endpoint=='equal_scenario_mean' else [endpoint]
  for m in methods:
   values=[np.mean([idx[s,k,m]['pooled_error'] for s in scenarios]) for k in seeds if all(valid(idx[s,k,m]) for s in scenarios)]
   groups.append(dict(endpoint=endpoint,method=m,n=len(values),excluded=len(seeds)-len(values),mean=float(np.mean(values)) if values else None,median=float(np.median(values)) if values else None,Q1=float(np.quantile(values,.25)) if values else None,Q3=float(np.quantile(values,.75)) if values else None))
  for m,base in reg['contrasts']:
   ds=[];batches=[];excluded=[];conflicts=0
   for bi,b in enumerate(reg['batches']):
    bd=[]
    for k in b['seeds']:
     if not all(valid(idx[s,k,m]) and valid(idx[s,k,base]) for s in scenarios):excluded.append(k);continue
     bd.append(float(np.mean([idx[s,k,m]['pooled_error']-idx[s,k,base]['pooled_error'] for s in scenarios])))
     for s in scenarios:conflicts+=int((idx[s,k,m]['pooled_error']-idx[s,k,base]['pooled_error'])*(idx[s,k,m]['truth_error']-idx[s,k,base]['truth_error'])<0)
    ds.extend(bd);batches.append(dict(batch=bi+1,n=len(bd),mean_delta=float(np.mean(bd)) if bd else None,CI95=paired_bootstrap_mean_ci(bd)))
   comparisons.append(dict(endpoint=endpoint,method=m,baseline=base,n=len(ds),excluded=excluded,mean_delta=float(np.mean(ds)) if ds else None,CI95=paired_bootstrap_mean_ci(ds),CI_four_family=None,batches=batches,deltas=ds,pooled_truth_conflicts=conflicts))
 family=[c for c in comparisons if c['endpoint']=='equal_scenario_mean'];usable=[c for c in family if c['n']==len(seeds)]
 if usable:
  mat=np.array([c['deltas'] for c in usable]).T;rng=np.random.default_rng(reg['analysis_rng']);rep=reg['bootstrap_replicates'];n=len(seeds);samples=np.lib.format.open_memmap(ROOT/'bootstrap_means.npy',mode='w+',dtype='float64',shape=(rep,len(usable)))
  for start in range(0,rep,1000):
   w=rng.multinomial(n,np.full(n,1/n),size=min(1000,rep-start));samples[start:start+len(w)]=w@mat/n
  samples.flush();ci=np.quantile(samples,[.05/(2*4),1-.05/(2*4)],axis=0).T
  for c,q in zip(usable,ci):c['CI_four_family']=q.tolist()
 for c in comparisons:del c['deltas']
 result=dict(inference=reg['inference'],groups=groups,comparisons=comparisons,labels=LABELS,new_conditions=540,combined_conditions=len(idx),execution_statuses={s:{m:sum(r['status']==s and r['method']==m for r in idx.values()) for m in methods} for s in sorted({r['status'] for r in idx.values()})})
 dump(ROOT/'analysis.json',result);dump(ROOT/'acceptance.json',dict(passed=True,errors=[],coverage=coverage,max_reload_difference=maxima,combined_conditions=len(idx),new_valid=sum(valid(r) for r in idx.values() if r['method'] in reg['methods']),old_results_unchanged=True))
 print(json.dumps(dict(means={g['method']:g['mean'] for g in groups if g['endpoint']=='equal_scenario_mean'},primary_comparisons=family)))
if __name__=='__main__':main()
