"""Independent completion evidence; checks actual rows, artifacts and inference."""
import argparse,hashlib,itertools,json
from pathlib import Path
import numpy as np
from peer_support_alpha import METHODS,SCENARIOS,CONTRASTS

def ordinary_interval(values,reg):
 if not values:return [None,None]
 values=np.asarray(values,float);rng=np.random.default_rng(reg['batch_ci_rng'])
 indices=rng.integers(0,len(values),size=(reg['batch_ci_replicates'],len(values)))
 return np.quantile(values[indices].mean(axis=1),[.025,.975]).tolist()

def interval_matches(actual,expected):
 if expected[0] is None:return actual is not None and list(actual)==[None,None]
 return actual is not None and len(actual)==2 and np.isfinite(actual).all() and np.max(abs(np.asarray(actual)-expected))<=1e-12

def audit(registration):
 reg=json.loads(Path(registration).read_text());root=Path(reg['output']);errors=[];index={};eligibility={};batches=[]
 for b in reg['batches']:
  d=Path(b['results']);m=json.loads((d/'manifest.json').read_text());v=json.loads((d/'verification.json').read_text());source=json.loads((Path(b['source'])/'verification.json').read_text())
  if m['status']!='complete' or not v['passed'] or not source['passed']:errors.append('source/batch incomplete or unverified')
  for name,h in reg['source_sha256'].items():
   if hashlib.sha256((d/'source_snapshot'/name).read_bytes()).hexdigest()!=h:errors.append('snapshot mismatch')
  rows=list(map(json.loads,(d/'results.jsonl').read_text().splitlines()));keys=[(r['scenario'],r['seed'],r['method']) for r in rows];expected=set(itertools.product(SCENARIOS,b['seeds'],METHODS))
  if len(keys)!=len(set(keys)) or set(keys)!=expected:errors.append('missing/duplicate conditions')
  for r,k in zip(rows,keys):
   if k in index:errors.append('cross-batch duplicate')
   index[k]=r
   if r['status']=='valid':
    x=np.asarray(r['X'],float);a=np.asarray(r['alpha'],float)
    if x.shape!=(2,) or not np.isfinite(x).all() or np.max(abs(x))>1+1e-8:errors.append('invalid delivery X')
    if a.shape!=(3,) or not np.isfinite(a).all() or np.min(a)<0 or abs(a.sum()-1)>1e-8:errors.append('invalid saved alpha')
    if any(r.get(f) is None or not np.isfinite(r[f]) or r[f]<0 for f in ['pooled_error','truth_error']):errors.append('invalid saved score')
   elif r['status'] not in ['qualification_failure','preparation_failure','solver_failure','invalid_output']:errors.append('unexpected program failure')
   elif r.get('pooled_error') is not None:errors.append('failure scored as success')
   if r['status'] not in ['qualification_failure','preparation_failure']:
    for f in ['initialization_requests','initialization_input_samples','initialization_seconds','stage3_requests','stage3_input_samples','stage3_seconds']:
     if f not in r or not np.isfinite(r[f]) or r[f]<0:errors.append('missing consumed method cost')
    cost=r['support_independent_cost'];expected_cost=6 if r['method'] in ['peer_support_shrink','target_peer_shrink'] else 0
    if cost['requests']!=expected_cost or cost['input_samples']!=expected_cost:errors.append('support cost misallocation')
  sr=list(map(json.loads,(d/'shared.jsonl').read_text().splitlines()));sk=[(r['scenario'],r['seed']) for r in sr]
  if len(sk)!=len(set(sk)) or set(sk)!=set(itertools.product(SCENARIOS,b['seeds'])):errors.append('common cost/eligibility coverage')
  for r in sr:
   k=(r['scenario'],r['seed']);eligibility[k]=r['eligibility']=='eligible'
   if not eligibility[k] and any(index[k+(m,)]['status'] not in ['qualification_failure','preparation_failure'] for m in METHODS):errors.append('shared exclusion differs across methods')
   if eligibility[k]:
    if r['preparation']['coalition_ids']!=[1,2,20]:errors.append('fixed IDs changed')
    for name,h in r['sha256'].items():
     if hashlib.sha256((Path(r['artifact'])/name).read_bytes()).hexdigest()!=h:errors.append('artifact changed')
  batches.append({'batch':b['number'],'saved':len(rows),'valid':sum(r['status']=='valid' for r in rows),'verification':v})
 analysis=json.loads((root/'formal_analysis.json').read_text());planned=[s for b in reg['batches'] for s in b['seeds']];eligible=[s for s in planned if all(eligibility[k,s] for k in SCENARIOS)]
 if len(set(planned))!=len(planned) or len(reg['batches'])!=3:errors.append('not three independent batches')
 if len(analysis['comparisons'])!=28 or {(c['endpoint'],c['method'],c['baseline']) for c in analysis['comparisons']}!=set((e,m,b) for e in ['equal_scenario_mean']+SCENARIOS for m,b in CONTRASTS):errors.append('comparison coverage')
 for c in analysis['comparisons']:
  sc=SCENARIOS if c['endpoint']=='equal_scenario_mean' else [c['endpoint']];ss=[s for s in planned if all(eligibility[k,s] for k in sc)];ds=[]
  for seed in ss:
   if all(index[k,seed,c['method']]['status']==index[k,seed,c['baseline']]['status']=='valid' for k in sc):ds.append(np.mean([index[k,seed,c['method']]['pooled_error']-index[k,seed,c['baseline']]['pooled_error'] for k in sc]))
  if len(ds)!=c['n'] or len(ss)!=c['eligible_n'] or ds and abs(np.mean(ds)-c['mean_delta'])>1e-12:errors.append('paired raw mean mismatch')
  if not interval_matches(c['CI95'],ordinary_interval(ds,reg)):errors.append('ordinary CI mismatch')
  if len(c['batches'])!=len(reg['batches']):errors.append('batch inference missing')
  for bi,batch in enumerate(reg['batches']):
   bd=[]
   for seed in batch['seeds']:
    if seed in ss and all(index[k,seed,c['method']]['status']==index[k,seed,c['baseline']]['status']=='valid' for k in sc):
     bd.append(np.mean([index[k,seed,c['method']]['pooled_error']-index[k,seed,c['baseline']]['pooled_error'] for k in sc]))
   if bi>=len(c['batches']):continue
   saved=c['batches'][bi]
   if saved['batch']!=bi or saved['n']!=len(bd):errors.append('batch population mismatch')
   if bd and (saved['mean_delta'] is None or abs(np.mean(bd)-saved['mean_delta'])>1e-12):errors.append('batch raw mean mismatch')
   if not bd and saved['mean_delta'] is not None:errors.append('empty batch scored')
   if not interval_matches(saved['CI95'],ordinary_interval(bd,reg)):errors.append('batch ordinary CI mismatch')
  ci=c['CI_simultaneous'];bs=c['batches'];should=bool(c['endpoint']=='equal_scenario_mean' and len(ds)==len(ss)>0 and ci is not None and ci[1]<0 and len(bs)==3 and all(b['mean_delta'] is not None and b['mean_delta']<0 for b in bs) and sum(b['CI95'][1] is not None and b['CI95'][1]<0 for b in bs)>=2)
  if c['stable_advantage']!=should:errors.append('unjustified stable flag')
 usable=[c for c in analysis['comparisons'] if c['endpoint']=='equal_scenario_mean' and c['CI_simultaneous'] is not None];quantile_max=0.
 if usable:
  samples=np.load(root/'bootstrap_means.npy',mmap_mode='r');tail=.05/(2*7)
  if samples.shape!=(reg['bootstrap_replicates'],len(usable)):errors.append('bootstrap dimensions')
  else:
   for j,c in enumerate(usable):quantile_max=max(quantile_max,float(np.max(abs(np.quantile(samples[:,j],[tail,1-tail])-c['CI_simultaneous']))))
  if quantile_max>1e-12:errors.append('saved corrected CI mismatch')
 for g in analysis['groups']:
  sc=SCENARIOS if g['endpoint']=='equal_scenario_mean' else [g['endpoint']];ss=[s for s in planned if all(eligibility[k,s] for k in sc)];vals=[np.mean([index[k,s,g['method']]['pooled_error'] for k in sc]) for s in ss if all(index[k,s,g['method']]['status']=='valid' for k in sc)]
  if len(vals)!=g['valid_n'] or len(ss)!=g['eligible_n']:errors.append('group population mismatch')
  if vals:
   calculated=[np.mean(vals),np.median(vals),*np.quantile(vals,[.25,.75])]
   if max(abs(np.array(calculated)-np.array([g['mean'],g['median'],g['Q1'],g['Q3']])))>1e-12:errors.append('group summaries mismatch')
 changed=[p for p,h in reg['source_sha256'].items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
 if changed:errors.append('scientific source changed')
 result={'passed':not errors,'errors':errors,'batches':batches,'all_conditions':len(index),'expected_conditions':reg['expected_conditions'],'common_eligible_n':len(eligible),'all_methods_retained':len(METHODS)==6,'quantile_max_difference':quantile_max,'frozen_sources_unchanged':not changed,'qualification_failures_retained':True,'formal_stable_comparisons':[(c['method'],c['baseline']) for c in analysis['comparisons'] if c['stable_advantage']]}
 (root/'completion_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='batches'},ensure_ascii=False));return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('registration');a=p.parse_args()
 if not audit(a.registration)['passed']:raise SystemExit(1)
