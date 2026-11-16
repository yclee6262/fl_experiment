import argparse,csv,hashlib,json,sys,time,shutil
from pathlib import Path
import numpy as np
from kernel import bank,select,quality,NAMES
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'host_server.py').is_file())
def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v) for v in x]
 return x
def save(p,x):
 with Path(p).open('x') as f:json.dump(plain(x),f,indent=2,allow_nan=False)
def csvsave(p,rs):
 with Path(p).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def prepare(out):
 old=ROOT/'contribution_analysis/custom_direction_experiment_v2_20261020';cases=read(old/'cases.json')['development'];out.mkdir();(out/'records').mkdir();(out/'source_snapshot').mkdir()
 inputs={str(old/'participant_comparison.csv'):digest(old/'participant_comparison.csv')}
 for c in cases:
  for p in [Path(c['models']),Path(c['evaluator']),old/'baselines'/(c['key']+'.json')]:inputs[str(p)]=digest(p)
 for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,out/'source_snapshot'/p.name)
 save(out/'cases.json',cases);save(out/'input_sha256.json',inputs)
 plan=ROOT/'docs/contribution_search_plan_20261020.md';shutil.copy2(plan,out/plan.name)
 save(out/'registration.json',dict(old=str(old),methods=NAMES,source_sha256={p.name:digest(p) for p in (out/'source_snapshot').glob('*.py')},plan_sha256=digest(plan),development_cases=20,independent_confirmation=False,bootstrap_seed=20261020,bootstrap_repeats=10000,minimum_mean_spearman=.5,minimum_coverage=.9,max_new_samples=10000,max_seconds=300))
def check(out):
 cfg=read(out/'registration.json')
 for n,h in cfg['source_sha256'].items():assert digest(out/'source_snapshot'/n)==h
 for n,h in read(out/'input_sha256.json').items():assert digest(n)==h
 return cfg

def score(out):
 import torch
 torch.set_num_threads(1);cfg=check(out);old=Path(cfg['old'])
 sys.path.insert(0,str(ROOT/'contribution_analysis/custom_direction_v1'))
 from experiment import load_server
 from observer import solve
 total=dict(replay_calls=0,replay_samples=0,replay_solves=0,new_score_calls=0,new_score_samples=0,score_seconds=0.,load_seconds=0.)
 begin=time.perf_counter()
 for c in read(out/'cases.json'):
  print('SCORE',c['key'],flush=True);rec=read(old/'baselines'/(c['key']+'.json'));p,l=bank(rec)
  start=time.perf_counter();server=load_server(c);total['load_seconds']+=time.perf_counter()-start
  replay=solve(server,True,c['fixed_start']);total['replay_solves']+=1;total['replay_calls']+=replay['counter']['calls'];total['replay_samples']+=replay['counter']['samples']
  if replay['query_order_sha256']!=rec['query_order_sha256'] or replay['result']['history']!=rec['result']['history']:raise ValueError('baseline_replay_drift')
  pp,ll=bank(replay)
  if not np.array_equal(p,pp) or not np.array_equal(l,ll):raise ValueError('bank_replay_drift')
  start=time.perf_counter();cache={};calls=0
  def evaluate(x):
   nonlocal calls
   key=np.asarray(x,dtype=np.float32).tobytes()
   if key not in cache:
    cache[key]=np.array([float(a.api_predict(x)[0]) for a in server.trusted_agents]);calls+=len(server.trusted_agents)
   return cache[key]
  x0=np.array(rec['result']['solution']);q0=quality(evaluate(x0));rows=[]
  for k,i in enumerate(c['ids']):
   j,x,w=select(p,l,c['candidates'],c['alphas'],k);pred=evaluate(x)
   if not np.allclose(abs(pred),l[j],atol=0,rtol=0):raise ValueError('bank_response_mismatch')
   q=quality(pred);rows.append(dict(agent_id=i,index=k,bank_index=j,solution=x,remaining_ids=[i2 for i2 in c['ids'] if i2!=i],remaining_weights=w[w>0],predictions=pred,scores={n:q[n]-q0[n] for n in NAMES}))
  elapsed=time.perf_counter()-start;total['score_seconds']+=elapsed;total['new_score_calls']+=calls;total['new_score_samples']+=calls
  if total['new_score_samples']>cfg['max_new_samples'] or time.perf_counter()-begin>cfg['max_seconds']:raise RuntimeError('budget')
  save(out/'records'/(c['key']+'.json'),dict(case=c,bank_points=p,bank_losses=l,baseline_solution=x0,baseline_predictions=evaluate(x0),baseline_quality=q0,participants=rows,score_calls=calls,score_samples=calls,score_seconds=elapsed,replay_exact=True,evaluator_loaded=False,truth_loaded=False))
 save(out/'score_lock.json',dict(record_sha256={p.name:digest(p) for p in (out/'records').glob('*.json')},totals=total,evaluator_loaded=False,loo_labels_loaded=False))

def analyze(out):
 from scipy.stats import spearmanr
 cfg=check(out);lock=read(out/'score_lock.json')
 for n,h in lock['record_sha256'].items():assert digest(out/'records'/n)==h
 with (Path(cfg['old'])/'participant_comparison.csv').open() as f:labels={(x['key'],int(x['agent_id'])):x for x in csv.DictReader(f)}
 rows=[];teams=[]
 for c in read(out/'cases.json'):
  rs=read(out/'records'/(c['key']+'.json'))['participants']
  for x in rs:rows.append(dict(key=c['key'],scenario=c['scenario'],agent_id=x['agent_id'],C=float(labels[c['key'],x['agent_id']]['C_custom_full']),**x['scores']))
  rr=[x for x in rows if x['key']==c['key']]
  for n in NAMES:
   x=np.array([a[n] for a in rr]);y=np.array([a['C'] for a in rr]);valid=np.ptp(x)>0 and np.ptp(y)>0
   teams.append(dict(key=c['key'],scenario=c['scenario'],method=n,spearman=float(spearmanr(x,y).statistic) if valid else None,score_range=float(np.ptp(x)),reference_range=float(np.ptp(y)),near_flat_reference=bool(np.ptp(y)<=1e-4)))
 summary=[]
 for scenario in ['clean','controlled_poison']:
  for n in NAMES:
   vals=[t['spearman'] for t in teams if t['method']==n and t['scenario']==scenario];a=np.array([np.nan if v is None else v for v in vals]);rng=np.random.default_rng(20261020);ss=a[rng.integers(0,len(a),size=(10000,len(a)))];counts=np.isfinite(ss).sum(axis=1);means=np.nansum(ss,axis=1)[counts>0]/counts[counts>0]
   summary.append(dict(scenario=scenario,method=n,mean=float(np.nanmean(a)) if np.isfinite(a).any() else None,ci=np.quantile(means,[.025,.975]).tolist() if len(means) else None,valid_teams=int(np.isfinite(a).sum()),total_teams=len(a),coverage=float(np.isfinite(a).mean()),flat_score_teams=sum(t['score_range']<=1e-4 for t in teams if t['method']==n and t['scenario']==scenario)))
 eligible=[n for n in NAMES if all(s['mean'] is not None and s['mean']>.5 and s['coverage']>=.9 for s in summary if s['method']==n)]
 selected=max(eligible,key=lambda n:min(s['mean'] for s in summary if s['method']==n)) if eligible else None
 csvsave(out/'participant_scores.csv',rows);csvsave(out/'team_metrics.csv',teams);save(out/'development_summary.json',summary);save(out/'decision.json',dict(selected_method=selected,development_passed=bool(selected),independent_validity_established=False,all_methods_reported=True))
 print(json.dumps(dict(summary=summary,selected=selected),indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','score','analyze']);p.add_argument('out',type=Path);a=p.parse_args();globals()[a.phase](a.out.resolve())
