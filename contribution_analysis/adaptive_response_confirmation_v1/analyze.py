"""Frozen offline analysis; never used to choose a solver point or modify a score."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v) for v in x]
 return x
def save(p,x):
 with p.open('x') as f:json.dump(plain(x),f,indent=2,allow_nan=False)
def csvsave(p,rs):
 with p.open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def sign(x):return int(x>1e-4)-int(x < -1e-4)
def boot(vals):
 a=np.array([np.nan if x is None else x for x in vals]);rng=np.random.default_rng(20261020);s=a[rng.integers(0,len(a),size=(10000,len(a)))];ct=np.isfinite(s).sum(axis=1);means=np.nansum(s,axis=1)[ct>0]/ct[ct>0]
 return dict(mean=float(np.nanmean(a)) if np.isfinite(a).any() else None,ci=np.quantile(means,[.025,.975]).tolist() if len(means) else None,undefined_bootstraps=int(np.sum(ct==0)))
def analyze(out):
 import torch
 from models import PretrainedModel
 torch.set_num_threads(1);cfg=read(out/'registration.json');lockp=read(out/'proxy_lock.json');lockr=read(out/'reference_lock.json');dl=read(out/'data_lock.json');sl=read(out/'score_lock.json')
 for folder,ls in [('proxies',lockp['record_sha256']),('normal',lockp['normal_sha256']),('references',lockr['record_sha256']),('scores',sl['record_sha256'])]:
  for n,h in ls.items():assert sha(out/folder/n)==h
 for n,h in dl['input_sha256'].items():assert sha(Path(n))==h
 rows=[];teams=[];evaluations=[];poolpoints=0
 for c in read(out/'cases.json'):
  print('OFFLINE',c['key'],flush=True);model=PretrainedModel(2);model.load_state_dict(torch.load(c['evaluator'],map_location='cpu',weights_only=True)['state_dict']);model.eval()
  def score(x,kind,i):
   nonlocal poolpoints
   with torch.no_grad():pred=float(model(torch.tensor(x,dtype=torch.float32)[None,:]).item())
   poolpoints+=1;evaluations.append(dict(key=c['key'],kind=kind,agent_id=i,x=x,prediction=pred,error=abs(pred)));return abs(pred)
  base=read(out/'normal'/(c['key']+'.json'));E0=score(base['result']['solution'],'baseline',None);pr=read(out/'proxies'/(c['key']+'.json'))['participants'];scores=read(out/'scores'/(c['key']+'.json'))['participants'];refs=read(out/'references'/(c['key']+'.json'))['participants'];sr=[]
  for p,r in zip(pr,refs):
   if p['agent_id']!=r['agent_id']:raise ValueError('role_pair_mismatch')
   F=score(r['result']['solution'],'reference',p['agent_id']);C=F-E0;A=next(a['A'] for a in scores if a['agent_id']==p['agent_id']);E=None
   row=dict(key=c['key'],scenario=c['scenario'],seed=c['seed'],agent_id=p['agent_id'],A=A,C_LOO=C,baseline_error=E0,loo_error=F,score_reference_abs_difference=abs(A-C),endpoint_distance=float(np.linalg.norm(np.array(p['result']['solution'])-r['result']['solution'])),direction_agreement=sign(A)==sign(C));rows.append(row);sr.append(row)
  x=np.array([r['A'] for r in sr]);y=np.array([r['C_LOO'] for r in sr]);valid=np.ptp(x)>0 and np.ptp(y)>0
  teams.append(dict(key=c['key'],scenario=c['scenario'],seed=c['seed'],n=len(sr),status='available' if valid else 'exact_tie',spearman=float(spearmanr(x,y).statistic) if valid else None,proxy_range=float(np.ptp(x)),reference_range=float(np.ptp(y)),near_flat_reference=bool(np.ptp(y)<=1e-4),near_flat_proxy=bool(np.ptp(x)<=1e-4),direction_agreement=float(np.mean([r['direction_agreement'] for r in sr])),C_MAE=float(np.mean(abs(x-y)))))
 # Include registered qualification failures in coverage denominator and bootstrap slots.
 for seed in cfg['seeds']:
  for scenario in cfg['scenarios']:
   if not any(t['seed']==seed and t['scenario']==scenario for t in teams):teams.append(dict(key=f'{scenario}_{seed}',scenario=scenario,seed=seed,n=0,status='preparation_or_qualification_failed',spearman=None,proxy_range=None,reference_range=None,near_flat_reference=None,near_flat_proxy=None,direction_agreement=None,C_MAE=None))
 summary=[]
 for scenario in cfg['scenarios']:
  ts=sorted([t for t in teams if t['scenario']==scenario],key=lambda t:t['seed']);vals=[t['spearman'] for t in ts];a=boot(vals);coverage=sum(v is not None for v in vals)/20
  passed=a['mean'] is not None and a['mean']>.5 and a['ci'][0]>0 and coverage>=.9
  summary.append(dict(scenario=scenario,registered_teams=20,valid_teams=sum(v is not None for v in vals),coverage=coverage,**a,quality_gate_passed=passed,lower_bound_above_half=bool(a['ci'] and a['ci'][0]>.5),mean_direction_agreement=float(np.mean([t['direction_agreement'] for t in ts if t['direction_agreement'] is not None])),near_flat_reference_teams=sum(t['near_flat_reference'] is True for t in ts)))
 normal=lockp['totals']['normal_samples'];extra=lockp['totals']['probe_samples'];ref=lockr['totals']['samples'];ratio=(normal+extra)/(normal+ref)
 cost=dict(normal_samples=normal,additional_probe_samples=extra,additional_loo_samples=ref,normal_plus_proxy_samples=normal+extra,normal_plus_loo_samples=normal+ref,sample_ratio=ratio,saving=1-ratio,sample_gate_passed=ratio<=.7,offline_evaluator_points=poolpoints,time_gate_established=False,client_observations_processed=sl['client_observations_processed'],client_regression_degree=2,client_provider_count=3)
 save(out/'pooled_evaluations.json',evaluations);csvsave(out/'participant_comparison.csv',rows);csvsave(out/'team_metrics.csv',teams);save(out/'summary.json',summary);save(out/'cost.json',cost);save(out/'decision.json',dict(quality_passed=all(s['quality_gate_passed'] for s in summary),sample_cost_passed=ratio<=.7,global_free_quality_gate_passed=all(s['quality_gate_passed'] for s in summary),strict_nonLOO_method=False,all_cost_gates_passed=False,scope='Fixed global-free score from client summaries and surrogate removal; pooled only scores baseline and real LOO reference. Runtime and data-access boundaries still apply.'))
 print(json.dumps(dict(summary=summary,cost=cost),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();analyze(a.out.resolve())
