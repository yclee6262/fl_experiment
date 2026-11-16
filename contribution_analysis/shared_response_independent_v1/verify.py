import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(out):
 import torch
 from models import PretrainedModel
 from host_server import HostServer
 from core import Proxy,simulate
 torch.set_num_threads(1);cfg=read(out/'registration.json');checks={};root=next(p for p in out.parents if (p/'host_server.py').is_file())
 checks['frozen_solver_sources']=all(sha(out/'source_snapshot'/n)==h for n,h in cfg['source_sha256'].items());checks['historical_sources_unchanged']=all(sha(root/n)==h for n,h in cfg['historical_sha256'].items());checks['frozen_plan']=sha(out/'shared_response_independent_preregistration_20261020.md')==cfg['plan_sha256'];ar=read(out/'analysis_registration.json');checks['analysis_frozen_before_outcomes']=not ar['proxy_or_reference_outcomes_exist'] and all(sha(out/'analysis_snapshot'/n)==h for n,h in ar['source_sha256'].items())
 dl=read(out/'data_lock.json');pl=read(out/'proxy_lock.json');rl=read(out/'reference_lock.json');checks['all_data_hashes']=all(sha(Path(n))==h for n,h in dl['input_sha256'].items());checks['all_solution_locks']=all(sha(out/folder/n)==h for folder,lock in [('normal',pl['normal_sha256']),('proxies',pl['record_sha256']),('references',rl['record_sha256'])] for n,h in lock.items());cases=read(out/'cases.json');checks['all_40_registered_cases']=len(cases)==40 and set((c['seed'],c['scenario']) for c in cases)==set((seed,scenario) for seed in cfg['seeds'] for scenario in cfg['scenarios']);checks['new_data_unique']=len(set(c['semantic_data_sha256'] for c in cases))==len(cases)
 old=root/'contribution_analysis/full_removal_budget_experiment_v1_20261016';oldhash={read(p).get('semantic_data_sha256') for p in old.glob('*/*/*/preparation.json')};checks['new_data_disjoint_from_development']=all(c['semantic_data_sha256'] not in oldhash for c in cases)
 with (out/'participant_comparison.csv').open() as f:rows=list(csv.DictReader(f))
 with (out/'team_metrics.csv').open() as f:teams=list(csv.DictReader(f))
 checks['roles_and_feasible_endpoints']=True;checks['all_160_proxy_replays']=True;checks['raw_pooled_rebuild']=True;max_pool=0.;max_x=0.;count=0;pool_points=0;recalc={}
 for c in cases:
  print('VERIFY',c['key'],flush=True);base=read(out/'normal'/(c['key']+'.json'));p=read(out/'proxies'/(c['key']+'.json'));rs=read(out/'references'/(c['key']+'.json'))['participants'];proxy=Proxy(np.array(p['bank_points']),np.array(p['bank_values']),'linear');server=HostServer(0.,2,constraint_mode='box_constrained');model=PretrainedModel(2);model.load_state_dict(torch.load(c['evaluator'],map_location='cpu',weights_only=True)['state_dict']);model.eval()
  def score(x):
   nonlocal pool_points
   pool_points+=1
   with torch.no_grad():return abs(float(model(torch.tensor(x,dtype=torch.float32)[None,:]).item()))
  E0=score(base['result']['solution']);a=[];b=[]
  for k,(pr,rr) in enumerate(zip(p['participants'],rs)):
   ids=[i for j,i in enumerate(c['ids']) if j!=k];w=np.array([v for j,v in enumerate(c['alphas']) if j!=k]);w=w/w.sum();checks['roles_and_feasible_endpoints'] &= pr['agent_id']==rr['agent_id']==c['ids'][k] and pr['remaining_ids']==rr['remaining_ids']==ids and bool(np.allclose(pr['remaining_weights'],w,rtol=0,atol=1e-15)) and bool(np.allclose(rr['remaining_weights'],w,rtol=0,atol=1e-15)) and max(abs(np.array(pr['result']['solution'])))<=1+1e-8 and max(abs(np.array(rr['result']['solution'])))<=1+1e-8
   replay=simulate(server,proxy,c,k);delta=float(np.max(abs(np.array(replay['result']['solution'])-pr['result']['solution'])));max_x=max(max_x,delta);checks['all_160_proxy_replays'] &= delta<1e-12 and replay['result']['history']==pr['result']['history'];count+=1
   A=score(pr['result']['solution'])-E0;C=score(rr['result']['solution'])-E0;r=next(r for r in rows if r['key']==c['key'] and int(r['agent_id'])==pr['agent_id']);dp=max(abs(A-float(r['C_proxy_offline'])),abs(C-float(r['C_LOO'])));max_pool=max(max_pool,dp);checks['raw_pooled_rebuild'] &= dp<1e-12;a.append(A);b.append(C)
  recalc[c['key']]=None if np.ptp(a)==0 or np.ptp(b)==0 else float(spearmanr(a,b).statistic)
 checks['all_40_team_correlations']=all((t['spearman']=='' and recalc[t['key']] is None) or(t['spearman']!='' and recalc[t['key']] is not None and abs(float(t['spearman'])-recalc[t['key']])<1e-12) for t in teams)
 checks['both_seed_bootstrap_summaries']=True
 for s in read(out/'summary.json'):
  vals=np.array([np.nan if recalc[f"{s['scenario']}_{seed}"] is None else recalc[f"{s['scenario']}_{seed}"] for seed in cfg['seeds']]);rng=np.random.default_rng(20261020);ss=vals[rng.integers(0,20,size=(10000,20))];ct=np.isfinite(ss).sum(axis=1);bs=np.nansum(ss,axis=1)[ct>0]/ct[ct>0];checks['both_seed_bootstrap_summaries'] &= abs(np.nanmean(vals)-s['mean'])<1e-12 and bool(np.allclose(np.quantile(bs,[.025,.975]),s['ci'],rtol=0,atol=1e-12))
 cost=read(out/'cost.json');checks['cost_and_budget']=cost['normal_samples']==pl['totals']['normal_samples'] and cost['additional_probe_samples']==pl['totals']['probe_samples'] and cost['additional_loo_samples']==rl['totals']['samples'] and abs(cost['sample_ratio']-(cost['normal_samples']+cost['additional_probe_samples'])/(cost['normal_samples']+cost['additional_loo_samples']))<1e-15 and cost['normal_samples']+cost['additional_probe_samples']+cost['additional_loo_samples']<=cfg['max_samples'] and dl['preparation_wall_seconds']<cfg['data_seconds_budget']
 checks={k:bool(v) for k,v in checks.items()};result=dict(accepted=all(checks.values()),checks=checks,replayed_proxies=count,max_proxy_endpoint_delta=max_x,max_pooled_delta=max_pool,offline_verification_evaluator_points=pool_points,new_participant_model_queries=0,scope='Integrity only; Clean quality gate failed, overall objective not achieved.')
 with (out/'verification.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result,indent=2));assert result['accepted']
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();sys.path.insert(0,str(a.out.resolve()/'source_snapshot'));run(a.out.resolve())
