"""Independent reconstruction of client summaries, adaptive probes, references and statistics."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'docs/contribution_analysis_research.md').is_file())
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def boot(v):
 a=np.array([np.nan if x is None else x for x in v]);rng=np.random.default_rng(20261020);s=a[rng.integers(0,len(a),size=(10000,len(a)))];ct=np.isfinite(s).sum(axis=1);b=np.nansum(s,axis=1)[ct>0]/ct[ct>0];return float(np.nanmean(a)),np.quantile(b,[.025,.975])
def independent_client_fit(X,y,center):
 distances=np.sqrt(np.sum((X-center)**2,axis=1));idx=np.argsort(distances,kind='stable')[:128];h=max(float(distances[idx[-1]]),1e-6);u=(X[idx]-center)/h;phi=np.array([[1,a,b,a*a,a*b,b*b] for a,b in u]);w=np.exp(-.5*(distances[idx]/h)**2);G=phi.T@(w[:,None]*phi);lam=1e-6*np.diag(G)[1:].mean();pen=np.diag([0,lam,lam,lam,lam,lam]);beta=np.linalg.solve(G+pen,phi.T@(w*y[idx]));return idx,h,beta

def verify(out):
 import torch
 from models import PretrainedModel
 from experiment import load_server
 from observer import solve
 torch.set_num_threads(1)
 sys.path.insert(0,str(ROOT/'contribution_analysis/adaptive_shared_response_v1'))
 from verify_fast_proxy_v2 import replay
 from fast_proxy import FastProxy
 from proxy_core import Proxy
 cfg=read(out/'registration.json');dl=read(out/'data_lock.json');pl=read(out/'proxy_lock.json');rl=read(out/'reference_lock.json');sl=read(out/'score_lock.json');cases=read(out/'cases.json');checks={};counts=dict(real_baseline_replays=0,real_removal_replays=0,verification_model_samples=0,pooled_points=0,adaptive_team_replays=0);maxA=0.;maxC=0.;max_beta=0.;max_endpoint=0.
 checks['historical_algorithms_unchanged']=all(sha(ROOT/n)==h for n,h in cfg['historical_sha256'].items());checks['frozen_source_snapshot']=all(sha(out/'source_snapshot'/n)==h for n,h in cfg['source_sha256'].items());checks['frozen_preregistration']=sha(out/'adaptive_response_confirmation_preregistration_20261020.md')==cfg['plan_sha256'];ar=read(out/'analysis_registration.json');checks['analysis_frozen_before_outcomes']=not ar['proxy_or_reference_outcomes_exist'] and all(sha(out/'analysis_snapshot'/n)==h for n,h in ar['source_sha256'].items());checks['all_input_hashes']=all(sha(Path(n))==h for n,h in dl['input_sha256'].items());checks['all_output_locks']=all(sha(out/folder/n)==h for folder,lock in [('normal',pl['normal_sha256']),('proxies',pl['record_sha256']),('references',rl['record_sha256']),('scores',sl['record_sha256'])] for n,h in lock.items())
 checks['all_registered_cases']=len(cases)==40 and set((c['seed'],c['scenario']) for c in cases)==set((s,t) for s in cfg['seeds'] for t in cfg['scenarios']);checks['unique_semantic_data']=len({c['semantic_data_sha256'] for c in cases})==40
 oldhash=set()
 for parent,pattern in [(ROOT/'contribution_analysis/full_removal_budget_experiment_v1_20261016','*/*/*/preparation.json'),(ROOT/'contribution_analysis/shared_response_independent_experiment_v1_20261020/data','*/preparation.json')]:
  for p in parent.glob(pattern):oldhash.add(read(p).get('semantic_data_sha256'))
 checks['independent_of_all_previous_study_data']=all(c['semantic_data_sha256'] not in oldhash for c in cases);checks['technical_retries_before_data_only']=True
 for f in out.glob('recovery_registration*.json'):
  r=read(f);p=Path(r['failed_log']);checks['technical_retries_before_data_only'] &= sha(p)==r['failed_log_sha256'] and 'OMP: Error #179' in p.read_text() and sorted(x.name for x in p.parent.iterdir())==['prepare.log'] and r['same_seed_retry'] and not r['method_or_sample_selection_changed']
 with (out/'participant_comparison.csv').open() as f:rows=list(csv.DictReader(f))
 with (out/'team_metrics.csv').open() as f:teams=list(csv.DictReader(f))
 for name in ['all_adaptive_probes_and_paths','all_true_baseline_and_removal_replays','all_roles_weights_and_starts','all_current_client_summaries','all_scores_independent_of_pooled','all_reference_pooled_rebuilds']:checks[name]=True
 corrs={}
 for c in cases:
  print('VERIFY',c['key'],flush=True);rec=read(out/'records'/(c['key']+'.json'));normal=read(out/'normal'/(c['key']+'.json'));proxy=read(out/'proxies'/(c['key']+'.json'))['participants'];refs=read(out/'references'/(c['key']+'.json'))['participants'];sc=read(out/'scores'/(c['key']+'.json'));queries,delta=replay(rec,Proxy,False);optimized_queries,optimized_delta=replay(rec,FastProxy,True);assert queries==optimized_queries and optimized_delta<1e-12;max_endpoint=max(max_endpoint,delta);checks['all_adaptive_probes_and_paths'] &= delta<1e-12 and len(queries)*len(c['ids'])==rec['new_probe_samples'];counts['adaptive_team_replays']+=1
  base_replay=solve(load_server(c),False,c['fixed_start']);counts['real_baseline_replays']+=1;counts['verification_model_samples']+=base_replay['counter']['samples'];checks['all_true_baseline_and_removal_replays'] &= base_replay['query_order_sha256']==normal['query_order_sha256'] and base_replay['counter']==normal['counter'] and base_replay['result']['history']==normal['result']['history'] and bool(np.array_equal(base_replay['result']['solution'],normal['result']['solution']))
  meta=read(Path(c['models']).parents[2]/'preparation.json');provider_ids=[t['agent_id'] for t in sorted(meta['phase1_report'],key=lambda t:(t['mse'],t['agent_id']))[:3]];checks['all_current_client_summaries'] &= provider_ids==sc['provider_ids'];base=np.array(normal['result']['solution']);coeff=[]
  with np.load(Path(c['models']).parent/'pooled_data.npz') as data:X=data['X'];y=data['observed_y'];owners=data['agent_ids']
  for i,s in zip(provider_ids,sc['provider_summaries']):
   idx,h,beta=independent_client_fit(X[owners==i],y[owners==i],base);max_beta=max(max_beta,float(np.max(abs(beta-s['beta']))));checks['all_current_client_summaries'] &= idx.tolist()==s['indices'] and abs(h-s['h'])<1e-12 and bool(np.allclose(beta,s['beta'],atol=1e-10,rtol=0));coeff.append((h,beta))
  def Q(x):
   ps=[]
   for h,beta in coeff:
    a,b=(np.array(x)-base)/h;ps.append(float(np.array([1,a,b,a*a,a*b,b*b])@beta))
   return float(np.median(abs(np.array(ps))))
  q0=Q(base);model=PretrainedModel(2);model.load_state_dict(torch.load(c['evaluator'],map_location='cpu',weights_only=True)['state_dict']);model.eval()
  def E(x):
   counts['pooled_points']+=1
   with torch.no_grad():return abs(float(model(torch.tensor(x,dtype=torch.float32)[None,:]).item()))
  e0=E(base);aa=[];cc=[]
  for k,(p,r,s) in enumerate(zip(proxy,refs,sc['participants'])):
   ids=[i for j,i in enumerate(c['ids']) if j!=k];w=np.array([v for j,v in enumerate(c['alphas']) if j!=k]);w/=w.sum();checks['all_roles_weights_and_starts'] &= p['agent_id']==r['agent_id']==s['agent_id']==c['ids'][k] and p['remaining_ids']==r['remaining_ids']==ids and bool(np.allclose(p['remaining_weights'],w,atol=1e-15,rtol=0)) and bool(np.allclose(r['remaining_weights'],w,atol=1e-15,rtol=0))
   repeat=solve(load_server(c,k),False,c['fixed_start']);counts['real_removal_replays']+=1;counts['verification_model_samples']+=repeat['counter']['samples'];checks['all_true_baseline_and_removal_replays'] &= repeat['query_order_sha256']==r['query_order_sha256'] and repeat['counter']==r['counter'] and repeat['result']['history']==r['result']['history'] and bool(np.array_equal(repeat['result']['solution'],r['result']['solution']))
   A=Q(s['solution'])-q0;C=E(r['result']['solution'])-e0;row=next(row for row in rows if row['key']==c['key'] and int(row['agent_id'])==p['agent_id']);maxA=max(maxA,abs(A-s['A']),abs(A-float(row['A'])));maxC=max(maxC,abs(C-float(row['C_LOO'])));checks['all_scores_independent_of_pooled'] &= abs(A-s['A'])<1e-10 and abs(A-float(row['A']))<1e-10 and sc['pooled_loaded'] is False and sc['truth_loaded'] is False;checks['all_reference_pooled_rebuilds'] &= abs(C-float(row['C_LOO']))<1e-12;aa.append(A);cc.append(C)
  corrs[c['key']]=None if np.ptp(aa)==0 or np.ptp(cc)==0 else float(spearmanr(aa,cc).statistic)
 checks['all_40_team_correlations']=all((t['spearman']=='' and corrs[t['key']] is None) or(t['spearman']!='' and corrs[t['key']] is not None and abs(float(t['spearman'])-corrs[t['key']])<1e-12) for t in teams);checks['both_summaries_and_bootstraps']=True
 for s in read(out/'summary.json'):
  vals=[corrs[f"{s['scenario']}_{seed}"] for seed in cfg['seeds']];mean,ci=boot(vals);checks['both_summaries_and_bootstraps'] &= abs(mean-s['mean'])<1e-12 and bool(np.allclose(ci,s['ci'],rtol=0,atol=1e-12)) and s['quality_gate_passed']==(mean>.5 and ci[0]>0 and np.isfinite([np.nan if v is None else v for v in vals]).mean()>=.9)
 cost=read(out/'cost.json');checks['cost_and_budgets']=cost['normal_samples']==pl['totals']['normal_samples'] and cost['additional_probe_samples']==pl['totals']['probe_samples'] and cost['additional_loo_samples']==rl['totals']['samples'] and abs(cost['sample_ratio']-(cost['normal_samples']+cost['additional_probe_samples'])/(cost['normal_samples']+cost['additional_loo_samples']))<1e-15 and sl['client_observations_processed']==40*384 and sum([cost['normal_samples'],cost['additional_probe_samples'],cost['additional_loo_samples']])+counts['verification_model_samples']<=cfg['max_samples'] and dl['preparation_wall_seconds']<=cfg['data_seconds_budget']
 checks={k:bool(v) for k,v in checks.items()};result=dict(accepted=all(checks.values()),checks=checks,counts=counts,max_score_rebuild_delta=maxA,max_reference_pooled_delta=maxC,max_client_coefficient_delta=max_beta,max_cached_proxy_endpoint_delta=max_endpoint,quality_scope='Two independent sample means >0.5; CI lower bounds do not establish population correlation >0.5.',all_actual_reference_replays_equal=True)
 save=out/'verification.json'
 with save.open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result,indent=2));assert result['accepted']
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();sys.path.insert(0,str(a.out.resolve()/'source_snapshot'));verify(a.out.resolve())
