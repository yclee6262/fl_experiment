import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'docs/contribution_analysis_research.md').is_file())
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v) for v in x]
 return x
def save(p,x):
 with Path(p).open('x') as f:json.dump(plain(x),f,indent=2,allow_nan=False)
def prepare(out):
 out.mkdir()
 for name in ['source_snapshot','data','normal','proxies','references','records','scores']:(out/name).mkdir()
 snap=out/'source_snapshot';origin=ROOT/'contribution_analysis/adaptive_shared_response_experiment_v2_20261020/source_snapshot'
 for n in ['proxy_core.py','local_core.py']:shutil.copy2(origin/n,snap/n)
 adaptive=(origin/'study.py').read_text().replace("CONFIGS=['linear','rbf']", "CONFIGS=['rbf']")
 adaptive=adaptive.replace(";old=read(Path(cfg['old'])/'baselines'/(c['key']+'.json'))", "")
 adaptive=adaptive.replace("  if normal['query_order_sha256']!=old['query_order_sha256'] or normal['result']['history']!=old['result']['history']:raise ValueError('normal_invariance')\n", "")
 (snap/'adaptive_kernel.py').write_text(adaptive)
 for n in ['observer.py','experiment.py','common.py']:shutil.copy2(ROOT/'contribution_analysis/custom_direction_v1'/n,snap/n)
 shutil.copy2(ROOT/'contribution_analysis/full_removal_budget_v1/worker.py',snap/'worker.py')
 hist=['host_server.py','agent_client.py','models.py','dataset.py','alpha_weight_experiments.py','p3_contribution_validation.py','research_evaluation.py','run_experiments.py']
 for n in hist:shutil.copy2(ROOT/n,snap/n)
 for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,snap/p.name)
 doc=ROOT/'docs/adaptive_response_confirmation_preregistration_20261020.md';shutil.copy2(doc,out/doc.name)
 for n in ['adaptive_shared_response_preregistration_20261020.md','local_response_panel_preregistration_20261020.md']:shutil.copy2(ROOT/'docs'/n,out/n)
 save(out/'registration.json',dict(seeds=list(range(2026102300,2026102320)),scenarios=['clean','controlled_poison'],selected=dict(config='rbf',method='degree2_median_abs'),historical_sha256={n:sha(ROOT/n) for n in hist},source_sha256={p.name:sha(p) for p in snap.glob('*.py')},plan_sha256=sha(doc),data_seconds_budget=1800,max_samples=500000,max_new_samples=20000,max_proxy_solves=800,max_seconds=600,max_solver_seconds=600,started_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),global_free_score=True,requires_client_observation_summary=True,strict_nonLOO=False))

def check(out):
 c=read(out/'registration.json')
 for n,h in c['source_sha256'].items():assert sha(out/'source_snapshot'/n)==h
 for n,h in c['historical_sha256'].items():assert sha(ROOT/n)==h
 return c

def data(out):
 cfg=check(out);cases=[];events=[];elapsed=0.;previous=set()
 old=ROOT/'contribution_analysis/full_removal_budget_experiment_v1_20261016'
 for p in list(old.glob('*/*/*/preparation.json'))+list((ROOT/'contribution_analysis/shared_response_independent_experiment_v1_20261020/data').glob('*/preparation.json')):
  a=read(p)
  if a.get('semantic_data_sha256'):previous.add(a['semantic_data_sha256'])
 for seed in cfg['seeds']:
  for scenario in cfg['scenarios']:
   name=f'{scenario}_{seed}';d=out/'data'/name;d.mkdir();print('PREPARE',name,flush=True);begin=time.perf_counter()
   cmd=[sys.executable,str(out/'source_snapshot/worker.py'),'prepare','--snapshot',str(out/'source_snapshot'),'--directory',str(d),'--output',str(d/'preparation.json'),'--seed',str(seed),'--scenario',scenario]
   try:
    with (d/'prepare.log').open('x') as log:res=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=min(120,cfg['data_seconds_budget']-elapsed))
    status=read(d/'preparation.json') if (d/'preparation.json').exists() else dict(status='missing_preparation')
   except subprocess.TimeoutExpired:status=dict(status='process_timeout');save(d/'timeout.json',status)
   elapsed+=time.perf_counter()-begin;event=dict(key=name,status=status.get('status'),seconds=time.perf_counter()-begin)
   if status.get('status')=='complete':
    h=status['semantic_data_sha256']
    if h in previous:raise ValueError('duplicate_semantic_data')
    previous.add(h);art=d/status['artifact'];cases.append(dict(key=name,scenario=scenario,seed=seed,ids=status['ids'],alphas=status['alphas'],candidates=status['candidates'],fixed_start=status['fixed_start'],models=str(art/'local_models.pt'),evaluator=str(art/'global_evaluator.pt'),semantic_data_sha256=h))
   else:event['error']=status.get('error','preparation_failed')
   events.append(event)
   with (out/'preparation_events.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
   if elapsed>cfg['data_seconds_budget']:raise RuntimeError('data_budget')
 save(out/'input_sha256.json',{str(p):sha(p) for c in cases for p in [Path(c['models']),Path(c['evaluator']),Path(c['models']).parent/'pooled_data.npz',Path(c['models']).parents[2]/'preparation.json']});save(out/'cases.json',cases);save(out/'data_lock.json',dict(events=events,complete=len(cases),registered=40,preparation_wall_seconds=elapsed,input_sha256={str(p):sha(p) for c in cases for p in [Path(c['models']),Path(c['evaluator']),Path(c['models']).parent/'pooled_data.npz',Path(c['models']).parents[2]/'preparation.json']},no_outcomes_read=True))

def proxies(out):
 from adaptive_kernel import endpoints
 check(out);endpoints(out);lock=read(out/'endpoint_lock.json');total=dict(normal_solves=0,normal_calls=0,normal_samples=0,normal_seconds=0.,probe_samples=0,probe_calls=0,proxy_solves=lock['totals']['proxy_solves'],proxy_seconds=lock['totals']['proxy_seconds'])
 for c in read(out/'cases.json'):
  r=read(out/'records'/(c['key']+'.json'));normal=r['normal'];save(out/'normal'/(c['key']+'.json'),dict(case=c,**normal));save(out/'proxies'/(c['key']+'.json'),dict(case=c,participants=r['final']['rbf'],bank_points=r['bank_points'],bank_values=r['bank_values'],extra_calls=r['new_probe_calls'],extra_samples=r['new_probe_samples'],evaluator_loaded=False,truth_loaded=False))
  total['normal_solves']+=1;total['normal_calls']+=normal['counter']['calls'];total['normal_samples']+=normal['counter']['samples'];total['normal_seconds']+=normal['seconds'];total['probe_samples']+=r['new_probe_samples'];total['probe_calls']+=r['new_probe_calls']
 save(out/'proxy_lock.json',dict(record_sha256={p.name:sha(p) for p in (out/'proxies').glob('*.json')},normal_sha256={p.name:sha(p) for p in (out/'normal').glob('*.json')},totals=total,evaluator_loaded=False,loo_labels_loaded=False))

def scores(out):
 from local_core import fit,predict
 check(out);lock=read(out/'proxy_lock.json')
 for n,h in lock['record_sha256'].items():assert sha(out/'proxies'/n)==h
 for c in read(out/'cases.json'):
  print('SCORE',c['key'],flush=True);base=read(out/'normal'/(c['key']+'.json'))['result']['solution'];meta=read(Path(c['models']).parents[2]/'preparation.json');ids=[t['agent_id'] for t in sorted(meta['phase1_report'],key=lambda t:(t['mse'],t['agent_id']))[:3]]
  with np.load(Path(c['models']).parent/'pooled_data.npz') as data:X=data['X'];y=data['observed_y'];owner=data['agent_ids']
  summaries=[fit(X[owner==i],y[owner==i],base,2) for i in ids];baseline_predictions=[predict(s,base) for s in summaries];q0=float(np.median(abs(np.array(baseline_predictions))));rs=[]
  for p in read(out/'proxies'/(c['key']+'.json'))['participants']:
   z=p['result']['solution'];pred=[predict(s,z) for s in summaries];A=float(np.median(abs(np.array(pred)))-q0);rs.append(dict(agent_id=p['agent_id'],solution=z,provider_predictions=pred,A=A))
  save(out/'scores'/(c['key']+'.json'),dict(case=c,provider_ids=ids,provider_summaries=summaries,baseline_predictions=baseline_predictions,baseline_Q=q0,participants=rs,client_observations_processed=384,neural_model_queries=0,pooled_loaded=False,truth_loaded=False))
 save(out/'score_lock.json',dict(record_sha256={p.name:sha(p) for p in (out/'scores').glob('*.json')},client_observations_processed=384*len(read(out/'cases.json')),neural_model_queries=0,pooled_loaded=False,loo_labels_loaded=False))

def references(out):
 import torch
 from experiment import load_server
 from observer import solve
 torch.set_num_threads(1);cfg=check(out);lock=read(out/'proxy_lock.json');assert (out/'score_lock.json').exists();total=dict(solves=0,calls=0,samples=0,seconds=0.)
 for n,h in lock['record_sha256'].items():assert sha(out/'proxies'/n)==h
 for c in read(out/'cases.json'):
  print('REFERENCE',c['key'],flush=True);rs=[]
  for k,i in enumerate(c['ids']):
   server=load_server(c,k);r=solve(server,False,c['fixed_start'],max_samples=10000,max_seconds=30);rs.append(dict(agent_id=i,excluded_index=k,remaining_ids=[a.agent_id for a in server.trusted_agents],remaining_weights=server.alphas,**r));total['solves']+=1;total['calls']+=r['counter']['calls'];total['samples']+=r['counter']['samples'];total['seconds']+=r['total_seconds']
   if total['samples']+lock['totals']['normal_samples']+lock['totals']['probe_samples']>cfg['max_samples'] or total['seconds']+lock['totals']['normal_seconds']+lock['totals']['proxy_seconds']>cfg['max_solver_seconds']:raise RuntimeError('reference_budget')
  save(out/'references'/(c['key']+'.json'),dict(case=c,participants=rs,evaluator_loaded=False,truth_loaded=False))
 save(out/'reference_lock.json',dict(record_sha256={p.name:sha(p) for p in (out/'references').glob('*.json')},totals=total,evaluator_loaded=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','data','proxies','scores','references']);p.add_argument('out',type=Path);a=p.parse_args();sys.path.insert(0,str(a.out.resolve()/'source_snapshot'));globals()[a.phase](a.out.resolve())
