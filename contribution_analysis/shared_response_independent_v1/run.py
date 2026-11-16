import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'host_server.py').is_file())
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
 out.mkdir();(out/'source_snapshot').mkdir();(out/'data').mkdir();(out/'normal').mkdir();(out/'proxies').mkdir();(out/'references').mkdir()
 snap=out/'source_snapshot';origin=ROOT/'contribution_analysis/shared_response_experiment_v2_20261020/source_snapshot'
 for n in ['core.py']:shutil.copy2(origin/n,snap/n)
 for p in (ROOT/'contribution_analysis/custom_direction_v1').glob('*.py'):
  if p.name in ['observer.py','experiment.py','common.py']:shutil.copy2(p,snap/p.name)
 shutil.copy2(ROOT/'contribution_analysis/full_removal_budget_v1/worker.py',snap/'worker.py')
 hist=['host_server.py','agent_client.py','models.py','dataset.py','alpha_weight_experiments.py','p3_contribution_validation.py','research_evaluation.py','run_experiments.py']
 for n in hist:shutil.copy2(ROOT/n,snap/n)
 for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,snap/p.name)
 doc=ROOT/'docs/shared_response_independent_preregistration_20261020.md';shutil.copy2(doc,out/doc.name)
 save(out/'registration.json',dict(seeds=list(range(2026102200,2026102220)),scenarios=['clean','controlled_poison'],config='linear_9',offline_only=True,global_free_validity_established=False,historical_sha256={n:sha(ROOT/n) for n in hist},source_sha256={p.name:sha(p) for p in snap.glob('*.py')},plan_sha256=sha(doc),data_seconds_budget=1800,max_samples=500000,max_solver_seconds=300,started_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()))
def check(out):
 c=read(out/'registration.json')
 for n,h in c['source_sha256'].items():assert sha(out/'source_snapshot'/n)==h
 for n,h in c['historical_sha256'].items():assert sha(ROOT/n)==h
 return c

def data(out):
 cfg=check(out);cases=[];events=[];elapsed=0.;previous=set()
 old=ROOT/'contribution_analysis/full_removal_budget_experiment_v1_20261016'
 for p in old.glob('*/*/*/preparation.json'):
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
 save(out/'cases.json',cases);save(out/'data_lock.json',dict(events=events,complete=len(cases),registered=40,preparation_wall_seconds=elapsed,input_sha256={str(p):sha(p) for c in cases for p in [Path(c['models']),Path(c['evaluator'])]},no_outcomes_read=True))

def proxies(out):
 import torch
 from core import record_normal,grid,Proxy,key,simulate
 from experiment import load_server
 torch.set_num_threads(1);cfg=check(out);cases=read(out/'cases.json');total=dict(normal_solves=0,normal_calls=0,normal_samples=0,normal_seconds=0.,probe_samples=0,probe_calls=0,proxy_solves=0,proxy_seconds=0.)
 for c in cases:
  print('PROXY',c['key'],flush=True);server=load_server(c);normal=record_normal(server,c['fixed_start']);cache={key(p):(p,v) for p,v in zip(normal['points'],normal['values'])};extra=0;begin=time.perf_counter()
  for p in grid(normal['points'],9):
   if key(p) not in cache:cache[key(p)]=(p.astype(np.float32).astype(float),np.array([float(a.api_predict(p)[0]) for a in server.trusted_agents]));extra+=len(server.trusted_agents)
  probe_seconds=time.perf_counter()-begin;pp=np.array([v[0] for v in cache.values()]);vv=np.array([v[1] for v in cache.values()]);begin=time.perf_counter();proxy=Proxy(pp,vv,'linear');build_seconds=time.perf_counter()-begin;rs=[]
  for k,i in enumerate(c['ids']):rs.append(dict(agent_id=i,excluded_index=k,**simulate(server,proxy,c,k)))
  save(out/'normal'/(c['key']+'.json'),dict(case=c,**normal));save(out/'proxies'/(c['key']+'.json'),dict(case=c,participants=rs,bank_points=pp,bank_values=vv,extra_calls=extra,extra_samples=extra,probe_seconds=probe_seconds,build_seconds=build_seconds,evaluator_loaded=False,truth_loaded=False))
  total['normal_solves']+=1;total['normal_calls']+=normal['counter']['calls'];total['normal_samples']+=normal['counter']['samples'];total['normal_seconds']+=normal['seconds'];total['probe_samples']+=extra;total['probe_calls']+=extra;total['proxy_solves']+=len(rs);total['proxy_seconds']+=sum(r['seconds'] for r in rs)+build_seconds
  if total['probe_samples']+total['normal_samples']>cfg['max_samples'] or total['proxy_seconds']+total['normal_seconds']>cfg['max_solver_seconds']:raise RuntimeError('proxy_budget')
 save(out/'proxy_lock.json',dict(record_sha256={p.name:sha(p) for p in (out/'proxies').glob('*.json')},normal_sha256={p.name:sha(p) for p in (out/'normal').glob('*.json')},totals=total,evaluator_loaded=False,loo_labels_loaded=False))

def references(out):
 import torch
 from experiment import load_server
 from observer import solve
 torch.set_num_threads(1);cfg=check(out);lock=read(out/'proxy_lock.json');total=dict(solves=0,calls=0,samples=0,seconds=0.)
 for n,h in lock['record_sha256'].items():assert sha(out/'proxies'/n)==h
 for c in read(out/'cases.json'):
  print('REFERENCE',c['key'],flush=True);rs=[]
  for k,i in enumerate(c['ids']):
   server=load_server(c,k);r=solve(server,False,c['fixed_start'],max_samples=10000,max_seconds=30);rs.append(dict(agent_id=i,excluded_index=k,remaining_ids=[a.agent_id for a in server.trusted_agents],remaining_weights=server.alphas,**r));total['solves']+=1;total['calls']+=r['counter']['calls'];total['samples']+=r['counter']['samples'];total['seconds']+=r['total_seconds']
   if total['samples']+lock['totals']['normal_samples']+lock['totals']['probe_samples']>cfg['max_samples'] or total['seconds']+lock['totals']['normal_seconds']+lock['totals']['proxy_seconds']>cfg['max_solver_seconds']:raise RuntimeError('reference_budget')
  save(out/'references'/(c['key']+'.json'),dict(case=c,participants=rs,evaluator_loaded=False,truth_loaded=False))
 save(out/'reference_lock.json',dict(record_sha256={p.name:sha(p) for p in (out/'references').glob('*.json')},totals=total,evaluator_loaded=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','data','proxies','references']);p.add_argument('out',type=Path);a=p.parse_args();sys.path.insert(0,str(a.out.resolve()/'source_snapshot'));globals()[a.phase](a.out.resolve())
