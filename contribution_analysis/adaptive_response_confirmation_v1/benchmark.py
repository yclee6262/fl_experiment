"""Paired prototype timing with cached-model LOO and behavior-preserving proxy caches."""
import argparse,hashlib,json,sys,time,shutil
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'docs/contribution_analysis_research.md').is_file());sys.path.insert(0,str(Path(__file__).resolve().parent if (Path(__file__).resolve().parent/'fast_proxy.py').is_file() else ROOT/'contribution_analysis/adaptive_shared_response_v1'))
from proxy_core import Proxy,record_normal,grid,key,simulate
from fast_proxy import FastProxy,batch_values
from local_core import fit,predict

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
class Trace:
 def __init__(self,cls,p,v):self.proxy=cls(p,v,'linear');self.observed=[]
 def __call__(self,x):self.observed.append(np.array(x));return self.proxy(x)
 @property
 def calls(self):return self.proxy.calls
 @property
 def outside_calls(self):return self.proxy.outside_calls

def method(case,fast):
 from experiment import load_server
 server=load_server(case);normal=record_normal(server,case['fixed_start']);cache={key(p):(p,v) for p,v in zip(normal['points'],normal['values'])};gp=grid(normal['points'],5);spacing=1e-3*np.linalg.norm(np.ptp(gp,axis=0));queries=[];samples=normal['counter']['samples'];cls=FastProxy if fast else Proxy
 def acquire(p):
  nonlocal samples
  p=np.array(p,dtype=np.float32).astype(float);k=key(p)
  if k not in cache:cache[k]=(p,np.array([float(a.api_predict(p)[0]) for a in server.trusted_agents]));samples+=len(case['ids']);queries.append(k.hex())
 for p in gp:acquire(p)
 for round_ in range(3):
  pp=np.array([v[0] for v in cache.values()]);vv=np.array([v[1] for v in cache.values()]);trace=Trace(cls,pp,vv)
  for k in range(len(case['ids'])):simulate(server,trace,case,k)
  cand={key(x):np.array(x,dtype=np.float32).astype(float) for x in trace.observed if key(x) not in cache};xs=np.array(list(cand.values()));chosen=[]
  if len(xs):
   l=cls(pp,vv,'linear');r=cls(pp,vv,'rbf');scores=np.mean(abs(batch_values(l,xs)-batch_values(r,xs)),axis=1) if fast else np.array([np.mean(abs(l(x)-r(x))) for x in xs]);order=sorted(range(len(xs)),key=lambda j:(-scores[j],tuple(xs[j])))
   for j in order:
    p=xs[j]
    if not chosen or min(np.linalg.norm(p-q) for q in chosen)>spacing:chosen.append(p)
    if len(chosen)==12:break
  for p in chosen:acquire(p)
 pp=np.array([v[0] for v in cache.values()]);vv=np.array([v[1] for v in cache.values()]);proxy=cls(pp,vv,'rbf');endpoints=[simulate(server,proxy,case,k) for k in range(len(case['ids']))]
 meta=read(Path(case['models']).parents[2]/'preparation.json');ids=[t['agent_id'] for t in sorted(meta['phase1_report'],key=lambda t:(t['mse'],t['agent_id']))[:3]];base=normal['result']['solution']
 with np.load(Path(case['models']).parent/'pooled_data.npz') as data:X=data['X'];y=data['observed_y'];owner=data['agent_ids']
 ss=[fit(X[owner==i],y[owner==i],base,2) for i in ids];q0=np.median(abs(np.array([predict(s,base) for s in ss])));A=[float(np.median(abs(np.array([predict(s,r['result']['solution']) for s in ss])))-q0) for r in endpoints]
 return dict(normal=normal,endpoints=endpoints,A=A,samples=samples,queries=queries,client_observations=384)

def control(case):
 from experiment import load_server
 from observer import solve
 server=load_server(case);normal=solve(server,False,case['fixed_start']);agents=list(server.trusted_agents);candidates=np.array(server.I_list);alphas=np.array(server.alphas);samples=normal['counter']['samples'];rs=[]
 try:
  for k in range(len(agents)):
   server.trusted_agents=[a for j,a in enumerate(agents) if j!=k];server.I_list=np.delete(candidates,k,axis=0);w=np.delete(alphas,k);server.alphas=list(w/w.sum());r=solve(server,False,case['fixed_start']);samples+=r['counter']['samples'];rs.append(r)
 finally:server.trusted_agents=agents;server.I_list=candidates;server.alphas=list(alphas)
 return dict(normal=normal,endpoints=rs,samples=samples)

def benchmark(out):
 import torch
 torch.set_num_threads(1);sys.path.insert(0,str(ROOT/'contribution_analysis/custom_direction_v1'));cases=[c for c in read(out/'cases.json') if c['seed'] in [2026102300,2026102301]];assert len(cases)==4
 lock=read(out/'reference_lock.json');scorelock=read(out/'score_lock.json')
 for n,h in lock['record_sha256'].items():assert sha(out/'references'/n)==h
 reg=dict(cases=[c['key'] for c in cases],warmups=2,repeats=10,methods=['original','cached','LOO_cached_models'],model_loading_included=True,client_data_scan_and_fit_included=True,pooled_evaluation_excluded=True,network_latency_not_simulated=True,source_sha256={str(p):sha(p) for p in [Path(__file__),Path(sys.modules['fast_proxy'].__file__)]})
 save(out/'benchmark_registration_v2.json',reg);trials=[];resources=[]
 for c in cases:
  print('BENCHMARK',c['key'],flush=True);normal=read(out/'normal'/(c['key']+'.json'));proxy=read(out/'proxies'/(c['key']+'.json'))['participants'];ref=read(out/'references'/(c['key']+'.json'))['participants'];expectedA=[a['A'] for a in read(out/'scores'/(c['key']+'.json'))['participants']]
  def one(name):
   start=time.perf_counter();r=control(c) if name=='LOO_cached_models' else method(c,name=='cached');elapsed=time.perf_counter()-start
   if r['normal']['result']['history']!=normal['result']['history'] or not np.array_equal(r['normal']['result']['solution'],normal['result']['solution']):raise ValueError('benchmark_baseline_drift')
   expected=ref if name=='LOO_cached_models' else proxy
   for rr,ee in zip(r['endpoints'],expected):
    if rr['result']['history']!=ee['result']['history'] or not np.array_equal(rr['result']['solution'],ee['result']['solution']):raise ValueError('benchmark_endpoint_drift:'+name)
   if name!='LOO_cached_models' and not np.allclose(r['A'],expectedA,atol=1e-15,rtol=0):raise ValueError('benchmark_score_drift')
   resources.append(dict(key=c['key'],method=name,samples=r['samples']));return elapsed
  names=reg['methods']
  for warm in range(2):
   for n in names:one(n)
  for repeat in range(10):
   order=names[repeat%3:]+names[:repeat%3];times={n:one(n) for n in order};trials.append(dict(key=c['key'],repeat=repeat,**times))
 sums={n:sum(np.median([t[n] for t in trials if t['key']==c['key']]) for c in cases) for n in reg['methods']};controlsum=sums['LOO_cached_models']
 saved=0
 for c in cases:
  seen={n:next(r['samples'] for r in resources if r['key']==c['key'] and r['method']==n) for n in reg['methods']};saved+=seen['LOO_cached_models']-seen['cached']
 extra=max(0,sums['cached']-controlsum)
 result=dict(passed=True,trials=trials,median_seconds_summed_across_four_cases=sums,original_vs_loo_ratio=sums['original']/controlsum,cached_vs_loo_ratio=sums['cached']/controlsum,local_time_gate_passed=sums['cached']<=controlsum,saved_model_samples_per_four_cases=saved,serialized_remote_call_latency_break_even_ms=1000*extra/saved if saved else None,total_research_model_samples=sum(r['samples'] for r in resources),scope=reg,all_normal_endpoints_and_scores_identical=True)
 save(out/'benchmark_results_v2.json',result);print(json.dumps(plain({k:v for k,v in result.items() if k not in ['trials','scope']}),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();sys.path.insert(0,str(a.out.resolve()/'source_snapshot'));benchmark(a.out.resolve())
