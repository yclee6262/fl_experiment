import argparse,json,time,sys
from pathlib import Path
import numpy as np
from proxy_core import Proxy,simulate,key,grid
from fast_proxy import FastProxy,batch_values
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'docs/contribution_analysis_research.md').is_file());sys.path.insert(0,str(ROOT));from host_server import HostServer

def read(p):return json.loads(p.read_text())
class Trace:
 def __init__(self,cls,p,v):self.proxy=cls(p,v,'linear');self.observed=[]
 def __call__(self,x):self.observed.append(np.array(x));return self.proxy(x)
 @property
 def calls(self):return self.proxy.calls
 @property
 def outside_calls(self):return self.proxy.outside_calls

def replay(rec,cls,batch=False):
 c=rec['case'];normal=rec['normal'];truth={key(p):(np.array(p),np.array(v)) for p,v in zip(rec['bank_points'],rec['bank_values'])};cache={key(p):(np.array(p),np.array(v)) for p,v in zip(normal['points'],normal['values'])};gp=grid(normal['points'],5);spacing=1e-3*np.linalg.norm(np.ptp(gp,axis=0));queries=[];server=HostServer(0.,2,constraint_mode='box_constrained')
 def acquire(p):
  k=key(p)
  if k not in cache:
   if k not in truth:raise ValueError('optimized_method_requested_new_point')
   cache[k]=truth[k];queries.append(k.hex())
 for p in gp:acquire(p)
 rounds=[]
 for round_ in range(3):
  pp=np.array([v[0] for v in cache.values()]);vv=np.array([v[1] for v in cache.values()]);tracer=Trace(cls,pp,vv)
  for k in range(len(c['ids'])):simulate(server,tracer,c,k)
  cand={key(x):np.array(x,dtype=np.float32).astype(float) for x in tracer.observed if key(x) not in cache};xs=np.array(list(cand.values()));chosen=[]
  if len(xs):
   l=cls(pp,vv,'linear');r=cls(pp,vv,'rbf');scores=np.mean(abs(batch_values(l,xs)-batch_values(r,xs)),axis=1) if batch else np.array([np.mean(abs(l(x)-r(x))) for x in xs]);order=sorted(range(len(xs)),key=lambda j:(-scores[j],tuple(xs[j])))
   for j in order:
    p=xs[j]
    if not chosen or min(np.linalg.norm(p-q) for q in chosen)>spacing:chosen.append(p)
    if len(chosen)==12:break
  for p in chosen:acquire(p)
  if [key(p) for p in chosen]!=[key(p) for p in rec['rounds'][round_]['chosen']]:raise ValueError('changed_adaptive_selection')
 pp=np.array([v[0] for v in cache.values()]);vv=np.array([v[1] for v in cache.values()]);proxy=cls(pp,vv,'rbf');maxdelta=0.
 for k,saved in enumerate(rec['final']['rbf']):
  r=simulate(server,proxy,c,k);delta=np.max(abs(np.array(r['result']['solution'])-saved['result']['solution']));maxdelta=max(maxdelta,float(delta))
  if delta>1e-12 or r['result']['history']!=saved['result']['history']:raise ValueError('changed_final_path')
 return queries,maxdelta

def main(out):
 source=ROOT/'contribution_analysis/adaptive_shared_response_experiment_v2_20261020';cases=read(source/'cases.json');a=[];b=[];maxdelta=0
 for c in cases:
  rec=read(source/'records'/(c['key']+'.json'));print('CHECK',c['key'],flush=True);start=time.perf_counter();old,delta=replay(rec,Proxy);a.append(time.perf_counter()-start);start=time.perf_counter();new,delta=replay(rec,FastProxy,True);b.append(time.perf_counter()-start);maxdelta=max(maxdelta,delta);assert old==new
 result=dict(accepted=True,n_teams=20,all_selected_queries_identical=True,all_rbf_histories_identical=True,max_final_endpoint_delta=maxdelta,original_numeric_seconds=sum(a),cached_numeric_seconds=sum(b),numeric_ratio=sum(b)/sum(a),time_scope='Numerical pipeline replay with saved model replies; not full end-to-end model/API time.')
 with out.open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();main(a.out)
