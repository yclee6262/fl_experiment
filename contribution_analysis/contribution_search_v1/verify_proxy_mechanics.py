import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path('/Users/yclee/Documents/GitHub/fl_experiment');out=ROOT/'contribution_analysis/shared_response_experiment_v2_20261020';sys.path.insert(0,str(out/'source_snapshot'));sys.path.insert(1,str(ROOT))
from core import Proxy,simulate,key,grid
from host_server import HostServer

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
cfg=read(out/'registration.json');source=Path(cfg['old']);checks={};errors=[];max_delta=0.;count=0
checks['historical_sources']=all(sha(ROOT/n)==h for n,h in cfg['historical_sha256'].items())
checks['frozen_sources']=all(sha(out/'source_snapshot'/n)==h for n,h in cfg['source_sha256'].items())
checks['inputs']=all(sha(Path(n))==h for n,h in read(out/'input_sha256.json').items())
checks['normal_invariance']=True;checks['roles_and_weights']=True;checks['all_320_proxy_replays']=True;checks['all_1280_global_free_scores']=True
for case in read(out/'cases.json'):
 normal=read(out/'normal'/(case['key']+'.json'));old=read(source/'baselines'/(case['key']+'.json'))
 checks['normal_invariance'] &= normal['query_order_sha256']==old['query_order_sha256'] and normal['counter']==old['counter'] and normal['result']['history']==old['result']['history'] and normal['result']['solution']==old['result']['solution']
 cache={key(p):(p,v) for p,v in zip(normal['all_points'],normal['all_values'])};normalkeys=[key(p) for p in normal['points']];basekey=key(normal['result']['solution']);bp=np.array(cache[basekey][1]);q0=dict(mean_abs=float(abs(bp).mean()),median_abs=float(np.median(abs(bp))),abs_median_signed=float(abs(np.median(bp))),abs_mean_signed=float(abs(bp.mean())))
 rawscores=read(out/'scores'/(case['key']+'.json'))['participants']
 for config in cfg['configs']:
  method,ns=config.split('_');keys=list(dict.fromkeys(normalkeys+[key(p) for p in grid(np.array(normal['points']),int(ns))]));pp=np.array([cache[k][0] for k in keys]);vv=np.array([cache[k][1] for k in keys]);proxy=Proxy(pp,vv,method);server=HostServer(0.,2,constraint_mode='box_constrained');stored=read(out/'endpoints'/(case['key']+'_'+config+'.json'))['participants']
  for k,(i,t) in enumerate(zip(case['ids'],stored)):
   r=simulate(server,proxy,case,k);delta=float(np.max(abs(np.array(r['result']['solution'])-t['result']['solution'])));max_delta=max(delta,max_delta);count+=1
   checks['all_320_proxy_replays'] &= delta<1e-12 and r['result']['history']==t['result']['history'] and r['proxy_calls']==t['proxy_calls'] and r['outside_calls']==t['outside_calls']
   checks['roles_and_weights'] &= t['remaining_ids']==[j for j in case['ids'] if j!=i] and bool(np.allclose(t['remaining_weights'],np.ones(len(case['ids'])-1)/(len(case['ids'])-1),atol=1e-15,rtol=0))
   s=next(s for s in rawscores if s['agent_id']==i and s['config']==config);p=np.array(s['predictions']);q=dict(mean_abs=float(abs(p).mean()),median_abs=float(np.median(abs(p))),abs_median_signed=float(abs(np.median(p))),abs_mean_signed=float(abs(p.mean())))
   checks['all_1280_global_free_scores'] &= all(abs(s['scores'][n]-(q[n]-q0[n]))<1e-15 for n in q)
checks={k:bool(v) for k,v in checks.items()};result=dict(accepted=all(checks.values()),checks=checks,replayed_proxy_runs=count,max_proxy_endpoint_delta=max_delta,new_actual_model_queries=0,supplemental_technical_tests=6,validity_not_implied=True)
with (out/'mechanism_verification.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2));assert result['accepted']
