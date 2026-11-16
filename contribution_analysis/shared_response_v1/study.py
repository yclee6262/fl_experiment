import argparse,csv,hashlib,json,sys,time,shutil
from pathlib import Path
import numpy as np
from core import record_normal,grid,Proxy,simulate,key
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'host_server.py').is_file())
NAMES=['mean_abs','median_abs','abs_median_signed','abs_mean_signed']
CONFIGS=[f'{m}_{n}' for m in ['linear','rbf'] for n in [5,9]]
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
def quality(p):
 p=np.asarray(p);return dict(mean_abs=float(abs(p).mean()),median_abs=float(np.median(abs(p))),abs_median_signed=float(abs(np.median(p))),abs_mean_signed=float(abs(p.mean())))
def prepare(out):
 old=ROOT/'contribution_analysis/custom_direction_experiment_v2_20261020';cases=read(old/'cases.json')['development'];out.mkdir()
 for n in ['source_snapshot','normal','endpoints','scores']:(out/n).mkdir()
 inputs={str(old/'participant_comparison.csv'):digest(old/'participant_comparison.csv')}
 for c in cases:
  for p in [Path(c['models']),Path(c['evaluator']),old/'baselines'/(c['key']+'.json')]:inputs[str(p)]=digest(p)
 for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,out/'source_snapshot'/p.name)
 plan=ROOT/'docs/shared_response_preregistration_20261020.md';shutil.copy2(plan,out/plan.name)
 hist={n:digest(ROOT/n) for n in ['host_server.py','agent_client.py','models.py','dataset.py','alpha_weight_experiments.py','p3_contribution_validation.py']}
 save(out/'cases.json',cases);save(out/'input_sha256.json',inputs)
 save(out/'registration.json',dict(old=str(old),source_sha256={p.name:digest(p) for p in (out/'source_snapshot').glob('*.py')},historical_sha256=hist,plan_sha256=digest(plan),methods=NAMES,configs=CONFIGS,max_samples=20000,max_proxy_solves=320,max_proxy_seconds=600,confirmation=False))
def check(out):
 c=read(out/'registration.json')
 for n,h in c['source_sha256'].items():assert digest(out/'source_snapshot'/n)==h
 for n,h in c['historical_sha256'].items():assert digest(ROOT/n)==h
 for n,h in read(out/'input_sha256.json').items():assert digest(n)==h
 return c

def endpoints(out):
 import torch
 torch.set_num_threads(1);cfg=check(out);sys.path.insert(0,str(ROOT/'contribution_analysis/custom_direction_v1'));from experiment import load_server
 totals=dict(normal_calls=0,normal_samples=0,normal_solves=0,probe_calls=0,probe_samples=0,proxy_solves=0,proxy_seconds=0.)
 for c in read(out/'cases.json'):
  print('PROXY',c['key'],flush=True);server=load_server(c);normal=record_normal(server,c['fixed_start']);old=read(Path(cfg['old'])/'baselines'/(c['key']+'.json'))
  if normal['query_order_sha256']!=old['query_order_sha256'] or normal['result']['history']!=old['result']['history'] or not np.array_equal(normal['result']['solution'],old['result']['solution']) or normal['counter']!=old['counter']:raise ValueError('normal_changed')
  totals['normal_solves']+=1;totals['normal_calls']+=normal['counter']['calls'];totals['normal_samples']+=normal['counter']['samples']
  cache={key(p):(p,y) for p,y in zip(normal['points'],normal['values'])};p5=grid(normal['points'],5);p9=grid(normal['points'],9);calls=0
  for p in p9:
   kk=key(p)
   if kk not in cache:cache[kk]=(p.astype(np.float32).astype(float),np.array([float(a.api_predict(p)[0]) for a in server.trusted_agents]));calls+=len(server.trusted_agents)
  totals['probe_calls']+=calls;totals['probe_samples']+=calls
  save(out/'normal'/(c['key']+'.json'),dict(case=c,**normal,extra_probe_calls=calls,extra_probe_samples=calls,grid5=p5,grid9=p9,all_points=np.array([x[0] for x in cache.values()]),all_values=np.array([x[1] for x in cache.values()])))
  normal_keys=[key(p) for p in normal['points']]
  for name in CONFIGS:
   method,ns=name.split('_');n=int(ns);keys=list(dict.fromkeys(normal_keys+[key(p) for p in (p5 if n==5 else p9)]));pp=np.array([cache[k][0] for k in keys]);vv=np.array([cache[k][1] for k in keys]);proxy=Proxy(pp,vv,method);rs=[]
   for k,i in enumerate(c['ids']):
    z=simulate(server,proxy,c,k);totals['proxy_solves']+=1;totals['proxy_seconds']+=z['seconds'];rs.append(dict(agent_id=i,index=k,**z))
   save(out/'endpoints'/(c['key']+'_'+name+'.json'),dict(case_key=c['key'],config=name,n_proxy_points=len(pp),participants=rs,evaluator_loaded=False,truth_loaded=False))
   if totals['probe_samples']>cfg['max_samples'] or totals['proxy_solves']>cfg['max_proxy_solves'] or totals['proxy_seconds']>cfg['max_proxy_seconds']:raise RuntimeError('proxy_budget')
 save(out/'endpoint_lock.json',dict(record_sha256={p.name:digest(p) for p in (out/'endpoints').glob('*.json')},normal_sha256={p.name:digest(p) for p in (out/'normal').glob('*.json')},totals=totals,evaluator_loaded=False,loo_labels_loaded=False))
def scores(out):
 import torch
 torch.set_num_threads(1);cfg=check(out);lock=read(out/'endpoint_lock.json');sys.path.insert(0,str(ROOT/'contribution_analysis/custom_direction_v1'));from experiment import load_server
 for n,h in lock['record_sha256'].items():assert digest(out/'endpoints'/n)==h
 calls=0
 for c in read(out/'cases.json'):
  print('QUALITY',c['key'],flush=True);server=load_server(c);normal=read(out/'normal'/(c['key']+'.json'));cache={key(x):np.array(y) for x,y in zip(normal['all_points'],normal['all_values'])}
  q0=quality(cache[key(normal['result']['solution'])]);rs=[];casecalls=0
  for name in CONFIGS:
   for t in read(out/'endpoints'/(c['key']+'_'+name+'.json'))['participants']:
    x=t['result']['solution'];kk=key(x)
    if kk not in cache:cache[kk]=np.array([float(a.api_predict(x)[0]) for a in server.trusted_agents]);casecalls+=len(server.trusted_agents)
    q=quality(cache[kk]);rs.append(dict(config=name,agent_id=t['agent_id'],solution=x,predictions=cache[kk],scores={n:q[n]-q0[n] for n in NAMES}))
  calls+=casecalls;save(out/'scores'/(c['key']+'.json'),dict(case_key=c['key'],baseline_quality=q0,participants=rs,score_calls=casecalls,score_samples=casecalls,evaluator_loaded=False))
  if calls+lock['totals']['probe_samples']>cfg['max_samples']:raise RuntimeError('score_budget')
 save(out/'score_lock.json',dict(record_sha256={p.name:digest(p) for p in (out/'scores').glob('*.json')},extra_samples=calls,extra_calls=calls,evaluator_loaded=False,loo_labels_loaded=False))
def analyze(out):
 import torch
 from scipy.stats import spearmanr
 from models import PretrainedModel
 torch.set_num_threads(1);cfg=check(out);lock=read(out/'score_lock.json')
 for n,h in lock['record_sha256'].items():assert digest(out/'scores'/n)==h
 with (Path(cfg['old'])/'participant_comparison.csv').open() as f:labels={(a['key'],int(a['agent_id'])):a for a in csv.DictReader(f)}
 rows=[];teams=[]
 for c in read(out/'cases.json'):
  print('OFFLINE',c['key'],flush=True);model=PretrainedModel(2);model.load_state_dict(torch.load(c['evaluator'],map_location='cpu',weights_only=True)['state_dict']);model.eval();ts=read(out/'scores'/(c['key']+'.json'))['participants']
  for t in ts:
   lab=labels[c['key'],t['agent_id']]
   with torch.no_grad():E=abs(float(model(torch.tensor(t['solution'],dtype=torch.float32)[None,:]).item()))
   rows.append(dict(key=c['key'],scenario=c['scenario'],config=t['config'],agent_id=t['agent_id'],C=float(lab['C_custom_full']),offline_E=E-float(lab['full_error']),**t['scores']))
  for name in CONFIGS:
   rs=[a for a in rows if a['key']==c['key'] and a['config']==name]
   for n in NAMES+['offline_E']:
    x=np.array([a[n] for a in rs]);y=np.array([a['C'] for a in rs]);valid=np.ptp(x)>0 and np.ptp(y)>0
    teams.append(dict(key=c['key'],scenario=c['scenario'],config=name,method=n,spearman=float(spearmanr(x,y).statistic) if valid else None,score_range=float(np.ptp(x)),reference_range=float(np.ptp(y))))
 summary=[]
 for scenario in ['clean','controlled_poison']:
  for name in CONFIGS:
   for n in NAMES+['offline_E']:
    ts=[t for t in teams if t['scenario']==scenario and t['config']==name and t['method']==n];a=np.array([np.nan if t['spearman'] is None else t['spearman'] for t in ts]);rng=np.random.default_rng(20261020);ss=a[rng.integers(0,len(a),size=(10000,len(a)))];ct=np.isfinite(ss).sum(axis=1);bs=np.nansum(ss,axis=1)[ct>0]/ct[ct>0]
    summary.append(dict(scenario=scenario,config=name,method=n,mean=float(np.nanmean(a)) if np.isfinite(a).any() else None,ci=np.quantile(bs,[.025,.975]).tolist() if len(bs) else None,valid_teams=int(np.isfinite(a).sum()),coverage=float(np.isfinite(a).mean()),near_flat_teams=sum(t['score_range']<=1e-4 for t in ts)))
 candidates=[(name,n) for name in CONFIGS for n in NAMES if all(s['mean'] is not None and s['mean']>.5 and s['coverage']>=.9 for s in summary if s['config']==name and s['method']==n)]
 offline=[name for name in CONFIGS if all(s['mean'] is not None and s['mean']>.5 and s['coverage']>=.9 for s in summary if s['config']==name and s['method']=='offline_E')]
 save(out/'development_summary.json',summary);csvsave(out/'participant_scores.csv',rows);csvsave(out/'team_metrics.csv',teams);save(out/'decision.json',dict(global_free_candidates=candidates,offline_only_candidates=offline,independent_validity_established=False))
 print(json.dumps(dict(summary=summary,candidates=candidates,offline=offline),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','endpoints','scores','analyze']);p.add_argument('out',type=Path);a=p.parse_args();globals()[a.phase](a.out.resolve())
