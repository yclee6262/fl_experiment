import argparse,csv,json,hashlib,shutil,time,sys
from pathlib import Path
import numpy as np
from proxy_core import record_normal,grid,Proxy,simulate,key
from local_core import fit,predict
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'docs/contribution_analysis_research.md').is_file());CONFIGS=['linear','rbf'];QUAL=['mean_abs','median_abs','abs_mean_signed','abs_median_signed'];METHODS=['team_'+q for q in QUAL]+[f'degree{d}_'+q for d in [1,2] for q in QUAL]
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
def csvsave(p,rs):
 with Path(p).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def quality(p):
 p=np.asarray(p);return dict(mean_abs=float(abs(p).mean()),median_abs=float(np.median(abs(p))),abs_median_signed=float(abs(np.median(p))),abs_mean_signed=float(abs(p.mean())))
def prepare(out):
 out.mkdir();(out/'records').mkdir();(out/'scores').mkdir();(out/'source_snapshot').mkdir();old=ROOT/'contribution_analysis/custom_direction_experiment_v2_20261020';cases=read(old/'cases.json')['development'];inputs={}
 for c in cases:
  for p in [Path(c['models']),Path(c['models']).parent/'pooled_data.npz',Path(c['models']).parents[2]/'preparation.json',old/'baselines'/(c['key']+'.json')]:inputs[str(p)]=sha(p)
 ref=old/'participant_comparison.csv';inputs[str(ref)]=sha(ref)
 for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,out/'source_snapshot'/p.name)
 for n in ['host_server.py','agent_client.py','models.py']:shutil.copy2(ROOT/n,out/'source_snapshot'/n)
 doc=ROOT/'docs/adaptive_shared_response_preregistration_20261020.md';shutil.copy2(doc,out/doc.name);save(out/'cases.json',cases);save(out/'input_sha256.json',inputs);save(out/'registration.json',dict(reference=str(ref),old=str(old),source_sha256={p.name:sha(p) for p in (out/'source_snapshot').glob('*.py')},plan_sha256=sha(doc),max_new_samples=10000,max_proxy_solves=500,max_seconds=300))
def check(out):
 cfg=read(out/'registration.json')
 for n,h in cfg['source_sha256'].items():assert sha(out/'source_snapshot'/n)==h
 for n,h in read(out/'input_sha256.json').items():assert sha(n)==h
 return cfg
class Tracer(Proxy):
 def __init__(self,p,v):super().__init__(p,v,'linear');self.observed=[]
 def __call__(self,x):self.observed.append(np.array(x));return super().__call__(x)

def endpoints(out):
 import torch
 torch.set_num_threads(1);cfg=check(out);sys.path.insert(0,str(ROOT/'contribution_analysis/custom_direction_v1'));from experiment import load_server
 total=dict(normal_solves=0,normal_samples=0,new_probe_samples=0,new_probe_calls=0,proxy_solves=0,proxy_seconds=0.);begin=time.perf_counter()
 for c in read(out/'cases.json'):
  print('ADAPT',c['key'],flush=True);server=load_server(c);normal=record_normal(server,c['fixed_start']);old=read(Path(cfg['old'])/'baselines'/(c['key']+'.json'))
  if normal['query_order_sha256']!=old['query_order_sha256'] or normal['result']['history']!=old['result']['history']:raise ValueError('normal_invariance')
  cache={key(p):(p,v) for p,v in zip(normal['points'],normal['values'])};gp=grid(normal['points'],5);new=0;events=[]
  def acquire(p):
   nonlocal new
   p=np.array(p,dtype=np.float32).astype(float)
   if key(p) not in cache:cache[key(p)]=(p,np.array([float(a.api_predict(p)[0]) for a in server.trusted_agents]));new+=len(c['ids'])
  for p in gp:acquire(p)
  spacing=1e-3*np.linalg.norm(np.ptp(gp,axis=0))
  for round_ in range(3):
   pp=np.array([v[0] for v in cache.values()]);vv=np.array([v[1] for v in cache.values()]);tracer=Tracer(pp,vv);runs=[]
   for k,i in enumerate(c['ids']):runs.append(dict(agent_id=i,**simulate(server,tracer,c,k)));total['proxy_solves']+=1
   total['proxy_seconds']+=sum(z['seconds'] for z in runs);cand={key(x):np.array(x,dtype=np.float32).astype(float) for x in tracer.observed if key(x) not in cache};candidates=np.array(list(cand.values()));chosen=[];disagreements=[]
   if len(candidates):
    lin=Proxy(pp,vv,'linear');rbf=Proxy(pp,vv,'rbf');scores=np.array([np.mean(abs(lin(x)-rbf(x))) for x in candidates]);order=sorted(range(len(candidates)),key=lambda j:(-scores[j],tuple(candidates[j])))
    for j in order:
     p=candidates[j]
     if not chosen or min(np.linalg.norm(p-q) for q in chosen)>spacing:chosen.append(p);disagreements.append(float(scores[j]))
     if len(chosen)==12:break
   for p in chosen:acquire(p)
   events.append(dict(round=round_,simulations=runs,n_candidates=len(candidates),chosen=chosen,disagreements=disagreements))
  pp=np.array([v[0] for v in cache.values()]);vv=np.array([v[1] for v in cache.values()]);final={}
  for name in CONFIGS:
   proxy=Proxy(pp,vv,name);runs=[]
   for k,i in enumerate(c['ids']):runs.append(dict(agent_id=i,**simulate(server,proxy,c,k)));total['proxy_solves']+=1
   final[name]=runs;total['proxy_seconds']+=sum(z['seconds'] for z in runs)
  total['normal_solves']+=1;total['normal_samples']+=normal['counter']['samples'];total['new_probe_samples']+=new;total['new_probe_calls']+=new
  save(out/'records'/(c['key']+'.json'),dict(case=c,normal=normal,rounds=events,bank_points=pp,bank_values=vv,final=final,new_probe_samples=new,new_probe_calls=new,pooled_loaded=False,truth_loaded=False))
  if total['new_probe_samples']>cfg['max_new_samples'] or total['proxy_solves']>cfg['max_proxy_solves'] or time.perf_counter()-begin>cfg['max_seconds']:raise RuntimeError('adaptive_budget')
 save(out/'endpoint_lock.json',dict(record_sha256={p.name:sha(p) for p in (out/'records').glob('*.json')},totals=total,wall_seconds=time.perf_counter()-begin,pooled_loaded=False,loo_labels_loaded=False))

def score(out):
 import torch
 torch.set_num_threads(1);cfg=check(out);lock=read(out/'endpoint_lock.json');sys.path.insert(0,str(ROOT/'contribution_analysis/custom_direction_v1'));from experiment import load_server
 total=0
 for n,h in lock['record_sha256'].items():assert sha(out/'records'/n)==h
 for c in read(out/'cases.json'):
  print('SCORE',c['key'],flush=True);rec=read(out/'records'/(c['key']+'.json'));base=rec['normal']['result']['solution'];cache={key(p):np.array(v) for p,v in zip(rec['bank_points'],rec['bank_values'])};server=load_server(c);calls=0;meta=read(Path(c['models']).parents[2]/'preparation.json');ids=[t['agent_id'] for t in sorted(meta['phase1_report'],key=lambda t:(t['mse'],t['agent_id']))[:3]];summaries={f'degree{d}':[] for d in [1,2]}
  with np.load(Path(c['models']).parent/'pooled_data.npz') as data:X=data['X'];y=data['observed_y'];owner=data['agent_ids']
  for i in ids:
   for d in [1,2]:summaries[f'degree{d}'].append(fit(X[owner==i],y[owner==i],base,d))
  q0={'team':quality(cache[key(base)])}
  for name,ss in summaries.items():q0[name]=quality([predict(s,base) for s in ss])
  rows=[]
  for name in CONFIGS:
   for t in rec['final'][name]:
    x=t['result']['solution']
    if key(x) not in cache:cache[key(x)]=np.array([float(a.api_predict(x)[0]) for a in server.trusted_agents]);calls+=len(c['ids'])
    values={'team':quality(cache[key(x)])}
    for mode,ss in summaries.items():values[mode]=quality([predict(s,x) for s in ss])
    scores={mode+'_'+q:values[mode][q]-q0[mode][q] for mode in values for q in QUAL};rows.append(dict(config=name,agent_id=t['agent_id'],solution=x,team_predictions=cache[key(x)],scores=scores))
  total+=calls;save(out/'scores'/(c['key']+'.json'),dict(case=c,selected_provider_ids=ids,local_summaries=summaries,baseline_quality=q0,participants=rows,new_samples=calls,client_observations=384,pooled_loaded=False,truth_loaded=False))
  if total+lock['totals']['new_probe_samples']>cfg['max_new_samples']:raise RuntimeError('score_budget')
 save(out/'score_lock.json',dict(record_sha256={p.name:sha(p) for p in (out/'scores').glob('*.json')},new_samples=total,client_observations=7680,pooled_loaded=False,loo_labels_loaded=False))

def analyze(out):
 import torch
 from models import PretrainedModel
 from scipy.stats import spearmanr
 torch.set_num_threads(1);cfg=check(out);lock=read(out/'score_lock.json')
 for n,h in lock['record_sha256'].items():assert sha(out/'scores'/n)==h
 with Path(cfg['reference']).open() as f:labels={(t['key'],int(t['agent_id'])):t for t in csv.DictReader(f)}
 rows=[];teams=[]
 for c in read(out/'cases.json'):
  print('OFFLINE',c['key'],flush=True);m=PretrainedModel(2);m.load_state_dict(torch.load(c['evaluator'],map_location='cpu',weights_only=True)['state_dict']);m.eval()
  for t in read(out/'scores'/(c['key']+'.json'))['participants']:
   lab=labels[c['key'],t['agent_id']]
   with torch.no_grad():E=abs(float(m(torch.tensor(t['solution'],dtype=torch.float32)[None,:]).item()))
   rows.append(dict(key=c['key'],scenario=c['scenario'],config=t['config'],agent_id=t['agent_id'],C=float(lab['C_custom_full']),offline_E=E-float(lab['full_error']),**t['scores']))
  for name in CONFIGS:
   rr=[r for r in rows if r['key']==c['key'] and r['config']==name]
   for method in METHODS+['offline_E']:
    a=[r[method] for r in rr];b=[r['C'] for r in rr];v=None if np.ptp(a)==0 or np.ptp(b)==0 else float(spearmanr(a,b).statistic);teams.append(dict(key=c['key'],scenario=c['scenario'],config=name,method=method,spearman=v))
 summary=[]
 for scenario in ['clean','controlled_poison']:
  for name in CONFIGS:
   for method in METHODS+['offline_E']:
    a=np.array([np.nan if t['spearman'] is None else t['spearman'] for t in teams if t['scenario']==scenario and t['config']==name and t['method']==method]);rng=np.random.default_rng(20261020);s=a[rng.integers(0,len(a),size=(10000,len(a)))];ct=np.isfinite(s).sum(axis=1);bs=np.nansum(s,axis=1)[ct>0]/ct[ct>0];summary.append(dict(scenario=scenario,config=name,method=method,mean=float(np.nanmean(a)) if np.isfinite(a).any() else None,ci=np.quantile(bs,[.025,.975]).tolist() if len(bs) else None,coverage=float(np.isfinite(a).mean())))
 eligible=[];offline=[]
 for name in CONFIGS:
  for method in METHODS+['offline_E']:
   ss=[s for s in summary if s['config']==name and s['method']==method]
   if all(s['mean'] is not None and s['mean']>.5 and s['coverage']>=.9 for s in ss):
    v=dict(config=name,method=method,min_mean=min(s['mean'] for s in ss),means={s['scenario']:s['mean'] for s in ss});(offline if method=='offline_E' else eligible).append(v)
 eligible.sort(key=lambda s:(-s['min_mean'],s['config']=='rbf',not s['method'].startswith('team'),s['method'].startswith('degree2'),s['method']));save(out/'development_summary.json',summary);csvsave(out/'participant_scores.csv',rows);csvsave(out/'team_metrics.csv',teams);save(out/'decision.json',dict(eligible=eligible,offline=offline,selected=eligible[0] if eligible else None,independent_validity_established=False));print(json.dumps(dict(eligible=eligible,offline=offline),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','endpoints','score','analyze']);p.add_argument('out',type=Path);a=p.parse_args();globals()[a.phase](a.out.resolve())
