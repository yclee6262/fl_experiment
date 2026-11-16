import argparse,csv,json,hashlib,shutil,time
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'host_server.py').is_file())
SOURCE=ROOT/'contribution_analysis/shared_response_experiment_v2_20261020'
CONFIGS=['linear_5','linear_9','rbf_5','rbf_9'];NAMES=['abs_mean_signed','abs_median_signed','mean_abs','median_abs'];PANELS=['all','noncoalition']
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
 out.mkdir();(out/'records').mkdir();(out/'source_snapshot').mkdir();inputs={}
 for p in [SOURCE/'endpoint_lock.json',SOURCE/'score_lock.json',SOURCE/'cases.json']+list((SOURCE/'endpoints').glob('*.json'))+list((SOURCE/'normal').glob('*.json')):inputs[str(p)]=digest(p)
 cases=read(SOURCE/'cases.json')
 for c in cases:inputs[c['models']]=digest(c['models'])
 old=ROOT/'contribution_analysis/custom_direction_experiment_v2_20261020';inputs[str(old/'participant_comparison.csv')]=digest(old/'participant_comparison.csv')
 for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,out/'source_snapshot'/p.name)
 plan=ROOT/'docs/distributed_panel_preregistration_20261020.md';shutil.copy2(plan,out/plan.name);save(out/'cases.json',cases);save(out/'input_sha256.json',inputs)
 save(out/'registration.json',dict(source=str(SOURCE),reference=str(old/'participant_comparison.csv'),source_sha256={p.name:digest(p) for p in (out/'source_snapshot').glob('*.py')},plan_sha256=digest(plan),configs=CONFIGS,panels=PANELS,qualities=NAMES,max_samples=20000,max_seconds=300,confirmation=False))
def check(out):
 cfg=read(out/'registration.json')
 for n,h in cfg['source_sha256'].items():assert digest(out/'source_snapshot'/n)==h
 for n,h in read(out/'input_sha256.json').items():assert digest(n)==h
 return cfg

def score(out):
 import torch
 from models import PretrainedModel
 torch.set_num_threads(1);cfg=check(out);source=Path(cfg['source']);total=0;start=time.perf_counter()
 for c in read(out/'cases.json'):
  print('PANEL',c['key'],flush=True);payload=torch.load(c['models'],map_location='cpu',weights_only=True);ids=sorted(payload['agents']);models=[]
  if len(ids)!=20:raise ValueError('expected_original_20_model_pool')
  for i in ids:
   m=PretrainedModel(2);m.load_state_dict(payload['agents'][i]);m.eval();models.append(m)
  normal=read(source/'normal'/(c['key']+'.json'));cache={};calls=0
  def evaluate(x):
   nonlocal calls
   k=np.asarray(x,dtype=np.float32).tobytes()
   if k not in cache:
    with torch.no_grad():cache[k]=np.array([float(m(torch.tensor(x,dtype=torch.float32)[None,:]).item()) for m in models])
    calls+=len(models)
   return cache[k]
  baseline=evaluate(normal['result']['solution']);rs=[]
  for name in CONFIGS:
   for t in read(source/'endpoints'/(c['key']+'_'+name+'.json'))['participants']:
    pred=evaluate(t['result']['solution'])
    for panel in PANELS:
     mask=np.ones(len(ids),dtype=bool) if panel=='all' else np.array([i not in c['ids'] for i in ids]);q=quality(pred[mask]);q0=quality(baseline[mask]);rs.append(dict(config=name,panel=panel,panel_ids=np.array(ids)[mask],agent_id=t['agent_id'],solution=t['result']['solution'],predictions=pred[mask],baseline_predictions=baseline[mask],scores={n:q[n]-q0[n] for n in NAMES}))
  total+=calls;save(out/'records'/(c['key']+'.json'),dict(case_key=c['key'],original_pool_ids=ids,participants=rs,calls=calls,samples=calls,evaluator_loaded=False,truth_loaded=False))
  if total>cfg['max_samples'] or time.perf_counter()-start>cfg['max_seconds']:raise RuntimeError('panel_budget')
 save(out/'score_lock.json',dict(record_sha256={p.name:digest(p) for p in (out/'records').glob('*.json')},calls=total,samples=total,seconds=time.perf_counter()-start,evaluator_loaded=False,loo_labels_loaded=False))

def analyze(out):
 from scipy.stats import spearmanr
 cfg=check(out);lock=read(out/'score_lock.json')
 for n,h in lock['record_sha256'].items():assert digest(out/'records'/n)==h
 with Path(cfg['reference']).open() as f:labels={(a['key'],int(a['agent_id'])):float(a['C_custom_full']) for a in csv.DictReader(f)}
 teams=[];rows=[]
 for c in read(out/'cases.json'):
  ts=read(out/'records'/(c['key']+'.json'))['participants']
  for t in ts:rows.append(dict(key=c['key'],scenario=c['scenario'],agent_id=t['agent_id'],config=t['config'],panel=t['panel'],C=labels[c['key'],t['agent_id']],**t['scores']))
  for name in CONFIGS:
   for panel in PANELS:
    rs=[t for t in rows if t['key']==c['key'] and t['config']==name and t['panel']==panel]
    for n in NAMES:
     x=np.array([t[n] for t in rs]);y=np.array([t['C'] for t in rs]);valid=np.ptp(x)>0 and np.ptp(y)>0;teams.append(dict(key=c['key'],scenario=c['scenario'],config=name,panel=panel,method=n,spearman=float(spearmanr(x,y).statistic) if valid else None,score_range=float(np.ptp(x))))
 summary=[]
 for scenario in ['clean','controlled_poison']:
  for name in CONFIGS:
   for panel in PANELS:
    for n in NAMES:
     vals=[t['spearman'] for t in teams if t['scenario']==scenario and t['config']==name and t['panel']==panel and t['method']==n];a=np.array([np.nan if x is None else x for x in vals]);rng=np.random.default_rng(20261020);s=a[rng.integers(0,len(a),size=(10000,len(a)))];ct=np.isfinite(s).sum(axis=1);bs=np.nansum(s,axis=1)[ct>0]/ct[ct>0]
     summary.append(dict(scenario=scenario,config=name,panel=panel,method=n,mean=float(np.nanmean(a)) if np.isfinite(a).any() else None,ci=np.quantile(bs,[.025,.975]).tolist() if len(bs) else None,valid_teams=int(np.isfinite(a).sum()),coverage=float(np.isfinite(a).mean())))
 eligible=[]
 for name in CONFIGS:
  for panel in PANELS:
   for n in NAMES:
    ss=[s for s in summary if s['config']==name and s['panel']==panel and s['method']==n]
    if all(s['mean'] is not None and s['mean']>.5 and s['coverage']>=.9 for s in ss):eligible.append(dict(config=name,panel=panel,method=n,min_mean=min(s['mean'] for s in ss),means={s['scenario']:s['mean'] for s in ss}))
 eligible.sort(key=lambda s:(-s['min_mean'],int(s['config'].split('_')[1]),s['config'].startswith('rbf'),s['panel']=='all',s['method']))
 save(out/'development_summary.json',summary);csvsave(out/'participant_scores.csv',rows);csvsave(out/'team_metrics.csv',teams);save(out/'decision.json',dict(eligible=eligible,selected=eligible[0] if eligible else None,confirmation_passed=False))
 print(json.dumps(dict(eligible=eligible,selected=eligible[0] if eligible else None),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','score','analyze']);p.add_argument('out',type=Path);a=p.parse_args();globals()[a.phase](a.out.resolve())
