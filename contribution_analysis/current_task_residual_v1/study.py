import argparse,csv,json,hashlib,shutil,time
from pathlib import Path
import numpy as np
from core import fit_local,corrected
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'host_server.py').is_file());SOURCE=ROOT/'contribution_analysis/shared_response_experiment_v2_20261020';CONFIGS=['linear_5','linear_9','rbf_5','rbf_9'];MODES=['intercept','linear'];QUAL=['abs_mean_signed','abs_median_signed','mean_abs','median_abs'];METHODS=[m+'_'+q for m in MODES for q in QUAL]
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
 out.mkdir();(out/'records').mkdir();(out/'source_snapshot').mkdir();cases=read(SOURCE/'cases.json');inputs={}
 for c in cases:
  for p in [Path(c['models']),Path(c['models']).parent/'pooled_data.npz',Path(c['models']).parents[2]/'preparation.json',SOURCE/'normal'/(c['key']+'.json')]+[SOURCE/'endpoints'/(c['key']+'_'+n+'.json') for n in CONFIGS]:inputs[str(p)]=sha(p)
 ref=ROOT/'contribution_analysis/custom_direction_experiment_v2_20261020/participant_comparison.csv';inputs[str(ref)]=sha(ref)
 for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,out/'source_snapshot'/p.name)
 doc=ROOT/'docs/current_task_residual_preregistration_20261020.md';shutil.copy2(doc,out/doc.name);save(out/'cases.json',cases);save(out/'input_sha256.json',inputs);save(out/'registration.json',dict(reference=str(ref),source=str(SOURCE),source_sha256={p.name:sha(p) for p in (out/'source_snapshot').glob('*.py')},plan_sha256=sha(doc),data_keys_allowed=['X','observed_y','agent_ids'],max_samples=10000,max_seconds=300,new_client_data_access_assumption=True))
def check(out):
 c=read(out/'registration.json')
 for n,h in c['source_sha256'].items():assert sha(out/'source_snapshot'/n)==h
 for n,h in read(out/'input_sha256.json').items():assert sha(n)==h
 return c

def score(out):
 import torch
 from models import PretrainedModel
 torch.set_num_threads(1);cfg=check(out);source=Path(cfg['source']);total=0;begin=time.perf_counter()
 for c in read(out/'cases.json'):
  print('CLIENT',c['key'],flush=True);meta=read(Path(c['models']).parents[2]/'preparation.json');selected=sorted(meta['phase1_report'],key=lambda t:(t['mse'],t['agent_id']))[:3];ids=[t['agent_id'] for t in selected];normal=read(source/'normal'/(c['key']+'.json'));base=np.array(normal['result']['solution']);payload=torch.load(c['models'],map_location='cpu',weights_only=True);models=[];summaries=[];calls=0
  with np.load(Path(c['models']).parent/'pooled_data.npz') as data:X=data['X'];y=data['observed_y'];owner=data['agent_ids']
  for i in ids:
   model=PretrainedModel(2);model.load_state_dict(payload['agents'][i]);model.eval();models.append(model);xx=X[owner==i];yy=y[owner==i];d=np.linalg.norm(xx-base,axis=1);order=np.argsort(d,kind='stable')[:32]
   with torch.no_grad():pred=np.array([float(model(torch.tensor(x,dtype=torch.float32)[None,:]).item()) for x in xx[order]])
   calls+=len(order);summaries.append(fit_local(xx,yy,pred,base))
  cache={}
  def evaluate(x):
   nonlocal calls
   key=np.asarray(x,dtype=np.float32).tobytes()
   if key not in cache:
    with torch.no_grad():cache[key]=np.array([float(m(torch.tensor(x,dtype=torch.float32)[None,:]).item()) for m in models])
    calls+=len(models)
   return cache[key]
  bpred=evaluate(base);qs={mode:quality([corrected(f,s,base,mode) for f,s in zip(bpred,summaries)]) for mode in MODES};rs=[]
  for name in CONFIGS:
   for t in read(source/'endpoints'/(c['key']+'_'+name+'.json'))['participants']:
    x=np.array(t['result']['solution']);pred=evaluate(x);values={};scores={}
    for mode in MODES:
     z=np.array([corrected(f,s,x,mode) for f,s in zip(pred,summaries)]);values[mode]=z;q=quality(z)
     for n in QUAL:scores[mode+'_'+n]=q[n]-qs[mode][n]
    rs.append(dict(config=name,agent_id=t['agent_id'],solution=x,predictions=pred,corrected_predictions=values,scores=scores))
  total+=calls;save(out/'records'/(c['key']+'.json'),dict(case=c,selected_model_ids=ids,summaries=summaries,baseline_predictions=bpred,baseline_quality=qs,participants=rs,samples=calls,calls=calls,pooled_loaded=False,truth_loaded=False,read_data_keys=['X','observed_y','agent_ids']))
  if total>cfg['max_samples'] or time.perf_counter()-begin>cfg['max_seconds']:raise RuntimeError('residual_budget')
 save(out/'score_lock.json',dict(record_sha256={p.name:sha(p) for p in (out/'records').glob('*.json')},samples=total,calls=total,seconds=time.perf_counter()-begin,pooled_loaded=False,loo_labels_loaded=False))
def analyze(out):
 from scipy.stats import spearmanr
 c=check(out);lock=read(out/'score_lock.json')
 for n,h in lock['record_sha256'].items():assert sha(out/'records'/n)==h
 with Path(c['reference']).open() as f:labels={(t['key'],int(t['agent_id'])):float(t['C_custom_full']) for t in csv.DictReader(f)}
 rows=[];teams=[]
 for case in read(out/'cases.json'):
  for t in read(out/'records'/(case['key']+'.json'))['participants']:rows.append(dict(key=case['key'],scenario=case['scenario'],config=t['config'],agent_id=t['agent_id'],C=labels[case['key'],t['agent_id']],**t['scores']))
  for name in CONFIGS:
   rs=[t for t in rows if t['key']==case['key'] and t['config']==name]
   for n in METHODS:
    x=[t[n] for t in rs];y=[t['C'] for t in rs];v=None if np.ptp(x)==0 or np.ptp(y)==0 else float(spearmanr(x,y).statistic);teams.append(dict(key=case['key'],scenario=case['scenario'],config=name,method=n,spearman=v))
 summary=[]
 for scenario in ['clean','controlled_poison']:
  for name in CONFIGS:
   for n in METHODS:
    a=np.array([np.nan if t['spearman'] is None else t['spearman'] for t in teams if t['scenario']==scenario and t['config']==name and t['method']==n]);rng=np.random.default_rng(20261020);s=a[rng.integers(0,len(a),size=(10000,len(a)))];ct=np.isfinite(s).sum(axis=1);bs=np.nansum(s,axis=1)[ct>0]/ct[ct>0];summary.append(dict(scenario=scenario,config=name,method=n,mean=float(np.nanmean(a)) if np.isfinite(a).any() else None,ci=np.quantile(bs,[.025,.975]).tolist() if len(bs) else None,coverage=float(np.isfinite(a).mean())))
 eligible=[]
 for name in CONFIGS:
  for n in METHODS:
   ss=[s for s in summary if s['config']==name and s['method']==n]
   if all(s['mean'] is not None and s['mean']>.5 and s['coverage']>=.9 for s in ss):eligible.append(dict(config=name,method=n,min_mean=min(s['mean'] for s in ss),means={s['scenario']:s['mean'] for s in ss}))
 eligible.sort(key=lambda s:(-s['min_mean'],int(s['config'].split('_')[1]),s['config'].startswith('rbf'),s['method'].startswith('linear'),s['method']));save(out/'development_summary.json',summary);csvsave(out/'participant_scores.csv',rows);csvsave(out/'team_metrics.csv',teams);save(out/'decision.json',dict(eligible=eligible,selected=eligible[0] if eligible else None,independent_validity_established=False,new_client_data_access_assumption=True));print(json.dumps(dict(eligible=eligible),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','score','analyze']);p.add_argument('out',type=Path);a=p.parse_args();globals()[a.phase](a.out.resolve())
