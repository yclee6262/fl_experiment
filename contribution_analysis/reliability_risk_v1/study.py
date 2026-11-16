import argparse,csv,json,hashlib,shutil
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'host_server.py').is_file())
SOURCE=ROOT/'contribution_analysis/distributed_panel_experiment_v1_20261020';METHODS=[f'best{k}_{op}' for k in [2,3,5] for op in ['mean_abs','median_abs']]+['inverse_mse_mean_abs'];CONFIGS=['linear_5','linear_9','rbf_5','rbf_9'];PANELS=['all','noncoalition']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def csvsave(p,rs):
 with Path(p).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def prepfile(c):return Path(c['models']).parents[2]/'preparation.json'
def weights(ids,mses,method):
 order=sorted(range(len(ids)),key=lambda j:(mses[j],ids[j]));w=np.zeros(len(ids))
 if method.startswith('inverse_mse'):w=1/(np.array(mses)+1e-8)
 else:w[order[:int(method.split('_')[0][4:])]]=1
 return w/w.sum()
def prepare(out):
 out.mkdir();(out/'records').mkdir();(out/'source_snapshot').mkdir();cases=read(SOURCE/'cases.json');inputs={str(SOURCE/'score_lock.json'):sha(SOURCE/'score_lock.json')}
 for c in cases:
  for p in [prepfile(c),SOURCE/'records'/(c['key']+'.json')]:inputs[str(p)]=sha(p)
 for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,out/'source_snapshot'/p.name)
 ref=ROOT/'contribution_analysis/custom_direction_experiment_v2_20261020/participant_comparison.csv';inputs[str(ref)]=sha(ref);doc=ROOT/'docs/reliability_risk_preregistration_20261020.md';shutil.copy2(doc,out/doc.name)
 save(out/'cases.json',cases);save(out/'input_sha256.json',inputs);save(out/'registration.json',dict(source=str(SOURCE),reference=str(ref),source_sha256={p.name:sha(p) for p in (out/'source_snapshot').glob('*.py')},plan_sha256=sha(doc),new_queries=0,alpha_unchanged=True))
def check(out):
 c=read(out/'registration.json')
 for n,h in c['source_sha256'].items():assert sha(out/'source_snapshot'/n)==h
 for n,h in read(out/'input_sha256.json').items():assert sha(n)==h
 return c

def score(out):
 cfg=check(out);source=Path(cfg['source'])
 for c in read(out/'cases.json'):
  ms={t['agent_id']:t['mse'] for t in read(prepfile(c))['phase1_report']};rs=[]
  for t in read(source/'records'/(c['key']+'.json'))['participants']:
   ids=t['panel_ids'];mse=[ms[i] for i in ids];scores={};ws={}
   for method in METHODS:
    w=weights(ids,mse,method);pred=abs(np.array(t['predictions']));base=abs(np.array(t['baseline_predictions']));scores[method]=float(np.median(pred[w>0])-np.median(base[w>0])) if method.endswith('median_abs') else float(w@(pred-base));ws[method]=w.tolist()
   rs.append(dict(config=t['config'],panel=t['panel'],agent_id=t['agent_id'],panel_ids=ids,panel_mse=mse,weights=ws,scores=scores))
  save(out/'records'/(c['key']+'.json'),dict(key=c['key'],participants=rs,pooled_loaded=False,truth_loaded=False,new_queries=0))
 save(out/'score_lock.json',dict(record_sha256={p.name:sha(p) for p in (out/'records').glob('*.json')},pooled_loaded=False,loo_labels_loaded=False,new_queries=0))
def analyze(out):
 from scipy.stats import spearmanr
 cfg=check(out);lock=read(out/'score_lock.json')
 for n,h in lock['record_sha256'].items():assert sha(out/'records'/n)==h
 with Path(cfg['reference']).open() as f:labels={(a['key'],int(a['agent_id'])):float(a['C_custom_full']) for a in csv.DictReader(f)}
 teams=[];rows=[]
 for c in read(out/'cases.json'):
  for t in read(out/'records'/(c['key']+'.json'))['participants']:rows.append(dict(key=c['key'],scenario=c['scenario'],config=t['config'],panel=t['panel'],agent_id=t['agent_id'],C=labels[c['key'],t['agent_id']],**t['scores']))
  for name in CONFIGS:
   for panel in PANELS:
    rs=[t for t in rows if t['key']==c['key'] and t['config']==name and t['panel']==panel]
    for n in METHODS:
     x=np.array([a[n] for a in rs]);y=np.array([a['C'] for a in rs]);v=None if np.ptp(x)==0 or np.ptp(y)==0 else float(spearmanr(x,y).statistic);teams.append(dict(key=c['key'],scenario=c['scenario'],config=name,panel=panel,method=n,spearman=v))
 summary=[]
 for scenario in ['clean','controlled_poison']:
  for name in CONFIGS:
   for panel in PANELS:
    for n in METHODS:
     vals=[t['spearman'] for t in teams if t['scenario']==scenario and t['config']==name and t['panel']==panel and t['method']==n];a=np.array([np.nan if t is None else t for t in vals]);rng=np.random.default_rng(20261020);s=a[rng.integers(0,len(a),size=(10000,len(a)))];ct=np.isfinite(s).sum(axis=1);bs=np.nansum(s,axis=1)[ct>0]/ct[ct>0];summary.append(dict(scenario=scenario,config=name,panel=panel,method=n,mean=float(np.nanmean(a)) if np.isfinite(a).any() else None,ci=np.quantile(bs,[.025,.975]).tolist() if len(bs) else None,coverage=float(np.isfinite(a).mean())))
 eligible=[]
 for name in CONFIGS:
  for panel in PANELS:
   for n in METHODS:
    ss=[s for s in summary if s['config']==name and s['panel']==panel and s['method']==n]
    if all(s['mean'] is not None and s['mean']>.5 and s['coverage']>=.9 for s in ss):eligible.append(dict(config=name,panel=panel,method=n,min_mean=min(s['mean'] for s in ss),means={s['scenario']:s['mean'] for s in ss}))
 eligible.sort(key=lambda s:(-s['min_mean'],int(s['config'].split('_')[1]),s['config'].startswith('rbf'),s['panel']=='all',s['method']))
 save(out/'development_summary.json',summary);csvsave(out/'participant_scores.csv',rows);csvsave(out/'team_metrics.csv',teams);save(out/'decision.json',dict(eligible=eligible,selected=eligible[0] if eligible else None,independent_validity_established=False));print(json.dumps(dict(eligible=eligible,selected=eligible[0] if eligible else None),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','score','analyze']);p.add_argument('out',type=Path);a=p.parse_args();globals()[a.phase](a.out.resolve())
