import argparse,csv,json,hashlib,shutil
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'host_server.py').is_file());SOURCE=ROOT/'contribution_analysis/shared_response_experiment_v2_20261020';METHODS=['trim20','upper20','winsor20'];CONFIGS=['linear_5','linear_9','rbf_5','rbf_9']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def csvsave(p,rs):
 with Path(p).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def q(pred,method):
 v=np.sort(abs(np.array(pred)));k=int(np.floor(len(v)*.2))
 if not k:return float(v.mean())
 if method=='trim20':return float(v[k:-k].mean())
 if method=='upper20':return float(v[:-k].mean())
 if method=='winsor20':return float(np.clip(v,v[k],v[-k-1]).mean())
 raise ValueError('method')
def prepare(out):
 out.mkdir();(out/'records').mkdir();(out/'source_snapshot').mkdir();cases=read(SOURCE/'cases.json');inputs={}
 for c in cases:
  for p in [SOURCE/'normal'/(c['key']+'.json'),SOURCE/'scores'/(c['key']+'.json')]:inputs[str(p)]=sha(p)
 ref=ROOT/'contribution_analysis/custom_direction_experiment_v2_20261020/participant_comparison.csv';inputs[str(ref)]=sha(ref)
 for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,out/'source_snapshot'/p.name)
 doc=ROOT/'docs/robust_endpoint_preregistration_20261020.md';shutil.copy2(doc,out/doc.name);save(out/'cases.json',cases);save(out/'input_sha256.json',inputs);save(out/'registration.json',dict(reference=str(ref),source=str(SOURCE),source_sha256={p.name:sha(p) for p in (out/'source_snapshot').glob('*.py')},plan_sha256=sha(doc),new_queries=0))
def check(out):
 c=read(out/'registration.json')
 for n,h in c['source_sha256'].items():assert sha(out/'source_snapshot'/n)==h
 for n,h in read(out/'input_sha256.json').items():assert sha(n)==h
 return c

def score(out):
 c=check(out);source=Path(c['source'])
 for case in read(out/'cases.json'):
  norm=read(source/'normal'/(case['key']+'.json'));base=np.array(norm['result']['solution']);j=next(j for j,p in enumerate(norm['points']) if np.array_equal(np.array(p,dtype=np.float32),np.array(base,dtype=np.float32)));bp=norm['values'][j];rs=[]
  for t in read(source/'scores'/(case['key']+'.json'))['participants']:rs.append(dict(config=t['config'],agent_id=t['agent_id'],scores={n:q(t['predictions'],n)-q(bp,n) for n in METHODS}))
  save(out/'records'/(case['key']+'.json'),dict(participants=rs,pooled_loaded=False,truth_loaded=False))
 save(out/'score_lock.json',dict(record_sha256={p.name:sha(p) for p in (out/'records').glob('*.json')},loo_labels_loaded=False,new_queries=0))
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
 eligible.sort(key=lambda s:(-s['min_mean'],int(s['config'].split('_')[1]),s['config'].startswith('rbf'),s['method']));save(out/'development_summary.json',summary);csvsave(out/'participant_scores.csv',rows);csvsave(out/'team_metrics.csv',teams);save(out/'decision.json',dict(eligible=eligible,selected=eligible[0] if eligible else None,independent_validity_established=False));print(json.dumps(dict(eligible=eligible),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','score','analyze']);p.add_argument('out',type=Path);a=p.parse_args();globals()[a.phase](a.out.resolve())
