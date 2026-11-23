"""Validate actual cold-cache replay counts and summarize comparable cost terms."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def savecsv(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def analyze(source,output):
 output.mkdir(parents=True,exist_ok=False)
 assert read(source/'manifest.json')['status']=='complete'
 old=ROOT/'contribution_analysis/model_pool_v_cm_validation_20261107_attempt_02'
 reg=read(old/'registration.json');cases=reg['fresh'];assert len(cases)==24
 hashes={str(old/'registration.json'):sha(old/'registration.json')};data=[]
 for c in cases:
  key=c['key'];oldpath=old/'records'/f'{key}_cost.json';reference=read(oldpath);hashes[str(oldpath)]=sha(oldpath)
  for arm in ['AV','ACM','J','LOO']:
   p=source/f'{key}_{arm}.json';r=read(p);hashes[str(p.resolve())]=sha(p)
   assert r['arm']==arm and r['base']==reference[arm]['base'] and r['baseline_digest']==reference[arm]['baseline_digest']
   assert r['endpoint_checks_passed']==r['endpoint_checks']
   assert r['calls']==r['requests']-r['hits']==sum(z['calls'] for z in r['stages'].values())
   assert len(r['events'])==r['requests']
   assert sum(not z['hit'] for z in r['events'])==r['calls']
   stage=r['stages'];base=stage['baseline']['calls'];anchor=sum(v['calls'] for k,v in stage.items() if k in ['features','anchor','anchor_peer_score'])
   score=stage.get('peer_score',{}).get('calls',0)
   branch=r['calls']-base-anchor-score
   assert branch>=0
   data.append(dict(key=key,arm=arm,n=c['n'],baseline=base,anchor=anchor,model_joint_LOO=branch,final_peer_score=score,total=r['calls'],requests=r['requests'],cache_hits=r['hits'],endpoint_checks=r['endpoint_checks']))
 summary=[]
 for arm in ['AV','ACM','J','LOO']:
  rr=[r for r in data if r['arm']==arm]
  v={'arm':arm,'teams':len(rr)}
  for field in ['baseline','anchor','model_joint_LOO','final_peer_score','total']:
   xs=[r[field] for r in rr];v[field+'_mean']=float(np.mean(xs));v[field+'_sd']=float(np.std(xs,ddof=1));v[field+'_sum']=int(sum(xs))
  summary.append(v)
 loo=next(r['total_mean'] for r in summary if r['arm']=='LOO')
 for r in summary:r['relative_total_to_LOO']=r['total_mean']/loo-1
 assert np.allclose([r['total_mean'] for r in summary],[1580.9583333333333,2211.4583333333333,1187.5833333333333,1888.75],rtol=0,atol=1e-10)
 for p in [ROOT/'contribution_analysis/peer_exclusion_cost_v1/run.py',ROOT/'experiments/contribution/replay_cost_comparison.py',Path(__file__)]:hashes[str(p.resolve())]=sha(p)
 savecsv(output/'teams.csv',data);savecsv(output/'summary.csv',summary)
 audit=dict(passed=True,teams=24,arms=4,outputs=96,all_endpoint_checks_passed=True,total_actual_calls=sum(r['total'] for r in data),scope='Actual cold-cache analytic-model replay; not allocation equivalence or network timing',source_sha256=hashes)
 (output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({k:v for k,v in audit.items() if k!='source_sha256'}))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();analyze(a.source,a.output_dir)
