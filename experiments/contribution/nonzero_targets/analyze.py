"""Team-paired description of nonzero role interventions (truth stays offline)."""
import argparse,csv,gzip,hashlib,json
from functools import lru_cache
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
TOL=1e-10
REPS=200000
SEED=20261123

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvout(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def share(x,tau):
 w=np.where(np.array(x)>tau,x,0.);s=w/w.sum() if w.sum() else w
 return np.r_[s,1-s.sum()]
def rho(x,y):
 return None if len(x)<3 or np.ptp(x)<=TOL or np.ptp(y)<=TOL else float(spearmanr(x,y).statistic)
@lru_cache(None)
def weights(n):
 rng=np.random.default_rng(SEED)
 return np.concatenate([rng.multinomial(n,np.full(n,1/n),size=min(2000,REPS-begin)) for begin in range(0,REPS,2000)])

def ci(vals):
 values=[x for x in vals if x is not None and np.isfinite(x)];n=len(values)
 if not n:return dict(n=0,missing=len(vals),mean=None,ci95_low=None,ci95_high=None)
 bs=weights(n)@np.array(values)/n
 return dict(n=n,missing=len(vals)-n,mean=float(np.mean(values)),ci95_low=float(np.quantile(bs,.025)),ci95_high=float(np.quantile(bs,.975)))

def ratio_ci(nums,dens):
 nums=np.asarray(nums);dens=np.asarray(dens);n=len(nums)
 if not n:return dict(teams=0,numerator=0,denominator=0,ratio=None,ci95_low=None,ci95_high=None,zero_denominator_draws=REPS)
 rng=np.random.default_rng(SEED)
 w=rng.multinomial(n,np.full(n,1/n),size=REPS);bn=w@nums;bd=w@dens;valid=bd>0
 values=bn[valid]/bd[valid]
 return dict(teams=n,numerator=int(nums.sum()),denominator=int(dens.sum()),ratio=float(nums.sum()/dens.sum()) if dens.sum() else None,ci95_low=float(np.quantile(values,.025)) if len(values) else None,ci95_high=float(np.quantile(values,.975)) if len(values) else None,zero_denominator_draws=int(sum(~valid)))

def analyze(source,output):
 output.mkdir(parents=True,exist_ok=False);manifest=read(source/'manifest.json');index=read(source/'index.json')
 assert manifest['status']=='complete' and not manifest['smoke'] and len(index)==80
 for f,h in manifest['source_sha256'].items():assert sha(Path(f))==h
 tasks=[];owners=[];roles=[];flips=[];hashes={}
 for ix in index:
  p=source/ix['file'];assert sha(p)==ix['sha256'];hashes[str(p.resolve())]=ix['sha256']
  with gzip.open(p,'rt') as f:r=json.load(f)
  target=ix['target'];scenario=ix['scenario'];key=ix['key']
  if r['status']!='valid':
   tasks.append(dict(key=key,scenario=scenario,target=target,status=r['status'],truth_error=None,actual_rounds=None,cache_miss_calls=None,inversion_forward_samples=None,observer_equivalence_cache_miss_calls=None,inverse_bad=None));continue
  assert abs(r['witness'][0]+r['witness'][1]+r['witness'][0]*r['witness'][1]-target)<1e-12
  tasks.append(dict(key=key,scenario=scenario,target=target,status=r['status'],truth_error=r['truth_error'],actual_rounds=r['actual_rounds'],cache_miss_calls=r['cache_miss_calls'],inversion_forward_samples=r['inversion_forward_samples'],observer_equivalence_cache_miss_calls=r['observer_equivalence_cache_miss_calls'],inverse_bad=r['inverse_target_error_above_point1']))
  people=r['participants'];n=len(people);truth=np.array([v['truth_LOO'] for v in people])
  for role in ['A','M','J']:
   f=np.array([v['scores']['F'][role] for v in people]);peer=np.array([v['scores']['Fminus'][role] for v in people]);tau=np.array([v['tauA'] for v in people]) if role=='A' else TOL
   sf=share(f,tau);sp=share(peer,tau)
   for judge,x in [('F',f),('Fminus',peer)]:
    sign=np.where(abs(x)>TOL,np.sign(x),0);sy=np.where(abs(truth)>TOL,np.sign(truth),0)
    roles.append(dict(key=key,scenario=scenario,target=target,role=role,judge=judge,n=n,rank_truth_LOO=rho(x,truth),sign_truth_LOO=float(np.mean(sign==sy)),reserved=bool(share(x,tau)[-1]>.5),TV_peer_full=float(abs(sf-sp).sum()/2)))
   flips.append(dict(key=key,scenario=scenario,target=target,role=role,positive_F=int(sum(f>TOL)),Fpositive_peernegative=int(sum((f>TOL)&(peer< -TOL)))))
   for i,v in enumerate(people):owners.append(dict(key=key,scenario=scenario,target=target,role=role,i=i,id=v['id'],F=float(f[i]),Fminus=float(peer[i]),truth_LOO=float(truth[i]),share_F=float(sf[i]),share_peer=float(sp[i])))
 summary=[];changes=[]
 for scenario in ['clean','controlled_poison']:
  for target in [0.,-.5,.5,2.7]:
   tt=[x for x in tasks if x['scenario']==scenario and x['target']==target]
   for metric in ['truth_error','actual_rounds','cache_miss_calls','inversion_forward_samples','observer_equivalence_cache_miss_calls','inverse_bad']:
    summary.append(dict(kind='task',scenario=scenario,target=target,role='all',judge='none',metric=metric,planned=10,**ci([x[metric] for x in tt])))
   for role in ['A','M','J']:
    for judge in ['F','Fminus']:
     rr=[x for x in roles if x['scenario']==scenario and x['target']==target and x['role']==role and x['judge']==judge]
     for metric in ['rank_truth_LOO','sign_truth_LOO','reserved','TV_peer_full']:
      summary.append(dict(kind='role',scenario=scenario,target=target,role=role,judge=judge,metric=metric,planned=10,**ci([float(x[metric]) if x[metric] is not None else None for x in rr])))
     if target!=0:
      zero={x['key']:x for x in roles if x['scenario']==scenario and x['target']==0 and x['role']==role and x['judge']==judge}
      for metric in ['rank_truth_LOO','sign_truth_LOO','reserved','TV_peer_full']:
       ds=[None if x[metric] is None or zero.get(x['key'],{}).get(metric) is None else float(x[metric])-float(zero[x['key']][metric]) for x in rr]
       changes.append(dict(scenario=scenario,target=target,role=role,judge=judge,metric=metric,planned=10,**ci(ds)))
 flip_summary=[]
 for scenario in ['clean','controlled_poison']:
  for target in [0.,-.5,.5,2.7]:
   for role in ['A','M','J']:
    rr=[x for x in flips if x['scenario']==scenario and x['target']==target and x['role']==role]
    flip_summary.append(dict(scenario=scenario,target=target,role=role,**ratio_ci([x['Fpositive_peernegative'] for x in rr],[x['positive_F'] for x in rr])))
 for row in summary+changes:row['missing_planned_teams']=row['planned']-row['n']
 for name,data in [('tasks',tasks),('owners',owners),('roles',roles),('sign_flips',flips),('sign_flip_summary',flip_summary),('summary',summary),('paired_target_changes',changes)]:csvout(output/(name+'.csv'),data)
 audit=dict(passed=True,tasks=len(tasks),valid=sum(x['status']=='valid' for x in tasks),failed=sum(x['status']!='valid' for x in tasks),source_sha256=hashes,analysis_sha256=sha(Path(__file__)),new_statistical_model_queries=0,bootstrap_replicates=REPS,rng=SEED,scope='Existing models, fixed teams, fresh target-specific endpoints; descriptive team intervals, not across-start stability or independent model confirmation')
 (output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({k:v for k,v in audit.items() if k!='source_sha256'}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();analyze(a.source,a.output_dir)
