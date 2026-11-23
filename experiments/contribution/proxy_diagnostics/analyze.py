"""Historical V/CM and signed anchor-projection audits, without model execution."""
import argparse,csv,hashlib,json,subprocess
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[3]
TOL=1e-10
REPS=200000
SEED=20261123


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def csvout(p,rows):
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def share(x,tau=TOL):
 x=np.asarray(x);w=np.where(x>tau,x,0.);p=w/w.sum() if w.sum() else w
 return np.r_[p,max(0.,1-p.sum())]
def metrics(x,y,tau=TOL):
 x=np.asarray(x);y=np.asarray(y);sx=np.where(abs(x)>TOL,np.sign(x),0);sy=np.where(abs(y)>TOL,np.sign(y),0)
 valid=len(x)>=3 and np.ptp(x)>TOL and np.ptp(y)>TOL
 return dict(owners=len(x),spearman=float(spearmanr(x,y).statistic) if valid else None,
  sign_agreement=float(np.mean(sx==sy)),opposite=int(sum(sx*sy<0)),both_nonzero=int(sum((sx!=0)&(sy!=0))),
  both_zero=int(sum((sx==0)&(sy==0))),one_zero=int(sum((sx==0)^(sy==0))),
  TV=float(abs(share(x,tau)-share(y,tau)).sum()/2),proxy_reserved=bool(share(x,tau)[-1]>.5),reference_reserved=bool(share(y,tau)[-1]>.5))
def projection(H,g,gain):
 H=np.asarray(H);g=np.asarray(g);den=float(g@g)
 if not np.allclose(H.sum(0),g,atol=1e-10,rtol=1e-10):raise ValueError('Direction decomposition mismatch')
 if den<=1e-20:
  if abs(gain)>1e-10:raise ValueError('Nonzero gain on zero direction')
  return np.zeros(len(H)),True
 out=gain*(H@g)/den
 if abs(out.sum()-gain)>1e-10:raise ValueError('Projection conservation mismatch')
 return out,False


def summarize(rows,groupfields):
 groups=defaultdict(list)
 for r in rows:groups[tuple(r[k] for k in groupfields)].append(r)
 result=[]
 for group,rs in sorted(groups.items()):
  for field in ['spearman','sign_agreement','TV','opposite_ratio']:
   if field=='opposite_ratio':
    vals=[r['opposite'] for r in rs];dens=[r['owners'] for r in rs]
   else:
    keep=[r for r in rs if r[field] is not None];vals=[r[field] for r in keep];dens=None
   if not vals:mean=lo=hi=None;n=0
   else:
    n=len(vals);rng=np.random.default_rng(SEED);boot=np.empty(REPS)
    for start in range(0,REPS,2000):
     w=rng.multinomial(n,np.full(n,1/n),size=min(2000,REPS-start))
     boot[start:start+len(w)]=(w@np.asarray(vals))/(w@np.asarray(dens) if dens else n)
    mean=float(sum(vals)/sum(dens)) if dens else float(np.mean(vals));lo,hi=map(float,np.quantile(boot,[.025,.975]))
   result.append(dict(zip(groupfields,group))|dict(metric=field,teams=len(rs),valid_teams=n,missing_teams=len(rs)-n,mean=mean,ci95_low=lo,ci95_high=hi,
      owners=sum(r['owners'] for r in rs),opposite=sum(r['opposite'] for r in rs),both_nonzero=sum(r['both_nonzero'] for r in rs),both_zero=sum(r['both_zero'] for r in rs),one_zero=sum(r['one_zero'] for r in rs),proxy_reserved_teams=sum(r['proxy_reserved'] for r in rs),reference_reserved_teams=sum(r['reference_reserved'] for r in rs)))
 return result


def analyze(output):
 output.mkdir(parents=True,exist_ok=False)
 src=ROOT/'contribution_analysis/model_pool_v_cm_validation_20261107_attempt_02';reg=read(src/'registration.json');hashes={str(src/'registration.json'):sha(src/'registration.json')}
 team=[];owners=[];checks=0;costchecks=0;fails=[]
 for c in reg['known']+reg['fresh']:
  p=src/'records'/f"{c['key']}_30.json";hashes[str(p)]=sha(p);r=read(p);a=np.asarray(r['weights']);n=r['n']
  assert len(c['ids'])==n and len(r['rows'])%n==0
  for z in r['rows']:
   for method,loss in [('V','minusV_losses'),('CM','minusM_losses')]:
    d=np.asarray(z[loss])-z['full_losses'];i=z['i'];total=float(a@d);peer=float(sum(a[j]*d[j] for j in range(n) if j!=i)/(1-a[i]))
    assert abs(total-z[method]['total'])<1e-10 and abs(peer-z[method]['peer_normalized'])<1e-10;checks+=2
  for judge in ['total','peer_normalized']:
   x=np.array([sum(z['V'][judge] for z in r['rows'] if z['i']==i) for i in range(n)])
   y=np.array([sum(z['CM'][judge] for z in r['rows'] if z['i']==i) for i in range(n)])
   team.append(dict(key=c['key'],phase=c['phase'],scenario=c['scenario'],n=n,judge=judge,**metrics(x,y)))
   for i in range(n):owners.append(dict(key=c['key'],phase=c['phase'],judge=judge,i=i,V=float(x[i]),CM=float(y[i])))
  if c['phase']=='new_confirmation':
   p=src/'records'/f"{c['key']}_cost.json";hashes[str(p)]=sha(p);cost=read(p)
   assert cost['AV']['base']==r['base']==cost['ACM']['base']
   for z,av,cm in zip(r['rows'],cost['AV']['rows'],cost['ACM']['rows']):
    assert (z['t'],z['i'])==(av['t'],av['i'])==(cm['t'],cm['i'])
    assert abs(av['V']-z['V']['total'])<1e-10
    assert cm['minusV']==z['minusV'] and cm['minusM']==z['minusM'];costchecks+=1
  assert r['baseline_only_V_new_calls']==0
 assert len(team)==248 and len({r['key'] for r in team})==124
 cases=ROOT/'contribution_analysis/combined_contribution_layer2_experiment_v1_20261021/cases.json';hashes[str(cases)]=sha(cases)
 proj=[];prowners=[];rounds=[];conservation=0.;decomp=0.
 for c in read(cases)['development']:
  p=ROOT/'contribution_analysis/method5_cost_experiment_20261030_attempt_04/records'/f"{c['key']}_New.json";hashes[str(p)]=sha(p);r=read(p);a=np.asarray(r['weights']);n=len(a);A=np.zeros(n);CA=np.zeros(n);tau=np.zeros(n)
  for t,f in enumerate(r['full']):
   rs=[z for z in r['rows'] if z['t']==t];assert len(rs)==n
   full=np.asarray(rs[0]['full_losses']);gain=float(a@(np.asarray(f['current_losses'])-full));v,zero=projection(f['H'],f['g'],gain)
   conservation=max(conservation,abs(float(sum(v))-gain));decomp=max(decomp,float(np.max(abs(np.asarray(f['H']).sum(0)-f['g']))))
   A+=v
   for z in rs:CA[z['i']]+=z['C_A'];tau[z['i']]+=z['rho']
   rounds.append(dict(key=c['key'],scenario=c['scenario'],t=t,gain=gain,projection_total=float(sum(v)),zero_direction=zero))
  tau=np.maximum(1e-8,tau)
  proj.append(dict(key=c['key'],phase='historical_projection20',scenario=c['scenario'],n=n,**metrics(A,CA,tau)))
  for i in range(n):prowners.append(dict(key=c['key'],scenario=c['scenario'],i=i,projection=float(A[i]),CA=float(CA[i]),tau=float(tau[i]),projection_share=float(share(A,tau)[i]),CA_share=float(share(CA,tau)[i])))
 for name,rows in [('V_CM_teams',team),('V_CM_owners',owners),('V_CM_summary',summarize(team,['phase','judge'])),('V_CM_stratified',summarize(team,['phase','scenario','n','judge'])),('projection_teams',proj),('projection_owners',prowners),('projection_rounds',rounds),('projection_summary',summarize(proj,['scenario']))]:csvout(output/(name+'.csv'),rows)
 audit=dict(passed=True,source_sha256=hashes,analysis_sha256=sha(Path(__file__)),design_sha256=sha(Path(__file__).with_name('design.md')),git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),V_CM_teams=124,V_CM_metric_teams=248,V_CM_score_checks=checks,fresh_cost_endpoint_checks=costchecks,projection_teams=20,projection_owners=80,projection_rounds=len(rounds),projection_conservation_max_error=conservation,direction_decomposition_max_error=decomp,bootstrap_replicates=REPS,rng=SEED,new_model_queries=0,new_solver_calls=0,scope='Reused historical records; V is observed-trial scoring, signed projection is thesis equation, CA/CM are different interventions; not substitution validity or fairness proof.')
 dump(output/'audit.json',audit);print(json.dumps({k:v for k,v in audit.items() if k!='source_sha256'}))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();analyze(a.output_dir)
