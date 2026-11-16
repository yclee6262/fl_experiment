"""Exploratory read-only recomputation; no model loading or calls."""
import json,hashlib
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parents[1]/'senior_remaining_20261109_attempt_01'
OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
reg=read(BASE/'registration.json');out=[];rows=[];threshold=[];locks={}
for c in reg['cases']:
 pth=BASE/'records'/f"{c['key']}_base.json";b=read(pth);locks[str(pth)]=hashlib.sha256(pth.read_bytes()).hexdigest();n=b['n'];T=len(b['rows'])//n
 ex=np.array([r['exact'] for r in b['rows']]);pr=np.array([r['proxy'] for r in b['rows']]);R=ex-pr
 for i in range(n):
  for role in [0,1]:
   e=R[np.array([r['i'] for r in b['rows']])==i,role,0];energy=float(e@e)
   rows.append(dict(key=c['key'],i=i,role=['A','M'][role],rounds=T,nonzero_residuals=int(np.sum(abs(e)>1e-10)),top1_variance_share=float(max(e*e)/energy) if energy else None,effective_residual_count=float(energy**2/np.sum(e**4)) if energy else None))
 for rate in [.05,.2,.5]:
  rng=np.random.default_rng(20261109000+c['seed']+int(rate*100));ind=rng.random((1000,len(ex),2,1))<rate
  est=(pr[None]+ind*R[None]/rate).reshape(1000,T,n,2,2).sum(1);var=(ind*R[None]**2*(1-rate)/rate**2).reshape(1000,T,n,2,2).sum(1)
  true=np.array(b['sums']);oracle=((1-rate)/rate*R**2).reshape(T,n,2,2).sum(0)
  for role in [0,1]:
   err=np.abs(est[:,:,role,0]-true[:,role,0]);normal=err<=1.96*np.sqrt(var[:,:,role,0])+1e-10;oracleok=err<=1.96*np.sqrt(oracle[:,role,0])+1e-10
   zero=var[:,:,role,0]==0
   out.append(dict(key=c['key'],rate=rate,role=['A','M'][role],estimated_coverage=float(normal.mean()),oracle_variance_normal_coverage=float(oracleok.mean()),zero_variance_fraction=float(zero.mean()),zero_variance_misses=int(np.sum(zero&~normal)),zero_variance_count=int(zero.sum())))
 for pp in sorted((BASE/'records').glob(c['key']+'_*.json')):
  if 'sparse' in pp.name:continue
  d=read(pp);S=np.array(d['sums'])[:,:,0];old=(S[:,0]<-1e-8)&(S[:,1]<-1e-10);new=(S[:,0]<-1e-10)&(S[:,1]<-1e-10)
  def share(t):
   w=np.where(S[:,0]>t,S[:,0],0);return np.r_[w/w.sum(),0] if w.sum() else np.r_[w,1]
  threshold.append(dict(file=pp.name,basic_diff=int(np.sum(old!=new)),anchor_TV=float(abs(share(1e-8)-share(1e-10)).sum()/2)))
summary={}
for role in ['A','M']:
 rr=[r for r in rows if r['role']==role];summary[role]=dict(participants=len(rr),median_rounds=float(np.median([r['rounds'] for r in rr])),median_effective_residuals=float(np.median([r['effective_residual_count'] for r in rr if r['effective_residual_count'] is not None])),median_top1_variance_share=float(np.median([r['top1_variance_share'] for r in rr if r['top1_variance_share'] is not None])),top1_over_half=sum(r['top1_variance_share']>.5 for r in rr if r['top1_variance_share'] is not None),rates=[])
 for rate in [.05,.2,.5]:
  vals=[r for r in out if r['role']==role and r['rate']==rate];summary[role]['rates'].append(dict(rate=rate,**{k:float(np.mean([r[k] for r in vals])) for k in ['estimated_coverage','oracle_variance_normal_coverage','zero_variance_fraction']},zero_misses=sum(r['zero_variance_misses'] for r in vals),zero_count=sum(r['zero_variance_count'] for r in vals)))
summary['threshold_change']=dict(records=len(threshold),basic_changes=sum(r['basic_diff'] for r in threshold),max_anchor_TV=max(r['anchor_TV'] for r in threshold))
result=dict(scope='new exploratory audit using old arrays, not new confirmation',model_calls=0,summary=summary,per_owner=rows,per_team=out,threshold=threshold,source_hashes=locks)
(OUT/'diagnostics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False));print(json.dumps(summary,indent=2))
