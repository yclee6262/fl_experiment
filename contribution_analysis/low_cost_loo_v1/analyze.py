import json,csv,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]/'low_cost_loo_exploration_20261116'
def read(n):return json.loads((P/n).read_text())
def save(n,x):
 with (P/n).open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def stats(xs):
 a=np.array([x for x in xs if x is not None],float)
 return dict(n=len(a),mean=float(a.mean()),p50=float(np.median(a)),p90=float(np.quantile(a,.9)),max=float(a.max())) if len(a) else dict(n=0,mean=None,p50=None,p90=None,max=None)
def rank(x):
 x=np.asarray(x);return np.array([1+sum(x<v)+(sum(x==v)-1)/2 for v in x])
def rho(x,y):
 x=rank(x);y=rank(y)
 return float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else None

def allocation(x):
 p=np.maximum(x,0);s=float(sum(p));return (p/s if s>1e-8 else None),s

def sign(x,eps):return np.where(np.asarray(x)>eps,1,np.where(np.asarray(x)<-eps,-1,0))
def main():
 data=read('scored.json');teams=[];people=[]
 for rec in data:
  key=rec['case']['key'];sc=rec['case']['scenario'];ref=rec['methods']['LOO'];check=rec['checks']
  for name,v in rec['methods'].items():
   if name=='LOO':continue
   pos=np.linalg.norm(np.array(v['x'])-ref['x'],axis=1);rawpos=np.linalg.norm(np.array(v['raw'])-ref['x'],axis=1)
   for ev in ['pooled','full','peer']:
    pred=np.array(v[ev]);true=np.array(ref[ev]);err=np.abs(pred-true);q,ps=allocation(pred);qr,rs=allocation(true)
    tv=float(abs(q-qr).sum()/2) if q is not None and qr is not None else None
    maxq=float(max(abs(q-qr))) if tv is not None else None;valid=abs(true)>1e-4
    signs=float(np.mean(sign(pred,1e-4)[valid]==sign(true,1e-4)[valid])) if any(valid) else None
    row=dict(key=key,scenario=sc,method=name,evaluator=ev,n=len(pred),rho=rho(pred,true),sign_match=signs,
     reference_nonzero=int(sum(valid)),sign_hits=int(sum(sign(pred,1e-4)[valid]==sign(true,1e-4)[valid])),
     all_three_class_match=float(np.mean(sign(pred,1e-4)==sign(true,1e-4))),
     q=None if q is None else q.tolist(),q_ref=None if qr is None else qr.tolist(),positive_sum=ps,reference_positive_sum=rs,
     fragile=bool(ps<=1e-4 or rs<=1e-4),tv=tv,max_share_error=maxq,position_mean=float(pos.mean()),
     quality_contribution_mae=float(err.mean()),max_quality_error=float(err.max()),repairs=int(sum(np.array(v['repairs'])>0)),
     total_samples=v['total_samples'],extra_samples=v['extra_samples'],loo_samples=ref['total_samples'],
     signed_nmae=float(err.sum()/abs(true).sum()) if abs(true).sum()>1e-8 else None,
     sign_sensitivity={str(e):float(np.mean(sign(pred,e)[abs(true)>e]==sign(true,e)[abs(true)>e])) if any(abs(true)>e) else None for e in [1e-6,1e-4,1e-3]})
    for th in [.05,.1,.2]:row['pass_'+str(th)]=bool(tv is not None and not row['fragile'] and tv<=th and maxq<=.1 and signs is not None and signs>=.8)
    teams.append(row)
   for i,aid in enumerate(rec['case']['ids']):
    diag=check['diagnostics'][i];D=np.array(diag['cumulative_vector']);trueD=np.array(diag['true_loo_vector']);dn=np.linalg.norm(D);tn=np.linalg.norm(trueD)
    pp=np.array(v['pooled']);rr=np.array(ref['pooled'])
    people.append(dict(key=key,scenario=sc,method=name,agent_id=aid,position_error=float(pos[i]),raw_position_error=float(rawpos[i]),
     normalized_box_position=float(pos[i]/np.sqrt(8)),pooled_score=float(pp[i]),loo_pooled_score=float(rr[i]),
     quality_contribution_error=float(abs(pp[i]-rr[i])),native_full_score=v['full'][i],native_peer_score=v['peer'][i],
     repair_distance=v['repairs'][i],early=diag['early_ratio'] is not None and diag['early_ratio']<.1,
     early_ratio=diag['early_ratio'],strong_cancellation=any(x is not None and x<.1 for x in check['cancellation']),
     model_dominates=diag['model_dominates'],path_length_changed=diag['loo_history_length']!=diag['base_history_length'],
     loo_vector_norm=float(tn),cumulative_vector_norm=float(dn),
     cum_vector_cosine=float(np.clip(np.dot(D,trueD)/dn/tn,-1,1)) if dn>1e-10 and tn>1e-10 else None,
     cum_vector_length_ratio=float(dn/tn) if tn>1e-10 else None))
 summary=[]
 for sc in ['all','clean','controlled_poison']:
  for ev in ['pooled','full','peer']:
   for name in data[0]['methods']:
    if name=='LOO':continue
    ts=[r for r in teams if r['method']==name and r['evaluator']==ev and (sc=='all' or r['scenario']==sc)]
    ps=[r for r in people if r['method']==name and (sc=='all' or r['scenario']==sc)]
    summary.append(dict(scenario=sc,evaluator=ev,method=name,teams=len(ts),
     **{f:stats([r[f] for r in ts]) for f in ['rho','sign_match','tv','max_share_error','quality_contribution_mae','signed_nmae']},
     position=stats([r['position_error'] for r in ps]),quality_error=stats([r['quality_contribution_error'] for r in ps]),
     invalid_allocations=sum(r['tv'] is None for r in ts),fragile_teams=sum(r['fragile'] for r in ts),
     strict_pass={str(th):sum(r['pass_'+str(th)] for r in ts) for th in [.05,.1,.2]},
     tv_only={str(th):sum(r['tv'] is not None and r['tv']<=th for r in ts) for th in [.05,.1,.2]},
     samples=sum(r['total_samples'] for r in ts),extra=sum(r['extra_samples'] for r in ts),loo_samples=sum(r['loo_samples'] for r in ts),
     savings=1-sum(r['total_samples'] for r in ts)/sum(r['loo_samples'] for r in ts),repairs=sum(r['repairs'] for r in ts)))
 hypotheses=[]
 for name in ['LAST_ANCHOR_REPLAY','CUM']:
  for hyp in ['early','strong_cancellation','model_dominates','path_length_changed']:
   for val in [False,True]:
    rows=[p for p in people if p['method']==name and p[hyp]==val]
    hypotheses.append(dict(method=name,hypothesis=hyp,value=val,n=len(rows),position=stats([r['position_error'] for r in rows]),quality=stats([r['quality_contribution_error'] for r in rows])))
 time_rows=read('streaming_timing.json')+[r for r in read('timing.json') if r['method']=='LOO'];timing=[]
 for name in ['LAST_ANCHOR_REPLAY','CUM','LOO']:
  vals=[float(np.median([r['seconds'] for r in time_rows if r['key']==rec['case']['key'] and r['method']==name])) for rec in data]
  timing.append(dict(method=name,total_team_median_seconds=sum(vals),per_team=stats(vals)))
 save('team_metrics.json',teams);save('participant_metrics.json',people);save('summary.json',summary);save('hypotheses.json',hypotheses);save('timing_summary.json',timing)
 for file,rows in [('participant_metrics.csv',people),('team_metrics.csv',teams)]:
  with (P/file).open('x',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 for r in summary:
  if r['scenario']=='all' and r['evaluator']=='pooled':print(r['method'],'TV',r['tv'],'rho',r['rho']['mean'],'sign',r['sign_match']['mean'],'save',r['savings'],'pass',r['strict_pass'],'pos',r['position'],'quality',r['quality_error'],'invalid',r['invalid_allocations'])
 print('timing',timing)
if __name__=='__main__':main()
