import sys,copy,json,time,hashlib,gzip
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parents[2];sys.path.insert(0,str(W))
from contribution_analysis.model_pool_v_cm_validation_v1.run import CA,Replies,analytic_server,run,endpoint,construct_directions,decide,plain,digest
from contribution_analysis.model_pool_combinations_v1.run import HostServer
from contribution_analysis.model_step_decision_v1.engine import branch_state
OLD=CA/'model_pool_v_cm_validation_20261107_attempt_02'
OUT=CA/'peer_exclusion_cost_20261112_attempt_02'
def save(p,v):p.write_text(json.dumps(plain(v),ensure_ascii=False,indent=2,allow_nan=False))
def execute(c,arm,old):
 start=time.perf_counter();s=analytic_server(c);a=np.asarray(s.alphas);a=a/a.sum();ids=list(range(len(a)));q=Replies(s,c['key']);q.stage='baseline';Replies.total=0
 if arm=='LOO':base=HostServer._solve_subspace_custom(s,s.I_list,q.objective(ids),num_iterations=30,direction_weights=a,initial_solution=np.asarray(c['fixed_start']));tr=None
 else:base,tr=run(s,q.objective(ids),30,initial=np.asarray(c['fixed_start']))
 bd=digest(q.events);obs={(e['model'],np.asarray(e['x'],np.float64).tobytes()):e['loss'] for e in q.events}
 def read(x):return np.array([obs[(i,np.asarray(x,np.float64).tobytes())] for i in ids])
 def peer(i,x,y):
  p=[j for j in ids if j!=i];w=a[p]/a[p].sum();return float(w@np.array([q.loss(j,x)-q.loss(j,y) for j in p]))
 rows=[];checks=[]
 if arm=='LOO':
  for i in ids:
   p=[j for j in ids if j!=i];b=copy.copy(s);b.I_list=np.asarray(s.I_list)[p];b.alphas=a[p]/a[p].sum();q.stage='LOO';r=HostServer._solve_subspace_custom(b,b.I_list,q.objective(p),num_iterations=30,direction_weights=b.alphas,initial_solution=np.asarray(c['fixed_start']));q.stage='peer_score';score=peer(i,r['solution'],base['solution']);ref=old['removed'][i];checks.append(np.allclose(r['solution'],ref['result']['solution'],atol=1e-12,rtol=0));rows.append(dict(i=i,solution=r['solution'],score=score))
 else:
  for t in range(len(tr.pre)):
   f=endpoint(tr,t);p=f['pre'];x=np.asarray(p['S_current']);xf=np.asarray(f['post']['S_current']);cur=read(x);lf=read(xf)
   q.stage='features';_,_,H,G=construct_directions(s,q,a,f) if arm in ('AV','ACM') else (None,None,None,None)
   trials=[z for z in tr.trials if z['iteration']==p['next_iteration']]
   if arm=='AV':q.stage='anchor';q.losses(xf)
   for i in ids:
    panel=[j for j in ids if j!=i];w=a[panel]/a[panel].sum();row=dict(t=t,i=i,full=xf)
    if arm=='AV':
     chosen=cur;xv=x
     for z in trials:
      val=read(z['x'])
      if w@val[panel]<w@cur[panel]:chosen=val;xv=np.asarray(z['x']);break
     row.update(V=float(w@(chosen-lf)[panel]),minusV=xv)
    if arm in ('ACM','J'):
     b=copy.copy(s)
     if arm=='J':b.I_list=np.asarray(s.I_list)[panel];b.alphas=w
     q.stage='joint' if arm=='J' else 'model';objective=q.objective(panel);state=branch_state(p,objective,b.I_list);_,bt=run(b,objective,1,resume=state);xm=np.asarray(bt.final['S_current']);q.stage='peer_score';row['J' if arm=='J' else 'CM']=peer(i,xm,xf);row['minusM']=xm
     if arm=='ACM':checks.append(np.allclose(xm,old['rows'][len(rows)]['minusM'],atol=1e-12,rtol=0))
     else:
      # Recompute old full-panel value outside measured cache for endpoint verification only.
      fullscore=float(a@np.array([abs(float(np.array([1.,xm[0],xm[1],xm[0]*xm[1],xm@xm])@np.asarray(c['coefficients'][j]))-s.target_T)-lf[j] for j in ids]));checks.append(abs(fullscore-old['rows'][len(rows)]['J'])<1e-10)
    if arm in ('AV','ACM'):
     q.stage='anchor';alt=decide(s,q.objective(ids),x,np.asarray(f['g'])-H[i],p['best_loss'],p['eta'],p['current_method']);q.stage='anchor_peer_score';row['CA']=peer(i,alt['x'],xf);row['minusA']=alt['x'];checks.append(abs(float(alt['loss']-a@lf)-old['rows'][len(rows)]['CA'])<1e-10)
     if arm=='AV':checks.append(np.allclose(xv,old_acm['rows'][len(rows)]['minusV'],atol=1e-12,rtol=0))
    rows.append(row)
 assert plain(base)==old['base'] and bd==old['baseline_digest'] and all(checks)
 stages={k:dict(requests=sum(e['stage']==k for e in q.events),hits=sum(e['hit'] and e['stage']==k for e in q.events),calls=sum(not e['hit'] and e['stage']==k for e in q.events)) for k in sorted({e['stage'] for e in q.events})}
 return dict(arm=arm,base=base,baseline_digest=bd,rows=rows,events=q.events,stages=stages,requests=len(q.events),hits=sum(e['hit'] for e in q.events),calls=sum(not e['hit'] for e in q.events),cache_entries=len(q.cache),seconds=time.perf_counter()-start,endpoint_checks=len(checks),endpoint_checks_passed=sum(checks),model_seconds=sum(q.seconds))
if __name__=='__main__':
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 OUT.mkdir(exist_ok=False);(OUT/'records').mkdir();cases=json.loads((OLD/'registration.json').read_text())['fresh'];results=[];start=time.perf_counter()
 sources=[Path(__file__), CA/'model_pool_v_cm_validation_v1/run.py', CA/'model_pool_combinations_v1/run.py',CA/'anchor_ca_model_v_cost_v1/run.py', CA/'model_qv_v1/run.py', CA/'model_pool_observed_vd_cost_v1/run.py',CA/'method5_cost_v1/run.py', CA/'model_step_decision_v1/engine.py',CA/'three_pool_t2_anchor_v1/core.py',OLD/'registration.json']+list((CA/'model_step_decision_experiment_v1_20261021/source_snapshot').glob('*.py'))+[OLD/'records'/f"{c['key']}_cost.json" for c in cases]
 locks={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources};save(OUT/'source_lock.json',locks)
 save(OUT/'registration.json',dict(cases=cases,arms=['AV','ACM','J','LOO'],iterations=30,score='normalized remaining alpha: F_-i(alternative)-F_-i(full)',independent_cold_cache=True,source=str(OLD)))
 for c in cases:
  print(c['key'],flush=True);old=json.loads((OLD/'records'/f"{c['key']}_cost.json").read_text());old_acm=old['ACM'];item={}
  for arm in ['AV','ACM','J','LOO']:
   item[arm]=execute(c,arm,old[arm]);events=item[arm].pop('events');save(OUT/'records'/f"{c['key']}_{arm}.json",item[arm]);
   with gzip.open(OUT/'records'/f"{c['key']}_{arm}_events.json.gz",'wt') as f: json.dump(plain(events),f,allow_nan=False,separators=(',',':'))
  results.append(dict(key=c['key'],n=c['n'],scenario=c['scenario'],arms={k:{z:v[z] for z in ['calls','requests','hits','cache_entries','seconds','model_seconds','stages','endpoint_checks','endpoint_checks_passed']} for k,v in item.items()}));save(OUT/'progress.json',results)
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in locks.items())
 save(OUT/'completion.json',dict(source_and_historical_hashes_unchanged=True,complete=True,teams=len(results),wall_seconds=time.perf_counter()-start,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
