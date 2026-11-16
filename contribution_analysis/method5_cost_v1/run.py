"""Frozen real-model paired offline cost experiment; no evaluator/training/payment."""
import os
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import copy,hashlib,json,math,resource,shutil,signal,sys,time
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parents[2];CA=W/'contribution_analysis';sys.path.insert(0,str(W))
FROZEN=CA/'model_step_decision_experiment_v1_20261021/source_snapshot';sys.path.insert(0,str(FROZEN))
from contribution_analysis.custom_realized_gain_v1.runtime import load_server
from host_server import HostServer
from contribution_analysis.three_pool_t0_t1_v1.engine import settle as basic_settle
from contribution_analysis.three_pool_t0_t1_v1.fixtures import contract as basic_contract,base_events
from contribution_analysis.model_step_decision_v1.engine import run,branch_state
from contribution_analysis.three_pool_t2_anchor_v1.core import decide
from contribution_analysis.anchor_pool_end_to_end_v1.settlement import settle
from contribution_analysis.model_cn_three_way_v1.run import allocation
from contribution_analysis.custom_realized_gain_v1.schema import plain
OUT=CA/'method5_cost_experiment_20261030_attempt_04'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(plain(v),indent=2,allow_nan=False))
class Replies:
 total=0
 def __init__(self,s,context):
  self.s=s;self.context=context;self.cache={};self.events=[];self.stage='baseline';self.seconds=[0.]*len(s.alphas)
 def loss(self,i,x):
  x=np.asarray(x,np.float64);k=self.context+':'+str(i)+':'+x.tobytes().hex();hit=k in self.cache
  if hit:v=self.cache[k]
  else:
   if Replies.total>=1200000:raise RuntimeError('sample_cap')
   t=time.perf_counter();v=abs(float(self.s.trusted_agents[i].api_predict(x)[0])-self.s.target_T);self.seconds[i]+=time.perf_counter()-t;Replies.total+=1;self.cache[k]=v
  if not np.isfinite(v):raise ValueError('nonfinite')
  self.events.append(dict(stage=self.stage,model=i,key=k,x=x.copy(),loss=v,hit=hit));return v
 def losses(self,x):return np.array([self.loss(i,x) for i in range(len(self.s.alphas))])
 def objective(self,ids):
  ids=list(ids);a=np.asarray(self.s.alphas)[ids];a=a/a.sum()
  return lambda x:float(a@np.array([self.loss(i,x) for i in ids]))
def event(tr,t):
 pre=tr.pre[t];post=next((z for z in tr.post if z['next_iteration']==pre['next_iteration']+1),tr.final)
 return dict(pre=pre,post=post,g=tr.gradients[t]['g'])
def base_ledger(c,q):
 # Standard previously validated fixed complete-service accounting fixture,
 # symmetric per arm; not actual dynamic authorization/identity integration.
 contract=basic_contract();events=base_events();result=basic_settle(contract,events)
 return dict(scope='symmetric_fixed_T0T1_fixture_not_live_team_service_integration',contract=contract,events=events,result=result)
def execute(c,s,arm,path):
 start=time.perf_counter();q=Replies(s,sha(c['models'])+':target0:abs:float64:cpu');n=len(s.alphas);ids=list(range(n));a=np.asarray(s.alphas);a=a/a.sum();owners=[str(i) for i in c['ids']]
 probes=[];scale=None
 if arm=='New':
  q.stage='probe';probes=np.array([q.losses(np.array([x,y])) for x in [-1,-.5,0,.5,1] for y in [-1,-.5,0,.5,1]])
  scale=np.quantile(probes,.9,axis=0)-np.quantile(probes,.1,axis=0)
  if (scale<=1e-12).any():raise ValueError('undefined_scale')
 q.stage='baseline';t=time.perf_counter();
 if arm=='New':base,tr=run(s,q.objective(ids),30,initial=np.asarray(c['fixed_start']))
 else:base=HostServer._solve_subspace_custom(s,s.I_list,q.objective(ids),num_iterations=30,direction_weights=s.alphas,initial_solution=np.asarray(c['fixed_start']))
 base_seconds=time.perf_counter()-t
 full=[event(tr,t) for t in range(len(tr.pre))] if arm=='New' else [];baseline_end=len(q.events);rows=[];branches=[]
 if arm=='LOO':
  for i in ids:
   panel=[j for j in ids if j!=i];b=copy.copy(s);b.I_list=np.asarray(s.I_list)[panel];b.alphas=a[panel]/a[panel].sum();q.stage='full_removal'
   result=HostServer._solve_subspace_custom(b,b.I_list,q.objective(panel),num_iterations=30,direction_weights=b.alphas,initial_solution=np.asarray(c['fixed_start']));q.stage='common';lf=q.losses(base['solution']);lm=q.losses(result['solution'])
   rows.append(dict(i=i,owner=owners[i],D=float(a@(lm-lf)),full=base['solution'],minus=result['solution'],full_losses=lf,minus_losses=lm,result=result))
 else:
  anchor_events=[]
  for t,fe in enumerate(full):
   pre=fe['pre'];x=np.asarray(pre['S_current']);q.stage='features';lx=q.losses(x);ds=[];seen=set();H=np.zeros((n,len(x)))
   for k,z in enumerate(s.I_list):
    distance=float(np.linalg.norm(z-x))
    if distance<1e-8:continue
    u=(z-x)/distance;h=distance if pre['current_method']=='secant' else s._bounded_step(x,u,pre['delta'])
    if h<=1e-12:continue
    p=np.asarray(z) if pre['current_method']=='secant' else x+h*u;lp=q.losses(p)
    slope=math.fsum(w*(b-v)/h for w,v,b in zip(a,lx,lp));H[k]+=a[k]*slope*u
    dk=(p.tobytes(),np.float64(h).tobytes(),u.tobytes())
    if dk not in seen:ds.append(dict(k=k,point=p,h=h,u=u,losses=lp,g=(lp-lx)/(scale*h)));seen.add(dk)
   assert np.allclose(H.sum(0),fe['g'],atol=1e-10,rtol=1e-10)
   if not ds:raise ValueError('undefined_directions')
   gs=np.array([d['g'] for d in ds]).T;dist=np.sqrt(np.mean((gs[:,None,:]-gs[None,:,:])**2,axis=2));np.fill_diagonal(dist,np.inf);U=dist.min(1)
   q.stage='common';lf=q.losses(fe['post']['S_current'])
   for i in ids:
    q.stage='anchor';alt=decide(s,q.objective(ids),x,np.asarray(fe['g'])-H[i],pre['best_loss'],pre['eta'],pre['current_method']);q.stage='common';la=q.losses(alt['x']);CAi=float(a@la-a@lf);rho=max(1e-12,32*np.finfo(float).eps*(1+abs(float(a@la))+abs(float(a@lf))))
    anchor_events.append(dict(owner=owners[i],t=t,C=str(CAi),rho=str(rho)))
    q.stage='model';f=q.objective([j for j in ids if j!=i]);state=branch_state(pre,f,s.I_list);_,bt=run(s,f,1,resume=state);xm=bt.final['S_current'];q.stage='common';lm=q.losses(xm);CM=float(a@(lm-lf));I=float(np.linalg.norm(np.asarray(fe['post']['S_current'])-xm)/pre['eta']);N=float((U[i]/10)/(1+U[i]/10)*I/(1+I))
    rows.append(dict(t=t,round=t,i=i,owner=owners[i],C=CM,C_A=CAi,C_M=CM,rho=rho,U=float(U[i]),I=I,N=N,full=fe['post']['S_current'],minusA=alt['x'],minusM=xm,full_losses=lf,minusA_losses=la,minusM_losses=lm))
    branches.append(dict(t=t,i=i,anchor=alt,model_pre=state,model_final=bt.final,model_trials=bt.trials))
   fe.update(directions=ds,H=H,current_losses=lx)
  anchor_allocation=settle(owners,list(range(len(full))),anchor_events);model_allocation=allocation(rows,owners,len(full),.2,True)
 ledger=base_ledger(c,q)
 data=dict(arm=arm,key=c['key'],scenario=c['scenario'],base=base,full=full,rows=rows,branches=branches,events=q.events,probe=probes,scale=scale,weights=a,owners=owners,ledger=ledger,allocation=None if arm=='LOO' else dict(anchor=anchor_allocation,model=model_allocation),model_seconds=q.seconds,baseline_seconds=base_seconds,baseline_samples=sum(not e['hit'] and e['stage']=='baseline' for e in q.events[:baseline_end]),samples=sum(not e['hit'] for e in q.events),requests=len(q.events),hits=sum(e['hit'] for e in q.events),per_model_samples=[sum(e['model']==i and not e['hit'] for e in q.events) for i in ids],peak_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 data['core_seconds']=time.perf_counter()-start;minimal={k:data[k] for k in ['arm','key','base','rows','ledger','allocation','samples']};save(path.with_suffix('.ledger.json'),minimal);data['required_output_seconds']=time.perf_counter()-start;save(path,data);data['wall_seconds']=time.perf_counter()-start
 return data

def main():
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 OUT.mkdir(exist_ok=False);(OUT/'records').mkdir();(OUT/'source_snapshot').mkdir()
 cases=json.loads((CA/'combined_contribution_layer2_experiment_v1_20261021/cases.json').read_text())['development'];dev=json.loads((CA/'anchor_pool_end_to_end_experiment_v1_20261028/cases.json').read_text())[:2]
 paths=[Path(__file__),W/'docs/method5_vs_full_loo_cost_design_20261030.md',W/'docs/method5_three_pool_proposal_20261030.md']
 paths+=list(FROZEN.glob('*.py'))
 for package in ['custom_realized_gain_v1','model_step_decision_v1','three_pool_t2_anchor_v1','anchor_pool_end_to_end_v1','model_cn_three_way_v1','model_consensus_pool_v1','model_raw_c_validation_v1','three_pool_t0_t1_v1']:
  paths+=list((CA/package).glob('*.py'))
 paths+=list({Path(c['models']) for c in cases+dev});paths+= [Path(m.__file__) for m in list(sys.modules.values()) if getattr(m,'__file__',None) and str(getattr(m,'__file__')).startswith(str(W)) and str(getattr(m,'__file__')).endswith('.py')];paths=list(dict.fromkeys(paths));lock={str(p):sha(p) for p in paths};save(OUT/'source_lock.json',lock)
 for p in paths:
  if p.suffix=='.py':shutil.copy2(p,OUT/'source_snapshot'/('__'.join(p.relative_to(W).parts)))
 save(OUT/'registration.json',dict(cases=cases,development=dev,warmups=1,repeats=3,order='rotate New/LOO by case_index+rep',grid='[-1,-.5,0,.5,1]^2',cache='each arm/rep empty; probe precedes baseline; exact model/version/input target0 cpu',sample_cap=1200000,wall_cap=900,baseline_observer='New AST engine; LOO original frozen solver without observer',training=0,pooled=0,truth=0,actual_payment=0))
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('wall_cap')));signal.alarm(900);beg=time.perf_counter();timing=[];done=[]
 try:
  for ci,c in enumerate(dev+cases):
   print(c['key'],flush=True);t=time.perf_counter();s=load_server(c);load=time.perf_counter()-t;saved={};development=ci<2
   for rep in range(1 if development else 4):
    order=['New','LOO'] if (ci+rep)%2==0 else ['LOO','New']
    for arm in order:
     path=OUT/'records'/(c['key']+'_'+arm+'.json');out=execute(c,s,arm,path if rep==0 else OUT/'temporary.json')
     if rep==0:saved[arm]=out
     else:
      assert plain(out['rows'])==plain(saved[arm]['rows']);assert out['samples']==saved[arm]['samples']
      timing.append(dict(key=c['key'],scenario=c['scenario'],arm=arm,repeat=rep,core_seconds=out['core_seconds'],wall_seconds=out['wall_seconds'],required_output_seconds=out['required_output_seconds'],load_seconds=load,model_seconds=sum(out['model_seconds'])))
    if rep==0 and not development:
     oldroot=CA/'custom_direction_experiment_v2_20261020'
     historical=json.loads((oldroot/'baselines'/(c['key']+'.json')).read_text())['result']
     for k,v in plain(saved['LOO']['base']).items():assert v==historical[k]
     assert plain(saved['New']['base'])==plain(saved['LOO']['base'])
     for r in saved['LOO']['rows']:
      historical=json.loads((oldroot/'removals'/(c['key']+'_'+str(c['ids'][r['i']])+'.json')).read_text())['result']
      for k,v in plain(r['result']).items():assert v==historical[k]
   assert plain(saved['New']['base'])==plain(saved['LOO']['base'])
   if development:
    old=json.loads((CA/'anchor_pool_end_to_end_experiment_v1_20261028'/c['key']/'base.json').read_text())
    assert plain(saved['New']['base'])==old['run']['result']
    for row in saved['New']['rows']:
     ref=next(b for b in old['branches'] if b['t']==row['t'] and b['owner']==row['owner'])
     assert abs(row['C_A']-ref['C'])<1e-8
    assert saved['New']['allocation']['anchor']['payments']==old['allocation']['payments']
    save(OUT/(c['key']+'_development_gate.json'),dict(original_result_exact=True,anchor_C_max_error=max(abs(r['C_A']-next(b for b in old['branches'] if b['t']==r['t'] and b['owner']==r['owner'])['C']) for r in saved['New']['rows']),anchor_payments_exact=True))
    if ci==1:
     save(OUT/'development_gate.json',dict(passed=True,teams=2));print('DEVELOPMENT_GATE_PASSED: exit before20; rerun --main to continue',flush=True)
     if '--main' not in sys.argv:return
   else:
    oldroot=CA/'custom_direction_experiment_v2_20261020'
    old=json.loads((oldroot/'baselines'/(c['key']+'.json')).read_text())['result']
    for k,v in plain(saved['LOO']['base']).items():assert v==old[k]
    for r in saved['LOO']['rows']:
     historical=json.loads((oldroot/'removals'/(c['key']+'_'+str(c['ids'][r['i']])+'.json')).read_text())['result']
     for k,v in plain(r['result']).items():assert v==historical[k]
   done.append(dict(key=c['key'],scenario=c['scenario'],development=development,load_seconds=load,arms={k:{f:v for f,v in z.items() if f in ['samples','requests','hits','baseline_samples','core_seconds','wall_seconds','per_model_samples','model_seconds']} for k,z in saved.items()}));save(OUT/'progress.json',done);save(OUT/'timing.json',timing)
  for p,h in lock.items():assert sha(p)==h
  save(OUT/'completion.json',dict(complete=True,teams=20,development_teams=2,physical_model_samples=Replies.total,wall_seconds=time.perf_counter()-beg,training=0,pooled=0,truth=0,actual_payment=0))
 except Exception as e:save(OUT/'failure.json',dict(error=repr(e),completed=len(done),samples=Replies.total));raise
 finally:signal.alarm(0)
if __name__=='__main__':main()
