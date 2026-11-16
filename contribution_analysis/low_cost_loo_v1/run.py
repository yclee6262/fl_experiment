"""Old-cohort exploratory endpoints; frozen solver, independent method caches."""
import sys,json,time,hashlib,copy,signal
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parents[2]; CA=W/'contribution_analysis'
OUT=CA/'low_cost_loo_exploration_20261116'; OLD=CA/'custom_direction_experiment_v2_20261020'
MOD=CA/'model_step_decision_experiment_v1_20261021'; FOUR=CA/'contribution_four_way_experiment_v2_20261022'
L2=CA/'combined_contribution_layer2_experiment_v1_20261021'
sys.path.insert(0,str(MOD/'source_snapshot'));sys.path.insert(0,str(CA/'contribution_four_way_v1'))
from study import load_server
from host_server import HostServer
# avoid importing another run.py by module name
import importlib.util
spec=importlib.util.spec_from_file_location('four_runner',CA/'contribution_four_way_v1/run.py');four=importlib.util.module_from_spec(spec);spec.loader.exec_module(four)
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from research_evaluation import load_evaluator_artifact,score_solution

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,dict):return {str(k):plain(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [plain(x) for x in v]
 return v

def save(n,v):
 with (OUT/n).open('x') as f:json.dump(plain(v),f,indent=2,ensure_ascii=False,allow_nan=False)

def vectors(base,n):
 parts=[];mparts=[];checks=[];cancellation=[]
 for it in base['iterations']:
  h=np.zeros((n,2))
  for v in it['directions']:h[v['k']]=v['weight']*v['derivative']*np.asarray(v['u'])
  g=np.asarray(it['g']);assert np.linalg.norm(h.sum(0)-g)<1e-9
  ac=[v for v in it['trials'] if v['accepted']];assert len(ac)<=1
  dx=np.asarray(ac[0]['x'])-it['x'] if ac else np.zeros(2)
  scale=-np.linalg.norm(dx)/np.linalg.norm(g) if ac else 0.
  d=scale*h;m=scale*np.asarray(it['model_vectors'])
  checks.extend([np.linalg.norm(d.sum(0)-dx),np.linalg.norm(m.sum(0)-dx)])
  assert max(checks)<1e-9
  parts.append(d);mparts.append(m)
  den=np.linalg.norm(h,axis=1).sum();cancellation.append(float(np.linalg.norm(g)/den) if den else None)
 d=np.asarray(parts);md=np.asarray(mparts);xf=np.asarray(base['result']['solution'])
 err=np.linalg.norm(d.sum((0,1))-(xf-base['case']['fixed_start']));assert err<1e-9
 return d,md,dict(max_step_reconstruction=max(checks),total_reconstruction=err,cancellation=cancellation)

class Q(four.Queries):
 total_new=0
 def reply(self,i,x):
  hit=self.stage!='baseline' and four.key(i,x) in self.cache
  if not hit:
   if Q.total_new>=300000:raise RuntimeError('local_budget')
   Q.total_new+=1
  return super().reply(i,x)

def prep(c):
 base=read(OLD/'baselines'/(c['key']+'.json'));model=read(MOD/'development'/(c['key']+'.json'))
 layer=read(L2/'development'/(c['key']+'.json'));groups={(g['t'],g['i']):g for g in layer['groups']}
 rem=[read(OLD/'removals'/(c['key']+'_'+str(i)+'.json')) for i in c['ids']]
 return base,model,groups,rem

def points(c,base,model,groups,rem):
 n=len(c['ids']);a=np.asarray(c['alphas']);d,md,check=vectors(base,n);xf=np.asarray(base['result']['solution'])
 ac=[t for t,it in enumerate(base['iterations']) if it['accepted']];k=ac[-1] if ac else None;last=len(d)-1
 dk=d[k] if k is not None else np.zeros((n,2));pre=np.asarray(base['iterations'][k]['x']) if k is not None else xf
 p={'LAST':xf-dk,'LAST_TRY':xf-d[last],'CUM':xf-d.sum(0),'CUM_MODEL':xf-md.sum(0),
    'LAST_RENORM':pre+(dk.sum(0)-dk)/(1-a[:,None]),'LOO':np.asarray([r['result']['solution'] for r in rem])}
 for name,t,role in [('LAST_ANCHOR_REPLAY',k,'anchor'),('LAST_JOINT_REPLAY',k,'joint'),('LAST_TRY_REPLAY',last,'anchor')]:
  p[name]=np.asarray([groups[t,i]['roles'][role]['endpoint'] for i in range(n)]) if t is not None else np.tile(xf,(n,1))
 for r in rem:
  assert r['fixed_start']==c['fixed_start'];assert r['result']['box_feasible']
  assert np.allclose(r['alphas'],np.delete(a,r['excluded_index'])/(1-a[r['excluded_index']]),atol=1e-15)
 final=xf
 assert np.linalg.norm(np.asarray(model['full'][-1]['post']['S_current'])-final)<1e-10
 path=np.linalg.norm(d,axis=2).sum(0)
 diagnostics=[]
 for i in range(n):
  group=groups[k,i] if k is not None else None
  aa=np.linalg.norm(np.asarray(group['roles']['anchor']['endpoint'])-xf) if group else 0
  mm=np.linalg.norm(np.asarray(group['roles']['model']['endpoint'])-xf) if group else 0
  diagnostics.append(dict(agent_id=c['ids'][i],early_ratio=float(np.linalg.norm(dk[i])/path[i]) if path[i]>0 else None,
   cumulative_vector=d[:,i].sum(0),true_loo_vector=xf-p['LOO'][i],model_last_distance=mm,anchor_last_distance=aa,
   model_dominates=bool(mm>aa),loo_history_length=len(rem[i]['result']['history']),base_history_length=len(base['result']['history']),
   cumulative_norm=float(np.linalg.norm(d[:,i].sum(0))),individual_path=float(path[i])))
 return p,dict(**check,last_accepted=k,last_attempted=last,accepted_rounds=len(ac),rounds=len(d),
   last_rejected=not base['iterations'][-1]['accepted'],diagnostics=diagnostics)

def execute(c,s,base,model,groups,rem,p,name,replay=False):
 q=Q(s);a=np.asarray(c['alphas']);n=len(a);tic=time.perf_counter()
 if replay:
  br=HostServer._solve_subspace_custom(s,s.I_list,q.objective(list(range(n))),num_iterations=30,direction_weights=a,initial_solution=np.asarray(c['fixed_start']))
  assert np.array_equal(br['solution'],base['result']['solution']);assert br['history']==base['result']['history']
 else:
  # Only baseline requests may enter a method's independent cache.
  for e in model['events']:
   if e['stage']=='baseline':q.cache[four.key(e['i'],e['x'])]=e['y']
 baseline_sec=time.perf_counter()-tic;q.stage='post';vec_start=time.perf_counter()
 if name in ['CUM','CUM_MODEL','LAST','LAST_TRY','LAST_RENORM']:
  pp,_=points(c,base,model,groups,rem);x=pp[name]
 else:x=p[name].copy()
 vector_sec=time.perf_counter()-vec_start
 if name in ['LAST_ANCHOR_REPLAY','LAST_JOINT_REPLAY','LAST_TRY_REPLAY']:
  ac=[t for t,it in enumerate(base['iterations']) if it['accepted']]
  t=len(base['iterations'])-1 if name=='LAST_TRY_REPLAY' else (ac[-1] if ac else None)
  role='joint' if name=='LAST_JOINT_REPLAY' else 'anchor'
  x=np.array([four.local(s,q,model['full'][t]['pre'],i,role) for i in range(n)]) if t is not None else p[name]
  assert np.allclose(x,p[name],rtol=0,atol=1e-10)
 if name=='LOO' and replay:
  xx=[]
  for i in range(n):
   panel=[j for j in range(n) if j!=i];b=copy.copy(s);b.I_list=s.I_list[panel];b.alphas=a[panel]/sum(a[panel]);q.stage='loo'
   r=HostServer._solve_subspace_custom(b,b.I_list,q.objective(panel),num_iterations=30,direction_weights=b.alphas,initial_solution=np.asarray(c['fixed_start']))
   assert np.array_equal(r['solution'],rem[i]['result']['solution']);assert r['history']==rem[i]['result']['history'];xx.append(r['solution'])
  x=np.array(xx)
 raw=x.copy();repair_start=time.perf_counter();x=np.clip(x,-1,1);repair_sec=time.perf_counter()-repair_start
 q.stage='common';lf=q.losses(base['result']['solution']);lm=np.array([q.losses(xx) for xx in x]);delta=lm-lf
 full=delta@a;peer=np.array([np.dot(np.delete(a,i),np.delete(delta[i],i))/(1-a[i]) for i in range(n)])
 total_sec=time.perf_counter()-tic
 extra=sum(not e['hit'] and e['stage']!='baseline' for e in q.events)
 return dict(x=x,raw=raw,full=full,peer=peer,baseline_samples=base['counter']['samples'],extra_samples=extra,
   total_samples=base['counter']['samples']+extra,cache_hits=sum(e['hit'] for e in q.events),
   seconds=total_sec,baseline_seconds=baseline_sec,vector_seconds=vector_sec,repair_seconds=repair_sec,
   query_events=q.events)

def main():
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('900sec')));signal.alarm(900)
 cases=read(FOUR/'cases.json');paths=[OUT/'design.md',Path(__file__),FOUR/'cases.json',FOUR/'comparisons.json']
 for c in cases:
  paths.extend([Path(c['models']),Path(c['evaluator']),OLD/'baselines'/(c['key']+'.json'),MOD/'development'/(c['key']+'.json'),L2/'development'/(c['key']+'.json')])
  paths.extend(OLD/'removals'/(c['key']+'_'+str(i)+'.json') for i in c['ids'])
 paths.extend((MOD/'source_snapshot').glob('*.py'));paths.append(CA/'contribution_four_way_v1/run.py')
 hashes={str(p):sha(p) for p in paths};save('source_lock.json',hashes)
 records=[];timings=[];start=time.perf_counter()
 for ci,c in enumerate(cases):
  print(c['key'],flush=True);base,model,groups,rem=prep(c);p,checks=points(c,base,model,groups,rem);s=load_server(c)
  rec=dict(case=c,checks=checks,methods={})
  for name in p:
   if name=='LOO':continue
   rec['methods'][name]=execute(c,s,base,model,groups,rem,p,name)
  # Same present machine and frozen models, 1 warmup and 3 measured passes.
  methods=['LAST_ANCHOR_REPLAY','CUM','LOO']
  for rep in range(4):
   order=methods[(ci+rep)%3:]+methods[:(ci+rep)%3]
   for name in order:
    v=execute(c,s,base,model,groups,rem,p,name,True)
    if name=='LOO' and rep==0:rec['methods'][name]=v
    if name!='LOO':
     assert np.array_equal(v['x'],rec['methods'][name]['x']);assert v['extra_samples']==rec['methods'][name]['extra_samples']
    if rep:timings.append(dict(key=c['key'],scenario=c['scenario'],method=name,repeat=rep,seconds=v['seconds'],vector_seconds=v['vector_seconds'],baseline_seconds=v['baseline_seconds']))
  records.append(rec)
 save('endpoints_locked.json',records);save('timing.json',timings)
 # Evaluation begins only after endpoints are locked.
 old={(r['key'],r['agent_id']):r for r in read(FOUR/'comparisons.json')};pooled_count=0
 for rec in records:
  c=rec['case'];model,_=load_evaluator_artifact(Path(c['evaluator']).parent);cache={}
  def E(x):
   nonlocal pooled_count
   k=np.asarray(x,np.float64).tobytes().hex()
   if k not in cache:
    if pooled_count>=3000:raise RuntimeError('pooled_cap')
    cache[k]=score_solution(model,x,0)[1];pooled_count+=1
   return cache[k]
  xf=read(OLD/'baselines'/(c['key']+'.json'))['result']['solution'];ef=E(xf);rec['pooled_full_loss']=ef
  for name,v in rec['methods'].items():
   v['pooled_loss']=[E(x) for x in v['x']];v['pooled']=np.array(v['pooled_loss'])-ef
   v['repairs']=np.linalg.norm(v['raw']-v['x'],axis=1)
  for i,aid in enumerate(c['ids']):
   assert abs(rec['methods']['LOO']['pooled'][i]-old[c['key'],aid]['D_E'])<1e-7
   assert abs(rec['methods']['LOO']['full'][i]-old[c['key'],aid]['D_L'])<1e-10
  rec['pooled_unique_samples']=len(cache)
  for v in rec['methods'].values():v.pop('query_events')
 save('scored.json',records)
 for p,h in hashes.items():assert sha(p)==h,p
 save('completion.json',dict(teams=len(records),participants=sum(len(r['case']['ids']) for r in records),
  local_physical_samples=Q.total_new,pooled_unique_samples=pooled_count,wall_seconds=time.perf_counter()-start,
  historical_sources_unchanged=True,training=0,independent_confirmation=False))
 signal.alarm(0)
if __name__=='__main__':main()
