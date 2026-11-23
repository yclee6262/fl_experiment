"""Fresh role endpoints for multiple targets on fixed historical model teams."""
import argparse,copy,gzip,hashlib,json,subprocess,time,inspect,platform
from pathlib import Path
import numpy as np
import torch
# These maintained imports register the frozen solver before any core alias.
from contribution_analysis.model_pool_observed_vd_cost_v1.run import (
    load_server,HostServer,run,endpoint,construct_directions,decide,FROZEN)
from contribution_analysis.model_step_decision_v1.engine import branch_state
from experiments.baselines.paired_benchmark.training import invert_model
ROOT=Path(__file__).resolve().parents[3]
CA=ROOT/'contribution_analysis'
TARGETS=[0.,-.5,.5,2.7]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
 if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
 if isinstance(x,(tuple,list,np.ndarray)):return [plain(v) for v in x]
 if isinstance(x,np.generic):return plain(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x

def dump(p,x):p.write_text(json.dumps(plain(x),indent=2,allow_nan=False)+'\n')
def truth(x,T):
 x=np.asarray(x);return float(abs(x.sum()+x[0]*x[1]-T))

def witness(T):
 if abs(T)<=1:return [T,0.]
 t=-1+np.sqrt(1+T);return [t,t]

class Replies:
 def __init__(self,s):self.s=s;self.cache={};self.events=[];self.stage='full'
 def loss(self,i,x):
  x=np.asarray(x,dtype=float);key=(i,x.tobytes());hit=key in self.cache
  if hit:value=self.cache[key]
  else:
   if len(self.cache)>=100000:raise RuntimeError('per-task cache-miss cap exceeded')
   value=abs(float(self.s.trusted_agents[i].api_predict(x)[0])-self.s.target_T)
   if not np.isfinite(value):raise ValueError('Nonfinite model reply')
   self.cache[key]=value
  self.events.append(dict(stage=self.stage,i=i,X=x.copy(),loss=value,hit=hit));return value
 def losses(self,x):return np.array([self.loss(i,x) for i in range(len(self.s.alphas))])
 def objective(self,ids):
  ids=list(ids);a=np.asarray(self.s.alphas)[ids];a=a/a.sum()
  return lambda x:float(a@np.array([self.loss(i,x) for i in ids]))


def split(delta,a,i):
 delta=np.asarray(delta);peer=[j for j in range(len(a)) if j!=i]
 return dict(F=float(a@delta),Fminus=float((a[peer]/a[peer].sum())@delta[peer]))


def execute(c,T,iterations=30,steps=500):
 start=time.perf_counter();s=load_server(c);s.target_T=T;n=len(s.alphas);a=np.ones(n)/n;s.alphas=a
 inv=[]
 for i,agent in enumerate(s.trusted_agents):
  result=invert_model(agent.model,T,c['seed']*100+int(c['ids'][i]),starts=1,steps=steps)
  inv.append(result)
 if any(z['X'] is None for z in inv):return dict(status='inversion_failure',inversions=inv)
 s.I_list=np.array([z['X'] for z in inv]);fixed=s.I_list.mean(0);q=Replies(s);ids=list(range(n))
 q.stage='full';full,tr=run(s,q.objective(ids),iterations,initial=fixed)
 # Verify the observer does not alter the original custom engine.
 control=Replies(s);control.stage='observer_equivalence'
 check=HostServer._solve_subspace_custom(s,s.I_list,control.objective(ids),num_iterations=iterations,direction_weights=a,initial_solution=fixed)
 assert np.allclose(check['solution'],full['solution'],atol=1e-12,rtol=0) and abs(check['loss']-full['loss'])<1e-12
 rows=[]
 for t in range(len(tr.pre)):
  f=endpoint(tr,t);pre=f['pre'];xf=np.asarray(f['post']['S_current']);q.stage='directions'
  _,_,H,_=construct_directions(s,q,a,f);lf=q.losses(xf)
  for i in ids:
   peer=[j for j in ids if j!=i];q.stage='anchor'
   alt=decide(s,q.objective(ids),pre['S_current'],np.asarray(f['g'])-H[i],pre['best_loss'],pre['eta'],pre['current_method'])
   la=q.losses(alt['x']);rho=max(1e-12,32*np.finfo(float).eps*(1+abs(float(a@la))+abs(float(a@lf))))
   role={'A':dict(X=alt['x'],losses=la,score=split(la-lf,a,i))}
   for label in ['M','J']:
    b=copy.copy(s)
    if label=='J':b.I_list=s.I_list[peer];b.alphas=a[peer]/a[peer].sum()
    q.stage=label;objective=q.objective(peer);state=branch_state(pre,objective,b.I_list)
    _,bt=run(b,objective,1,resume=state);x=np.asarray(bt.final['S_current']);loss=q.losses(x)
    role[label]=dict(X=x,losses=loss,score=split(loss-lf,a,i))
   rows.append(dict(t=t,i=i,full_X=xf,full_losses=lf,rho=rho,roles=role))
 q.stage='full_LOO';loo=[]
 for i in ids:
  peer=[j for j in ids if j!=i];b=copy.copy(s);b.I_list=s.I_list[peer];b.alphas=a[peer]/a[peer].sum()
  result=HostServer._solve_subspace_custom(b,b.I_list,q.objective(peer),num_iterations=iterations,direction_weights=b.alphas,initial_solution=fixed)
  loss=q.losses(result['solution']);base=q.losses(full['solution'])
  loo.append(dict(i=i,X=result['solution'],losses=loss,full_losses=base,score=split(loss-base,a,i),truth_effect=truth(result['solution'],T)-truth(full['solution'],T)))
 participants=[]
 for i in ids:
  own=[z for z in rows if z['i']==i]
  participants.append(dict(i=i,id=c['ids'][i],tauA=max(1e-8,sum(z['rho'] for z in own)),scores={judge:{role:sum(z['roles'][role]['score'][judge] for z in own) for role in ['A','M','J']} for judge in ['F','Fminus']},truth_LOO=loo[i]['truth_effect']))
 finite=[full['solution']]+[v['X'] for z in rows for v in z['roles'].values()]+[z['X'] for z in loo]
 assert all(np.isfinite(x).all() and max(abs(np.asarray(x)))<=1+1e-8 for x in finite)
 return dict(status='valid',key=c['key'],scenario=c['scenario'],seed=c['seed'],target=T,n=n,ids=c['ids'],anchors=s.I_list,start=fixed,alpha=a,
  inversions=inv,witness=witness(T),full=full,truth_error=truth(full['solution'],T),rows=rows,LOO=loo,participants=participants,
  actual_rounds=len(tr.pre),cache_miss_calls=len(q.cache),requests=len(q.events),events=q.events,
  inversion_forward_samples=sum(z['forward_samples'] for z in inv),
  inverse_absolute_errors=[z['loss'] for z in inv],inverse_target_error_above_point1=sum(z['loss']>.1 for z in inv),
  observer_equivalence_cache_miss_calls=len(control.cache),observer_equivalence_requests=len(control.events),
  seconds=time.perf_counter()-start)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 a.output_dir.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
 source=CA/'combined_contribution_layer2_experiment_v1_20261021/cases.json';cases=json.loads(source.read_text())['development']
 actual_functions=[HostServer._solve_subspace_custom,load_server,run,endpoint,construct_directions,decide,invert_model]
 files=[Path(inspect.getsourcefile(f)).resolve() for f in actual_functions]+[Path(__file__),Path(__file__).with_name('design.md'),ROOT/'experiments/baselines/paired_benchmark/training.py',CA/'model_step_decision_v1/engine.py',CA/'three_pool_t2_anchor_v1/core.py']+list(FROZEN.glob('*.py'))+[Path(c['models']) for c in cases]
 hashes={str(f.resolve()):sha(f) for f in files};hashes[str(source)]=sha(source)
 jobs=[(cases[0],.5)] if a.smoke else [(c,T) for c in cases for T in TARGETS]
 manifest=dict(status='running',python=platform.python_version(),versions=dict(numpy=np.__version__,torch=torch.__version__),actual_function_sources={f.__name__:str(Path(inspect.getsourcefile(f)).resolve()) for f in actual_functions},frozen_host_sha256=sha(FROZEN/'host_server.py'),smoke=a.smoke,jobs=len(jobs),targets=TARGETS,source_sha256=hashes,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
 dump(a.output_dir/'manifest.json',manifest)
 completed=[]
 try:
  for c,T in jobs:
   name=f"{c['key']}_target{T:g}";result=execute(c,T,iterations=3 if a.smoke else 30,steps=20 if a.smoke else 500)
   with gzip.open(a.output_dir/(name+'.json.gz'),'wt') as f:json.dump(plain(result),f,allow_nan=False)
   completed.append(dict(key=c['key'],scenario=c['scenario'],seed=c['seed'],target=T,status=result['status'],file=name+'.json.gz',sha256=sha(a.output_dir/(name+'.json.gz'))))
   dump(a.output_dir/'index.json',completed);print(name,result['status'],flush=True)
  for f,h in hashes.items():assert sha(Path(f))==h
  manifest.update(status='complete',completed=len(completed),cache_miss_calls=sum(json.loads(gzip.open(a.output_dir/z['file'],'rt').read()).get('cache_miss_calls',0) for z in completed))
 except BaseException as exc:manifest.update(status='interrupted',exception=repr(exc));raise
 finally:dump(a.output_dir/'manifest.json',manifest)
if __name__=='__main__':main()
