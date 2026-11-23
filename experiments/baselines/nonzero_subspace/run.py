"""Nonzero-target restricted-space study; frozen source models are read only."""
import argparse,gzip,hashlib,itertools,json,platform,subprocess,time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linprog
from core.agent_client import AgentNode
from core.host_server import HostServer
from experiments.baselines.paired_benchmark.training import invert_model
from experiments.baselines.paired_benchmark.search import run_search

ROOT=Path(__file__).resolve().parents[3]
PROTOCOL=Path(__file__).with_name('protocol.json')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x

def dump(p,x):p.write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n')
def h(x):
 x=np.asarray(x);return float(x.sum()+(x[:-1]*x[1:]).sum())
def targets(d):return dict(zero=0.,negative=-.5,positive=.5,near_upper=.9*(2*d-1))
def witness(d,target):
 if d<2:raise ValueError('This study requires dimension >=2')
 disc=d*d+4*(d-1)*target
 if disc<0:raise ValueError('No equal-coordinate witness')
 t=2*target/(d+np.sqrt(disc))
 x=np.full(d,t)
 if max(abs(x))>1+1e-12 or abs(h(x)-target)>1e-10:raise ValueError('Invalid target witness')
 return x

def distances(anchors,point):
 A=np.asarray(anchors,dtype=float);point=np.asarray(point)
 linear=np.linalg.lstsq(A.T,point,rcond=None)[0]@A
 centered=A[1:]-A[0]
 affine=A[0]+(np.linalg.lstsq(centered.T,point-A[0],rcond=None)[0]@centered if len(centered) else np.zeros(A.shape[1]))
 return float(np.linalg.norm(point-linear)),float(np.linalg.norm(point-affine))

def geometry(anchors,point):
 A=np.asarray(anchors,dtype=float);dl,da=distances(A,point);centered=A[1:]-A[0];matrix=A.T
 linear_lp=linprog(np.zeros(len(A)),A_ub=np.r_[matrix,-matrix],b_ub=np.ones(2*A.shape[1]),bounds=[(None,None)]*len(A),method='highs')
 affine_lp=linprog(np.zeros(len(A)),A_ub=np.r_[matrix,-matrix],b_ub=np.ones(2*A.shape[1]),A_eq=np.ones((1,len(A))),b_eq=[1.],bounds=[(None,None)]*len(A),method='highs')
 return dict(linear_rank=int(np.linalg.matrix_rank(A)),affine_rank=int(np.linalg.matrix_rank(centered)) if len(centered) else 0,
             witness_linear_distance=dl,witness_affine_distance=da,
             linear_box_feasible=bool(linear_lp.success),affine_box_feasible=bool(affine_lp.success),
             zero_anchor_count=int(sum(np.linalg.norm(A,axis=1)<=1e-12)))

def model_agents(case,state,d,ids):
 agents=[]
 for ident in ids:
  a=AgentNode(ident,None,n_features=d);a.device=torch.device('cpu');a.model.load_state_dict(state['local_states'][ident-1]);a.model.eval();agents.append(a)
 return agents

def execute_task(directory,case,agents,inversions,k,label,target,model_hash):
 directory.mkdir(parents=True,exist_ok=False);start=time.perf_counter();d=case['dimension'];seed=case['seed'];ids=[a.agent_id for a in agents[:k]];valid=len(ids)==k and all(v['status']=='valid' for v in inversions[:k])
 info=dict(seed=seed,dimension=d,rho=case['rho'],requested_k=k,selected_ids=ids,target_label=label,target=target,
           model_sha256=model_hash,witness=witness(d,target),source_selected_ids=case['selected_ids'],
           inversion_forward_samples=sum(v['forward_samples'] for v in inversions[:k]),inversion_backward_steps=sum(v['backward_steps'] for v in inversions[:k]))
 server=HostServer(target,d,constraint_mode='box_constrained',coefficient_eps=1e-5)
 if valid:
  server.trusted_agents=agents[:k];server.alphas=[1/k]*k;server.I_list=[v['X'] for v in inversions[:k]]
  info.update(anchors=server.I_list,alpha=server.alphas,**geometry(server.I_list,info['witness']))
 else:info.update(anchors=[],alpha=[],reason='insufficient_source_coalition' if len(ids)<k else 'inverse_failure')
 dump(directory/'case.json',info);rows=[]
 for method in ['anchor_mean','best_anchor','origin','coefficient','custom','best_of','nelder_mead','direct_slsqp']:
  for budget in ([0] if method in ['anchor_mean','best_anchor','origin'] else [100,200,400]):
   if valid:
    assigned=k if method=='best_anchor' else 1 if budget==0 else budget
    result=run_search(server,method,assigned);trace=directory/f'{method}_{budget}.json.gz'
    events=result.pop('events')
    with gzip.open(trace,'wt') as f:json.dump(clean(events),f,allow_nan=False)
    if result['X'] is not None:
     chosen=next(e for e in events if e.get('X')==result['X'] and e.get('loss')==result['loss'])
     predictions=np.asarray(chosen['predictions']);true=h(result['X'])
     result.update(mean_model_absolute_truth_error=float(np.mean(abs(predictions-true))),ensemble_absolute_truth_error=float(abs(predictions.mean()-true)),prediction_disagreement_sd=float(predictions.std()))
    result.update(trace=trace.name,trace_sha256=sha(trace),assigned_panel_budget=assigned)
   else:result=dict(method=method,X=None,loss=None,status=info['reason'],success=False,model_queries=0,panel_evaluations=0)
   x=result['X'];result.update(seed=seed,dimension=d,rho=case['rho'],requested_k=k,target_label=label,target=target,panel_budget=budget,
       truth_error=None if x is None else abs(h(x)-target),
       operational_loss_truth_gap=None if x is None else abs(result['loss']-abs(h(x)-target)))
   if x is not None:
    dl,da=distances(server.I_list,np.asarray(x));result.update(endpoint_linear_distance=dl,endpoint_affine_distance=da)
   rows.append(clean(result))
 with (directory/'results.jsonl').open('w') as f:
  for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
 dump(directory/'completion.json',dict(status='complete',rows=18,case_sha256=sha(directory/'case.json'),results_sha256=sha(directory/'results.jsonl'),seconds=time.perf_counter()-start))


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args();torch.set_num_threads(1)
 spec=read(PROTOCOL);out=a.output_dir.resolve()
 assert read(a.source/'manifest.json')['status']=='complete' and read(a.source/'acceptance.json')['passed']
 if out.exists():
  if not a.resume:raise FileExistsError(out)
  m=read(out/'manifest.json');assert m['protocol_sha256']==sha(PROTOCOL) and m['smoke']==a.smoke and m['source']==str(a.source.resolve())
 else:
  out.mkdir(parents=True)
  paths=[Path(__file__),PROTOCOL,ROOT/'core/host_server.py',ROOT/'core/agent_client.py',ROOT/'core/models.py',ROOT/'experiments/baselines/paired_benchmark/search.py',ROOT/'experiments/baselines/paired_benchmark/training.py']
  m=dict(status='running',source=str(a.source.resolve()),protocol=spec,protocol_sha256=sha(PROTOCOL),smoke=a.smoke,code_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},input_sha256={},git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),versions=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__))
  dump(out/'manifest.json',m)
 for f,hsh in m['code_sha256'].items():assert sha(ROOT/f)==hsh,f
 conditions=list(itertools.product(spec['seeds'],spec['dimensions'],spec['rhos']))
 if a.smoke:conditions=conditions[:1]
 try:
  for f in ['manifest.json','acceptance.json']:m['input_sha256'][str((a.source/f).resolve())]=sha(a.source/f)
  for seed,d,rho in conditions:
   src=a.source/f'seed{seed}_d{d}_rho{rho:g}';case=read(src/'case.json');payload=torch.load(src/'models.pt',map_location='cpu',weights_only=True)
   for f in ['case.json','models.pt']:
    m['input_sha256'][str((src/f).resolve())]=sha(src/f)
   assert sha(src/'models.pt')==case['models_sha256']
   agents=model_agents(case,payload,d,case['selected_ids'][:max(spec['coalition_sizes'])])
   parent=out/f'seed{seed}_d{d}_rho{rho:g}';parent.mkdir(exist_ok=True)
   for label,target in targets(d).items():
    pool=parent/f'{label}_inversions.json'
    if pool.exists():
     saved=read(pool);assert saved['model_sha256']==sha(src/'models.pt') and saved['target']==target;inversions=saved['inversions']
    else:
     inversions=[invert_model(agent.model,target,seed*100+agent.agent_id,starts=1,steps=500) for agent in agents]
     dump(pool,dict(target=target,model_sha256=sha(src/'models.pt'),ids=[a.agent_id for a in agents],inversions=inversions))
    for k in spec['coalition_sizes']:
     task=parent/f'{label}_k{k}'
     if (task/'completion.json').exists():
      c=read(task/'completion.json');assert c['status']=='complete' and sha(task/'case.json')==c['case_sha256'] and sha(task/'results.jsonl')==c['results_sha256'];continue
     if task.exists():raise RuntimeError(f'Incomplete task preserved: {task}')
     execute_task(task,case,agents,inversions,k,label,target,sha(src/'models.pt'))
   print(seed,d,rho,'8 tasks complete',flush=True)
  for f,hsh in m['input_sha256'].items():assert sha(Path(f))==hsh,f
  m.update(status='complete',source_cases=len(conditions),tasks=8*len(conditions),rows=144*len(conditions));dump(out/'manifest.json',m)
 except BaseException as exc:
  m.update(status='interrupted',exception=repr(exc));dump(out/'manifest.json',m);raise
if __name__=='__main__':main()
