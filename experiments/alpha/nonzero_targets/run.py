"""Reinvert fixed historical coalitions and replay ten rules at four targets."""
import argparse,copy,gzip,hashlib,json,subprocess,platform,time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from project_support.legacy import prepare
prepare()
from experiments.alpha.replay import load_base,SCENARIOS
from experiments.alpha.compare_initializations import METHODS
from four_alpha_initialization import initializer
from pure_es_initialization import initializer as pure_initializer
from se_qu_initialization import initializer as pair_initializer
from experiments.baselines.paired_benchmark.training import invert_model
from experiments.baselines.paired_benchmark.search import run_search
from experiments.shared.solver_policy import apply_solver_policy
ROOT=Path(__file__).resolve().parents[3]
PLAN=Path(__file__).with_name('protocol.json')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def plain(x):
 if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple,np.ndarray)):return [plain(v) for v in x]
 if isinstance(x,np.generic):return plain(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x

def dump(p,x):p.write_text(json.dumps(plain(x),indent=2,allow_nan=False)+'\n')
def h(x):x=np.asarray(x);return float(x.sum()+(x[:-1]*x[1:]).sum())
def witness(t):return [-1+np.sqrt(1+t)]*2 if t==2.7 else [t,0.]

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
 spec=json.loads(PLAN.read_text());regpath=ROOT/'alpha_weight_outputs/se_qu_extension_v1_20261107/registration.json';reg=json.loads(regpath.read_text())
 runtime=[Path(__file__),PLAN,ROOT/'experiments/baselines/paired_benchmark/search.py',ROOT/'experiments/baselines/paired_benchmark/training.py',ROOT/'core/host_server.py',ROOT/'core/agent_client.py',ROOT/'core/models.py',ROOT/'experiments/alpha/replay.py',ROOT/'experiments/alpha/compare_initializations.py',ROOT/'experiments/shared/solver_policy.py',ROOT/'experiments/shared/solver_policy_v1.json']+list((ROOT/'experiments/alpha/methods').glob('*.py'))
 manifest=dict(status='running',protocol=spec,smoke=a.smoke,source_registration_sha256=sha(regpath),runtime_sha256={str(f.relative_to(ROOT)):sha(f) for f in runtime},input_sha256={},git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),versions=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__),conditions=0)
 dump(a.output_dir/'manifest.json',manifest)
 try:
  for bi,b in enumerate(reg['batches'][:1 if a.smoke else 3],1):
   source=ROOT/f'alpha_weight_outputs/four_alpha_v1_20261014/formal_batch_{bi}/source'
   for seed in b['seeds'][:1] if a.smoke else b['seeds']:
    for scenario in SCENARIOS[:1] if a.smoke else SCENARIOS:
     server,item,artifact=load_base(source,scenario,seed)
     assert [g.agent_id for g in server.trusted_agents]==[1,2,20]
     for f in [artifact/'local_models.pt',source/scenario/'shared_costs.jsonl']:manifest['input_sha256'][str(f.relative_to(ROOT))]=sha(f)
     count=Counter()
     for agent in server.trusted_agents:
      old=agent.api_predict
      def call(x,old=old):
       v=np.asarray(x);count['requests']+=1;count['samples']+=1 if v.ndim==1 else len(v);return old(x)
      agent.api_predict=call
     predictions=np.column_stack([g.api_predict(server.test_X) for g in server.trusted_agents]);mse=((predictions-server.test_y[:,None])**2).mean(0)
     blindcost=dict(count)
     for target in spec['targets']:
      w=witness(target);assert max(abs(np.array(w)))<=1 and abs(h(w)-target)<1e-12
      inverses=[invert_model(g.model,target,seed*100+g.agent_id,starts=1,steps=500) for g in server.trusted_agents]
      directory=a.output_dir/f'{scenario}_{seed}_T{target:g}';directory.mkdir(exist_ok=False)
      if any(v['X'] is None for v in inverses):raise ValueError('Inversion produced no finite anchor; preserve failed run')
      server.target_T=target;server.I_list=[np.array(v['X']) for v in inverses];apply_solver_policy(server,'float32_slsqp_v1')
      count.clear();peer=np.full((3,3),np.nan)
      for j,g in enumerate(server.trusted_agents):
       for i,x in enumerate(server.I_list):
        if i!=j:peer[j,i]=g.api_predict(x)[0]
      supportcost=dict(count)
      meta=dict(batch=bi,seed=seed,scenario=scenario,target=target,source_artifact=str(artifact.relative_to(ROOT)),witness=w,blind_X=server.test_X,blind_y=server.test_y,mse=mse,blind_predictions=predictions,peer_predictions=peer,blind_cost=blindcost,support_cost=supportcost,inverses=inverses,anchors=server.I_list,linear_rank=int(np.linalg.matrix_rank(server.I_list)),affine_rank=int(np.linalg.matrix_rank(np.array(server.I_list)[1:]-server.I_list[0])))
      dump(directory/'case.json',meta);decisions=[]
      for label,method in METHODS.items():
       fn=pure_initializer(peer) if label in ['E','S'] else pair_initializer(peer) if label in ['SE','QU'] else initializer(peer)
       count.clear();alpha,audit=fn(method,server.trusted_agents,mse,server.test_X,server.test_y,server.I_list,target,cached_blind_predictions=predictions,bootstrap_seed=seed,bootstrap_replicates=1000)
       initcost=dict(count);server.alphas=alpha.tolist();count.clear()
       result=run_search(server,'best_of',400)
       assert result['model_queries']==count.get('requests',0)==count.get('samples',0)
       trace=directory/f'{label}_trace.json.gz'
       with gzip.open(trace,'wt') as f:json.dump(plain(result.pop('events')),f,allow_nan=False)
       row=dict(**result,label=label,initializer=method,seed=seed,scenario=scenario,target=target,alpha=alpha,alpha_audit=audit,initialization_requests=initcost.get('requests',0),initialization_samples=initcost.get('samples',0),trace_path=trace.name,trace_sha256=sha(trace),truth_error=None)
       with (directory/'decisions.jsonl').open('a') as f:f.write(json.dumps(plain(row),allow_nan=False)+'\n')
       decisions.append(row)
      for row in decisions:
       if row['X'] is not None:row['truth_error']=abs(h(row['X'])-target)
       with (directory/'results.jsonl').open('a') as f:f.write(json.dumps(plain(row),allow_nan=False)+'\n')
      dump(directory/'completion.json',dict(status='complete',rows=len(decisions),case_sha256=sha(directory/'case.json'),results_sha256=sha(directory/'results.jsonl')))
      manifest['conditions']+=1
     print(bi,seed,scenario,'complete',flush=True)
  manifest['status']='complete'
 except BaseException as exc:
  manifest.update(status='interrupted',exception=repr(exc));raise
 finally:dump(a.output_dir/'manifest.json',manifest)
if __name__=='__main__':main()
