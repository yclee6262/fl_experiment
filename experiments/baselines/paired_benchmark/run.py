"""Run the frozen paired benchmark; all models and search traces stay in outputs."""
import argparse,contextlib,gzip,hashlib,io,itertools,json,platform,subprocess,time
from pathlib import Path
import numpy as np
import torch
import scipy
from core.agent_client import AgentNode
from core.host_server import HostServer
from core.models import PretrainedModel
from experiments.shared.dataset import generate_agent_datasets
from experiments.shared.solver_policy import apply_solver_policy
from experiments.baselines.paired_benchmark.training import train_paired,train_fedavg,invert_model
from experiments.baselines.paired_benchmark.search import run_search

ROOT=Path(__file__).resolve().parents[3]
PROTOCOL=Path(__file__).with_name('protocol.json')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x

def dump(p,x):p.write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n')
def truth(x):
 x=np.array(x);return float(abs(x.sum()+(x[:-1]*x[1:]).sum()))

def make_model(state,d):
 net=PretrainedModel(d);net.load_state_dict(state);return net.eval()

class Node(AgentNode):
 def __init__(self,i,state,d,seed,steps):
  super().__init__(i,None,n_features=d);self.device=torch.device('cpu');self.model=make_model(state,d)
  self.seed=seed;self.steps=steps;self.inverse=None;self.remote_calls=0;self.remote_samples=0
 def api_predict(self,x):
  self.remote_calls+=1;xx=np.asarray(x);self.remote_samples+=1 if xx.ndim==1 else len(xx)
  return super().api_predict(x)
 def infer_parameters_D(self,target,steps=500,n_features=2):
  self.inverse=invert_model(self.model,target,self.seed,starts=1,steps=self.steps)
  if self.inverse['X'] is None: return np.full(self.n_feature,np.nan)
  return np.array(self.inverse['X'])


def execute_case(seed,d,rho,directory,protocol,smoke=False):
 directory.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
 epochs=1 if smoke else 30;samples=32 if smoke else 500;steps=20 if smoke else 500
 records=generate_agent_datasets(num_agents=20,n_samples_per_agent=samples,poison_ratio=rho,n_features=d,seed=seed)
 np.savez_compressed(directory/'data.npz',X=np.stack([r['X'] for r in records]),observed_y=np.stack([r['observed_y'] for r in records]),clean_y=np.stack([r['clean_y'] for r in records]))
 training=train_paired(records,seed,epochs=epochs)
 agents=[Node(i+1,state,d,seed*100+i+1,steps) for i,state in enumerate(training['local_states'])]
 server=HostServer(0,d,test_seed=seed+910000,n_test=20,constraint_mode='box_constrained',coefficient_eps=1e-5)
 selection_error=None
 try:
  with contextlib.redirect_stdout(io.StringIO()) as log:
   server.phase1_filter_agents(agents,mse_threshold=.1,budget_fraction=.8,diversity_eta=.5,enable_inverse_check=True,inverse_target=0.,inverse_steps=steps,inverse_loss_threshold=.1)
  (directory/'qualification.log').write_text(log.getvalue())
 except Exception as exc:
  selection_error=repr(exc)
  (directory/'qualification.log').write_text(log.getvalue())
 selected=server.trusted_agents if selection_error is None else []
 server.I_list=[a.inverse['X'] for a in selected];server.alphas=[1/len(selected)]*len(selected) if selected else []
 if selected:apply_solver_policy(server,'float32_slsqp_v1')
 selected_ids=[a.agent_id for a in selected]
 matched=None
 if selected:
  matched=train_fedavg([r for r in records if r['agent_id'] in selected_ids],seed,training['initial_state'],epochs=epochs)
 payload={k:v for k,v in training.items() if k!='metrics'}
 payload['fedavg_selected_state']=None if matched is None else matched['state']
 torch.save(payload,directory/'models.pt')
 qualifications=[{k:v for k,v in r.items() if k not in ['agent','residual']} for r in server.phase1_report]
 host_samples=sum(a.remote_samples for a in agents);host_calls=sum(a.remote_calls for a in agents)
 local_forward=sum(a.inverse['forward_samples'] for a in agents if a.inverse);local_back=sum(a.inverse['backward_steps'] for a in agents if a.inverse)
 # Each inverse-feasibility proposal is transferred once before its API gate score.
 proposals=sum(a.inverse is not None for a in agents)
 qualification_bytes=4*(d+1)*host_samples+4*d*proposals+8*len(agents)
 parameter_scalars=sum(v.numel() for v in training['initial_state'].values())
 common=dict(seed=seed,dimension=d,rho=rho,selected_ids=selected_ids,selected_count=len(selected),qualified_reports=qualifications,selection_error=selection_error,
   blind_X=server.test_X,blind_y=server.test_y,anchors=server.I_list,alpha=server.alphas,
   training=training['metrics'],fedavg_selected_training=None if matched is None else matched['metrics'],
   parameter_scalars=parameter_scalars,qualification_host_requests=host_calls,qualification_host_samples=host_samples,
   qualification_payload_bytes=qualification_bytes,local_inversion_forward_samples=local_forward,local_inversion_backward_steps=local_back,
   proposal_count=proposals,gate_reuse=True,models_sha256=sha(directory/'models.pt'),data_sha256=sha(directory/'data.npz'))
 dump(directory/'case.json',common)
 allrows=[]
 def write_row(row):
  row.update(seed=seed,dimension=d,rho=rho,selected_count=len(selected),truth_error=None if row.get('X') is None else truth(row['X']))
  # Qualification is a shared prerequisite only for collaborative and matched-FedAvg arms.
  row['qualification_payload_bytes']=0 if row['method']=='fedavg_all' else qualification_bytes
  row['qualification_host_samples']=0 if row['method']=='fedavg_all' else host_samples
  row['local_inversion_forward_samples']=0 if row['method']=='fedavg_all' else local_forward
  row['training_forward_samples']=training['metrics']['training_forward_samples']
  row['training_communication_bytes']=0
  row['training_communication_rounds']=0
  if row['method'].startswith('fedavg'):
   n=20 if row['method']=='fedavg_all' else len(selected)
   row['training_communication_bytes']=2*n*epochs*parameter_scalars*4
   row['training_communication_rounds']=epochs if row['method']=='fedavg_all' or matched is not None else 0
   if row['method']=='fedavg_selected':row['training_forward_samples']+=(0 if matched is None else matched['metrics']['training_forward_samples'])
  row['total_payload_bytes']=row['qualification_payload_bytes']+row.get('transmitted_bytes',0)+row['training_communication_bytes']
  row['qualification_host_requests']=0 if row['method']=='fedavg_all' else host_calls
  row['host_prediction_requests']=row['qualification_host_requests']+row.get('requests',0)
  row['host_prediction_input_samples']=row['qualification_host_samples']+row.get('model_queries',0)
  row['client_inversion_forward_samples_total']=row['local_inversion_forward_samples']+row.get('inversion_forward_samples',0)
  row['qualification_parallel_stages_estimate']=0 if row['method']=='fedavg_all' else (3 if proposals else 1)
  row['search_sequential_panels']=row.get('panel_evaluations',0)
  allrows.append(clean(row))
 for method in protocol['search']['simple_methods']+protocol['search']['methods']:
  budgets=[0] if method in protocol['search']['simple_methods'] else protocol['search']['panel_budgets']
  for budget in budgets:
   if selected:
    allowance=(len(selected) if method=='best_anchor' else 1) if budget==0 else budget
    result=run_search(server,method,allowance)
    trace=directory/f'{method}_{budget}_trace.json.gz'
    with gzip.open(trace,'wt') as f:json.dump(clean(result.pop('events')),f,allow_nan=False)
    result.update(trace_path=trace.name,trace_sha256=sha(trace),panel_budget=budget,actual_assigned_panel_budget=allowance)
   else:result=dict(method=method,panel_budget=budget,X=None,loss=None,status='no_coalition',success=False,stop_reason=selection_error,model_queries=0,transmitted_bytes=0,panel_evaluations=0)
   write_row(result)
 for name,state in [('fedavg_all',training['fedavg_state']),('fedavg_selected',None if matched is None else matched['state'])]:
  if state is None:result=dict(X=None,loss=None,status='no_coalition',start_results=[],forward_samples=0,backward_steps=0)
  else:result=invert_model(make_model(state,d),0.,seed*100+77,starts=5,steps=steps)
  result.update(method=name,panel_budget=0,model_queries=0,transmitted_bytes=0,panel_evaluations=0,
    success=result['status']=='valid',stop_reason='fixed_multistart_local_gradient_inversion' if state is not None else (selection_error or 'no_selected_coalition'),inversion_forward_samples=result.pop('forward_samples'),inversion_backward_steps=result.pop('backward_steps'))
  write_row(result)
 with (directory/'results.jsonl').open('w') as f:
  for r in allrows:f.write(json.dumps(r,allow_nan=False)+'\n')
 dump(directory/'completion.json',dict(status='complete',rows=len(allrows),seconds=time.perf_counter()-start,result_sha256=sha(directory/'results.jsonl'),case_sha256=sha(directory/'case.json')))
 return allrows


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--validation-case',action='store_true');a=p.parse_args()
 if a.smoke and a.validation_case:p.error('Choose one validation mode')
 torch.set_num_threads(1)
 spec=json.loads(PROTOCOL.read_text());out=a.output_dir.resolve();phash=sha(PROTOCOL)
 if out.exists():
  if not a.resume:raise FileExistsError(out)
  manifest=json.loads((out/'manifest.json').read_text());assert manifest['protocol_sha256']==phash and manifest['smoke']==a.smoke and manifest.get('validation_case',False)==a.validation_case
 else:
  out.mkdir(parents=True)
  files=[Path(__file__),Path(__file__).with_name('training.py'),Path(__file__).with_name('search.py'),ROOT/'experiments/shared/solver_policy.py',ROOT/'experiments/shared/solver_policy_v1.json']+[ROOT/'core/host_server.py',ROOT/'core/agent_client.py',ROOT/'core/models.py',ROOT/'experiments/shared/dataset.py',PROTOCOL]
  manifest=dict(status='running',protocol=spec,protocol_sha256=phash,smoke=a.smoke,validation_case=a.validation_case,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),git_status=subprocess.check_output(['git','status','--short'],text=True),code_sha256={str(f.relative_to(ROOT)):sha(f) for f in files},python=platform.python_version(),versions=dict(numpy=np.__version__,torch=torch.__version__,scipy=scipy.__version__),platform=platform.platform())
  dump(out/'manifest.json',manifest)
 for f,h in manifest['code_sha256'].items():assert sha(ROOT/f)==h,f
 cases=[(39999,2,0.)] if a.smoke else [(39998,2,.3)] if a.validation_case else list(itertools.product(spec['seeds'],spec['dimensions'],spec['poison_ratios']))
 for seed,d,rho in cases:
  directory=out/f'seed{seed}_d{d}_rho{rho:g}'
  if (directory/'completion.json').exists():
   complete=json.loads((directory/'completion.json').read_text())
   assert complete['status']=='complete' and complete['rows']==20
   assert sha(directory/'results.jsonl')==complete['result_sha256'] and sha(directory/'case.json')==complete['case_sha256']
   stored=[json.loads(v) for v in (directory/'results.jsonl').read_text().splitlines()]
   assert len(stored)==20 and all((v['seed'],v['dimension'],v['rho'])==(seed,d,rho) for v in stored)
   continue
  if directory.exists():raise RuntimeError(f'Incomplete case preserved: {directory}; choose new output or inspect before retry')
  try:
   rows=execute_case(seed,d,rho,directory,spec,a.smoke)
   print(seed,d,rho,'complete',len(rows),flush=True)
  except BaseException as exc:
   manifest.update(status='interrupted',exception=repr(exc),case=str(directory));dump(out/'manifest.json',manifest);raise
 manifest.update(status='complete',cases=len(cases),rows=20*len(cases));dump(out/'manifest.json',manifest)
if __name__=='__main__':main()
