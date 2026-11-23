"""Independent accounting and saved-model endpoint acceptance for task eight."""
import argparse,gzip,hashlib,itertools,json
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from core.models import PretrainedModel

ROOT=Path(__file__).resolve().parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def records(p):return [json.loads(v) for v in p.read_text().splitlines() if v.strip()]
def model(state,d):
 net=PretrainedModel(d);net.load_state_dict(state);return net.eval()
def prediction(net,x):
 with torch.no_grad():return float(net(torch.tensor(x,dtype=torch.float32)[None,:]).item())

def validate(root):
 torch.set_num_threads(1);out=root/'acceptance.json'
 if out.exists():raise FileExistsError(out)
 manifest=read(root/'manifest.json');assert manifest['status']=='complete'
 for f,h in manifest['code_sha256'].items():assert sha(ROOT/f)==h,f
 files=sorted(root.glob('seed*/completion.json'));assert len(files)==manifest['cases']
 spec=manifest['protocol']
 expected={(39999,2,0.)} if manifest['smoke'] else {(39998,2,.3)} if manifest.get('validation_case') else set(itertools.product(spec['seeds'],spec['dimensions'],spec['poison_ratios']))
 observed=set()
 result_count=event_count=model_queries=wire_bytes=endpoint_evaluations=0;statuses=Counter();hashes={}
 for completefile in files:
  p=completefile.parent;complete=read(completefile);case=read(p/'case.json');rr=records(p/'results.jsonl')
  assert len(rr)==complete['rows']==20 and sha(p/'case.json')==complete['case_sha256'] and sha(p/'results.jsonl')==complete['result_sha256']
  for f,h in [('models.pt',case['models_sha256']),('data.npz',case['data_sha256'])]:assert sha(p/f)==h
  for f in ['completion.json','case.json','results.jsonl','models.pt','data.npz']:hashes[str(p/f)]=sha(p/f)
  assert len({(r['method'],r['panel_budget']) for r in rr})==20
  payload=torch.load(p/'models.pt',map_location='cpu',weights_only=True);d=case['dimension'];k=case['selected_count']
  identity=(case['seed'],d,case['rho']);assert identity in expected and identity not in observed;observed.add(identity)
  n=20;epochs=1 if manifest['smoke'] else 30;samples=32 if manifest['smoke'] else 500
  assert case['training']['training_forward_samples']==n*epochs*samples
  assert case['training']['training_backward_samples']==n*epochs*samples
  assert case['training']['optimizer_steps_all_agents']==n*epochs*((samples+31)//32)
  assert case['qualification_host_samples']==n*20+case['proposal_count']
  assert case['qualification_host_requests']==n+case['proposal_count']
  assert case['qualification_payload_bytes']==4*(d+1)*case['qualification_host_samples']+4*d*case['proposal_count']+8*n
  from experiments.shared.dataset import generate_agent_datasets
  regenerated=generate_agent_datasets(num_agents=n,n_samples_per_agent=samples,poison_ratio=case['rho'],n_features=d,seed=case['seed'])
  saved=np.load(p/'data.npz')
  for field in ['X','observed_y','clean_y']:assert np.array_equal(saved[field],np.stack([z[field] for z in regenerated]))
  nets=[model(payload['local_states'][i-1],d) for i in case['selected_ids']]
  fed={name:None if payload[key] is None else model(payload[key],d) for name,key in [('fedavg_all','fedavg_state'),('fedavg_selected','fedavg_selected_state')]}
  P=sum(v.numel() for v in payload['initial_state'].values());assert P==case['parameter_scalars']
  for r in rr:
   assert (r['seed'],r['dimension'],r['rho'])==(case['seed'],d,case['rho'])
   statuses[r['status']]+=1;result_count+=1
   assert r['host_prediction_requests']==r['qualification_host_requests']+r.get('requests',0)
   assert r['host_prediction_input_samples']==r['qualification_host_samples']+r.get('model_queries',0)
   assert r['total_payload_bytes']==r['qualification_payload_bytes']+r['transmitted_bytes']+r['training_communication_bytes']
   if r['method'].startswith('fedavg'):
    n=20 if r['method']=='fedavg_all' else k
    assert r['training_communication_bytes']==2*n*r['training_communication_rounds']*P*4
    if r['X'] is not None:
     assert abs(abs(prediction(fed[r['method']],r['X']))-r['loss'])<1e-10;endpoint_evaluations+=1
     valid=[z for z in r['start_results'] if z['valid']];assert len(r['start_results'])==5
     assert r['loss']==min(z['loss'] for z in valid)
   elif 'trace_path' in r:
    trace=p/r['trace_path'];assert sha(trace)==r['trace_sha256']
    with gzip.open(trace,'rt') as f:events=json.load(f)
    counted=[e for e in events if e['counted']];finite=[e for e in counted if e.get('loss') is not None]
    assert len(counted)==r['panel_evaluations']<=r['actual_assigned_panel_budget']
    assert sum(e['model_queries'] for e in counted)==r['model_queries']
    assert r['input_payload_bytes']==r['model_queries']*4*d
    assert r['output_payload_bytes']==4*sum(len(e.get('predictions',[])) for e in events)
    assert r['transmitted_bytes']==r['input_payload_bytes']+r['output_payload_bytes']
    if r['method']=='best_of':
     quotas=r['actual_assigned_panel_budget']//2
     assert all(sum(e['engine']==engine for e in counted)<=quotas for engine in ['coefficient','custom'])
    for e in finite:
     assert np.isfinite(e['X']).all() and max(abs(np.array(e['X'])))<=1
     assert len(e['predictions'])==k
     assert abs(np.mean(np.abs(e['predictions']))-e['loss'])<1e-12
    if r['X'] is not None:
     assert any(e['X']==r['X'] and e['loss']==r['loss'] for e in finite)
     assert r['loss']==min(e['loss'] for e in finite)
     predictions=[prediction(net,r['X']) for net in nets];endpoint_evaluations+=k
     assert abs(np.mean(np.abs(predictions))-r['loss'])<1e-10
    event_count+=len(events);model_queries+=r['model_queries'];wire_bytes+=r['transmitted_bytes']
   if r['X'] is not None:
    x=np.array(r['X']);assert np.isfinite(x).all() and max(abs(x))<=1+1e-8
    truth=abs(x.sum()+(x[:-1]*x[1:]).sum());assert abs(truth-r['truth_error'])<1e-12
   else:assert r['truth_error'] is None
 assert result_count==manifest['rows'] and observed==expected
 audit=dict(passed=True,cases=len(files),records=result_count,statuses=dict(statuses),search_trace_events=event_count,
            actual_search_model_queries=model_queries,actual_search_payload_bytes=wire_bytes,
            independent_endpoint_model_evaluations=endpoint_evaluations,all_hard_limits_respected=True,
            observed_best_delivery=True,resource_arithmetic_verified=True,
            source_sha256=hashes,validator_sha256=sha(Path(__file__)))
 out.write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({k:v for k,v in audit.items() if k!='source_sha256'}))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();validate(a.run_dir)
