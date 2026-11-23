"""Independent saved-model, trace and subspace acceptance for task nine."""
import argparse,gzip,hashlib,itertools,json
from pathlib import Path
import numpy as np
import torch
from core.models import PretrainedModel
from experiments.baselines.nonzero_subspace.run import h,witness,distances

ROOT=Path(__file__).resolve().parents[3]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def validate(source):
 torch.set_num_threads(1);output=source/'acceptance.json'
 if output.exists():raise FileExistsError(output)
 m=read(source/'manifest.json');assert m['status']=='complete' and not m['smoke'];spec=m['protocol']
 for f,v in m['code_sha256'].items():assert sha(ROOT/f)==v
 for f,v in m['input_sha256'].items():assert sha(Path(f))==v
 expected=set(itertools.product(spec['seeds'],spec['dimensions'],spec['rhos'],spec['coalition_sizes'],spec['targets']));observed=set();rows=events_count=queries=endpoint_calls=0;maxloss=0.;maxspan=0.;maxaffine=0.
 cases={}
 for done in sorted(source.glob('seed*/*_k*/completion.json')):
  p=done.parent;c=read(p/'case.json');complete=read(done);identity=tuple(c[k] for k in ['seed','dimension','rho','requested_k','target_label']);assert identity in expected and identity not in observed;observed.add(identity)
  assert sha(p/'case.json')==complete['case_sha256'] and sha(p/'results.jsonl')==complete['results_sha256']
  src=Path(m['source'])/f"seed{c['seed']}_d{c['dimension']}_rho{c['rho']:g}"
  sourcecase=read(src/'case.json');assert c['selected_ids']==sourcecase['selected_ids'][:c['requested_k']]
  if c['target_label']=='zero' and c.get('anchors'):assert c['anchors']==sourcecase['anchors'][:c['requested_k']]
  key=(c['seed'],c['dimension'],c['rho'])
  if key not in cases:
   payload=torch.load(src/'models.pt',weights_only=True,map_location='cpu');nets={}
   for ident in sourcecase['selected_ids'][:3]:
    net=PretrainedModel(c['dimension']);net.load_state_dict(payload['local_states'][ident-1]);nets[ident]=net.eval()
   cases[key]=nets
  netlist=[cases[key][i] for i in c['selected_ids']]
  w=witness(c['dimension'],c['target']);assert np.allclose(w,c['witness'],atol=0,rtol=0)
  assert abs(h(w)-c['target'])<1e-10
  rr=[json.loads(s) for s in (p/'results.jsonl').read_text().splitlines()];assert len(rr)==18
  for r in rr:
   rows+=1
   if r['X'] is None:assert r['truth_error'] is None;continue
   x=np.array(r['X']);assert max(abs(x))<=1+1e-8 and abs(abs(h(x)-c['target'])-r['truth_error'])<1e-10
   with torch.no_grad():pred=np.array([float(n(torch.tensor(x,dtype=torch.float32)[None,:]).item()) for n in netlist])
   endpoint_calls+=len(netlist);loss=float(np.mean(abs(pred-c['target'])));maxloss=max(maxloss,abs(loss-r['loss']));assert abs(loss-r['loss'])<1e-10
   assert abs(np.mean(abs(pred-h(x)))-r['mean_model_absolute_truth_error'])<1e-10
   dl,da=distances(c['anchors'],x)
   if r['method'] in ['coefficient','custom','best_of','anchor_mean','best_anchor']:maxspan=max(maxspan,dl);assert dl<1e-7
   if r['method'] in ['custom','anchor_mean','best_anchor']:maxaffine=max(maxaffine,da);assert da<1e-7
   assert sha(p/r['trace'])==r['trace_sha256']
   with gzip.open(p/r['trace'],'rt') as f:trace=json.load(f)
   counted=[e for e in trace if e['counted']];finite=[e for e in counted if e.get('loss') is not None]
   assert len(counted)==r['panel_evaluations']<=r['assigned_panel_budget']
   assert sum(e['model_queries'] for e in counted)==r['model_queries']
   assert r['model_queries']<=r['assigned_panel_budget']*c['requested_k']
   assert r['loss']==min(e['loss'] for e in finite) and any(e['X']==r['X'] and e['loss']==r['loss'] for e in finite)
   if r['method']=='best_of':
    assert all(sum(e['engine']==method for e in counted)<=r['assigned_panel_budget']//2 for method in ['custom','coefficient'])
   for e in finite:
    assert len(e['predictions'])==c['requested_k']
    assert abs(np.mean(abs(np.array(e['predictions'])-c['target']))-e['loss'])<1e-10
   events_count+=len(trace);queries+=r['model_queries']
 assert observed==expected and rows==17280
 report=dict(passed=True,tasks=len(observed),records=rows,trace_events=events_count,study_model_queries=queries,validation_endpoint_model_queries=endpoint_calls,max_Lopt_difference=maxloss,max_restricted_linear_residual=maxspan,max_custom_affine_residual=maxaffine,source_models_unchanged=True,validator_sha256=sha(Path(__file__)))
 output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);a=p.parse_args();validate(a.source)
