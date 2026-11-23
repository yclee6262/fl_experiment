"""Independent source, target, endpoint and hard-budget checks."""
import argparse,gzip,hashlib,json,itertools
from pathlib import Path
import numpy as np
import torch
from core.models import PretrainedModel
ROOT=Path(__file__).resolve().parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(s) for s in p.read_text().splitlines()]
def h(x):x=np.array(x);return float(x.sum()+x[0]*x[1])
def validate(source):
 out=source/'acceptance.json'
 if out.exists():raise FileExistsError(out)
 torch.set_num_threads(1);m=read(source/'manifest.json');assert m['status']=='complete' and not m['smoke']
 for f,v in {**m['runtime_sha256'],**m['input_sha256']}.items():assert sha(ROOT/f)==v,f
 count=queries=events=0;models={};starts={};identities=set()
 for donefile in source.glob('*/completion.json'):
  p=donefile.parent;c=read(p/'case.json');done=read(donefile)
  assert done['status']=='complete' and sha(p/'case.json')==done['case_sha256'] and sha(p/'results.jsonl')==done['results_sha256']
  identity=(c['scenario'],c['seed'],c['target']);assert identity not in identities;identities.add(identity)
  artifact=ROOT/c['source_artifact'];key=str(artifact)
  if key not in models:
   payload=torch.load(artifact/'local_models.pt',map_location='cpu',weights_only=True);models[key]=[]
   for i in [1,2,20]:
    net=PretrainedModel(2);net.load_state_dict(payload['agents'][i]);models[key].append(net.eval())
  nets=models[key];target=c['target'];w=np.array(c['witness']);assert max(abs(w))<=1 and abs(h(w)-target)<1e-12
  for j,(net,inv) in enumerate(zip(nets,c['inverses'])):
   first=inv['start_results'][0]['initial_X'];skey=(c['scenario'],c['seed'],j)
   if skey in starts:assert starts[skey]==first
   else:starts[skey]=first
   with torch.no_grad():pred=float(net(torch.tensor(inv['X'],dtype=torch.float32)[None,:]).item())
   assert abs(abs(pred-target)-inv['loss'])<1e-10;queries+=1
  decisions=lines(p/'decisions.jsonl');results=lines(p/'results.jsonl');assert len(results)==len(decisions)==10
  assert len({r['label'] for r in results})==10
  for r,d in zip(results,decisions):
   assert d['truth_error'] is None and r['X']==d['X']
   assert r['panel_evaluations']<=400 and r['model_queries']<=1200
   trace=p/r['trace_path'];assert sha(trace)==r['trace_sha256']
   with gzip.open(trace,'rt') as f:ee=json.load(f)
   counted=[e for e in ee if e['counted']];assert len(counted)==r['panel_evaluations']
   assert all(sum(e['engine']==engine for e in counted)<=200 for engine in ['coefficient','custom'])
   assert sum(e['model_queries'] for e in counted)==r['model_queries']
   a=np.array(r['alpha']);a/=a.sum();valid=[e for e in counted if e['loss'] is not None]
   for e in valid:assert abs(a@abs(np.array(e['predictions'])-target)-e['loss'])<1e-12
   if r['X'] is not None:
    assert r['loss']==min(e['loss'] for e in valid)
    with torch.no_grad():v=np.array([float(net(torch.tensor(r['X'],dtype=torch.float32)[None,:]).item()) for net in nets])
    assert abs(a@abs(v-target)-r['loss'])<1e-10;queries+=3
    assert abs(abs(h(r['X'])-target)-r['truth_error'])<1e-12 and max(abs(np.array(r['X'])))<=1
   count+=1;events+=len(ee)
 seeds=sorted({x[1] for x in identities});assert len(models)==270 and len(seeds)==90 and identities==set(itertools.product(['reference_clean','low_quality_25pct','sample_imbalance_4x'],seeds,[0.,-.5,.5,2.7]))
 assert count==10800
 out.write_text(json.dumps(dict(passed=True,records=count,conditions=len(identities),model_sets=len(models),validation_model_queries=queries,trace_events=events,all_target_residuals_and_budgets_verified=True,all_cross_target_initial_points_identical=True,validator_sha256=sha(Path(__file__))),indent=2)+'\n');print(out.read_text())
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);a=p.parse_args();validate(a.source)
