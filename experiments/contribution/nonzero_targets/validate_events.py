"""Audit every cached model response and paired inversion start independently."""
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np
import torch
from core.models import PretrainedModel
ROOT=Path(__file__).resolve().parents[3]
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);a=p.parse_args();out=a.source/'event_acceptance.json'
 if out.exists():raise FileExistsError(out)
 torch.set_num_threads(1);cases=read(ROOT/'contribution_analysis/combined_contribution_layer2_experiment_v1_20261021/cases.json')['development'];sources={c['key']:c for c in cases};models={};starts={};checks=hits=invchecks=0
 for entry in read(a.source/'index.json'):
  with gzip.open(a.source/entry['file'],'rt') as f:r=json.load(f)
  assert r['status']=='valid';key=r['key'];c=sources[key];target=r['target']
  if key not in models:
   payload=torch.load(c['models'],map_location='cpu',weights_only=True);models[key]=[]
   for id in c['ids']:
    net=PretrainedModel(2);net.load_state_dict(payload['agents'][id]);models[key].append(net.eval())
  nets=models[key];cache={}
  for e in r['events']:
   x=np.array(e['X'],dtype=np.float64);k=(e['i'],x.tobytes())
   assert np.isfinite(x).all() and max(abs(x))<=1+1e-8
   if e['hit']:assert k in cache and cache[k]==e['loss'];hits+=1
   else:
    assert k not in cache
    with torch.no_grad():loss=abs(float(nets[e['i']](torch.tensor(x,dtype=torch.float32)[None,:]).item())-target)
    assert abs(loss-e['loss'])<1e-12
    cache[k]=e['loss'];checks+=1
  assert len(cache)==r['cache_miss_calls']
  for i,inv in enumerate(r['inversions']):
   k=(key,i);initial=inv['start_results'][0]['initial_X']
   if k in starts:assert starts[k]==initial
   else:starts[k]=initial
   with torch.no_grad():residual=abs(float(nets[i](torch.tensor(inv['X'],dtype=torch.float32)[None,:]).item())-target)
   assert abs(residual-inv['loss'])<1e-12;invchecks+=1
 result=dict(passed=True,independently_evaluated_cache_miss_responses=checks,cache_hits_verified=hits,inverse_endpoint_checks=invchecks,paired_initial_points=len(starts),new_validation_model_evaluations=checks+invchecks,validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
