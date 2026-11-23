"""Independent saved-endpoint and model-response audit for role target expansion."""
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np
import torch
from core.models import PretrainedModel
ROOT=Path(__file__).resolve().parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def truth(x,T):
 x=np.array(x);return float(abs(x.sum()+x[0]*x[1]-T))
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);a=p.parse_args();out=a.source/'acceptance.json'
 if out.exists():raise FileExistsError(out)
 torch.set_num_threads(1);m=json.loads((a.source/'manifest.json').read_text());idx=json.loads((a.source/'index.json').read_text())
 assert m['status']=='complete' and not m['smoke']
 for f,h in m['source_sha256'].items():assert sha(Path(f))==h
 cases=json.loads((ROOT/'contribution_analysis/combined_contribution_layer2_experiment_v1_20261021/cases.json').read_text())['development'];by={c['key']:c for c in cases}
 assert {(z['key'],z['target']) for z in idx}=={(c['key'],t) for c in cases for t in [0.,-.5,.5,2.7]}
 scorechecks=0;modelcalls=0;truthchecks=0
 for entry in idx:
  path=a.source/entry['file'];assert sha(path)==entry['sha256']
  with gzip.open(path,'rt') as f:r=json.load(f)
  assert r['status']=='valid';T=r['target'];n=r['n'];weights=np.array(r['alpha']);c=by[r['key']]
  payload=torch.load(c['models'],map_location='cpu',weights_only=True);models=[]
  for i in c['ids']:
   model=PretrainedModel(2);model.load_state_dict(payload['agents'][i]);models.append(model.eval())
  def evaluate(x):
   nonlocal modelcalls
   with torch.no_grad():v=np.array([abs(float(model(torch.tensor(x,dtype=torch.float32)[None,:]).item())-T) for model in models])
   modelcalls+=n;return v
  assert np.allclose(evaluate(r['full']['solution'])@weights,r['full']['loss'],rtol=0,atol=1e-12)
  assert abs(truth(r['full']['solution'],T)-r['truth_error'])<1e-12;truthchecks+=1
  for row in r['rows']:
   full=np.array(row['full_losses']);i=row['i'];peer=[j for j in range(n) if j!=i]
   for role in ['A','M','J']:
    alt=row['roles'][role];delta=np.array(alt['losses'])-full
    assert abs(weights@delta-alt['score']['F'])<1e-12
    assert abs((weights[peer]/weights[peer].sum())@delta[peer]-alt['score']['Fminus'])<1e-12;scorechecks+=2
  for participant in r['participants']:
   own=[row for row in r['rows'] if row['i']==participant['i']]
   for judge in ['F','Fminus']:
    for role in ['A','M','J']:assert abs(sum(z['roles'][role]['score'][judge] for z in own)-participant['scores'][judge][role])<1e-12
  for loo in r['LOO']:
   assert np.allclose(evaluate(loo['X']),loo['losses'],rtol=0,atol=1e-12)
   assert abs(truth(loo['X'],T)-truth(r['full']['solution'],T)-loo['truth_effect'])<1e-12;truthchecks+=1
  assert sum(not e['hit'] for e in r['events'])==r['cache_miss_calls']
  assert len(r['events'])==r['requests']
 result=dict(passed=True,tasks=len(idx),score_checks=scorechecks,truth_checks=truthchecks,independent_endpoint_model_evaluations=modelcalls,raw_source_hashes=len(m['source_sha256']),validator_sha256=sha(Path(__file__)))
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
