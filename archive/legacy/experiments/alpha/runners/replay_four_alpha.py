"""One independent-confirmation case fully regenerated; not added to statistics."""
import torch
import json,subprocess,sys,os
from pathlib import Path
import numpy as np
from oob_shrinkage_research import dump
from four_alpha_initialization import SCENARIOS
ROOT=Path('alpha_weight_outputs/four_alpha_v1_20261014')
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def main():
 reg=json.load(open(ROOT/'formal_registration.json'));original=reg['batches'][0];seed=original['seeds'][0];r=ROOT/'replay';r.mkdir(exist_ok=False)
 replay=dict(reg,output=str(r),batches=[dict(number=0,seeds=[seed],source=str(r/'source'),results=str(r/'results'))]);dump(r/'registration.json',replay)
 env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 commands=[['run_alpha_initialization_study.py','--output-dir',str(r/'source'),'--seeds',str(seed),'--scenarios',','.join(SCENARIOS),'--methods','A_uniform','--primary-baseline','A_uniform'],['verify_alpha_initialization_study.py',str(r/'source')],['run_four_alpha_batch.py','--registration',str(r/'registration.json'),'--batch','0'],['run_four_alpha_batch.py','--verify',str(r/'results')]]
 for i,cmd in enumerate(commands):
  with (r/f'stage{i}.log').open('w') as log:code=subprocess.call([sys.executable]+cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
  if code:raise RuntimeError('Replay failed '+str(cmd))
 errors=[];maxima=dict(X=0.,alpha=0.,pooled=0.,truth=0.);source_checks=[]
 before={(x['scenario'],x['seed'],x['method']):x for x in lines(Path(original['results'])/'results.jsonl') if x['seed']==seed};after=lines(r/'results/results.jsonl')
 for row in after:
  old=before[row['scenario'],seed,row['method']]
  if row['status']!=old['status']:errors.append('discrete outcome mismatch')
  if row['status']=='valid':
   for key in ['X','alpha']:maxima[key]=max(maxima[key],float(np.max(abs(np.array(row[key])-old[key]))))
   for key,col in [('pooled','pooled_error'),('truth','truth_error')]:maxima[key]=max(maxima[key],abs(row[col]-old[col]))
   if row['solver'].get('chosen_engine')!=old['solver'].get('chosen_engine'):errors.append('solver engine mismatch')
 for scenario in SCENARIOS:
  a=Path(original['source'])/scenario;b=r/'source'/scenario
  item=next(x for x in lines(a/'shared_costs.jsonl') if x['seed']==seed);re=lines(b/'shared_costs.jsonl')[0]
  if item['coalition_ids']!=re['coalition_ids'] or item['missing_fixed_ids']!=re['missing_fixed_ids']:errors.append('qualification/coalition mismatch')
  for key in ['blind_test_X','blind_test_y','proposals']:
   if not np.array_equal(item[key],re[key]):errors.append('source input mismatch '+key)
  pa=a/item['artifact_path'];pb=b/re['artifact_path'];da=np.load(pa/'pooled_data.npz');db=np.load(pb/'pooled_data.npz')
  equal=da.files==db.files and all(np.array_equal(da[k],db[k]) for k in da.files)
  if not equal:errors.append('pooled data/split mismatch')
  for filename in ['local_models.pt','global_evaluator.pt']:
   la=torch.load(pa/filename,map_location='cpu',weights_only=True);lb=torch.load(pb/filename,map_location='cpu',weights_only=True)
   def tensors(x,prefix=''):
    if isinstance(x,torch.Tensor):return {prefix:x}
    out={}
    if isinstance(x,dict):
     for k,v in x.items():out.update(tensors(v,prefix+'/'+str(k)))
    elif isinstance(x,(list,tuple)):
     for i,v in enumerate(x):out.update(tensors(v,prefix+'/'+str(i)))
    return out
   ta,tb=tensors(la),tensors(lb)
   if not ta or ta.keys()!=tb.keys() or not all(torch.equal(ta[k],tb[k]) for k in ta):errors.append('model tensor mismatch '+filename)
  source_checks.append(dict(scenario=scenario,pooled_arrays_equal=equal,inputs_models_checked=True))
 if max(maxima.values())>1e-7:errors.append('numerical replay tolerance exceeded')
 result=dict(passed=not errors,errors=errors,seed=seed,conditions=len(after),max_difference=maxima,source_checks=source_checks,cost_scope='research replay only, not added to comparisons')
 dump(r/'verification.json',result)
 if errors:raise RuntimeError(str(errors))
 print(json.dumps(result))
if __name__=='__main__':main()
