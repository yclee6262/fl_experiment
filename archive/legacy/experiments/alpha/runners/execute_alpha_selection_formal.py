"""Sequential registered 3x30 confirmation, three full replays, then one analysis release."""
import torch
import os,sys,json,time,subprocess
from pathlib import Path
import numpy as np
import oob_shrinkage_research as shared
from run_alpha_initialization_selection import lines
ROOT=Path('alpha_weight_outputs/alpha_initialization_selection_formal_v1_20261015')
ENV=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')

def tensor_map(x,p=''):
 if isinstance(x,torch.Tensor):return {p:x}
 out={}
 if isinstance(x,dict):
  for k,v in x.items():out.update(tensor_map(v,p+'/'+str(k)))
 elif isinstance(x,(list,tuple)):
  for i,v in enumerate(x):out.update(tensor_map(v,p+'/'+str(i)))
 return out

def compare_replay(reg,i):
 name=f'formal_batch_{i}';rep=f'replay_batch_{i}';conf=reg[name];rc=reg[rep];seed=rc['seeds'][0]
 before={(r['scenario'],r['seed'],r['method']):r for r in lines(Path(conf['results'])/'results.jsonl')};after=lines(Path(rc['results'])/'results.jsonl');errors=[];maxima={'X':0.,'alpha':0.,'pooled':0.};checks=[]
 for r in after:
  a=before[r['scenario'],seed,r['method']]
  if r['status']!=a['status'] or r.get('selected_method')!=a.get('selected_method'):errors.append('discrete replay mismatch')
  if r['status']=='valid':
   for k in ['X','alpha']:maxima[k]=max(maxima[k],float(np.max(np.abs(np.asarray(r[k])-a[k]))))
   maxima['pooled']=max(maxima['pooled'],abs(r['pooled_error']-a['pooled_error']))
   if r.get('solver',{}).get('chosen_engine')!=a.get('solver',{}).get('chosen_engine'):errors.append('engine replay mismatch')
 for sc in reg['scenarios']:
  a=Path(conf['source'])/sc;b=Path(rc['source'])/sc
  old=next(r for r in lines(a/'shared_costs.jsonl') if r['seed']==seed);new=lines(b/'shared_costs.jsonl')[0]
  for k in ['missing_fixed_ids','coalition_ids','blind_test_X','blind_test_y','proposals']:
   if old.get(k)!=new.get(k):errors.append('source replay mismatch '+k)
  if not old.get('artifact_path'):checks.append({'scenario':sc,'qualification_failure_reproduced':True});continue
  pa,pb=a/old['artifact_path'],b/new['artifact_path'];da,db=np.load(pa/'pooled_data.npz'),np.load(pb/'pooled_data.npz')
  if da.files!=db.files or not all(np.array_equal(da[k],db[k]) for k in da.files):errors.append('data/split mismatch')
  for name in ['local_models.pt','global_evaluator.pt']:
   ta=tensor_map(torch.load(pa/name,map_location='cpu',weights_only=True));tb=tensor_map(torch.load(pb/name,map_location='cpu',weights_only=True))
   if not ta or ta.keys()!=tb.keys() or not all(torch.equal(ta[k],tb[k]) for k in ta):errors.append('model replay mismatch '+name)
  checks.append({'scenario':sc,'source_arrays_and_model_tensors_equal':True})
 if max(maxima.values())>1e-7:errors.append('replay tolerance')
 result={'passed':not errors,'errors':errors,'seed':seed,'conditions':len(after),'max_difference':maxima,'source_checks':checks,'added_quality_samples':0}
 shared.dump(ROOT/f'replay_verification_batch_{i}.json',result)
 if errors:raise RuntimeError(str(errors))
 return result

def main():
 regpath=ROOT/'registration.json';reg=json.load(open(regpath));expected=json.load(open(ROOT/'registration_digest.json'))['sha256'];state={'status':'running','pid':os.getpid(),'started_unix':time.time(),'stages':[],'manual_interim_quality_inspection':False}
 if (ROOT/'pipeline_status.json').exists():raise FileExistsError('Use separate audited resume; preserve existing status')
 def check():
  if shared.digest(regpath)!=expected:raise RuntimeError('Registration changed')
  for n,h in reg['source_sha256'].items():
   if shared.digest(n)!=h:raise RuntimeError('Source changed '+n)
  for n,h in reg['prior_results_sha256'].items():
   if shared.digest(n)!=h:raise RuntimeError('Historical/pilot result changed '+n)
 def step(name,args):
  check();state['current_stage']=name;shared.dump(ROOT/'pipeline_status.json',state);started=time.time()
  with (ROOT/(name+'.log')).open('x') as log:code=subprocess.call([sys.executable]+args,stdout=log,stderr=subprocess.STDOUT,env=ENV)
  state['stages'].append({'name':name,'returncode':code,'seconds':time.time()-started});shared.dump(ROOT/'pipeline_status.json',state)
  if code:raise RuntimeError('Stage failed '+name)
 def source(name):
  conf=reg[name]
  step(name+'_source',['run_alpha_initialization_study.py','--output-dir',conf['source'],'--seeds',','.join(map(str,conf['seeds'])),'--scenarios',','.join(reg['scenarios']),'--methods','A_uniform','--primary-baseline','A_uniform'])
  step(name+'_source_verify',['verify_alpha_initialization_study.py',conf['source']])
 try:
  for name in reg['batches']:
   source(name);step(name+'_solve',['run_alpha_selection_formal_batch.py',str(regpath),name])
  state['all_formal_batches_complete_before_analysis']=True;shared.dump(ROOT/'pipeline_status.json',state)
  for i in range(1,4):
   name=f'replay_batch_{i}';source(name);step(name+'_solve',['run_alpha_selection_formal_batch.py',str(regpath),name]);compare_replay(reg,i)
  step('formal_analysis',['analyze_alpha_selection_formal.py',str(regpath)])
  step('independent_audit',['audit_alpha_selection_formal.py',str(regpath)])
  state.update(status='complete',finished_unix=time.time());shared.dump(ROOT/'pipeline_status.json',state)
 except Exception as exc:state.update(status='interrupted',reason=repr(exc));shared.dump(ROOT/'pipeline_status.json',state);raise
if __name__=='__main__':main()
