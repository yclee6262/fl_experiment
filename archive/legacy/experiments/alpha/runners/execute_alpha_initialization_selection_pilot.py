"""Sequential smoke/pilot/full replay only; never starts formal confirmation."""
import json,os,sys,subprocess,time
from pathlib import Path
ROOT=Path('alpha_weight_outputs/alpha_initialization_selection_v1')
ENV=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')

def main():
 from oob_shrinkage_research import dump,digest
 from run_alpha_initialization_selection import lines
 from alpha_initialization_selection import SCENARIOS
 regpath=ROOT/'registration.json';reg=json.load(open(regpath));state={'status':'running','pid':os.getpid(),'started':time.time(),'stages':[]}
 def step(name,args):
  for filename,h in reg['source_sha256'].items():
   if digest(filename)!=h:raise RuntimeError('Frozen source changed '+filename)
  state['current_stage']=name;dump(ROOT/'pipeline_status.json',state)
  with (ROOT/(name+'.log')).open('x') as log:code=subprocess.call([sys.executable]+args,stdout=log,stderr=subprocess.STDOUT,env=ENV)
  state['stages'].append({'stage':name,'returncode':code});dump(ROOT/'pipeline_status.json',state)
  if code:raise RuntimeError('Stage failed '+name)
 def generate(name,conf):
  step(name+'_source',['run_alpha_initialization_study.py','--output-dir',conf['source'],'--seeds',','.join(map(str,conf['seeds'])),'--scenarios',','.join(SCENARIOS),'--methods','A_uniform','--primary-baseline','A_uniform'])
  step(name+'_source_verify',['verify_alpha_initialization_study.py',conf['source']])
 try:
  for stage in ['smoke','pilot']:
   generate(stage,reg[stage]);step(stage+'_run',['run_alpha_initialization_selection.py',str(regpath),stage])
  replay=dict(reg);replay['replay']={'seeds':[reg['pilot']['seeds'][0]],'source':str(ROOT/'replay/source'),'results':str(ROOT/'replay/results')}
  rp=ROOT/'replay_registration.json';dump(rp,replay);generate('replay',replay['replay'])
  step('replay_run',['run_alpha_initialization_selection.py',str(rp),'replay'])
  import numpy as np,torch
  before={(r['scenario'],r['seed'],r['method']):r for r in lines(Path(reg['pilot']['results'])/'results.jsonl')};after=lines(Path(replay['replay']['results'])/'results.jsonl');errors=[];maxima={'X':0.,'alpha':0.,'pooled':0.};source_checks=[]
  for r in after:
   old=before[r['scenario'],r['seed'],r['method']]
   if r['status']!=old['status'] or r.get('selected_method')!=old.get('selected_method'):errors.append('discrete replay')
   if r['status']=='valid':
    for k in ['X','alpha']:maxima[k]=max(maxima[k],float(np.max(np.abs(np.array(r[k])-old[k]))))
    maxima['pooled']=max(maxima['pooled'],abs(r['pooled_error']-old['pooled_error']))
    if r.get('solver',{}).get('chosen_engine')!=old.get('solver',{}).get('chosen_engine'):errors.append('engine replay')
  def tensors(x,p=''):
   if isinstance(x,torch.Tensor):return {p:x}
   out={}
   if isinstance(x,dict):
    for k,v in x.items():out.update(tensors(v,p+'/'+str(k)))
   elif isinstance(x,(list,tuple)):
    for i,v in enumerate(x):out.update(tensors(v,p+'/'+str(i)))
   return out
  for sc in SCENARIOS:
   a=Path(reg['pilot']['source'])/sc;b=Path(replay['replay']['source'])/sc;seed=replay['replay']['seeds'][0]
   old=next(r for r in lines(a/'shared_costs.jsonl') if r['seed']==seed);new=lines(b/'shared_costs.jsonl')[0]
   if old.get('missing_fixed_ids')!=new.get('missing_fixed_ids'):errors.append('qualification replay')
   if not old.get('artifact_path'):continue
   for k in ['blind_test_X','blind_test_y','proposals','coalition_ids']:
    if old[k]!=new[k]:errors.append('source replay '+k)
   pa=a/old['artifact_path'];pb=b/new['artifact_path'];da=np.load(pa/'pooled_data.npz');db=np.load(pb/'pooled_data.npz')
   if da.files!=db.files or not all(np.array_equal(da[k],db[k]) for k in da.files):errors.append('source data/splits replay')
   for name in ['local_models.pt','global_evaluator.pt']:
    ta=tensors(torch.load(pa/name,map_location='cpu',weights_only=True));tb=tensors(torch.load(pb/name,map_location='cpu',weights_only=True))
    if ta.keys()!=tb.keys() or not all(torch.equal(ta[k],tb[k]) for k in ta):errors.append('model replay '+name)
   source_checks.append({'scenario':sc,'inputs_data_models_checked':True})
  if max(maxima.values())>1e-7:errors.append('replay tolerance')
  verification={'passed':not errors,'errors':errors,'max_difference':maxima,'conditions':len(after),'source_checks':source_checks,'not_added_to_pilot_samples':True}
  dump(ROOT/'replay_verification.json',verification)
  if errors:raise RuntimeError(str(errors))
  for stage in ['smoke','pilot']:
   ap=Path(reg[stage]['results'])/'acceptance.json';accept=json.load(open(ap));accept['full_training_replay_passed']=True;dump(ap,accept)
  state.update(status='complete',finished=time.time());dump(ROOT/'pipeline_status.json',state)
 except Exception as exc:state.update(status='interrupted',reason=repr(exc));dump(ROOT/'pipeline_status.json',state);raise
if __name__=='__main__':main()
