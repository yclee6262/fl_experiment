"""Frozen addition of pure controls; original six-method data remain untouched."""
import torch
import json,os,sys,subprocess,time
from pathlib import Path
from oob_shrinkage_research import dump,digest
ROOT=Path('alpha_weight_outputs/pure_es_extension_v1_20261014')
def main():
 ROOT.mkdir(exist_ok=False);old=json.load(open('alpha_weight_outputs/four_alpha_v1_20261014/formal_registration.json'))
 reg=dict(output=str(ROOT),kind='paired_posthoc_pure_control_extension',methods=['stacked_full','pure_support'],scenarios=old['scenarios'],
  batches=[dict(b,results=str(ROOT/f'batch_{i+1}'),six_method_results=b['results']) for i,b in enumerate(old['batches'])],
  source_sha256={p.name:digest(p) for p in Path('.').glob('*.py')},
  original_six_results_sha256={b['results']:digest(Path(b['results'])/'results.jsonl') for b in old['batches']},
  expected_new_conditions=540,combined_conditions=2160,paired_case_n=90,
  analysis_rng=20261014,bootstrap_replicates=200000,family_error=.05,
  contrasts=[['ensemble_shrink','stacked_full'],['B_stacked_half','stacked_full'],['peer_support_shrink','pure_support'],['support_quality_mix50','pure_support']],
  inference='reuses inspected six-method cohort; added controls frozen before their scores; exploratory evidence, not independent confirmation',
  delta='shrunk minus pure pooled error, negative supports shrinking',primary='within-seed equal mean of three scenarios',pooled_role='post-delivery only',truth_role='diagnostic',no_new_labels=True,alpha_update=False,pruning=False,settlement='not_run')
 dump(ROOT/'registration.json',reg);dump('docs/pure_es_extension_registration_20261014.json',reg)
 state=dict(status='running',started_unix=time.time(),pid=os.getpid(),stages=[]);env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 def step(name,args):
  state['current_stage']=name;dump(ROOT/'pipeline_status.json',state)
  with (ROOT/(name+'.log')).open('w') as log:code=subprocess.call([sys.executable]+args,stdout=log,stderr=subprocess.STDOUT,env=env)
  state['stages'].append(dict(name=name,returncode=code,command=args));dump(ROOT/'pipeline_status.json',state)
  if code:raise RuntimeError(name+' failed')
 try:
  for i,b in enumerate(reg['batches']):
   step(f'batch{i+1}_run',['run_pure_es_batch.py','--registration',str(ROOT/'registration.json'),'--batch',str(i)])
   step(f'batch{i+1}_verify',['run_pure_es_batch.py','--verify',b['results']])
  step('analysis',['analyze_pure_es_extension.py'])
  state.update(status='complete',finished_unix=time.time());dump(ROOT/'pipeline_status.json',state)
 except Exception as exc:state.update(status='interrupted',reason=repr(exc));dump(ROOT/'pipeline_status.json',state);raise
if __name__=='__main__':main()
