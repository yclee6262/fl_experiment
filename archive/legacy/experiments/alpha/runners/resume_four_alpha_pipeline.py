"""Explicit recovery after sandbox OMP startup abort; no numerical setting changes."""
import torch
import os,json,sys,time,subprocess
from pathlib import Path
import oob_shrinkage_research as shared
from four_alpha_initialization import SCENARIOS
ROOT=Path('alpha_weight_outputs/four_alpha_v1_20261014')
def main():
 statepath=ROOT/'pipeline_status.json';state=json.loads(statepath.read_text());regs={k:json.loads((ROOT/(k+'_registration.json')).read_text()) for k in ['pilot','development','formal']}
 state.update(status='running',pid=os.getpid(),recovery='sandbox shared-memory startup abort; pilot executed normally on local host; no numerical changes')
 hashes=regs['formal']['source_sha256'];env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 state['stages'].append(dict(name='pilot1_run_local_recovery',returncode=0,conditions=90));shared.dump(statepath,state)
 def step(name,args):
  for file,h in hashes.items():
   if shared.digest(file)!=h:raise RuntimeError('Frozen source changed '+file)
  state['current_stage']=name;shared.dump(statepath,state)
  with (ROOT/(name+'.log')).open('w') as log:code=subprocess.call([sys.executable]+args,stdout=log,stderr=subprocess.STDOUT,env=env)
  state['stages'].append(dict(name=name,command=args,returncode=code));shared.dump(statepath,state)
  if code:raise RuntimeError(name+' failed; inspect log')
 try:
  step('pilot1_verify',['run_four_alpha_batch.py','--verify',regs['pilot']['batches'][0]['results']])
  step('pilot_analysis',['analyze_four_alpha.py',str(ROOT/'pilot_registration.json')])
  for i,b in enumerate(regs['development']['batches']):
   step(f'development{i+1}_run',['run_four_alpha_batch.py','--registration',str(ROOT/'development_registration.json'),'--batch',str(i)])
   step(f'development{i+1}_verify',['run_four_alpha_batch.py','--verify',b['results']])
  step('development_analysis',['analyze_four_alpha.py',str(ROOT/'development_registration.json')])
  for i,b in enumerate(regs['formal']['batches']):
   prefix=f'formal{i+1}'
   step(prefix+'_source',['run_alpha_initialization_study.py','--output-dir',b['source'],'--seeds',','.join(map(str,b['seeds'])),'--scenarios',','.join(SCENARIOS),'--methods','A_uniform','--primary-baseline','A_uniform'])
   step(prefix+'_source_verify',['verify_alpha_initialization_study.py',b['source']])
   step(prefix+'_run',['run_four_alpha_batch.py','--registration',str(ROOT/'formal_registration.json'),'--batch',str(i)])
   step(prefix+'_verify',['run_four_alpha_batch.py','--verify',b['results']])
  step('formal_analysis',['analyze_four_alpha.py',str(ROOT/'formal_registration.json'),'--formal'])
  state.update(status='complete',finished_unix=time.time());shared.dump(statepath,state)
 except Exception as exc:state.update(status='interrupted',reason=repr(exc));shared.dump(statepath,state);raise
if __name__=='__main__':main()
