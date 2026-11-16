"""Sequential pilot gate followed by frozen independent comparison."""
import os,sys,json,subprocess,hashlib
from pathlib import Path
REG=Path('docs/next_alpha_preregistration_20261010.json')
r=json.loads(REG.read_text());root=Path('alpha_weight_outputs/next_alpha_pipeline_20261010');root.mkdir(exist_ok=False)
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
steps=[('pilot',['run_next_alpha_comparison.py','--source',r['pilot_source'],'--output',r['pilot_output'],'--seeds',','.join(map(str,r['pilot_seeds']))]),
('pilot_acceptance',['analyze_next_alpha_comparison.py',r['pilot_output']]),
('formal_source',['run_alpha_initialization_study.py','--output-dir',r['source_output'],'--seeds',','.join(map(str,r['seeds'])),'--scenarios',','.join(r['scenarios']),'--methods','A_uniform','--primary-baseline','A_uniform']),
('source_acceptance',['verify_alpha_initialization_study.py',r['source_output']]),
('formal',['run_next_alpha_comparison.py','--source',r['source_output'],'--output',r['study_output'],'--seeds',','.join(map(str,r['seeds']))]),
('formal_acceptance',['analyze_next_alpha_comparison.py',r['study_output'],'--registration',str(REG)])]
state={'status':'running','stages':[],'commands':steps};path=root/'status.json'
try:
 for name,args in steps:
  for f,h in r['source_sha256'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
  state['current_stage']=name;path.write_text(json.dumps(state,indent=2));print('Starting '+name,flush=True)
  with (root/(name+'.log')).open('w') as log:p=subprocess.run([sys.executable,*args],env=env,stdout=log,stderr=subprocess.STDOUT)
  state['stages'].append({'name':name,'returncode':p.returncode})
  if p.returncode:raise RuntimeError(name+' failed')
  if name=='pilot_acceptance':
   a=json.loads((Path(r['pilot_output'])/'analysis.json').read_text());assert a['verification']['passed']
 state['status']='complete'
except Exception as e:state.update(status='interrupted',reason=repr(e));raise
finally:path.write_text(json.dumps(state,indent=2))
