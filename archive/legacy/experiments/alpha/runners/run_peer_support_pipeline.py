"""Sequential pilot acceptance then frozen three-batch confirmation; no retuning."""
import argparse,hashlib,json,math,os,platform,subprocess,sys,time
from pathlib import Path
import numpy as np
from scipy.stats import norm
from peer_support_alpha import METHODS,SCENARIOS,CONTRASTS
ROOT=Path('alpha_weight_outputs/peer_support_v2_20261013')
PILOT=Path('docs/peer_support_pilot_registration_v2_20261013.json')
FORMAL=Path('docs/peer_support_formal_registration_v2_20261013.json')

def freeze_formal():
 if FORMAL.exists():raise FileExistsError('Formal registration already exists; inspect authoritative process before recovery')
 pilot=json.loads((ROOT/'pilot_analysis.json').read_text());assert pilot['status']=='pilot'
 sd=[];reg=json.loads(PILOT.read_text());index={}
 for line in (Path(reg['batches'][0]['results'])/'results.jsonl').read_text().splitlines():
  r=json.loads(line);index[r['scenario'],r['seed'],r['method']]=r
 eligible=[s for s in reg['batches'][0]['seeds'] if all(index[k,s,m]['status']=='valid' for k in SCENARIOS for m in METHODS)]
 if len(eligible)<3:raise RuntimeError('Too few complete pilot cases for variance planning')
 for m,b in CONTRASTS:
  ds=[np.mean([index[k,s,m]['pooled_error']-index[k,s,b]['pooled_error'] for k in SCENARIOS]) for s in eligible];sd.append({'method':m,'baseline':b,'SD':float(np.std(ds,ddof=1))})
 historical=json.loads(Path('docs/repeated_alpha_confirmation_registration_20261012.json').read_text())['power_planning']['maximum_SD']
 planning_sd=max(historical,max(x['SD'] for x in sd));delta=.003;family=7;power=.9
 n=math.ceil(((norm.ppf(1-.05/(2*family))+norm.ppf(power))*planning_sd/delta)**2);per=max(30,math.ceil(n/3));used=set();scanned=0
 for directory in ['docs','alpha_weight_outputs']:
  for p in Path(directory).rglob('*.json'):
   if not any(x in p.name for x in ['manifest','registration']):continue
   try:d=json.loads(p.read_text())
   except (OSError,ValueError):continue
   scanned+=1
   if isinstance(d,dict):used.update(x for x in d.get('seeds',[]) if isinstance(x,int))
 used.update(reg['batches'][0]['seeds']);start=12000;batches=[]
 for i in range(3):
  while used.intersection(range(start,start+per)):start+=1000
  seeds=list(range(start,start+per));used.update(seeds);d=ROOT/f'batch_{i+1}';d.mkdir(exist_ok=False);batches.append({'number':i+1,'seeds':seeds,'source':str(d/'source'),'results':str(d/'results')});start+=1000
 hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('.').glob('*.py')}
 frozen=ROOT/'formal_source_snapshot';frozen.mkdir()
 for p,h in hashes.items():(frozen/p).write_bytes(Path(p).read_bytes())
 out={**reg,'kind':'peer_support_three_batch_confirmation','batches':batches,'seeds':[s for b in batches for s in b['seeds']],'source_sha256':hashes,'expected_conditions':per*3*3*6,'comparison_family':7,'family_error':.05,'bootstrap_replicates':200000,'analysis_rng':20261013,'batch_ci_replicates':10000,'batch_ci_rng':20260924,'power_planning':{'pilot_SD':sd,'historical_maximum_SD':historical,'planning_SD':planning_sd,'delta':delta,'delta_role':'research resolution, not proven business value','power':power,'approximate_N':n,'per_batch':per,'limitations':'5-case pilot variance uncertain; use larger of pilot and historical maximum; normal approximation; no guaranteed power for smaller effects or qualification attrition'},'seed_audit':{'scanned_files':scanned,'batches_disjoint':True,'pilot_excluded':True},'stability_rule':'simultaneous CI <0; three negative batch means; at least2 negative ordinary95% batch CIs; complete valid common-eligible pairs; no increased observed method failures','conditional_population':'common pre-method eligibility in all three scenarios; shared failures retained, no replacement draws','secondary_scenarios':'exploratory only','solver':'best_of custom30; box[-1,1]; tol1e-8','pooled_role':'post-delivery primary only; never passed to initialization','environment':{'python':sys.version,'platform':platform.platform(),'numeric_threads':1}}
 FORMAL.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'formal_seeds':3*per,'conditions':out['expected_conditions'],'per_batch':per}),flush=True);return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--register-only',action='store_true');a=p.parse_args()
 if a.register_only:freeze_formal();return
 if not a.execute:raise ValueError('Choose execute')
 statepath=ROOT/'pipeline_status.json'
 if statepath.exists():raise FileExistsError('Existing pipeline; inspect handle, never silently restart')
 state={'status':'running','pid':os.getpid(),'started_unix':time.time(),'stages':[]};env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 def save():statepath.write_text(json.dumps(state,indent=2)+'\n')
 def step(name,args):
  state['current_stage']=name;save()
  with (ROOT/(name+'.log')).open('w') as log:code=subprocess.call([sys.executable]+args,stdout=log,stderr=subprocess.STDOUT,env=env)
  state['stages'].append({'name':name,'command':args,'returncode':code});save()
  if code:raise RuntimeError(name+' failed; see saved log')
 try:
  step('pilot_source_verify',['verify_alpha_initialization_study.py',json.loads(PILOT.read_text())['batches'][0]['source']])
  step('pilot_run',['run_peer_support_batch.py','--registration',str(PILOT),'--batch','0'])
  step('pilot_verify',['run_peer_support_batch.py','--verify',str(ROOT/'pilot/results')])
  step('pilot_analysis',['analyze_peer_support.py',str(PILOT)])
  reg=freeze_formal();state['formal_registration']=str(FORMAL);save()
  for i,b in enumerate(reg['batches']):
   # Hash check before every new run prevents tuning after registration.
   for name,h in reg['source_sha256'].items():
    if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=h:raise RuntimeError('Registered source changed: '+name)
   prefix='batch'+str(i+1)
   step(prefix+'_source',['run_alpha_initialization_study.py','--output-dir',b['source'],'--seeds',','.join(map(str,b['seeds'])),'--scenarios',','.join(SCENARIOS),'--methods','A_uniform','--primary-baseline','A_uniform'])
   step(prefix+'_source_verify',['verify_alpha_initialization_study.py',b['source']])
   step(prefix+'_run',['run_peer_support_batch.py','--registration',str(FORMAL),'--batch',str(i)])
   step(prefix+'_verify',['run_peer_support_batch.py','--verify',b['results']])
  step('formal_analysis',['analyze_peer_support.py',str(FORMAL),'--formal']);state['status']='complete';state['finished_unix']=time.time();save()
 except Exception as exc:state.update(status='interrupted',reason=repr(exc));save();raise

if __name__=='__main__':main()
