"""Six-method paired development and new independent three-batch confirmation."""
import torch
import json,os,sys,time,subprocess,platform
from pathlib import Path
import oob_shrinkage_research as shared
from four_alpha_initialization import METHODS,SCENARIOS,CONTRASTS,LABELS
ROOT=Path('alpha_weight_outputs/four_alpha_v1_20261014')
ENV=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')

def used_seeds():
 used=set();files=0
 def walk(d):
  if isinstance(d,dict):
   for k,v in d.items():
    if k=='seeds' and isinstance(v,list):used.update(x for x in v if isinstance(x,int))
    elif k=='seed' and isinstance(v,int):used.add(v)
    elif isinstance(v,(dict,list)):walk(v)
  elif isinstance(d,list):
   for x in d:walk(x)
 for directory in ['docs','alpha_weight_outputs']:
  for p in Path(directory).rglob('*.json'):
   if not any(k in p.name for k in ['manifest','registration','preregistration']):continue
   try:walk(json.loads(p.read_text()));files+=1
   except (ValueError,OSError):continue
 return used,files

def main():
 ROOT.mkdir(exist_ok=False);old=json.loads(Path('docs/peer_support_formal_registration_v2_20261013.json').read_text());pilot_old=json.loads(Path('docs/peer_support_pilot_registration_v2_20261013.json').read_text())
 used,files=used_seeds();newb=[];start=20000
 for i in range(3):
  while used.intersection(range(start,start+30)):start+=1000
  seeds=list(range(start,start+30));used.update(seeds);d=ROOT/f'formal_batch_{i+1}';d.mkdir();newb.append(dict(number=i+1,seeds=seeds,source=str(d/'source'),results=str(d/'results')));start+=1000
 hashes={p.name:shared.digest(p) for p in Path('.').glob('*.py')}
 common=dict(date='2026-10-14',methods=METHODS,labels=LABELS,scenarios=SCENARIOS,source_sha256=hashes,
  comparison_family=len(CONTRASTS),contrasts=CONTRASTS,family_error=.05,bootstrap_replicates=200000,analysis_rng=20261014,
  primary_endpoint='within-seed equal mean of three scenarios',conditional_population='shared pre-method qualification, no replacement',
  stability_rule='family CI wholly negative; all three batch means negative; >=2 ordinary95% batch CIs negative; complete eligible pairs; no higher observed failures',
  secondary_scenarios='exploratory ordinary95%',pooled_role='primary post-delivery only',truth_role='diagnostic only',
  shrinkage=.5,alpha_update=False,pruning=False,settlement='not_run',constraints='box[-1,1], tol1e-8',solver='best_of custom30',
  environment=dict(python=sys.version,platform=platform.platform(),numeric_threads=1),
  seed_audit=dict(scanned_files=files,new_formal=[b['seeds'] for b in newb],disjoint_from_saved_manifests=True),
  sample_size_rule='fixed 3 x 30 new seeds (90 paired seed units), not a guaranteed-power calculation; retain all methods; inconclusive intervals remain inconclusive',
  definitions=dict(EU='0.5*w+0.5*U',EB='0.5*w+0.5*B',SU='0.5*s+0.5*U',SB='0.5*s+0.5*B',w='20-label simplex mixture-MSE optimum',B='inverse individual 20-label MSE',s='off-diagonal peer errors, median, exp with median scale'))
 pilot=dict(common,output=str(ROOT/'pilot'),analysis_kind='pilot',batches=[dict(pilot_old['batches'][0],results=str(ROOT/'pilot/results'))])
 dev=dict(common,output=str(ROOT/'development'),analysis_kind='reused_data_exploration',batches=[dict(b,results=str(ROOT/f'development/batch_{i+1}')) for i,b in enumerate(old['batches'])])
 formal=dict(common,output=str(ROOT/'formal'),analysis_kind='independent_confirmation',batches=newb,expected_conditions=1620)
 for name,reg in [('pilot',pilot),('development',dev),('formal',formal)]:
  Path(reg['output']).mkdir(parents=True,exist_ok=True);shared.dump(ROOT/(name+'_registration.json'),reg)
 shared.dump(Path('docs/four_alpha_registration_20261014.json'),dict(pilot=pilot,development=dev,formal=formal))
 snap=ROOT/'source_snapshot';snap.mkdir()
 for name in hashes:(snap/name).write_bytes(Path(name).read_bytes())
 state=dict(status='running',pid=os.getpid(),started_unix=time.time(),stages=[]);status=ROOT/'pipeline_status.json'
 def step(name,args):
  for filename,h in hashes.items():
   if shared.digest(filename)!=h:raise RuntimeError('Frozen source changed '+filename)
  state['current_stage']=name;shared.dump(status,state)
  with (ROOT/(name+'.log')).open('w') as log:code=subprocess.call([sys.executable]+args,stdout=log,stderr=subprocess.STDOUT,env=ENV)
  state['stages'].append(dict(name=name,command=args,returncode=code));shared.dump(status,state)
  if code:raise RuntimeError(name+' failed; inspect preserved log')
 try:
  for kind,reg in [('pilot',pilot),('development',dev)]:
   path=str(ROOT/(kind+'_registration.json'))
   for i,b in enumerate(reg['batches']):
    step(f'{kind}{i+1}_run',['run_four_alpha_batch.py','--registration',path,'--batch',str(i)])
    step(f'{kind}{i+1}_verify',['run_four_alpha_batch.py','--verify',b['results']])
   step(kind+'_analysis',['analyze_four_alpha.py',path])
  # All six methods remain registered regardless of exploratory ranking.
  path=str(ROOT/'formal_registration.json')
  for i,b in enumerate(newb):
   prefix=f'formal{i+1}'
   step(prefix+'_source',['run_alpha_initialization_study.py','--output-dir',b['source'],'--seeds',','.join(map(str,b['seeds'])),'--scenarios',','.join(SCENARIOS),'--methods','A_uniform','--primary-baseline','A_uniform'])
   step(prefix+'_source_verify',['verify_alpha_initialization_study.py',b['source']])
   step(prefix+'_run',['run_four_alpha_batch.py','--registration',path,'--batch',str(i)])
   step(prefix+'_verify',['run_four_alpha_batch.py','--verify',b['results']])
  step('formal_analysis',['analyze_four_alpha.py',path,'--formal'])
  state.update(status='complete',finished_unix=time.time());shared.dump(status,state)
 except Exception as exc:state.update(status='interrupted',reason=repr(exc));shared.dump(status,state);raise
if __name__=='__main__':main()
