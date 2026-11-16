"""Prospectively sized clean replication of unchanged R5 formulas."""
import json,hashlib,os,sys,math,subprocess
from pathlib import Path
from statistics import NormalDist
ROOT=Path('alpha_weight_outputs/corr_replication_pipeline_20261011');ROOT.mkdir(exist_ok=False)
REG=Path('docs/corr_replication_preregistration_20261011.json')
PILOT='alpha_weight_outputs/corr_replication_pilot_v1_20261011'
SOURCE='alpha_weight_outputs/corr_replication_source_v1_20261011'
FORMAL='alpha_weight_outputs/corr_replication_formal_v1_20261011'
methods=['B_stage0','corr_precision_0p5','adaptive_corr_precision']
old=json.loads(Path('alpha_weight_outputs/adaptive_corr_formal_v1_20261010/analysis.json').read_text())
rows=[json.loads(s) for s in Path('alpha_weight_outputs/adaptive_corr_formal_v1_20261010/results.jsonl').read_text().splitlines()]
index={(r['seed'],r['method']):r for r in rows if r['scenario']=='reference_clean'}
priorseeds=list(range(5000,5034));pairs=[('adaptive_corr_precision','B_stage0'),('corr_precision_0p5','B_stage0'),('adaptive_corr_precision','corr_precision_0p5')]
sds=[]
for a,b in pairs:
 ds=[index[(s,a)]['pooled_error']-index[(s,b)]['pooled_error'] for s in priorseeds]
 mean=sum(ds)/len(ds);sd=math.sqrt(sum((d-mean)**2 for d in ds)/(len(ds)-1))
 sds.append({'method':a,'baseline':b,'n':34,'paired_SD':sd})
z=NormalDist().inv_cdf(1-.05/(2*3))+NormalDist().inv_cdf(.8)
n=max(30,math.ceil(z*z*max(d['paired_SD'] for d in sds)**2/.005**2))
used=set();scanned=0
for p in Path('alpha_weight_outputs').rglob('*manifest*.json'):
 try:m=json.loads(p.read_text())
 except (ValueError,OSError):continue
 scanned+=1
 if isinstance(m.get('seeds'),list):used.update(s for s in m['seeds'] if isinstance(s,int))
start=6000
while used.intersection(range(start,start+n)):start+=1000
seeds=list(range(start,start+n))
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('.').glob('*.py')}
r={'date':'2026-10-11','methods':methods,'scenarios':['reference_clean'],'seeds':seeds,'N':n,'expected_formal':3*n,
 'pilot_seeds':priorseeds[:5],'pilot_source':'alpha_weight_outputs/adaptive_corr_source_v1_20261010','pilot_output':PILOT,
 'source_output':SOURCE,'study_output':FORMAL,'source_sha256':hashes,'seed_audit':{'manifests_checked':scanned,'overlap':sorted(set(seeds)&used)},
 'N_rule':'max30,ceil((z_(1-.05/(2*3))+z_.8)^2 * maximum R5-clean paired_SD^2 /.005^2)',
 'N_inputs':sds,'research_resolution_delta':.005,'nominal_power':.8,'power_limit':'normal approximation, not guaranteed power or business tolerance',
 'R5_role':'prior SD planning and historical pilot only; not pooled into formal, no formula/parameter tuning',
 'contrasts':pairs,'bootstrap_replicates':10000,'analysis_RNG':20260924,'family_error':.05,'grid':[0,.25,.5,.75,1],
 'folds':5,'fit_rows':16,'heldout_rows':4,'tie_tolerance':1e-12,'formal_parameters_fixed':True,
 'pooled_role':'post-delivery only; truth diagnostic; no evaluator enters alpha/selection/solver',
 'constraints':'2D,target0,IDs1/2/20,box[-1,1], feasibility1e-8,reload1e-7; no alpha update/pruning/settlement',
 'advantage_rule':'all pairs valid, corrected CI wholly negative, no higher observed failure, acceptance passes',
 'decision_rule':'replication superiority only; no noninferiority/equivalence claim when CI crosses0; adaptive increment requires superiority to fixed'}
REG.write_text(json.dumps(r,indent=2)+'\n')
state={'status':'running','stages':[],'N':n,'expected_formal':3*n,'registration':str(REG)}
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
def save():(ROOT/'status.json').write_text(json.dumps(state,indent=2))
def step(name,args):
 for f,h in hashes.items():
  if hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h:raise RuntimeError('source changed '+f)
 state['current_stage']=name;save();print('Starting '+name,flush=True)
 with (ROOT/(name+'.log')).open('w') as log:p=subprocess.run([sys.executable,*args],env=env,stdout=log,stderr=subprocess.STDOUT)
 state['stages'].append({'name':name,'args':args,'returncode':p.returncode});save()
 if p.returncode:raise RuntimeError(name+' failed')
try:
 print(f'Preregistered N{n}, seeds{start}..{start+n-1}',flush=True)
 step('pilot',['run_correlation_replication.py','--source',r['pilot_source'],'--output',PILOT,'--seeds',','.join(map(str,r['pilot_seeds']))])
 step('pilot_acceptance',['analyze_correlation_replication.py',PILOT])
 pilot=[json.loads(s) for s in (Path(PILOT)/'results.jsonl').read_text().splitlines()]
 diffs=[]
 for row in pilot:
  previous=index[(row['seed'],row['method'])]
  for field in ['alpha','X','pooled_error','truth_error']:
   a=row[field];b=previous[field]
   diff=max(abs(x-y) for x,y in zip(a,b)) if isinstance(a,list) else abs(a-b)
   diffs.append(diff)
  if row['status']!=previous['status'] or row['stage3_requests']!=previous['stage3_requests']:raise RuntimeError('Pilot changed R5 discrete state')
 if max(diffs)>1e-7:raise RuntimeError('Pilot changed R5 results')
 (ROOT/'pilot_R5_identity.json').write_text(json.dumps({'passed':True,'conditions':len(pilot),'max_difference':max(diffs)},indent=2))
 step('formal_source',['run_alpha_initialization_study.py','--output-dir',SOURCE,'--seeds',','.join(map(str,seeds)),
  '--scenarios','reference_clean','--methods','A_uniform','--primary-baseline','A_uniform'])
 step('source_acceptance',['verify_alpha_initialization_study.py',SOURCE])
 step('formal',['run_correlation_replication.py','--source',SOURCE,'--output',FORMAL,'--seeds',','.join(map(str,seeds))])
 step('formal_acceptance',['analyze_correlation_replication.py',FORMAL,'--registration',str(REG)])
 state['status']='complete'
except Exception as e:state.update(status='interrupted',reason=repr(e));raise
finally:save()
