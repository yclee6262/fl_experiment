"""Freeze the all-method screen before results, then execute sequential gates."""
import os,sys,json,subprocess,hashlib
from pathlib import Path
from unified_alpha_methods import METHODS,DEFINITIONS

ROOT=Path('alpha_weight_outputs/unified_alpha_pipeline_v2_20261011');ROOT.mkdir(exist_ok=False)
PILOT='alpha_weight_outputs/unified_alpha_pilot_v1_20261011'
SOURCE='alpha_weight_outputs/unified_alpha_source_v1_20261011'
STUDY='alpha_weight_outputs/unified_alpha_screen_v1_20261011'
used=set();count=0
for p in Path('alpha_weight_outputs').rglob('*manifest*.json'):
    try:m=json.loads(p.read_text())
    except (ValueError,OSError):continue
    count+=1
    if isinstance(m,dict) and isinstance(m.get('seeds'),list):used.update(s for s in m['seeds'] if isinstance(s,int))
start=7000
while used.intersection(range(start,start+30)):start+=1000
seeds=list(range(start,start+30));pilot=list(range(5000,5005))
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('.').glob('*.py')}
reg={'experiment_kind':'exploratory_unified_screen','date':'2026-10-11','methods':METHODS,'definitions':DEFINITIONS,
    'scenarios':['reference_clean','low_quality_25pct','sample_imbalance_4x'],'seeds':seeds,'pilot_seeds':pilot,
    'source_output':SOURCE,'study_output':STUDY,'pilot_output':PILOT,'expected':3*30*len(METHODS),
    'source_sha256':hashes,'seed_audit':{'manifests':count,'overlap':sorted(set(seeds)&used)},
    'comparison_rule':'same models/data/proposals/evaluator; target0; 2D; IDs1/2/20; best_of custom30; weighted local absolute target error',
    'information_budget':'20 original qualification labels; no new labels; no alpha updates/pruning/settlement',
    'primary':'post-delivery frozen pooled error, lower is better; truth diagnostic only',
    'aggregate':'equal 1/3 weight for each scenario, descriptive only',
    'analysis':'10000 paired seed bootstrap; all 141 baseline contrasts one Bonferroni family; no pooled-selected parameters',
    'claim_limit':'screening ranks, not a confirmed universal winner; independent replication required',
    'deferred':['additional 9 proposal-local labels','local bias risk with additional 80 labels','multiple Stage3 solutions then local selection','alternative ensemble-target objective'],
    'aliases':{'uniform_stacked_half':'ensemble_shrink','sparse_top_two_alpha':'top_two_expertise'},
    'historical_scope':'sparse and stacked weight formulas standardized to common original Stage3; does not replay their former different-objective trials'}
Path('docs/unified_alpha_screen_registration_20261011.json').write_text(json.dumps(reg,ensure_ascii=False,indent=2)+'\n')
state={'status':'running','stages':[],'registration':'docs/unified_alpha_screen_registration_20261011.json'}
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
def save():(ROOT/'status.json').write_text(json.dumps(state,indent=2)+'\n')
def step(name,args):
    for f,h in hashes.items():
        if hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h:raise RuntimeError('source changed '+f)
    state['current_stage']=name;save();print('Starting '+name,flush=True)
    with (ROOT/(name+'.log')).open('w') as log:
        p=subprocess.run([sys.executable,*args],env=env,stdout=log,stderr=subprocess.STDOUT)
    state['stages'].append({'name':name,'command':args,'returncode':p.returncode});save()
    if p.returncode:raise RuntimeError(name+' failed; inspect log')
try:
    print('Frozen all-method screen; '+str(len(METHODS))+' methods; new replications '+str(seeds),flush=True)
    step('pilot',['run_unified_alpha_comparison.py','--source','alpha_weight_outputs/adaptive_corr_source_v1_20261010','--output',PILOT,'--seeds',','.join(map(str,pilot))])
    step('pilot_acceptance',['analyze_unified_alpha_comparison.py',PILOT])
    step('source',['run_alpha_initialization_study.py','--output-dir',SOURCE,'--seeds',','.join(map(str,seeds)),'--scenarios',','.join(reg['scenarios']),'--methods','A_uniform','--primary-baseline','A_uniform'])
    step('source_acceptance',['verify_alpha_initialization_study.py',SOURCE])
    step('screen',['run_unified_alpha_comparison.py','--source',SOURCE,'--output',STUDY,'--seeds',','.join(map(str,seeds))])
    step('screen_acceptance',['analyze_unified_alpha_comparison.py',STUDY])
    state['status']='complete'
except Exception as e:state.update(status='interrupted',reason=repr(e));raise
finally:save()
