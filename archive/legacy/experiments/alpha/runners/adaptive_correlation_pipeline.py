"""Pilot/development gate, prospective N registration, fresh sequential formal study."""
import hashlib, json, math, os, subprocess, sys
from pathlib import Path
from statistics import NormalDist

ROOT=Path('alpha_weight_outputs/adaptive_corr_pipeline_20261010')
ROOT.mkdir(exist_ok=False)
ENV=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
METHODS=['A_uniform','B_stage0','bootstrap_conservative_quality','corr_precision_0p5','adaptive_corr_precision']
SCENARIOS=['reference_clean','low_quality_25pct','sample_imbalance_4x']
PILOT=Path('alpha_weight_outputs/adaptive_corr_pilot_v1_20261010')
DEV=Path('alpha_weight_outputs/adaptive_corr_development_v1_20261010')
SOURCE=Path('alpha_weight_outputs/adaptive_corr_source_v1_20261010')
FORMAL=Path('alpha_weight_outputs/adaptive_corr_formal_v1_20261010')
REG=Path('docs/adaptive_corr_preregistration_20261010.json')
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('.').glob('*.py')}
state={'status':'running','stages':[],'source_sha256':hashes,
 'design':{'grid':[0,.25,.5,.75,1],'folds':5,'fit_rows':16,'heldout_rows':4,'tie_tolerance':1e-12,
 'pilot_seeds':list(range(4800,4805)),'development_seeds':list(range(4810,4820)),
 'delta':.005,'nominal_power':.8,'N_rule':'ceil((z_(1-.05/(2*6))+z_.8)^2 * max_development_paired_SD^2 / .005^2), minimum30; no power guarantee',
 'no_tuning_after_development':True,'methods':METHODS,'scenarios':SCENARIOS}}
def save():(ROOT/'status.json').write_text(json.dumps(state,indent=2))
def step(name,args):
    for name_,h in hashes.items():
        if hashlib.sha256(Path(name_).read_bytes()).hexdigest()!=h:raise RuntimeError('Source changed '+name_)
    state['current_stage']=name;save();print('Starting '+name,flush=True)
    with (ROOT/(name+'.log')).open('w') as log:
        p=subprocess.run([sys.executable,*args],env=ENV,stdout=log,stderr=subprocess.STDOUT)
    state['stages'].append({'name':name,'args':args,'returncode':p.returncode});save()
    if p.returncode:raise RuntimeError(name+' failed')
try:
    save()
    for label,seeds,out in [('pilot',range(4800,4805),PILOT),('development',range(4810,4820),DEV)]:
        step(label,['run_adaptive_correlation_study.py','--source','alpha_weight_outputs/conservative_shrink_source_v1_20261010',
            '--output',str(out),'--seeds',','.join(map(str,seeds))])
        step(label+'_acceptance',['analyze_adaptive_correlation.py',str(out)])
    # No scientific imports before child training: avoids OpenMP shared-memory collision.
    a=json.loads((DEV/'analysis.json').read_text())
    summaries=[]
    for c in a['comparisons']:
        if not c['confirmatory']:continue
        ds=[p['delta'] for p in c['seed_pairs']]
        if len(ds)!=10:raise RuntimeError('Cannot plan N from incomplete development pairs')
        mean=sum(ds)/len(ds);sd=math.sqrt(sum((v-mean)**2 for v in ds)/(len(ds)-1))
        summaries.append({'scenario':c['scenario'],'baseline':c['baseline'],'n':len(ds),'paired_SD':sd})
    sd=max(s['paired_SD'] for s in summaries)
    z=NormalDist().inv_cdf(1-.05/(2*6))+NormalDist().inv_cdf(.8)
    n=max(30,math.ceil(z*z*sd*sd/.005**2))
    # Scan all authoritative manifests before reserving independent formal seeds.
    used=set();scanned=0
    for path in Path('alpha_weight_outputs').rglob('*manifest*.json'):
        try:d=json.loads(path.read_text())
        except (ValueError,OSError):continue
        scanned+=1
        if isinstance(d.get('seeds'),list):used.update(v for v in d['seeds'] if isinstance(v,int))
    start=5000
    while used.intersection(range(start,start+n)):start+=1000
    seeds=list(range(start,start+n))
    reg={'date':'2026-10-10','seeds':seeds,'methods':METHODS,'primary':METHODS[-1],'scenarios':SCENARIOS,
         'source_output':str(SOURCE),'study_output':str(FORMAL),'pilot_output':str(PILOT),'development_output':str(DEV),
         'pilot_seeds':list(range(4800,4805)),'development_seeds':list(range(4810,4820)),
         'N':n,'expected_formal':n*15,'N_rule':state['design']['N_rule'],'development_variability':summaries,
         'research_resolution_delta':.005,'nominal_power':.8,'power_limit':'normal approximation; noisy10seed SD; not guaranteed power or business threshold',
         'seed_audit':{'manifests_checked':scanned,'overlap':sorted(set(seeds)&used)},
         'calibration':state['design'],'confirmatory_baselines':['B_stage0','corr_precision_0p5'],
         'comparison_rule':'6 comparisons; paired seed bootstrap10000 RNG20260924; Bonferroni family.05; complete valid pairs and no higher failures',
         'constraints':'box[-1,1], feasibility1e-8, reload1e-7; no alpha update/pruning/settlement',
         'pooled_role':'post-delivery only; truth diagnostic; neither enters weight or delivery selection',
         'source_sha256':hashes,'formal_parameters_fixed_before_execution':True}
    REG.write_text(json.dumps(reg,indent=2)+'\n');state['formal_N']=n;state['formal_seeds']=seeds;save()
    print(f'Preregistered N={n}, seeds{start}..{start+n-1}',flush=True)
    step('formal_source',['run_alpha_initialization_study.py','--output-dir',str(SOURCE),'--seeds',','.join(map(str,seeds)),
        '--scenarios',','.join(SCENARIOS),'--methods','A_uniform','--primary-baseline','A_uniform'])
    step('source_acceptance',['verify_alpha_initialization_study.py',str(SOURCE)])
    step('formal',['run_adaptive_correlation_study.py','--source',str(SOURCE),'--output',str(FORMAL),'--seeds',','.join(map(str,seeds))])
    step('formal_acceptance',['analyze_adaptive_correlation.py',str(FORMAL),'--registration',str(REG)])
    state['status']='complete'
except Exception as exc:
    state.update(status='interrupted',reason=repr(exc));raise
finally:save()
