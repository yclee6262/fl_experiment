"""Three frozen independent batches, without eliminating any candidate."""
import argparse,hashlib,json,math,os,platform,subprocess,sys,time
from pathlib import Path
import numpy as np
from scipy.stats import norm
from repeated_alpha_candidates import METHODS,CANDIDATES,DEFINITIONS
from analyze_repeated_alpha_confirmation import contrasts

REG=Path('docs/repeated_alpha_confirmation_registration_20261012.json')
ROOT=Path('alpha_weight_outputs/repeated_alpha_confirmation_v1_20261012')

def freeze():
    ROOT.mkdir(exist_ok=False)
    old=[json.loads(l) for l in Path('alpha_weight_outputs/unified_alpha_screen_v1_20261011/results.jsonl').read_text().splitlines()]
    old += [json.loads(l) for l in Path('alpha_weight_outputs/unified_alpha_extra_labels_v1_20261011/results.jsonl').read_text().splitlines()]
    idx={(r['scenario'],r['seed'],r['method']):r for r in old};scenarios=['reference_clean','low_quality_25pct','sample_imbalance_4x']
    priorseeds=sorted(set(r['seed'] for r in old));sds=[]
    for method,base in contrasts()[:-2]:
        ds=[np.mean([idx[s,k,method]['pooled_error']-idx[s,k,base]['pooled_error'] for s in scenarios]) for k in priorseeds]
        sds.append({'method':method,'baseline':base,'SD':float(np.std(ds,ddof=1))})
    family=len(contrasts())*4;resolution=.003;power=.9
    n=math.ceil(((norm.ppf(1-.05/(2*family))+norm.ppf(power))*max(x['SD'] for x in sds)/resolution)**2)
    per_batch=max(30,math.ceil(n/3));used=set();scanned=0
    for directory in ['alpha_weight_outputs','docs']:
        for p in Path(directory).rglob('*registration*.json') if directory=='docs' else Path(directory).rglob('*manifest*.json'):
            try:m=json.loads(p.read_text())
            except (ValueError,OSError):continue
            scanned+=1
            if isinstance(m,dict) and isinstance(m.get('seeds'),list):used.update(s for s in m['seeds'] if isinstance(s,int))
    batches=[];start=8000
    for i in range(3):
        while used.intersection(range(start,start+per_batch)):start+=1000
        seeds=list(range(start,start+per_batch));used.update(seeds);broot=ROOT/f'batch_{i+1}';broot.mkdir()
        batches.append({'number':i+1,'seeds':seeds,'source':str(broot/'source'),'results':str(broot/'results')});start+=1000
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('.').glob('*.py')}
    snapshot=ROOT/'source_snapshot';snapshot.mkdir()
    for name in hashes:(snapshot/name).write_bytes(Path(name).read_bytes())
    r={'date':'2026-10-12','kind':'all_candidate_three_batch_confirmation','output':str(ROOT),'methods':METHODS,'candidates':CANDIDATES,
       'definitions':DEFINITIONS,'scenarios':scenarios,'batches':batches,'seeds':[s for b in batches for s in b['seeds']],
       'source_sha256':hashes,'expected_conditions':3*per_batch*3*len(METHODS),
       'primary_endpoint':'per-seed equal mean pooled error across three scenarios','secondary_endpoints':scenarios,
       'contrasts':contrasts(),'comparison_family':family,'family_error':.05,'bootstrap_replicates':500000,'analysis_rng':20261012,
       'formula_policy':'no changes to any of27 candidate formulas; two new matched-label inverse-MSE controls; no pruning of methods',
       'power_planning':{'historical_paired_SD':sds,'maximum_SD':max(x['SD'] for x in sds),'minimum_detectable_delta':resolution,
            'delta_role':'research resolution for planning, not a business success margin','nominal_power':power,'normal_approximation_N':n,
            'rounded_total_N':3*per_batch,'per_batch':per_batch,
            'limitations':'historical SD estimates and normal approximation; matched-control SD unknown; no guaranteed power for smaller effects or individual scenarios'},
       'seed_audit':{'files_scanned':scanned,'new_batches_disjoint':True,'prior_results_role':'planning only, excluded from confirmation'},
       'stability_rule':'final simultaneous CI wholly below0; all three batch means negative; at least2/3 ordinary95% batch CIs below0; complete valid planned pairs; no higher observed failure count',
       'method_claim_rule':'stable versus both A and B on primary endpoint; scenario-specific stability reported separately; augmented scheme claim separate from matched-information increment',
       'retain_rule':'all methods in all three batches regardless of interim rank; insufficient evidence is not proof of no effect; no optional stopping or parameter tuning',
       'solver':'same weighted sum of local absolute target errors; best_of custom30; box[-1,1], tolerance1e-8; reload1e-7; no alpha update/pruning/settlement',
       'pooled_role':'frozen post-delivery primary evaluation only; never enters weights or selection','truth_role':'diagnostic scoring; trusted synthetic label acquisition separate',
       'pilot':{'source':'alpha_weight_outputs/unified_alpha_source_v1_20261011','seeds':priorseeds[:5],'results':str(ROOT/'pilot')},
       'environment':{'python':sys.version,'platform':platform.platform(),'numeric_threads':1}}
    REG.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    return r

def main():
    p=argparse.ArgumentParser();p.add_argument('--register-only',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.register_only:
        r=freeze();print(json.dumps({'N':len(r['seeds']),'conditions':r['expected_conditions'],'per_batch':r['power_planning']['per_batch']}));return
    if not a.execute:raise ValueError('Choose register-only or execute')
    r=json.loads(REG.read_text());state_path=ROOT/'status.json'
    if state_path.exists():raise FileExistsError('Existing pipeline state; inspect process before any recovery')
    state={'status':'running','pid':os.getpid(),'started_unix':time.time(),'stages':[],'registration':str(REG)}
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    def save():state_path.write_text(json.dumps(state,indent=2)+'\n')
    def step(name,args):
        for f,h in r['source_sha256'].items():
            if hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h:raise RuntimeError('frozen code changed '+f)
        state['current_stage']=name;save();print('Starting '+name,flush=True)
        with (ROOT/(name+'.log')).open('w') as log:
            worker=subprocess.Popen([sys.executable,*args],env=env,stdout=log,stderr=subprocess.STDOUT)
            state['worker_pid']=worker.pid;save();rc=worker.wait()
        state['stages'].append({'name':name,'command':args,'returncode':rc});state.pop('worker_pid',None);save()
        if rc:raise RuntimeError(name+' failed')
    try:
        pilot={**r,'batches':[{'seeds':r['pilot']['seeds'],'source':r['pilot']['source'],'results':r['pilot']['results']}]}
        pilot_path=ROOT/'pilot_registration.json';pilot_path.write_text(json.dumps(pilot,ensure_ascii=False,indent=2)+'\n')
        step('pilot',['run_repeated_alpha_batch.py','--registration',str(pilot_path),'--batch','0'])
        step('pilot_acceptance',['run_repeated_alpha_batch.py','--verify',r['pilot']['results']])
        for i,b in enumerate(r['batches']):
            step(f'batch{i+1}_source',['run_alpha_initialization_study.py','--output-dir',b['source'],'--seeds',','.join(map(str,b['seeds'])),
                '--scenarios',','.join(r['scenarios']),'--methods','A_uniform','--primary-baseline','A_uniform'])
            step(f'batch{i+1}_source_acceptance',['verify_alpha_initialization_study.py',b['source']])
            step(f'batch{i+1}_runs',['run_repeated_alpha_batch.py','--registration',str(REG),'--batch',str(i)])
            step(f'batch{i+1}_acceptance',['run_repeated_alpha_batch.py','--verify',b['results']])
            step(f'batch{i+1}_interim',['analyze_repeated_alpha_confirmation.py',str(REG)])
        step('final_analysis',['analyze_repeated_alpha_confirmation.py',str(REG),'--final'])
        state['status']='complete'
    except Exception as exc:state.update(status='interrupted',reason=repr(exc));raise
    finally:state['updated_unix']=time.time();save()

if __name__=='__main__':main()
