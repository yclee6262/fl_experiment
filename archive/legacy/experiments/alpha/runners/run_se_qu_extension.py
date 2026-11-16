"""Registered paired SE/QU extension; old results are read-only."""
import torch
import json, os, subprocess, sys, time
from pathlib import Path
from oob_shrinkage_research import dump, digest
ROOT=Path('alpha_weight_outputs/se_qu_extension_v1_20261107')

def main():
    ROOT.mkdir(exist_ok=False)
    old=json.loads(Path('alpha_weight_outputs/pure_es_extension_v1_20261014/registration.json').read_text())
    batches=[dict(b, pure_results=b['results'], results=str(ROOT/f'batch_{i+1}')) for i,b in enumerate(old['batches'])]
    eight=['A_uniform','B_stage0','stacked_full','ensemble_shrink','B_stacked_half','pure_support','peer_support_shrink','support_quality_mix50']
    new=['support_ensemble_mix50','quality_shrink']
    reg=dict(kind='paired_posthoc_extension_not_independent_confirmation', methods=new, original_methods=eight,
        formulas={'SE':'0.5*S+0.5*E','QU':'0.5*Q+0.5*U'}, batches=batches, scenarios=old['scenarios'],
        contrasts=[[m,b] for m in new for b in eight]+[[new[0],new[1]]],
        bootstrap_replicates=200000,analysis_rng=20261107,family_error=.05,
        primary='Equal mean of three scenarios within seed; paired bootstrap over 90 seeds',
        stability_rule='corrected upper<0; all three batch means<0; at least two batch ordinary CI upper<0; no excess failures',
        scope='270 case-scenario pairs, 540 added solves, no tuning; original 2160 results retained',
        feasibility_tolerance=1e-8,reload_tolerance=1e-7,
        pooled_role='post-delivery only, not passed to initializer or solver',truth_role='diagnostic only',
        additional_labels=0,alpha_update=False,pruning=False,payment=False,
        cost_rule='one Stage3 best_of invocation; SE adds six peer queries, QU zero; elapsed historical ratios descriptive only',
        source_sha256={p.name:digest(p) for p in Path('.').glob('*.py')},
        original_results_sha256={str(Path(b[k])/'results.jsonl'):digest(Path(b[k])/'results.jsonl') for b in batches for k in ['pure_results','six_method_results']})
    # Each batch's first seed is also rerun for an independent solve replay.
    reg['batches'] += [dict(b,seeds=b['seeds'][:1],results=str(ROOT/f'replay_{i+1}')) for i,b in enumerate(batches)]
    dump(ROOT/'registration.json',reg)
    state=dict(status='running',started=time.time(),stages=[]);dump(ROOT/'pipeline_status.json',state)
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    def step(name,args):
        state['stage']=name;dump(ROOT/'pipeline_status.json',state)
        with (ROOT/f'{name}.log').open('w') as log:
            code=subprocess.call([sys.executable]+args,stdout=log,stderr=subprocess.STDOUT,env=env)
        state['stages'].append(dict(name=name,returncode=code));dump(ROOT/'pipeline_status.json',state)
        if code:raise RuntimeError(name+' failed; see saved log')
    try:
        for i,b in enumerate(reg['batches']):
            step(f'run_{i+1}',['run_se_qu_batch.py','--registration',str(ROOT/'registration.json'),'--batch',str(i)])
            step(f'verify_{i+1}',['run_se_qu_batch.py','--verify',b['results']])
        step('analysis',['analyze_se_qu_extension.py'])
        state.update(status='complete',finished=time.time());dump(ROOT/'pipeline_status.json',state)
    except Exception as exc:
        state.update(status='interrupted',reason=repr(exc));dump(ROOT/'pipeline_status.json',state);raise
if __name__=='__main__':main()
