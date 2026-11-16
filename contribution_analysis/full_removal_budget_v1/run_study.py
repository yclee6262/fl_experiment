"""Sequential bounded scheduler with a pilot/replay gate before confirmation."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from worker import write, digest


def load(path):
    return json.loads(path.read_text())


def execute(root):
    config=load(root/'registration.json')
    snapshot=root/'source_snapshot'
    worker=snapshot/'worker.py'
    totals=dict(calls=0,input_samples=0,solver_seconds=0.0,preparation_seconds=0.0,solves=0)
    events=[]
    status='running'
    def launch(mode,directory,output,extra=(),timeout=120):
        command=[sys.executable,str(worker),mode,'--snapshot',str(snapshot),
                 '--directory',str(directory),'--output',str(output),*map(str,extra)]
        log=output.with_suffix('.log')
        with log.open('x') as f:
            try:
                run=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,
                       timeout=timeout,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
                                            'OPENBLAS_NUM_THREADS':'1','PYTHONHASHSEED':'0'})
            except subprocess.TimeoutExpired:
                if not output.exists():
                    write(output,dict(status='process_timeout',error=f'{mode}_process_timeout'))
                raise RuntimeError(f'{mode}:process_timeout')
        data=load(output)
        if run.returncode or data['status'] not in ['complete','valid_delivery']:
            raise RuntimeError(f'{output.relative_to(root)}:{data.get("status")}:{data.get("error")}')
        return data
    def one_solve(directory,label,excluded,iterations):
        if totals['solves']>=250:
            raise RuntimeError('solve_count_budget')
        remaining_calls=1000000-totals['calls']
        remaining_samples=1000000-totals['input_samples']
        remaining_seconds=1800-totals['solver_seconds']
        if min(remaining_calls,remaining_samples,remaining_seconds)<=0:
            raise RuntimeError('global_solver_budget')
        p=directory/'solves'/f'{label}_{excluded}.json'
        try:
            data=launch('solve',directory,p,['--excluded',excluded,'--iterations',iterations,
                      '--calls',min(20000,remaining_calls),'--samples',remaining_samples,
                      '--seconds',min(60,remaining_seconds)])
            import numpy as np
            state=load(directory/'preparation.json')
            expected_ids=[i for j,i in enumerate(state['ids']) if j!=excluded]
            if data['remaining_ids']!=expected_ids or abs(sum(data['alphas'])-1)>1e-12:
                raise RuntimeError('removal_state_mismatch')
            if data['fixed_start']!=state['fixed_start'] or data['evaluator_loaded']:
                raise RuntimeError('starting_state_or_evaluator_leak')
            if not np.isfinite(data['result']['solution']).all() or data['result']['max_box_violation']>1e-8:
                raise RuntimeError('invalid_delivery')
            if set(data['engines'])!={'bfgs','custom','scipy'}:
                raise RuntimeError('missing_engine_metadata')
        finally:
            if p.exists():
                d=load(p)
                totals['solves']+=1
                for source,target in [('calls','calls'),('input_samples','input_samples'),('seconds','solver_seconds')]:
                    totals[target]+=d.get(source,0)
        return data
    try:
        for phase in ['pilot','confirmation']:
            for seed in config[phase+'_seeds']:
                for scenario in ['clean','controlled_poison']:
                    directory=root/phase/scenario/str(seed)
                    directory.mkdir(parents=True,exist_ok=False)
                    (directory/'solves').mkdir()
                    print(f'PREPARE {phase} {scenario} seed={seed}',flush=True)
                    remaining=3600-totals['preparation_seconds']
                    if remaining<=0:
                        raise RuntimeError('preparation_budget')
                    data=launch('prepare',directory,directory/'preparation.json',
                                ['--seed',seed,'--scenario',scenario],timeout=remaining)
                    totals['preparation_seconds']+=data['preparation_seconds']+data['evaluator_seconds']
                    if totals['preparation_seconds']>3600:
                        raise RuntimeError('preparation_budget')
                    # Source and semantic dataset hashes are checked before any solves.
                    observed=[load(p)['semantic_data_sha256'] for p in root.glob('*/*/*/preparation.json')
                              if p!=directory/'preparation.json' and load(p).get('status')=='complete']
                    # Same seed across scenarios deliberately shares synthetic X/clean labels;
                    # observed labels differ. Independent seed datasets must not coincide.
                    if data['semantic_data_sha256'] in observed:
                        raise RuntimeError('duplicate_semantic_dataset')
                    for label,iterations in [('A',30),('B',120)]:
                        for excluded in [-1,*range(len(data['ids']))]:
                            one_solve(directory,label,excluded,iterations)
                    if phase=='pilot' and seed==config['pilot_seeds'][0]:
                        for excluded in [-1,*range(len(data['ids']))]:
                            replay=one_solve(directory,'replay_A',excluded,30)
                            original=load(directory/'solves'/f'A_{excluded}.json')
                            import numpy as np
                            for field in ['remaining_ids','alphas','candidates','fixed_start',
                                          'coalition_sha256','local_models_sha256']:
                                if replay[field]!=original[field]:
                                    raise RuntimeError(f'replay_state_mismatch:{field}')
                            if np.max(np.abs(np.array(replay['result']['solution'])-original['result']['solution']))>1e-7:
                                raise RuntimeError('replay_X_mismatch')
                            if abs(replay['result']['loss']-original['result']['loss'])>1e-7:
                                raise RuntimeError('replay_objective_mismatch')
                    launch('score',directory,directory/'scores.json')
                    scores=load(directory/'scores.json')['rows']
                    by_name={r['file']:r for r in scores}
                    for r in scores:
                        if r['file'].startswith('replay_A_'):
                            original=by_name[r['file'].replace('replay_A_','A_')]
                            if abs(r['pooled_error']-original['pooled_error'])>1e-7:
                                raise RuntimeError('replay_pooled_mismatch')
                    events.append(dict(phase=phase,scenario=scenario,seed=seed,status='complete'))
                    print(f'COMPLETE {phase} {scenario} seed={seed} totals={totals}',flush=True)
            # Numerical stability outcomes never determine whether confirmation runs.
            if phase=='pilot':
                # Re-load evaluation in an independent process and compare all saved scores.
                for directory in sorted((root/'pilot').glob('*/*')):
                    launch('score',directory,directory/'reload_scores.json')
                    original=load(directory/'scores.json')['rows']
                    reloaded=load(directory/'reload_scores.json')['rows']
                    if len(original)!=len(reloaded) or any(
                        a['file']!=b['file'] or abs(a['pooled_error']-b['pooled_error'])>1e-7
                        for a,b in zip(original,reloaded)):
                        raise RuntimeError('pilot_reload_mismatch')
                projected={k:totals[k]*6 for k in ['calls','input_samples','solver_seconds','preparation_seconds']}
                if projected['calls']>1000000 or projected['input_samples']>1000000 or projected['solver_seconds']>1800 or projected['preparation_seconds']>3600:
                    raise RuntimeError('pilot_projected_budget_exceeds_limit')
                write(root/'pilot_gate.json',dict(accepted=True,events=events,projected_costs=projected,
                        source_hashes_match=all(digest(snapshot/p)==h for p,h in config['source_sha256'].items())))
                if not load(root/'pilot_gate.json')['source_hashes_match']:
                    raise RuntimeError('source_hash_mismatch')
                print('PILOT GATE ACCEPTED; confirmation begins without changing methods or thresholds',flush=True)
        status='complete'
    except Exception as exc:
        status='stopped'
        events.append(dict(status='stop',error=f'{type(exc).__name__}:{exc}'))
        print(f'STOPPED: {exc}',flush=True)
    finally:
        write(root/'execution.json',dict(status=status,totals=totals,events=events))
    return status


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    args=p.parse_args()
    sys.exit(0 if execute(args.root.resolve())=='complete' else 1)
