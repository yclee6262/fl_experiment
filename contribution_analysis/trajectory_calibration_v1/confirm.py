"""Explicit conditional confirmation scheduler; never runs on import or prepare."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run(root):
    sys.path.insert(0,str(root/'source_snapshot'))
    from experiment import read,save,sha
    config=read(root/'registration.json')
    if not read(root/'development_gate.json')['accepted']:
        raise RuntimeError('development_gate_not_passed')
    if not read(root/'complete_online_cost_gate.json')['accepted']:
        raise RuntimeError('online_cost_gate_not_passed')
    for name,h in config['source_sha256'].items():
        if sha(root/'source_snapshot'/name)!=h:raise RuntimeError('source_hash_mismatch')
    manifest=read(root/'fitted_models_manifest.json')
    for name,h in manifest['model_sha256'].items():
        if sha(root/name)!=h:raise RuntimeError('model_hash_mismatch')
    confirmation=root/'confirmation';confirmation.mkdir(exist_ok=False)
    save(root/'confirmation_start_manifest.json',dict(status='locked_before_new_data',
        models=manifest,confirmation_seeds=config['confirmation_seeds'],
        source_sha256=config['source_sha256'],thresholds=dict(delta=1e-4,direction=.7,MAE_relative=.1,CI_lower_direction=.5)))
    totals=dict(calls=0,input_samples=0,solver_seconds=0.,preparation_seconds=0.,n_solves=0)
    events=[];status='stopped'
    def launch(script,arguments,output,timeout):
        with output.with_suffix('.log').open('x') as log:
            try:
                proc=subprocess.run([sys.executable,str(root/'source_snapshot'/script),*map(str,arguments)],
                    stdout=log,stderr=subprocess.STDOUT,timeout=timeout,
                    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
            except subprocess.TimeoutExpired:
                if not output.exists():save(output,dict(status='process_timeout'))
                raise RuntimeError('process_timeout:'+str(output))
        if not output.exists():
            save(output,dict(status='process_aborted_without_output',calls=None,input_samples=None))
            raise RuntimeError('unknown_failed_process_cost:'+str(output))
        data=read(output)
        if proc.returncode or data['status'] not in ['complete','valid_delivery']:raise RuntimeError('worker_failure:'+str(output))
        return data
    def account(data):
        totals['calls']+=data['calls'];totals['input_samples']+=data['input_samples'];totals['solver_seconds']+=data['seconds'];totals['n_solves']+=1
    try:
        for seed in config['confirmation_seeds']:
            for scenario in ['clean','controlled_poison']:
                d=confirmation/scenario/str(seed);d.mkdir(parents=True,exist_ok=False);(d/'solves').mkdir()
                print('CONFIRM',scenario,seed,flush=True)
                remaining=3600-totals['preparation_seconds']
                if remaining<=0:raise RuntimeError('preparation_budget')
                started=time.perf_counter()
                prep=launch('reference_worker.py',['prepare','--snapshot',root/'source_snapshot','--directory',d,
                     '--output',d/'preparation.json','--seed',seed,'--scenario',scenario],d/'preparation.json',remaining)
                totals['preparation_seconds']+=time.perf_counter()-started
                if len(prep['ids'])!=(3 if scenario=='clean' else 5):raise RuntimeError('coalition_size')
                for i in [-1,*range(len(prep['ids']))]:
                    remaining_calls=500000-totals['calls'];remaining_seconds=1800-totals['solver_seconds']
                    if min(remaining_calls,remaining_seconds)<=0 or totals['n_solves']>=200:raise RuntimeError('confirmation_budget')
                    output=d/'solves'/('baseline.json' if i==-1 else f'removal_{i}.json')
                    if i==-1:
                        data=launch('predict_new.py',['--root',root,'--directory',d,'--output',output,
                                  '--calls',min(20000,remaining_calls),'--seconds',min(60,remaining_seconds)],output,120)
                        if not (d/'predictions.json').is_file():raise RuntimeError('predictions_not_locked')
                        prediction_hash=sha(d/'predictions.json')
                    else:
                        if sha(d/'predictions.json')!=prediction_hash:raise RuntimeError('predictions_changed')
                        data=launch('reference_worker.py',['solve','--snapshot',root/'source_snapshot','--directory',d,
                             '--output',output,'--excluded',i,'--iterations',30,'--calls',min(20000,remaining_calls),
                             '--samples',min(20000,500000-totals['input_samples']),'--seconds',min(60,remaining_seconds)],output,120)
                    account(data)
                    if data.get('evaluator_loaded'):raise RuntimeError('pooled_loaded_during_solve')
                launch('reference_worker.py',['score','--snapshot',root/'source_snapshot','--directory',d,
                       '--output',d/'scores.json'],d/'scores.json',120)
                events.append(dict(scenario=scenario,seed=seed,status='complete'))
        status='complete'
    except Exception as exc:
        events.append(dict(status='failure',reason=f'{type(exc).__name__}:{exc}'))
    save(root/'confirmation_execution.json',dict(status=status,totals=totals,events=events))
    print(status)
    return status


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args()
    sys.exit(0 if run(args.root.resolve())=='complete' else 1)
