"""Explicit score/pilot stages. Neither stage reads removal labels/evaluator."""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,dict): return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [plain(v) for v in x]
    return x
def save(p,obj):
    with Path(p).open('x') as f: json.dump(plain(obj),f,indent=2,allow_nan=False); f.write('\n')
def csvsave(p,rows):
    with Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def check_sources(root):
    config=read(root/'registration.json')
    for name,h in config['source_sha256'].items():
        if sha(root/'source_snapshot'/name)!=h: raise ValueError('frozen_source:'+name)
    if sha(root/'cases.json')!=config['cases_sha256']: raise ValueError('cases_hash')
    return config


def score(root):
    from attribution import attribute
    from legacy_experiment import load_server
    config=check_sources(root);previous=Path(config['previous'])
    output=root/'records';output.mkdir(exist_ok=False)
    totals=dict(calls=0,samples=0,attribution_seconds=0.,model_load_seconds=0.,new_solves=0)
    rows=[]
    try:
        for case in read(root/'cases.json')['development']:
            print('SCORE',case['key'],flush=True)
            locked=read(previous/'records'/(case['key']+'.json'))
            x=np.asarray(locked['trace']['states']);solution=np.asarray(locked['solution'])
            if not np.array_equal(x[-1],solution): raise ValueError('trace_final_delivery')
            begin=time.perf_counter();server=load_server(case);totals['model_load_seconds']+=time.perf_counter()-begin
            result=attribute(server.trusted_agents,locked['trace'],case['alphas'],h=config['h'])
            totals['calls']+=result['calls'];totals['samples']+=result['samples'];totals['attribution_seconds']+=result['seconds']
            if totals['samples']>500000 or totals['attribution_seconds']>300: raise RuntimeError('score_budget')
            save(output/(case['key']+'.json'),dict(key=case['key'],ids=case['ids'],
                source_record_sha256=sha(previous/'records'/(case['key']+'.json')),
                labels_loaded=False,evaluator_loaded=False,**result))
            for i,agent in enumerate(case['ids']):
                rows.append(dict(key=case['key'],scenario=case['scenario'],seed=case['seed'],agent_id=agent,
                    gradient_work=float(result['gradient_work'][i]),endpoint_work=float(result['endpoint_work'][i]),
                    n_steps=result['n_steps'],chosen_engine=locked['chosen_engine']))
        csvsave(root/'scores.csv',rows)
        save(root/'scores_lock.json',dict(scores_sha256=sha(root/'scores.csv'),n_rows=len(rows),
            record_sha256={p.name:sha(p) for p in output.glob('*.json')},totals=totals,
            labels_loaded=False,evaluator_loaded=False,model_parameters_used_for_gradients=False))
    except Exception as e:
        save(root/'score_failure.json',dict(error=repr(e),totals=totals));raise


def pilot(root):
    from attribution import attribute
    from legacy_experiment import load_server,verify
    from recorder import solve
    check_sources(root)
    totals=dict(solves=0,baseline_calls=0,baseline_samples=0,gradient_calls=0,gradient_samples=0,seconds=0.)
    rows=[]
    def count(r,a=None):
        totals['solves']+=1;totals['baseline_calls']+=r['counter']['calls'];totals['baseline_samples']+=r['counter']['samples']
        totals['seconds']+=r['seconds']+(a['seconds'] if a else 0.)
        if a:
            totals['gradient_calls']+=a['calls'];totals['gradient_samples']+=a['samples']
        if totals['baseline_samples']+totals['gradient_samples']>500000 or totals['seconds']>300: raise RuntimeError('pilot_budget')
    try:
        for case in read(root/'cases.json')['pilot']:
            print('PILOT',case['key'],flush=True)
            server=load_server(case)
            for _ in range(2): count(solve(server,False,case['history_mode'],initial_solution=case['fixed_start']))
            controls=[];augmented=[];extra=[]
            for repeat in range(15):
                pair={}
                for enabled in ([False,True] if repeat%2==0 else [True,False]):
                    begin=time.perf_counter()
                    r=solve(server,enabled,case['history_mode'],initial_solution=case['fixed_start'],
                            max_calls=20000,max_samples=20000,max_seconds=60)
                    if enabled:
                        a=attribute(server.trusted_agents,r['trace'],server.alphas)
                        elapsed=time.perf_counter()-begin
                        augmented.append(elapsed);extra.append(dict(calls=a['calls'],samples=a['samples'],seconds=a['seconds']))
                    else:
                        elapsed=time.perf_counter()-begin;controls.append(elapsed);a=None
                    count(r,a);verify(case,r);pair[enabled]=r
                if pair[False]['counter']!=pair[True]['counter']: raise ValueError('changed_baseline_queries')
                if not np.array_equal(pair[False]['result']['solution'],pair[True]['result']['solution']): raise ValueError('changed_X')
                if pair[False]['result']['loss']!=pair[True]['result']['loss']: raise ValueError('changed_loss')
            rows.append(dict(key=case['key'],baseline_calls=case['expected_calls'],baseline_samples=case['expected_calls'],
                control_seconds=controls,augmented_seconds=augmented,extra=extra,
                median_ratio=float(np.median(augmented)/np.median(controls)),
                X_and_calls_unchanged=True))
        ratio=sum(np.median(r['augmented_seconds']) for r in rows)/sum(np.median(r['control_seconds']) for r in rows)
        save(root/'pilot_results.json',dict(technical_accepted=True,cost_target_met=bool(ratio<=1.10),
              aggregate_ratio=float(ratio),cases=rows,totals=totals,
              timing_scope='normal solver plus recorder plus FD scoring; excludes model loading and file IO',
              baseline_timing_includes_the_same_call_counters=True))
    except Exception as e:
        save(root/'pilot_failure.json',dict(error=repr(e),totals=totals));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['score','pilot']);p.add_argument('root',type=Path)
    a=p.parse_args();root=a.root.resolve();sys.path.insert(0,str(root/'source_snapshot'))
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if a.mode=='score':score(root)
    else:pilot(root)
