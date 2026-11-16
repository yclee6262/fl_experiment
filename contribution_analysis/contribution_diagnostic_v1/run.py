"""Two explicit phases; pooled evaluator is never loaded here."""
import argparse
import sys
import time
from pathlib import Path
import numpy as np
from common import read,save,sha,check

def full_revaluation(root):
    from legacy_experiment import load_server
    from core import fixed_evaluate
    cfg=check(root);old=Path(cfg['previous_custom']);out=root/'full_revaluation';out.mkdir();totals=dict(calls=0,samples=0,solves=0,evaluation_seconds=0.,model_load_seconds=0.)
    try:
        for case in read(root/'cases.json'):
            print('FIXED_PANEL',case['key'],flush=True);t=time.perf_counter();server=load_server(case);totals['model_load_seconds']+=time.perf_counter()-t
            base=read(old/'baselines'/(case['key']+'.json'));baseL=base['result']['loss'];rows=[]
            for i in case['ids']:
                removal=read(old/'removals'/(case['key']+'_'+str(i)+'.json'));x=removal['result']['solution'];t=time.perf_counter()
                r=fixed_evaluate(server.trusted_agents,server.alphas,np.array(x));totals['evaluation_seconds']+=time.perf_counter()-t
                totals['calls']+=r['calls'];totals['samples']+=r['samples']
                if totals['samples']>340:raise RuntimeError('phase_one_budget')
                rows.append(dict(agent_id=i,solution=x,original_panel_ids=case['ids'],original_alphas=case['alphas'],
                    baseline_full_L=baseL,full_removal_fixed_L=r['loss'],C_full_L=r['loss']-baseL,**r))
            save(out/(case['key']+'.json'),dict(key=case['key'],rows=rows,evaluator_loaded=False,truth_loaded=False))
        save(root/'full_revaluation_lock.json',dict(record_sha256={p.name:sha(p) for p in out.glob('*.json')},totals=totals))
    except Exception as e:save(root/'full_revaluation_failure.json',dict(reason=repr(e),totals=totals));raise

def anchor_runs(root):
    from legacy_experiment import load_server
    from core import anchor_only
    cfg=check(root);lock=read(root/'full_revaluation_lock.json')
    for n,h in lock['record_sha256'].items():
        if sha(root/'full_revaluation'/n)!=h:raise ValueError('phase_one_changed')
    out=root/'anchor_removals';out.mkdir();totals=dict(solves=0,calls=0,samples=0,solver_seconds=0.,model_load_seconds=0.)
    try:
        for case in read(root/'cases.json'):
            for k,i in enumerate(case['ids']):
                print('ANCHOR_ONLY',case['key'],i,flush=True);t=time.perf_counter();server=load_server(case);totals['model_load_seconds']+=time.perf_counter()-t
                r=anchor_only(server,case,k);totals['solves']+=1;totals['calls']+=r['counter']['calls'];totals['samples']+=r['counter']['samples'];totals['solver_seconds']+=r['seconds']
                if totals['solves']>80 or totals['samples']>200000 or totals['solver_seconds']>300:raise RuntimeError('phase_two_budget')
                save(out/(case['key']+'_'+str(i)+'.json'),dict(key=case['key'],excluded_anchor_id=i,excluded_anchor_index=k,**r))
        save(root/'anchor_removal_lock.json',dict(record_sha256={p.name:sha(p) for p in out.glob('*.json')},totals=totals,evaluator_loaded=False,truth_loaded=False))
    except Exception as e:save(root/'anchor_run_failure.json',dict(reason=repr(e),totals=totals));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['full_revaluation','anchor_runs']);p.add_argument('root',type=Path);a=p.parse_args();root=a.root.resolve();sys.path.insert(0,str(root/'source_snapshot'))
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if a.mode=='full_revaluation':full_revaluation(root)
    else:anchor_runs(root)
