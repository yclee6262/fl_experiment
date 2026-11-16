"""Paired same-machine pipelines. No evaluator, label reading, or tuning."""
import argparse
import gc
import resource
import sys
import time
import tracemalloc
from pathlib import Path
import numpy as np
from common import read,save,sha,check

def benchmark(root):
    from legacy_experiment import load_server
    from direction_observer import solve
    from kernel import score
    cfg=check(root);previous=Path(cfg['previous']);lock=read(root/'scores_lock.json')
    for n,h in lock['record_sha256'].items():
        if sha(root/'records'/n)!=h:raise ValueError('score_lock')
    totals=dict(solves=0,calls=lock['totals']['calls'],samples=lock['totals']['samples'],core_seconds=lock['totals']['core_seconds'])
    began=time.perf_counter();rows=[]
    def pipeline(case,method):
        start=time.perf_counter();server=load_server(case);loaded=time.perf_counter()-start
        baseline=solve(server,method=='anchor',case['fixed_start']);calls=baseline['counter']['calls'];samples=baseline['counter']['samples'];core=baseline['total_seconds'];solves=1
        originals=(server.trusted_agents,server.I_list,server.alphas)
        removal_X=[];anchor_score=None
        if method=='anchor':
            anchor_score=score(case,baseline,server.trusted_agents);calls+=anchor_score['counter']['calls'];samples+=anchor_score['counter']['samples'];core+=anchor_score['core_seconds']
        else:
            for k in range(len(case['ids'])):
                server.trusted_agents=[v for j,v in enumerate(originals[0]) if j!=k]
                server.I_list=np.array([v for j,v in enumerate(originals[1]) if j!=k]);w=np.array([v for j,v in enumerate(originals[2]) if j!=k]);server.alphas=list(w/w.sum())
                r=solve(server,False,case['fixed_start']);solves+=1;calls+=r['counter']['calls'];samples+=r['counter']['samples'];core+=r['total_seconds'];removal_X.append(r['result']['solution'])
        elapsed=time.perf_counter()-start
        totals['solves']+=solves;totals['calls']+=calls;totals['samples']+=samples;totals['core_seconds']+=core
        if totals['solves']>450 or totals['samples']>500000 or totals['core_seconds']>300 or time.perf_counter()-began>900:raise RuntimeError('benchmark_budget')
        old=read(previous/'baselines'/(case['key']+'.json'))
        if not np.array_equal(baseline['result']['solution'],old['result']['solution']):raise ValueError('fresh_baseline_changed')
        if baseline['query_order_sha256']!=old['query_order_sha256']:raise ValueError('fresh_baseline_queries_changed')
        if method=='anchor':
            target=read(root/'records'/(case['key']+'.json'))
            if anchor_score['status']!=target['status'] or not np.allclose(anchor_score['A_partial'],target['A_partial'],atol=1e-10,rtol=0):raise ValueError('fresh_A_changed')
        else:
            for i,x in zip(case['ids'],removal_X):
                target=read(previous/'removals'/(case['key']+'_'+str(i)+'.json'))
                if not np.array_equal(x,target['result']['solution']):raise ValueError('fresh_removal_changed')
        return dict(method=method,model_load_seconds=loaded,compute_seconds=core,pipeline_seconds=elapsed,
                    calls=calls,samples=samples,solves=solves,process_RSS_high_water_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    try:
        for case in read(root/'cases.json')['pilot']:
            print('BENCHMARK',case['key'],flush=True);warm=[];samples=[]
            for j in range(2):
                for method in ['anchor','full_LOO']:
                    if j==0:tracemalloc.start()
                    result=pipeline(case,method)
                    if j==0:
                        result['python_tracemalloc_peak_bytes']=tracemalloc.get_traced_memory()[1];tracemalloc.stop()
                    warm.append(result)
            for repeat in range(15):
                for method in (['anchor','full_LOO'] if repeat%2==0 else ['full_LOO','anchor']):
                    samples.append(dict(repeat=repeat,**pipeline(case,method)))
            rows.append(dict(key=case['key'],warmups=warm,timings=samples,technical_accepted=True))
        aggregation={}
        for metric in ['compute_seconds','pipeline_seconds','samples','calls']:
            a=sum(np.median([s[metric] for s in c['timings'] if s['method']=='anchor']) for c in rows)
            b=sum(np.median([s[metric] for s in c['timings'] if s['method']=='full_LOO']) for c in rows)
            aggregation[metric]=dict(anchor=float(a),full_LOO=float(b),ratio=float(a/b),saving=float(1-a/b))
        save(root/'benchmark_results.json',dict(cases=rows,aggregate=aggregation,totals_including_first_scores_and_warmups=totals,
            elapsed_seconds=time.perf_counter()-began,evaluator_loaded=False,LOO_labels_loaded=False,
            timing_excludes='quality evaluator, disk artifact writing and integrity checking after each pipeline'))
    except Exception as e:
        if tracemalloc.is_tracing():tracemalloc.stop()
        save(root/'benchmark_failure.json',dict(reason=repr(e),totals=totals));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root.resolve();sys.path.insert(0,str(root/'source_snapshot'))
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);benchmark(root)
