"""First score frozen trajectories, without labels/evaluator or new solves."""
import argparse
import resource
import sys
import time
from pathlib import Path
from common import read,save,sha,check

def run(root):
    from legacy_experiment import load_server
    from kernel import score
    cfg=check(root);prior=Path(cfg['previous']);out=root/'records';out.mkdir()
    totals=dict(calls=0,samples=0,core_seconds=0.,new_custom_solves=0);started=time.perf_counter()
    try:
        for phase,cases in read(root/'cases.json').items():
            for case in cases:
                print('SCORE',phase,case['key'],flush=True);start=time.perf_counter()
                b=read(prior/'baselines'/(case['key']+'.json'));read_seconds=time.perf_counter()-start
                begin=time.perf_counter();server=load_server(case);load_seconds=time.perf_counter()-begin
                s=score(case,b,server.trusted_agents)
                totals['calls']+=s['counter']['calls'];totals['samples']+=s['counter']['samples'];totals['core_seconds']+=s['core_seconds']
                if totals['samples']>cfg['sample_budget'] or totals['core_seconds']>cfg['compute_seconds_budget']:raise RuntimeError('score_budget')
                path=out/(case['key']+'.json');before_write=time.perf_counter()
                save(path,dict(key=case['key'],phase=phase,ids=case['ids'],source_baseline_sha256=sha(prior/'baselines'/(case['key']+'.json')),
                    input_read_seconds=read_seconds,model_load_seconds=load_seconds,
                    peak_process_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**s))
                write_seconds=time.perf_counter()-before_write
                # Timing metadata lives separately: it includes serializing the full trace.
                save(out/(case['key']+'.timing.json'),dict(key=case['key'],input_read_seconds=read_seconds,
                    model_load_seconds=load_seconds,core_seconds=s['core_seconds'],query_seconds=s['query_seconds'],
                    write_seconds=write_seconds,end_to_end_seconds=time.perf_counter()-start,record_bytes=path.stat().st_size,
                    process_RSS_high_water_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        save(root/'scores_lock.json',dict(record_sha256={p.name:sha(p) for p in out.glob('*.json')},totals=totals,
            end_to_end_seconds=time.perf_counter()-started,evaluator_loaded=False,LOO_labels_loaded=False))
    except Exception as e:save(root/'score_failure.json',dict(reason=repr(e),totals=totals));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root.resolve();sys.path.insert(0,str(root/'source_snapshot'))
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);run(root)
