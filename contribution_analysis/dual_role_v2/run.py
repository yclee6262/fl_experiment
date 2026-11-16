import sys
import time
from pathlib import Path
import numpy as np
from common import read,save,sha,check_sources


def qualification(case):
    from host_server import HostServer
    from core import quality
    # Reconstruct the ORIGINAL qualification set, not the evaluator's dataset.
    rng=np.random.get_state()
    try:s=HostServer(0.,2,test_seed=case['seed']+7919,n_test=20,constraint_mode='box_constrained')
    finally:np.random.set_state(rng)
    prior=Path(case['models']).parents[2]/'coalition.json';o=read(prior)
    mse=[next(r['mse'] for r in o['phase1_report'] if r['agent_id']==i) for i in case['ids']]
    variance=float(np.var(s.test_y))
    return dict(MSE=mse,variance=variance,q=quality(mse,variance),test_X=s.test_X,test_y=s.test_y,
                source=str(prior),test_seed=case['seed']+7919,n_test=20,
                source_kind='reconstructed original Stage1 common qualification',pooled_loaded=False)


def equivalent(a,b):
    assert a['counter']==b['counter'] and a['query_order_sha256']==b['query_order_sha256'],'query_behavior'
    for name in ['solution','history']:
        np.testing.assert_array_equal(a['result'][name],b['result'][name])
    for name in ['states','loss']:assert a['result'][name]==b['result'][name],name


def technical(root):
    from core import run
    from legacy_experiment import load_server
    from direction_observer import solve
    cfg=check_sources(root);out=root/'pilot';out.mkdir();timing=[];totals=dict(solves=0,samples=0)
    for c in read(root/'cases.json')['pilot']:
        print('TECHNICAL',c['key'],flush=True);s=load_server(c);q=qualification(c)['q']
        legacy=solve(s,False,c['fixed_start']);totals['solves']+=1;totals['samples']+=legacy['counter']['samples']
        plain=run(s,c['fixed_start']);equivalent(legacy,plain);totals['solves']+=1;totals['samples']+=plain['counter']['samples']
        # Two warmups (one of each); all measured pairs alternate execution order.
        for record in [False,True]:
            r=run(s,c['fixed_start'],record,q);totals['solves']+=1;totals['samples']+=r['counter']['samples']
        controls=[];records=[]
        for rep in range(10):
            pair={}
            for flag in ([False,True] if rep%2==0 else [True,False]):
                t=time.perf_counter();r=run(s,c['fixed_start'],flag,q);wall=time.perf_counter()-t
                pair[flag]=r;totals['solves']+=1;totals['samples']+=r['counter']['samples']
                (records if flag else controls).append(wall)
            equivalent(pair[False],pair[True]);equivalent(legacy,pair[True])
        t=time.perf_counter();save(out/(c['key']+'.json'),pair[True]);saving=time.perf_counter()-t
        timing.append(dict(key=c['key'],control_seconds=controls,record_seconds=records,
                           median_ratio=float(np.median(records)/np.median(controls)),saving_seconds=saving,
                           behavior_identical=True,samples=plain['counter']['samples']))
    ratio=sum(np.median(x['record_seconds']) for x in timing)/sum(np.median(x['control_seconds']) for x in timing)
    save(root/'technical.json',dict(accepted=True,cases=timing,aggregate_ratio=float(ratio),cost_gate=bool(ratio<=1.10),
         totals=totals,scope='loaded-model solver+observer+attribution, serialization separate',
         baseline_compatibility_solves=8,warmup_and_measured_solves=88))


def scores(root):
    from core import attribute
    cfg=check_sources(root);prior=Path(cfg['previous_custom']);out=root/'scores';out.mkdir();quals=root/'qualification';quals.mkdir()
    for c in read(root/'cases.json')['development']:
        print('LOCK_SCORE',c['key'],flush=True);b=read(prior/'baselines'/(c['key']+'.json'));q=qualification(c)
        sc=attribute(b['iterations'],c['alphas'],q['q'])
        gain=b['result']['history'][0]-b['result']['loss']
        assert abs(sc['G']-gain)<=1e-10
        save(quals/(c['key']+'.json'),q)
        save(out/(c['key']+'.json'),dict(case=c,**sc,baseline_solution=b['result']['solution'],
             baseline_loss=b['result']['loss'],baseline_counter=b['counter'],
             per_agent_samples=[b['counter']['samples']//len(c['ids'])]*len(c['ids']),
             evaluator_loaded=False,removal_labels_loaded=False,source_baseline_sha256=sha(prior/'baselines'/(c['key']+'.json'))))
    save(root/'scores_lock.json',dict(record_sha256={str(p.relative_to(root)):sha(p) for d in [out,quals] for p in d.glob('*.json')},
         evaluator_loaded=False,removal_labels_loaded=False,normal_solves_reused=20,new_model_samples=0))


def references(root):
    from core import run
    from legacy_experiment import load_server
    cfg=check_sources(root)
    for n,h in read(root/'scores_lock.json')['record_sha256'].items():assert sha(root/n)==h
    diag=Path(cfg['previous_diagnostic']);out=root/'references';out.mkdir();total=dict(solves=0,samples=0,solver_seconds=0.,load_seconds=0.)
    for c in read(root/'cases.json')['development']:
        for k,i in enumerate(c['ids']):
            print('MODEL_ONLY',c['key'],i,flush=True);t=time.perf_counter();s=load_server(c);total['load_seconds']+=time.perf_counter()-t
            r=run(s,c['fixed_start'],role='model',excluded=k)
            total['solves']+=1;total['samples']+=r['counter']['samples'];total['solver_seconds']+=r['solver_seconds']
            if total['solves']>cfg['new_solver_budget'] or total['samples']>cfg['model_sample_budget'] or total['solver_seconds']>cfg['solver_seconds_budget']:
                save(root/'budget_failure.json',total);raise RuntimeError('reference_budget')
            save(out/(c['key']+'_model_'+str(i)+'.json'),r)
            p=diag/'anchor_removals'/(c['key']+'_'+str(i)+'.json');a=read(p)
            assert a['scoring_model_ids']==c['ids'] and a['fixed_start']==c['fixed_start']
            np.testing.assert_array_equal(a['scoring_alphas'],c['alphas'])
            np.testing.assert_allclose(a['remaining_direction_weights'],np.delete(np.asarray(c['alphas']),k)/(1-c['alphas'][k]),rtol=0,atol=1e-15)
            save(out/(c['key']+'_anchor_'+str(i)+'.json'),dict(reused=True,source=str(p),source_sha256=sha(p),**a))
    save(root/'references_lock.json',dict(record_sha256={p.name:sha(p) for p in out.glob('*.json')},totals=total,
         anchor_solves_reused=80,evaluator_loaded=False))


if __name__=='__main__':
    root=Path(sys.argv[2]).resolve();sys.path.insert(0,str(root/'source_snapshot'))
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    {'technical':technical,'scores':scores,'references':references}[sys.argv[1]](root)
