"""Score all baselines before any removal or evaluator is loaded."""
import argparse
import sys
import time
from pathlib import Path
import numpy as np
from common import read,sha,save,check_sources


def load_server(case,excluded=-1):
    import torch
    from host_server import HostServer
    from agent_client import AgentNode
    payload=torch.load(case['models'],map_location='cpu',weights_only=True)
    server=HostServer(0.,2,constraint_mode='box_constrained')
    ids=[i for k,i in enumerate(case['ids']) if k!=excluded]
    for i in ids:
        agent=AgentNode(i,None,n_features=2);agent.model.load_state_dict(payload['agents'][i]);agent.model.eval();server.trusted_agents.append(agent)
    server.I_list=np.array([x for k,x in enumerate(case['candidates']) if k!=excluded])
    w=np.array([v for k,v in enumerate(case['alphas']) if k!=excluded]);server.alphas=list(w/w.sum())
    return server


def compare(a,b):
    if a['counter']!=b['counter'] or a['query_order_sha256']!=b['query_order_sha256']:raise ValueError('query_behavior_changed')
    if np.max(abs(np.array(a['result']['solution'])-b['result']['solution']))>1e-7:raise ValueError('delivery_changed')
    if abs(a['result']['loss']-b['result']['loss'])>1e-10:raise ValueError('loss_changed')
    for k in ['history','states']:
        if a['result'][k]!=b['result'][k]:raise ValueError('branch_history_changed')


def baseline(root):
    from observer import solve
    cfg=check_sources(root);cases=read(root/'cases.json');out=root/'baselines';out.mkdir()
    total=dict(solves=0,calls=0,samples=0,compute_seconds=0.);timing=[]
    def count(r):
        total['solves']+=1;total['calls']+=r['counter']['calls'];total['samples']+=r['counter']['samples'];total['compute_seconds']+=r['total_seconds']
        if total['samples']>cfg['model_samples_budget'] or total['compute_seconds']>cfg['compute_seconds_budget']:raise RuntimeError('baseline_budget')
    try:
        for case in cases['pilot']:
            print('PILOT',case['key'],flush=True);server=load_server(case)
            for _ in range(2):count(solve(server,False,case['fixed_start']))
            controls=[];records=[]
            for repeat in range(15):
                pair={}
                for record in ([False,True] if repeat%2==0 else [True,False]):
                    r=solve(server,record,case['fixed_start']);count(r);pair[record]=r
                    (records if record else controls).append(r['total_seconds'])
                compare(pair[False],pair[True])
            save(out/(case['key']+'.json'),dict(case=case,phase='pilot',**pair[True]))
            timing.append(dict(key=case['key'],control_seconds=controls,record_seconds=records,
                               calls=pair[False]['counter']['calls'],samples=pair[False]['counter']['samples'],
                               calls_and_query_order_equal=True,median_ratio=float(np.median(records)/np.median(controls))))
        for case in cases['development']:
            print('BASELINE',case['key'],flush=True);server=load_server(case);r=solve(server,True,case['fixed_start']);count(r)
            save(out/(case['key']+'.json'),dict(case=case,phase='development',**r))
        ratio=sum(np.median(c['record_seconds']) for c in timing)/sum(np.median(c['control_seconds']) for c in timing)
        save(root/'pilot_results.json',dict(technical_accepted=True,cases=timing,aggregate_ratio=float(ratio),cost_target_met=bool(ratio<=1.10)))
        save(root/'baseline_lock.json',dict(record_sha256={p.name:sha(p) for p in out.glob('*.json')},totals=total,
             evaluator_loaded=False,removal_labels_loaded=False))
    except Exception as e:save(root/'baseline_failure.json',dict(reason=repr(e),totals=total));raise


def removals(root):
    from observer import solve
    cfg=check_sources(root);lock=read(root/'baseline_lock.json');totals=dict(lock['totals']);out=root/'removals';out.mkdir()
    for n,h in lock['record_sha256'].items():
        if sha(root/'baselines'/n)!=h:raise ValueError('baseline_unlocked')
    try:
        for phase,cases in read(root/'cases.json').items():
            for case in cases:
                for k,i in enumerate(case['ids']):
                    print('REMOVE',case['key'],i,flush=True)
                    server=load_server(case,k);r=solve(server,False,case['fixed_start'])
                    totals['solves']+=1;totals['calls']+=r['counter']['calls'];totals['samples']+=r['counter']['samples'];totals['compute_seconds']+=r['total_seconds']
                    if totals['solves']>244 or totals['samples']>500000 or totals['compute_seconds']>300:raise RuntimeError('experiment_budget')
                    save(out/(case['key']+'_'+str(i)+'.json'),dict(case_key=case['key'],excluded_id=i,excluded_index=k,
                        remaining_ids=[a.agent_id for a in server.trusted_agents],alphas=server.alphas,
                        candidates=server.I_list,fixed_start=case['fixed_start'],**r))
        save(root/'removal_lock.json',dict(record_sha256={p.name:sha(p) for p in out.glob('*.json')},totals=totals,evaluator_loaded=False))
    except Exception as e:save(root/'removal_failure.json',dict(reason=repr(e),totals=totals));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['baseline','removals']);p.add_argument('root',type=Path);a=p.parse_args();root=a.root.resolve();sys.path.insert(0,str(root/'source_snapshot'))
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if a.mode=='baseline':baseline(root)
    else:removals(root)
