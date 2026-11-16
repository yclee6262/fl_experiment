"""Instrument original solver and evaluate all owner-round anchor ablations."""
import time
import math
import numpy as np
from contribution_analysis.custom_realized_gain_v1.runtime import load_server,manifest_for
from contribution_analysis.custom_realized_gain_v1.observer import record_custom_run
from contribution_analysis.custom_realized_gain_v1.objective_trace import ObjectiveTrace
from contribution_analysis.custom_realized_gain_v1.accounting import compute_accounts
from contribution_analysis.custom_realized_gain_v1.verify import verify_run
from contribution_analysis.custom_realized_gain_v1.schema import plain
from contribution_analysis.three_pool_t2_anchor_v1.core import owner_vectors,decide
from .settlement import settle
from .reference import verify_branches,verify_settlement
from .analysis import stress,prefixes,ledger_checks


def configured(case,variant,model_hash):
    t=time.perf_counter();s=load_server(case);load_seconds=time.perf_counter()-t
    original_owners=[str(i)for i in case['ids']];a=np.array(s.alphas,float);K=len(s.I_list)
    indices=list(range(K));weights=a.tolist()
    if variant.startswith('split_'):
        indices=[i for i in range(K)for _ in range(2)]if variant=='split_adjacent'else list(range(K))*2
        weights=[a[i]/2 for i in indices]
        s.I_list=np.asarray([case['candidates'][i]for i in indices]);s.direction_weights=np.asarray(weights)
    owners=[original_owners[i]for i in indices]
    config=dict(num_iterations=60 if variant=='budget60'else 30,use_annealing=True,allow_tangent=True)
    m=manifest_for(s,case['fixed_start'],config,anchor_owners=owners,model_owners=original_owners)
    m.update(case_id=case['key'],variant=variant,model_artifact=case['models'],model_artifact_sha256=model_hash,
             model_versions=[model_hash]*len(case['ids']),source_anchor_indices=indices)
    return s,m,load_seconds


def execute(case,variant,model_hash,record=True):
    s,m,load_time=configured(case,variant,model_hash);started=time.perf_counter()
    r=record_custom_run(s,case['fixed_start'],m,record)
    if not record:return dict(manifest=m,run=r,load_seconds=load_time,pipeline_seconds=time.perf_counter()-started)
    scoring=time.perf_counter();accounts=compute_accounts(m,r['evaluations'],r['iterations'],r['initial_eval_id'],r['final_eval_id'])
    original_check=verify_run(m,plain(r),plain(accounts));original_audit_seconds=time.perf_counter()-scoring
    branches=[];owners=list(dict.fromkeys(m['anchor_owners']));t=time.perf_counter()
    trace=ObjectiveTrace(s,True,m['model_versions'])
    with trace:
        for it in r['iterations']:
            os,H=owner_vectors(m,r,it)
            if os!=owners:raise ValueError('owner_order')
            for i,o in enumerate(owners):
                start_eid=len(trace.evaluations);trace.context=dict(purpose='anchor_ablation_trial',iteration=it['t'],anchor_id=None,attempt=None)
                alt=decide(s,trace,it['x_before'],np.asarray(it['g_solver'])-H[i],it['L_before'],it['eta_before'],it['mode_before'])
                C=alt['loss']-it['L_after'];rho=max(1e-12,32*np.finfo(float).eps*(1+abs(alt['loss'])+abs(it['L_after'])))
                branches.append(dict(t=it['t'],owner=o,H=H[i],decision=alt,C=C,rho=rho,full_loss=it['L_after'],
                                     eval_ids=list(range(start_eid,len(trace.evaluations)))))
    branch_seconds=time.perf_counter()-t
    btrace=dict(evaluations=trace.evaluations,queries=trace.queries,costs=trace.costs())
    branch_check_start=time.perf_counter();branch_check=verify_branches(m,plain(r),plain(branches),plain(btrace));branch_check_seconds=time.perf_counter()-branch_check_start
    events=[dict(owner=b['owner'],t=b['t'],C=str(b['C']),rho=str(b['rho']))for b in branches];rounds=[it['t']for it in r['iterations']]
    t=time.perf_counter();allocation=settle(owners,rounds,events);verify_settlement(owners,rounds,events,allocation)
    path=prefixes(owners,rounds,events);numeric=stress(owners,rounds,events,allocation);ledger=ledger_checks(owners,rounds,events,allocation)
    analysis_seconds=time.perf_counter()-t
    unique={(q['model_index'],q['input_hex'])for q in r['queries']+btrace['queries']}
    return dict(manifest=m,run=r,accounts=accounts,original_verification=original_check,branches=branches,branch_trace=btrace,
                branch_verification=branch_check,events=events,allocation=allocation,prefixes=path,numerical=numeric,ledger_checks=ledger,
                cost=dict(load_seconds=load_time,search_seconds=r['costs']['search_seconds'],branch_seconds=branch_seconds,
                          original_audit_seconds=original_audit_seconds,branch_audit_seconds=branch_check_seconds,analysis_seconds=analysis_seconds,
                          pipeline_seconds_before_save=time.perf_counter()-started,
                          algorithm_processing_seconds=r['costs']['search_seconds']+branch_seconds+analysis_seconds,
                          main_api_calls=r['costs']['api_calls'],branch_api_calls=trace.query_count,
                          main_input_samples=r['costs']['input_samples'],branch_input_samples=trace.samples,
                          unique_inputs_main_and_branch=len(unique),explicit_cache=False),actual_payments=0)


def compare(base,other):
    b=base['allocation'];v=other['allocation'];os=sorted(b['owners']);bs=base['run'];rs=other['run']
    same_status=b['status']==v['status'];maxq=max(abs(float(b['shares'][o])-float(v['shares'][o]))for o in os)if same_status and b['status']=='settled'else None
    maxnet=max(abs(float(b['stats'][o]['net'])-float(v['stats'][o]['net']))for o in os)
    dx=float(np.max(np.abs(np.asarray(bs['result']['solution'])-np.asarray(rs['result']['solution']))));dl=abs(bs['result']['loss']-rs['result']['loss'])
    original_iterations=bs['iterations'];new_iterations=rs['iterations'];common=min(len(original_iterations),len(new_iterations))
    decision_divergence=[t for t in range(common)if original_iterations[t]['mode_before']!=new_iterations[t]['mode_before']or
                         (original_iterations[t]['accepted_eval_id']is None)!=(new_iterations[t]['accepted_eval_id']is None)]
    passed=same_status and dx<=1e-10 and dl<=1e-10 and maxnet<=1e-8 and(maxq is not None and maxq<=1e-6)and b['payments']==v['payments']
    return dict(passed=passed,max_solution_difference=dx,loss_difference=dl,max_net_difference=maxnet,max_share_difference=maxq,
                payments_equal=b['payments']==v['payments'],base_rounds=len(original_iterations),other_rounds=len(new_iterations),
                common_round_mode_or_acceptance_divergence=decision_divergence,
                exact_result_equal=bs['result']==rs['result'],query_digest_equal=bs['costs']['query_sha256']==rs['costs']['query_sha256'])
