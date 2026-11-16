"""Freeze sources and execute two-team bounded validation on CPU."""
import argparse
import datetime
import json
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
from contribution_analysis.custom_realized_gain_v1.schema import read,save,sha,plain
from contribution_analysis.custom_realized_gain_v1.runtime import equivalent
from .experiment import execute,compare

WORK=Path(__file__).resolve().parents[2]
PILOT=WORK/'contribution_analysis/custom_realized_gain_pilot_v1_20261027'
DESIGN=WORK/'docs/anchor_pool_end_to_end_design_v1_20261028.md'
VARIANTS=['control0','base','control1','repeat','split_adjacent','split_separated','budget60']


def prior_hashes():
    locks={}
    for name in ['custom_realized_gain_pilot_v1_20261027','three_pool_t0_t1_experiment_v1_20261028','three_pool_t2_anchor_experiment_v1_20261028',
                 'three_pool_t3_model_experiment_v1_20261028','anchor_pool_shadow_experiment_v1_20261028']:
        root=WORK/'contribution_analysis'/name;p=root/'completion_receipt.json';locks[str(p)]=sha(p)
        for rel,h in read(p)['file_sha256'].items():
            if sha(root/rel)!=h:raise ValueError('old_artifact_changed:'+str(root/rel))
    return locks


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=WORK/'contribution_analysis/anchor_pool_end_to_end_experiment_v1_20261028');args=ap.parse_args()
    root=args.output.resolve();root.mkdir(exist_ok=False);(root/'source_snapshot').mkdir();started=time.perf_counter();total_calls=total_rounds=0
    paths=[DESIGN,WORK/'host_server.py',WORK/'agent_client.py',WORK/'models.py',
           *Path(__file__).parent.glob('*.py'),Path(__file__).parent/'README.md',
           *(WORK/'contribution_analysis/custom_realized_gain_v1').glob('*.py'),WORK/'contribution_analysis/three_pool_t2_anchor_v1/core.py']
    lock={}
    for p in paths:
        rel=p.relative_to(WORK);dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);lock[str(rel)]=sha(p)
    save(root/'source_lock.json',lock);cases=read(PILOT/'cases.json');save(root/'cases.json',cases)
    model_hashes={c['models']:sha(c['models'])for c in cases};save(root/'model_hashes.json',model_hashes)
    save(root/'registration.json',dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),design_sha256=sha(DESIGN),cases_sha256=sha(root/'cases.json'),
        variants=VARIANTS,old_data_exploration=True,new_formal_confirmation=False,max_main_rounds=250,max_model_api=100000,max_seconds=600,max_bytes=250000000,
        primary_q_tolerance=1e-6,net_tolerance=1e-8,endpoint_tolerance=1e-10,payment_budget=500,economic_release_threshold=None,pooled_calls=0,actual_payments=0))
    try:
        old=prior_hashes();save(root/'historical_receipts.json',old)
        test=subprocess.run([sys.executable,'-m','unittest','contribution_analysis.anchor_pool_end_to_end_v1.test_rules','-v'],cwd=WORK,text=True,capture_output=True)
        save(root/'technical_tests.json',dict(returncode=test.returncode,stdout=test.stdout,stderr=test.stderr))
        if test.returncode:raise RuntimeError('technical_test_failed')
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        summaries=[]
        for c in cases:
            folder=root/c['key'];folder.mkdir();observed={};controls={};costs={}
            for variant in VARIANTS:
                record=not variant.startswith('control');t=time.perf_counter();doc=execute(c,variant,model_hashes[c['models']],record)
                if record:
                    cost=doc['cost'];total_calls+=cost['main_api_calls']+cost['branch_api_calls'];total_rounds+=len(doc['run']['iterations']);observed[variant]=doc
                else:
                    cost=dict(main_api_calls=doc['run']['costs']['api_calls'],branch_api_calls=0,search_seconds=doc['run']['costs']['search_seconds'],
                              pipeline_seconds_before_save=doc['pipeline_seconds'],load_seconds=doc['load_seconds']);total_calls+=cost['main_api_calls'];controls[variant]=doc
                    # No iteration trace in control: match these rounds to base after digest agreement.
                save_start=time.perf_counter();save(folder/(variant+'.json'),doc);save_seconds=time.perf_counter()-save_start
                costs[variant]=dict(**plain(cost),save_seconds=save_seconds,bytes=(folder/(variant+'.json')).stat().st_size,wall_including_load_audit_save=time.perf_counter()-t)
                print(c['key'],variant,'calls',cost['main_api_calls']+cost['branch_api_calls'],flush=True)
                if total_calls>100000 or total_rounds>250 or time.perf_counter()-started>600:raise RuntimeError('budget_exceeded')
            base=observed['base'];equivalent(controls['control0']['run'],base['run']);equivalent(controls['control1']['run'],base['run'])
            total_rounds+=2*len(base['run']['iterations'])
            oldm=read(PILOT/c['key']/'run_metadata.json');old_result_match=base['run']['result']==oldm['result']
            comparisons={v:compare(base,d)for v,d in observed.items()if v!='base'}
            repeat=observed['repeat'];exact_repeat=base['run']['costs']['query_sha256']==repeat['run']['costs']['query_sha256']and base['events']==repeat['events']and base['allocation']==repeat['allocation']
            comparisons['repeat']['exact_events_and_shares']=exact_repeat
            expected_owner=len(set(base['manifest']['anchor_owners']));main_count=len(base['run']['iterations']);pending_old=[]
            # Cross-check existing 69 saved branches, without using them to compute new scores.
            oldrows=read(WORK/'contribution_analysis/three_pool_t2_anchor_experiment_v1_20261028/historical_reanalysis.json')
            for r in [r for r in oldrows if r['case']==c['key']]:
                b=next(b for b in base['branches']if b['owner']==r['owner']and b['t']==r['t']);pending_old.append(abs(b['C']-r['C']))
            controls_mean=sum(costs[v]['search_seconds']for v in ('control0','control1'))/2
            actual_algorithm_seconds=costs['base']['algorithm_processing_seconds']+costs['base']['save_seconds']
            result=dict(case=c['key'],base_rounds=main_count,base_owner_rounds=len(base['branches']),coverage=len(base['branches'])==main_count*expected_owner,
                        old_result_match=old_result_match,old_C_count=len(pending_old),old_C_max_error=max(pending_old,default=0),
                        base_allocation=base['allocation'],comparisons=comparisons,costs=costs,
                        base_processing_vs_control_ratio=actual_algorithm_seconds/controls_mean,
                        base_calls_vs_control_ratio=(costs['base']['main_api_calls']+costs['base']['branch_api_calls'])/costs['control0']['main_api_calls'],
                        exact_repeat=exact_repeat,base_numerical={k:{x:y for x,y in v.items()if x!='scenarios'}for k,v in base['numerical'].items()},
                        all_primary_numerical_passed=all(d['numerical']['arithmetic']['primary_passed']for d in observed.values()),
                        all_exact_reply_acceptance_preserved=all(d['branch_verification']['exact_reply_acceptance_flips']==0 for d in observed.values()),
                        extended_rounds=len(observed['budget60']['run']['iterations']),
                        prefix_tail={k:v for k,v in base['prefixes'].items()if k!='rows'})
            if not result['coverage']or not old_result_match or result['old_C_max_error']>1e-8 or not exact_repeat:raise ValueError('pipeline_regression')
            save(folder/'team_summary.json',result);summaries.append(result)
        if total_rounds>250:raise RuntimeError('round_budget')
        if prior_hashes()!=old:raise ValueError('old_file_changed')
        for rel,h in lock.items():
            if sha(WORK/rel)!=h or sha(root/'source_snapshot'/rel)!=h:raise ValueError('source_drift')
        for p,h in model_hashes.items():
            if sha(p)!=h:raise ValueError('model_drift')
        size=sum(p.stat().st_size for p in root.rglob('*')if p.is_file())
        if size>250000000:raise RuntimeError('size_budget')
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        save(root/'summary.json',dict(complete=True,teams=summaries,total_new_model_api_calls=total_calls,total_main_rounds=total_rounds,
            new_main_searches=len(cases)*len(VARIANTS),observed_searches=len(cases)*5,seconds=time.perf_counter()-started,bytes_before_audit=size,
            process_lifetime_peak_rss_raw=rss,peak_rss_unit='bytes'if platform.system()=='Darwin'else 'KiB',
            trained_models_reused=True,new_training=0,pooled_calls=0,truth_loaded=False,actual_payments=0,new_formal_confirmation=False,
            fixed_budget_share_candidate=True,economic_external_validity_established=False,low_cost_established=False))
        print('COMPLETE',total_calls,'calls',total_rounds,'main rounds',flush=True)
    except Exception as exc:
        save(root/'failure.json',dict(complete=False,error=repr(exc),model_api_calls_completed_runs=total_calls,main_rounds_completed_runs=total_rounds));raise


if __name__=='__main__':main()
