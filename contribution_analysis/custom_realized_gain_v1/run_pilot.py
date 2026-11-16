"""Bounded two-case pilot. Never opens evaluator artifacts or removal results."""
import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import tracemalloc
import numpy as np
from .schema import read, save, save_lines, plain, sha
from .runtime import load_server, manifest_for, equivalent
from .observer import record_custom_run
from .accounting import compute_accounts
from .verify import verify_run

WORK=Path(__file__).resolve().parents[2]
MODULE=Path(__file__).resolve().parent
DESIGN=WORK/'docs/custom_realized_gain_attribution_design_v1_20261027.md'
CASES=WORK/'contribution_analysis/combined_contribution_layer2_experiment_v1_20261021/cases.json'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=WORK/'contribution_analysis/custom_realized_gain_pilot_v1_20261027')
    args=ap.parse_args();root=args.output.resolve();root.mkdir(exist_ok=False)
    try:
        for name in ['source_snapshot','implementation']:(root/name).mkdir()
        for name in ['host_server.py','agent_client.py','models.py']:shutil.copy2(WORK/name,root/'source_snapshot'/name)
        for p in MODULE.rglob('*'):
            if p.is_file() and p.suffix in ['.py','.md']:
                dst=root/'implementation'/p.relative_to(MODULE);dst.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(p,dst)
        shutil.copy2(DESIGN,root/DESIGN.name)
        cases=read(CASES)['pilot'];selected=[next(c for c in cases if c['scenario']==name) for name in ['clean','controlled_poison']]
        selected=[{k:v for k,v in c.items() if k!='evaluator'} for c in selected]
        save(root/'cases.json',selected)
        save(root/'inputs.json',{str(p):sha(p) for p in [DESIGN,CASES,*[Path(c['models']) for c in selected]]})
        lock={str(p.relative_to(root)):sha(p) for name in ['source_snapshot','implementation'] for p in (root/name).rglob('*') if p.is_file()}
        save(root/'source_lock.json',lock)
        save(root/'registration.json',dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),user_authorization='根據設計文件去實作及執行實驗',
             method_version='custom_realized_gain_v1',old_data_exploration=True,new_confirmation=False,selected_cases=[c['key'] for c in selected],
             timing_repeats=3,technical_pairs=1,cost_target_ratio=1.15,max_samples_per_solve=20000,max_seconds_per_solve=60,max_bytes_per_case=100000000,
             sensitivity_probe=dict(type='relative_loss_component_shift',relative_scale=1e-6,signs=[-1,1],selection='all accepted steps/all model slope sources'),
             pooled_loaded=False,truth_loaded=False,new_training=False,LOO_comparison=False,design_sha256=sha(DESIGN)))
        # Run tests BEFORE loading either pilot model or starting pilot solves.
        completed=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(MODULE/'tests'),'-v'],cwd=WORK,text=True,capture_output=True)
        save(root/'technical_tests.json',dict(passed=completed.returncode==0,stdout=completed.stdout,stderr=completed.stderr,returncode=completed.returncode))
        if completed.returncode:raise RuntimeError('technical_tests_failed')
        sys.path.insert(0,str(root/'source_snapshot'))
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        summary=[];research_samples=research_calls=0
        for case in selected:
            print('PILOT',case['key'],flush=True)
            out=root/case['key'];out.mkdir();server=load_server(case)
            owners=[str(i) for i in case['ids']]
            manifest=manifest_for(server,case['fixed_start'],anchor_owners=owners,model_owners=owners)
            model_hash=sha(case['models']);manifest.update(case_id=case['key'],scenario=case['scenario'],source_hashes=lock,
                model_artifact=case['models'],model_artifact_sha256=model_hash,model_versions=[model_hash]*len(owners),
                torch_version=torch.__version__,threads=1,run_id=case['key'],platform=platform.platform())
            save(out/'manifest.json',manifest)
            first=None;timings=[];last_score=None
            for rep in range(4):
                pair={}
                for flag in ([False,True] if rep%2==0 else [True,False]):
                    if rep==0:tracemalloc.start()
                    run=record_custom_run(server,case['fixed_start'],manifest,flag)
                    if rep==0:
                        _,peak=tracemalloc.get_traced_memory();tracemalloc.stop();run['costs']['python_peak_bytes']=peak
                    research_samples+=run['costs']['input_samples'];research_calls+=run['costs']['api_calls'];pair[flag]=run
                equivalent(pair[False],pair[True])
                if first is None:
                    first=pair[True]
                    save(out/'control.json',dict(result=pair[False]['result'],costs=pair[False]['costs']))
                    t=time.perf_counter();last_score=compute_accounts(manifest,first['evaluations'],first['iterations'],first['initial_eval_id'],first['final_eval_id']);account_seconds=time.perf_counter()-t
                    verification=verify_run(manifest,plain(first),plain(last_score))
                    save(out/'verification.json',verification)
                else:
                    equivalent(first,pair[True])
                    timings.append(dict(rep=rep,control=pair[False]['costs'],record=pair[True]['costs'],exact_equivalence=True))
            serialization_started=time.perf_counter()
            save(out/'run_metadata.json',{k:v for k,v in first.items() if k not in ['evaluations','queries','iterations']})
            save_lines(out/'evaluations.jsonl',first['evaluations']);save_lines(out/'queries.jsonl',first['queries']);save_lines(out/'iterations.jsonl',first['iterations'])
            last_score['verification_status']='passed';save(out/'accounts.json',{k:v for k,v in last_score.items() if k!='steps'})
            save_lines(out/'diagnostics.jsonl',last_score['steps'])
            events_serialization_seconds=time.perf_counter()-serialization_started
            tracemalloc.start();t=time.perf_counter()
            repeat=compute_accounts(manifest,first['evaluations'],first['iterations'],first['initial_eval_id'],first['final_eval_id'])
            memory_account_seconds=time.perf_counter()-t;_,account_peak=tracemalloc.get_traced_memory();tracemalloc.stop()
            if plain(repeat['A_support_owner'])!=plain(last_score['A_support_owner']):raise ValueError('offline_repeat_changed')
            ctrl=float(np.median([r['control']['search_seconds'] for r in timings]));rec=float(np.median([r['record']['search_seconds'] for r in timings]));ratio=rec/ctrl
            costs=dict(single_search=first['costs'],timings=timings,control_median_seconds=ctrl,record_median_seconds=rec,record_control_ratio=ratio,
                       search_time_target_passed=ratio<=1.15,offline_accounting_seconds=account_seconds,
                       offline_python_peak_bytes=account_peak,offline_instrumented_memory_timing_seconds=memory_account_seconds,
                       observer_search_peak_measured=True,observer_search_python_peak_bytes=first['costs']['python_peak_bytes'],
                       events_serialization_seconds=events_serialization_seconds,native_memory_measured=False,rss_measured=False,
                       additional_predictions=0,research_solves=8,research_input_samples=first['costs']['input_samples']*8)
            t=time.perf_counter();save(out/'costs.json',costs);serialization_probe=time.perf_counter()-t
            nbytes=sum(p.stat().st_size for p in out.iterdir() if p.is_file())
            if nbytes>100000000:raise RuntimeError('output_size_cap')
            summary.append(dict(case=case['key'],scenario=case['scenario'],owners=owners,
                initial_loss=first['evaluations'][first['initial_eval_id']]['objective_value'],final_loss=first['result']['loss'],
                G=last_score['G_total'],A=last_score['A_support_owner'],share=last_score['owner_share'],E=last_score['E_model'],
                accepted=last_score['accepted_count'],iterations=len(first['iterations']),sensitive=last_score['sensitive_rounds'],
                verification=verification,costs=costs,bytes_before_supplement=nbytes,serialization_probe_seconds=serialization_probe))
        for path,h in read(root/'inputs.json').items():
            if sha(path)!=h:raise ValueError('input_drift')
        for name,h in lock.items():
            if sha(root/name)!=h:raise ValueError('source_drift')
        save(root/'summary.json',dict(complete=True,technical_accepted=True,cases=summary,research_input_samples=research_samples,
                                    research_api_calls=research_calls,research_solves=16,new_model_training=0,pooled_calls=0,truth_loaded=False,
                                    old_data_exploration=True,new_confirmation=False,economic_validity_established=False))
        print('COMPLETE',root,flush=True)
    except Exception as error:
        save(root/'failure.json',dict(error=repr(error),complete=False));raise

if __name__=='__main__':main()
