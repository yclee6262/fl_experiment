"""Frozen three-arm capture cost and equivalence experiment."""
import argparse
import datetime
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import tracemalloc
import numpy as np
from contribution_analysis.a_e_b_capture.compact import record_compact_run
from contribution_analysis.a_e_b_validation.run import generated,from_spec,score
from contribution_analysis.custom_realized_gain_v1.schema import save,read,sha,plain
from contribution_analysis.custom_realized_gain_v1.runtime import manifest_for,load_server,equivalent
from contribution_analysis.custom_realized_gain_v1.observer import record_custom_run
from contribution_analysis.custom_realized_gain_v1.verify import verify_run
from contribution_analysis.custom_realized_gain_v1.verify_balance import verify_balance

WORK=Path(__file__).resolve().parents[2]
DESIGN=WORK/'docs/a_e_b_capture_efficiency_design_v1_20261027.md'
CASES=WORK/'contribution_analysis/combined_contribution_layer2_experiment_v1_20261021/cases.json'
OLD=WORK/'contribution_analysis/a_e_b_validation_v1_20261027'
ARMS=['control','full','compact']


def compare_scores(m,full,new):
    a,b=score(m,full);aa,bb=score(m,new)
    if plain(a)!=plain(aa) or plain(b)!=plain(bb):raise ValueError('score_or_balance_changed')
    return a,b


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=WORK/'contribution_analysis/a_e_b_capture_efficiency_v1_20261027');args=ap.parse_args()
    root=args.output.resolve();root.mkdir(exist_ok=False);started=time.perf_counter();calls=solves=0
    (root/'source_snapshot').mkdir();lock={}
    sources=[DESIGN,WORK/'host_server.py',WORK/'agent_client.py',WORK/'models.py',*Path(__file__).parent.glob('*.py'),
             * (WORK/'contribution_analysis/custom_realized_gain_v1').rglob('*.py'),* (WORK/'contribution_analysis/a_e_b_validation').glob('*.py')]
    sources=[p for p in sources if '__pycache__' not in str(p)]
    for p in sources:
        rel=p.relative_to(WORK);dest=root/'source_snapshot'/rel;dest.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(p,dest);lock[str(rel)]=sha(p)
    save(root/'source_lock.json',lock)
    raw=read(CASES);cases=[]
    for scenario in ['clean','controlled_poison']:
        c=next(c for c in raw['pilot'] if c['scenario']==scenario);cases.append(dict(kind='development',case={k:v for k,v in c.items() if k!='evaluator'}))
    cases += [dict(kind='historical_holdout',case={k:v for k,v in c.items() if k!='evaluator'}) for c in raw['development']]
    for j,(K,M,D,N) in enumerate([(3,3,2,10),(8,8,8,20),(16,16,16,30),(8,3,8,20)]):
        cases.append(dict(kind='synthetic_cost',case=dict(key=f'synthetic_{28101+j}',spec=generated(28101+j,K,M,D),budget=N)))
    save(root/'cases.json',cases)
    inputs={str(CASES):sha(CASES),str(OLD/'completion_receipt.json'):sha(OLD/'completion_receipt.json')}
    for c in cases:
        if 'models' in c['case']:inputs[c['case']['models']]=sha(c['case']['models'])
    save(root/'inputs.json',inputs)
    save(root/'registration.json',dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),design_sha256=sha(DESIGN),
         cases=[c['case']['key'] for c in cases],timing_repeats=7,warmup_groups=1,memory_groups=1,case_order='development_then_historical_then_synthetic',
         arm_order='cyclic control/full/compact rotation',search_ratio_target=1.15,capture_ratio_target=1.15,
         formulas_changed=False,historical_holdout_not_independent_real_data=True,new_formal_confirmation=False,pooled_calls=0,truth_loaded=False,notion_modified=False))
    try:
        for name,h in read(OLD/'completion_receipt.json')['file_sha256'].items():
            if sha(OLD/name)!=h:raise ValueError('old_receipt_drift')
        tests=[]
        for folder in ['contribution_analysis/custom_realized_gain_v1/tests','contribution_analysis/custom_realized_gain_v1/tests_balance','contribution_analysis/a_e_b_capture']:
            done=subprocess.run([sys.executable,'-m','unittest','discover','-s',folder,'-p','test*.py','-v'],cwd=WORK,capture_output=True,text=True)
            tests.append(dict(folder=folder,returncode=done.returncode,stdout=done.stdout,stderr=done.stderr))
        save(root/'tests.json',tests)
        if any(t['returncode'] for t in tests):raise ValueError('tests_failed')
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        summaries=[]
        for entry in cases:
            c=entry['case'];caseid=c['key'];out=root/caseid;out.mkdir()
            if 'spec' in c:s,owners=from_spec(c['spec']);start=c['spec']['start'];N=c['budget']
            else:s=load_server(c);owners=[str(i) for i in c['ids']];start=c['fixed_start'];N=30
            m=manifest_for(s,start,dict(num_iterations=N,use_annealing=True,allow_tangent=True),anchor_owners=owners,
                           model_owners=owners if len(owners)==len(s.trusted_agents) else None)
            if 'models' in c:
                m.update(model_artifact=c['models'],model_artifact_sha256=sha(c['models']))
                m['model_versions']=[m['model_artifact_sha256']]*len(s.trusted_agents)
            save(out/'manifest.json',m)
            def run(arm):
                nonlocal calls,solves
                t=time.perf_counter()
                r=record_compact_run(s,start,m) if arm=='compact' else record_custom_run(s,start,m,arm=='full')
                wall=time.perf_counter()-t;r['costs']['capture_wall_seconds']=wall
                calls+=r['costs']['api_calls'];solves+=1
                if calls>1500000 or time.perf_counter()-started>600:raise RuntimeError('global_budget')
                return r
            samples=[];first=None
            for rep in range(8):
                pair={}
                order=ARMS[rep%3:]+ARMS[:rep%3]
                for arm in order:pair[arm]=run(arm)
                equivalent(pair['control'],pair['full']);equivalent(pair['full'],pair['compact'])
                if first is None:
                    first=pair
                    compare_scores(m,pair['full'],pair['compact'])
                if rep:samples.append(dict(rep=rep,order=order,arms={arm:r['costs'] for arm,r in pair.items()}))
            ac,b=compare_scores(m,first['full'],first['compact'])
            verification={}
            for arm in ['full','compact']:
                a,bb=score(m,first[arm]);verification[arm]=dict(account=verify_run(m,plain(first[arm]),plain(a)),balance=verify_balance(m,plain(first[arm]),plain(bb)))
                save(out/(arm+'.json'),dict(run=first[arm],accounts=a,balance=bb))
            save(out/'control.json',first['control'])
            memory={}
            for arm in ARMS:
                tracemalloc.start();memoryrun=run(arm);_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
                equivalent(first['control'],memoryrun);memory[arm]=dict(capture_python_peak_bytes=peak,api_calls=memoryrun['costs']['api_calls'],
                    result=memoryrun['result'],query_sha256=memoryrun['costs']['query_sha256'])
            phase={}
            for arm in ['full','compact']:
                offline=[];encoding=[]
                for _ in range(7):
                    t=time.perf_counter();a,bb=score(m,first[arm]);offline.append(time.perf_counter()-t)
                    t=time.perf_counter();payload=json.dumps(plain(dict(run=first[arm],accounts=a,balance=bb)),allow_nan=False,separators=(',',':')).encode();encoding.append(time.perf_counter()-t)
                t=time.perf_counter();packed=gzip.compress(payload,compresslevel=1,mtime=0);compress=time.perf_counter()-t
                t=time.perf_counter()
                with (out/(arm+'_encoding_probe.json')).open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
                disk=time.perf_counter()-t
                phase[arm]=dict(offline_samples=offline,encoding_samples=encoding,offline_median_seconds=float(np.median(offline)),
                               encoding_median_seconds=float(np.median(encoding)),disk_write_fsync_seconds=disk,bytes=len(payload),
                               gzip_level=1,gzip_seconds=compress,gzip_bytes=len(packed),gzip_roundtrip_passed=gzip.decompress(packed)==payload)
                with (out/(arm+'_encoding_probe.json.gz')).open('xb') as f:f.write(packed)
            med={arm:{metric:float(np.median([v['arms'][arm].get(metric,0.) for v in samples])) for metric in ['search_seconds','capture_wall_seconds','materialize_seconds']} for arm in ARMS}
            ratios=dict(compact_search_control=med['compact']['search_seconds']/med['control']['search_seconds'],
                        compact_capture_control=med['compact']['capture_wall_seconds']/med['control']['capture_wall_seconds'],
                        compact_capture_full=med['compact']['capture_wall_seconds']/med['full']['capture_wall_seconds'],
                        full_search_control=med['full']['search_seconds']/med['control']['search_seconds'])
            result=dict(case=caseid,kind=entry['kind'],K=len(m['anchors']),M=len(m['model_ids']),D=len(start),rounds=len(first['compact']['iterations']),
                        api_calls_per_solve=first['compact']['costs']['api_calls'],G=ac['G_total'],share=ac['owner_share'],A=ac['A_support_owner'],E=ac['E_model'],
                        medians=med,ratios=ratios,search_target_passed=ratios['compact_search_control']<=1.15,
                        capture_target_passed=ratios['compact_capture_control']<=1.15,capture_faster_than_full=ratios['compact_capture_full']<1,
                        samples=samples,phases=phase,memory=memory,verification=verification,exact_score_equivalence=True,exact_solver_and_digest_equivalence=True)
            save(out/'result.json',result);summaries.append(result)
            size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
            if size>300000000:raise RuntimeError('size_budget')
            print(caseid,'search',round(ratios['compact_search_control'],3),'capture',round(ratios['compact_capture_control'],3),'vs_full',round(ratios['compact_capture_full'],3),flush=True)
        for rel,h in lock.items():
            if sha(WORK/rel)!=h or sha(root/'source_snapshot'/rel)!=h:raise ValueError('source_drift')
        for path,h in inputs.items():
            if sha(path)!=h:raise ValueError('input_drift')
        save(root/'summary.json',dict(complete=True,cases=summaries,main_solves=solves,main_api_calls=calls,seconds=time.perf_counter()-started,
                                    original_algorithm_and_scores_unchanged=True,new_formal_confirmation=False,economic_validity_established=False,
                                    pooled_calls=0,new_training=0,truth_loaded=False,notion_modified=False))
        print('COMPLETE',root,flush=True)
    except Exception as exc:
        save(root/'failure.json',dict(complete=False,error=repr(exc),main_solves=solves,main_api_calls=calls));raise


if __name__=='__main__':main()
