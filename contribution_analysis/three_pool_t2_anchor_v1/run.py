"""Freeze 48 analytic fixtures and execute T2, no payment or model training."""
import argparse
import datetime
from fractions import Fraction
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from contribution_analysis.custom_realized_gain_v1.schema import save,read,sha,plain,lines
from contribution_analysis.custom_realized_gain_v1.runtime import manifest_for,equivalent
from contribution_analysis.custom_realized_gain_v1.observer import record_custom_run
from contribution_analysis.custom_realized_gain_v1.verify import verify_run
from contribution_analysis.custom_realized_gain_v1.verify_balance import verify_balance
from contribution_analysis.a_e_b_validation.run import server,score
from .fixtures import cases
from .core import diagnose
from .reference import verify

WORK=Path(__file__).resolve().parents[2]
OLD=WORK/'contribution_analysis/a_e_b_validation_v1_20261027'
PILOT=WORK/'contribution_analysis/custom_realized_gain_pilot_v1_20261027'
DESIGN=WORK/'docs/three_pool_t2_anchor_design_v1_20261028.md'


def historical():
    rows=[];inputs={}
    for root in [OLD,PILOT]:
        receipt=root/'completion_receipt.json';inputs[str(receipt)]=sha(receipt)
        for name,h in read(receipt)['file_sha256'].items():
            if sha(root/name)!=h:raise ValueError('historical_receipt_mismatch:'+name)
    for path in sorted(OLD.glob('*_branches.json')):
        doc=read(path);inputs[str(path)]=sha(path);case=doc['case'];m=doc['manifest'];owners=list(dict.fromkeys(m['anchor_owners']))
        diagnostics=lines(PILOT/case/'diagnostics.jsonl')
        for row in doc['rows']:
            if row['status']!='available':raise ValueError('historical_unsupported_status')
            i=owners.index(row['owner']);A=diagnostics[row['t']]['A_increment_owner'][i]
            rows.append(dict(case=case,t=row['t'],owner=row['owner'],p=row['p'],B=row['B'],A=A,C=row['round_D'],
                             full_accepted=row['full_round']['accepted'],without_accepted=row['without_round']['accepted'],
                             trial_difference=len(row['without_round']['trials'])-len(row['full_round']['trials']),
                             old_data_exploration=True,new_model_calls=0,source_path=str(path)))
    return rows,inputs


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=WORK/'contribution_analysis/three_pool_t2_anchor_experiment_v1_20261028');args=ap.parse_args()
    root=args.output.resolve();root.mkdir(exist_ok=False);(root/'source_snapshot').mkdir();(root/'cases').mkdir();started=time.perf_counter();calls=solves=branch_calls=0
    sources=[DESIGN,WORK/'docs/three_pool_contribution_validation_plan_v1_20261028.md',WORK/'host_server.py',
             *Path(__file__).parent.glob('*.py'),*(WORK/'contribution_analysis/custom_realized_gain_v1').glob('*.py'),WORK/'contribution_analysis/a_e_b_validation/run.py']
    lock={}
    for p in sources:
        rel=p.relative_to(WORK);dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);lock[str(rel)]=sha(p)
    save(root/'source_lock.json',lock);fixed=cases();save(root/'fixtures.json',fixed)
    save(root/'registration.json',dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),design_sha256=sha(DESIGN),fixtures_sha256=sha(root/'fixtures.json'),
        case_count=48,development=24,heldout=24,heldout_kind='analytic_mechanism_not_real_task',death_zone=1e-8,sensitivity_zone=1e-6,
        max_new_synthetic_calls=10000,max_seconds=60,max_bytes=50000000,original_algorithm_unchanged=True,actual_payments=0,pooled_calls=0))
    try:
        old,inputs=historical();save(root/'historical_inputs.json',inputs);save(root/'historical_reanalysis.json',old)
        from .audit import verify_historical
        prior=verify_historical(old);save(root/'historical_verification.json',prior)
        test=subprocess.run([sys.executable,'-m','unittest','contribution_analysis.three_pool_t2_anchor_v1.test_core','-v'],cwd=WORK,text=True,capture_output=True)
        save(root/'technical_tests.json',dict(returncode=test.returncode,stdout=test.stdout,stderr=test.stderr))
        if test.returncode:raise ValueError('technical_tests_failed')
        summaries=[]
        for split in ['development','heldout']:
            for c in [c for c in fixed if c['split']==split]:
                l=np.array([float(Fraction(z)) for z in c['linear']]);Q=np.array([[float(Fraction(z)) for z in r] for r in c['quadratic']]);offset=float(Fraction(c['offset']))
                def fn(z):return float(offset+np.dot(l,z)+z@Q@z)
                anchors=[[float(Fraction(z)) for z in row] for row in c['anchors']];start=[float(Fraction(z)) for z in c['start']]
                s=server([fn],anchors);m=manifest_for(s,start,dict(num_iterations=1,use_annealing=True,allow_tangent=True))
                control=record_custom_run(s,start,m,False);run=record_custom_run(s,start,m,True);solves+=2
                calls+=control['costs']['api_calls']+run['costs']['api_calls'];equivalent(control,run)
                a,b=score(m,run);baseverify=dict(account=verify_run(m,plain(run),plain(a)),balance=verify_balance(m,plain(run),plain(b)))
                queries=[]
                def objective(z):
                    nonlocal calls,branch_calls
                    y=s.trusted_agents[0].api_predict(z);value=abs(float(y[0]));calls+=1;branch_calls+=1
                    queries.append(dict(x=np.asarray(z),reply=float(y[0]),loss=value));return value
                t=time.perf_counter();result=diagnose(s,objective,m,run,a,'owner_'+str(c['target_owner']));seconds=time.perf_counter()-t
                independent=verify(c,plain(result))
                save(root/'cases'/(c['id']+'.json'),dict(fixture=c,manifest=m,control=control,run=run,accounts=a,balance=b,
                     baseline_verification=baseverify,result=result,branch_queries=queries,independent=independent,branch_seconds=seconds))
                summaries.append(dict(id=c['id'],category=c['category'],split=split,C=result['C'],A=result['A'],B=result['B'],p=result['p'],
                    full_accepted=result['full']['accepted'],without_accepted=result['without']['accepted'],trial_difference=result['trial_difference'],
                    full_zero=result['full']['q'] is None,without_zero=result['without']['q'] is None,verification=independent,
                    original_search_calls=control['costs']['api_calls']+run['costs']['api_calls'],
                    full_branch_calls=len(result['full']['trials']),ablation_branch_calls=len(result['without']['trials'])))
                if calls>10000 or time.perf_counter()-started>60:raise RuntimeError('budget_exceeded')
            print(split,'PASSED',sum(c['split']==split for c in summaries),flush=True)
        for rel,h in lock.items():
            if sha(WORK/rel)!=h or sha(root/'source_snapshot'/rel)!=h:raise ValueError('source_drift')
        size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        if size>50000000:raise RuntimeError('size_budget')
        save(root/'summary.json',dict(complete=True,analytic_cases=summaries,historical_owner_rounds=len(old),new_synthetic_searches=solves,
             new_synthetic_api_calls=calls,new_branch_api_calls=branch_calls,full_branch_decisions=48,ablation_branch_decisions=48,
             trained_model_calls=0,new_training=0,pooled_calls=0,truth_loaded=False,known_analytic_polynomial_used=True,actual_payments=0,
             seconds=time.perf_counter()-started,bytes_before_audit=size,local_result_reference_validated=True,low_cost_estimator_established=False,
             economic_payment_validity_established=False,new_formal_confirmation=False,notion_modified=False))
        print('COMPLETE',len(summaries),'analytic;',len(old),'historical;',calls,'new synthetic calls')
    except Exception as exc:
        save(root/'failure.json',dict(complete=False,error=repr(exc),new_synthetic_calls=calls,searches=solves));raise


if __name__=='__main__':main()
