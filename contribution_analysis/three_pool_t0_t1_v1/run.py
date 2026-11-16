"""Freeze golden inputs, then run T0/T1; no search or real transfers."""
import argparse
import copy
import datetime
import json
from pathlib import Path
import shutil
import time
import tracemalloc
from contribution_analysis.custom_realized_gain_v1.schema import save,read,sha
from .fixtures import cases,contract,event
from .engine import settle
from .reference import verify

WORK=Path(__file__).resolve().parents[2]
DESIGN=WORK/'docs/three_pool_t0_t1_contract_v1_20261028.md'


def view(result):return dict(owners=result['owners'],basic=result['basic'],contribution_pools=result['contribution_pools'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=WORK/'contribution_analysis/three_pool_t0_t1_experiment_v1_20261028');args=ap.parse_args()
    root=args.output.resolve();root.mkdir(exist_ok=False);(root/'source_snapshot').mkdir();started=time.perf_counter()
    lock={}
    sources=[DESIGN,WORK/'docs/three_pool_contribution_validation_plan_v1_20261028.md',*Path(__file__).parent.glob('*.py')]
    for p in sources:
        rel=p.relative_to(WORK);dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);lock[str(rel)]=sha(p)
    save(root/'source_lock.json',lock)
    fixed=cases();save(root/'contract.json',contract());save(root/'golden_cases.json',fixed)
    save(root/'registration.json',dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),design_sha256=sha(DESIGN),
        golden_sha256=sha(root/'golden_cases.json'),cases=len(fixed),unit='integer_test_units_not_currency',
        invariance_variants=['reverse_order','exact_replay','unapproved_noise','owner_alias','diagnostics_changed'],
        actual_transfers=0,model_calls=0,searches=0,new_training=0,pooled_calls=0,truth_loaded=False,notion_modified=False))
    try:
        results=[];checks=[];negative=[];latency=[]
        for case in fixed:
            t=time.perf_counter()
            try:result=settle(case['contract'],case['events'])
            except ValueError as exc:
                latency.append(time.perf_counter()-t)
                if case['expected_rejection'] is None or not str(exc).startswith(case['expected_rejection']):raise
                negative.append(dict(case=case['name'],error=str(exc),expected=case['expected_rejection'],shadow_settlement_available=False));continue
            latency.append(time.perf_counter()-t)
            if case['expected_rejection'] is not None:raise AssertionError('expected_rejection_not_raised:'+case['name'])
            verify(case,result);results.append(dict(case=case['name'],result=result))
            for variant in ['reverse_order','exact_replay','unapproved_noise','owner_alias','diagnostics_changed']:
                es=copy.deepcopy(case['events'])
                if variant=='reverse_order':es.reverse()
                elif variant=='exact_replay':es+=copy.deepcopy(es)
                elif variant=='unapproved_noise':
                    es += [event('noise_unknown','not_admitted','anchor_delivery',70,dict(slot='fake',x=[0,0])),
                           event('noise_unauthorized','D','score_response',60,dict(request_id='unknown',version='frozen_D',x=[0.,0.],value=123.))]
                elif variant=='owner_alias':
                    for e in es:
                        if e['participant']=='A':e['participant']='A_alias'
                        elif e['participant']=='A_alias2':e['participant']='A'
                else:
                    for e in es:e['diagnostics']=dict(A=99999,E=-99999,B=.49,G=0)
                candidate=settle(case['contract'],es)
                if view(candidate)!=view(result):raise AssertionError('invariance_failed:'+case['name']+':'+variant)
                changed=dict(case,events=es);verify(changed,candidate)
                checks.append(dict(case=case['name'],variant=variant,passed=True))
        save(root/'results.json',results);save(root/'expected_rejections.json',negative);save(root/'invariance_checks.json',checks)
        mutations=[]
        for row in results:
            case=next(c for c in fixed if c['name']==row['case'])
            for mutation in ['amount','role_status']:
                bad=copy.deepcopy(row['result'])
                if mutation=='amount':bad['owners'][0]['basic_shadow_amount']+=1
                else:
                    role=next(iter(bad['owners'][0]['roles']));bad['owners'][0]['roles'][role]['fulfilled']=not bad['owners'][0]['roles'][role]['fulfilled']
                detected=False
                try:verify(case,bad)
                except ValueError:detected=True
                if not detected:raise AssertionError('audit_mutation_not_detected')
                mutations.append(dict(case=row['case'],mutation=mutation,detected=True))
        save(root/'mutation_checks.json',mutations)
        tracemalloc.start();probe=settle(fixed[0]['contract'],fixed[0]['events']);_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        if view(probe)!=view(results[0]['result']):raise AssertionError('memory_probe_changed')
        for rel,h in lock.items():
            if sha(WORK/rel)!=h or sha(root/'source_snapshot'/rel)!=h:raise AssertionError('source_drift')
        duration=time.perf_counter()-started;size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        if duration>60 or size>50000000:raise RuntimeError('budget_exceeded')
        save(root/'summary.json',dict(complete=True,case_count=len(fixed),successful_settlements=len(results),expected_rejections=len(negative),
             exhaustive_cases=sum(c['name'].startswith('exhaustive_') for c in fixed),invariance_checks=len(checks),audit_mutation_checks=len(mutations),
             total_engine_invocations=len(fixed)+len(checks)+1,costs=dict(seconds=duration,per_case_seconds=latency,python_peak_bytes_single_demo=peak,bytes_before_audit=size),
             actual_transfers=0,model_calls=0,searches=0,new_training=0,pooled_calls=0,truth_loaded=False,
             t0_research_contract_complete=True,t1_technical_validation_passed=True,contribution_payment_validity_established=False,
             deployment_contract_complete=False,notion_modified=False))
        print('COMPLETE',len(fixed),'cases;',len(checks),'invariance;',len(mutations),'mutation checks',flush=True)
    except Exception as exc:
        save(root/'failure.json',dict(complete=False,error=repr(exc)));raise


if __name__=='__main__':main()
