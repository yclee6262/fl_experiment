"""Independent artifact audit; no production settlement/ablation engine imports."""
import argparse
import copy
from pathlib import Path
from contribution_analysis.custom_realized_gain_v1.schema import read,save,sha
from contribution_analysis.three_pool_t2_anchor_v1.audit import verify_historical
from .reference import verify
from .study import independent

WORK=Path(__file__).resolve().parents[2]
T2=WORK/'contribution_analysis/three_pool_t2_anchor_experiment_v1_20261028'


def audit(root):
    for rel,h in read(root/'source_lock.json').items():
        if sha(WORK/rel)!=h or sha(root/'source_snapshot'/rel)!=h:raise ValueError('source_hash')
    reg=read(root/'registration.json')
    if sha(root/'fixtures.json')!=reg['fixture_sha256']:raise ValueError('fixture_hash')
    for p,h in read(root/'historical_receipts.json').items():
        if sha(p)!=h:raise ValueError('old_receipt_hash')
        for rel,oldh in read(p)['file_sha256'].items():
            if sha(Path(p).parent/rel)!=oldh:raise ValueError('old_artifact_hash')
    fixed={f['id']:f for f in read(root/'fixtures.json')};checks=mutations=0
    rows=read(root/'ledger_results.json')
    if len(rows)!=len(fixed) or len({r['id'] for r in rows})!=len(fixed):raise ValueError('case_coverage')
    for r in rows:
        f=fixed[r['id']]
        if f['error']:
            if r.get('rejected')!=f['error']:raise ValueError('reject_reason')
            continue
        for rule,v in r['results'].items():
            verify(f['contract'],f['events'],rule,v);checks+=1
            owner=f['contract']['owners'][0]
            for field in ('payments','net','status'):
                bad=copy.deepcopy(v)
                if field=='payments':bad['payments'][owner]=1 if bad['payments'][owner] is None else bad['payments'][owner]+1
                elif field=='net':bad['rows'][owner]['net']='1' if bad['rows'][owner]['net'] is None else str(float(bad['rows'][owner]['net'])+1)
                else:bad['status']='settled' if bad['status']!='settled' else 'held'
                try:verify(f['contract'],f['events'],rule,bad)
                except ValueError:mutations+=1
                else:raise ValueError('mutation_not_detected')
    invariance=read(root/'invariance.json')
    originals={r['id']:r for r in rows if 'results' in r}
    for r in invariance:
        verify(r['contract'],r['events'],r['rule'],r['result'])
        base=originals[r['id']]['results'][r['rule']]
        if r['result']['payments']!=base['payments'] or r['result']['status']!=base['status']:raise ValueError('invariance_result')
    analytic=[]
    for p in sorted((root/'analytic').glob('*.json')):
        d=read(p);source=read(T2/'cases'/p.name)
        if d['fixture']!=source['fixture'] or d['manifest']!=source['manifest']:raise ValueError('analytic_input_change')
        v=independent(d);analytic.append(v)
        for r in d['rows']:
            c=dict(owners=list(r['owners']),rounds=[0],budget=500,epsilon='0.00000001',min_release='0.001',reference_id='frozen_internal_v1')
            es=[dict(owner=o,round=0,C=str(a['C']),reference_id=c['reference_id']) for o,a in r['owners'].items()]
            verify(c,es,'net',d['shadow'][str(r['factor'])])
    history=read(root/'historical_settlement.json');oldrows=read(T2/'historical_reanalysis.json')
    oldcheck=verify_historical(oldrows)
    for h in history:
        expect={(r['owner'],r['t']):r['C'] for r in oldrows if r['case']==h['case']}
        if {(e['owner'],e['round']):float(e['C']) for e in h['events']}!=expect:raise ValueError('historical_score_change')
        for rule,v in h['results'].items():verify(h['contract'],h['events'],rule,v)
    result=dict(passed=True,ledger_reference_checks=checks,mutation_detections=mutations,invariance_checks=len(invariance),
        analytic_states=len(analytic),analytic_branch_scenarios=3*len(analytic),analytic_trial_checks=sum(v['trial_checks'] for v in analytic),
        analytic_reply_checks=sum(v['reply_checks'] for v in analytic),max_analytic_error=max(v['max_error'] for v in analytic),
        historical_reference=oldcheck,historical_settlements=len(history),rejection_checks='frozen_expected_reason_only_not_independent_engine_rerun')
    save(root/'independent_audit.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();print(audit(args.root))
