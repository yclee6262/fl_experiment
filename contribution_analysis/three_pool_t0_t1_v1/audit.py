"""Re-read frozen golden inputs and settlement outputs; no engine dependency."""
import argparse
import copy
from pathlib import Path
from contribution_analysis.custom_realized_gain_v1.schema import read,save,sha
from .reference import verify

WORK=Path(__file__).resolve().parents[2]


def audit(root):
    for rel,h in read(root/'source_lock.json').items():
        if sha(WORK/rel)!=h or sha(root/'source_snapshot'/rel)!=h:raise ValueError('source_hash_mismatch')
    registration=read(root/'registration.json');golden=read(root/'golden_cases.json')
    if sha(root/'golden_cases.json')!=registration['golden_sha256']:raise ValueError('golden_hash_mismatch')
    byname={c['name']:c for c in golden};results=read(root/'results.json');rejected=read(root/'expected_rejections.json')
    if len(byname)!=len(golden) or {r['case'] for r in results+rejected}!=set(byname) or len(results)+len(rejected)!=len(golden):raise ValueError('coverage_mismatch')
    for row in results:
        case=byname[row['case']]
        if case['expected_rejection'] is not None:raise ValueError('unexpected_settlement')
        verify(case,row['result'])
    for row in rejected:
        expected=byname[row['case']]['expected_rejection']
        if expected is None or not row['error'].startswith(expected) or row['shadow_settlement_available'] is not False:raise ValueError('rejection_mismatch')
    checks=read(root/'invariance_checks.json');mutations=read(root/'mutation_checks.json')
    variants=set(registration['invariance_variants'])
    if len(checks)!=len(results)*len(variants) or {(r['case'],r['variant']) for r in checks}!={(r['case'],v) for r in results for v in variants} or not all(r['passed'] for r in checks):raise ValueError('invariance_coverage')
    # Independently reconstruct transformations and reference outcomes; no engine replay.
    from .reference import reference
    for row in results:
        original=byname[row['case']];base=reference(original['contract'],original['events'])
        if reference(original['contract'],list(reversed(original['events'])))!=base:raise ValueError('reference_order_dependency')
        if reference(original['contract'],original['events']*2)!=base:raise ValueError('reference_duplicate_dependency')
    detected=0
    for row in results:
        for mutation in ['amount','role_status']:
            altered=copy.deepcopy(row['result'])
            if mutation=='amount':altered['owners'][0]['basic_shadow_amount']+=1
            else:
                role=next(iter(altered['owners'][0]['roles']));altered['owners'][0]['roles'][role]['fulfilled']=not altered['owners'][0]['roles'][role]['fulfilled']
            try:verify(byname[row['case']],altered)
            except ValueError:detected+=1
            else:raise ValueError('mutation_not_detected')
    if len(mutations)!=detected or not all(m['detected'] for m in mutations):raise ValueError('mutation_coverage')
    summary=read(root/'summary.json')
    if summary['case_count']!=len(golden) or summary['successful_settlements']!=len(results) or summary['expected_rejections']!=len(rejected):raise ValueError('summary_coverage')
    save(root/'independent_audit.json',dict(accepted=True,golden_cases=len(golden),verified_settlements=len(results),verified_rejections=len(rejected),
         raw_event_reference_rebuild=True,engine_imported=False,source_and_golden_hashes_passed=True,mutation_detection=detected,
         actual_transfers=0,model_calls=0,scope='synthetic trusted-event contract correctness only'))
    print('AUDIT PASSED',len(results),'settlements;',len(rejected),'rejections;',detected,'mutations')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);audit(ap.parse_args().root.resolve())
