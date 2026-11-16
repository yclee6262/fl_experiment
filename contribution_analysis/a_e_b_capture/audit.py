"""Independent persisted-evidence gate; no compact recorder or scorer import."""
import argparse
from pathlib import Path
import numpy as np
from contribution_analysis.custom_realized_gain_v1.schema import read,save,sha,check_close
from contribution_analysis.custom_realized_gain_v1.verify import verify_run
from contribution_analysis.custom_realized_gain_v1.verify_balance import verify_balance

WORK=Path(__file__).resolve().parents[2]


def audit(root):
    for rel,h in read(root/'source_lock.json').items():
        if sha(WORK/rel)!=h or sha(root/'source_snapshot'/rel)!=h:raise ValueError('source_hash')
    for p,h in read(root/'inputs.json').items():
        if sha(p)!=h:raise ValueError('input_hash')
    rows=[]
    for c in read(root/'cases.json'):
        out=root/c['case']['key'];m=read(out/'manifest.json');ctrl=read(out/'control.json');result=read(out/'result.json')
        first=None;v={}
        for arm in ['full','compact']:
            doc=read(out/(arm+'.json'));r=doc['run']
            if r['result']!=ctrl['result']:raise ValueError('solver_result')
            for field in ['api_calls','input_samples','unique_inputs','query_sha256']:
                if r['costs'][field]!=ctrl['costs'][field]:raise ValueError('query_equivalence')
            v[arm]=dict(account=verify_run(m,r,doc['accounts']),balance=verify_balance(m,r,doc['balance']))
            if first is not None:
                if first['accounts']!=doc['accounts'] or first['balance']!=doc['balance']:raise ValueError('score_equivalence')
                # Verify actual event metadata too, excluding performance-only fields.
                if first['run']['evaluations']!=r['evaluations'] or first['run']['iterations']!=r['iterations']:raise ValueError('event_equivalence')
                for q,qq in zip(first['run']['queries'],r['queries']):
                    if {k:z for k,z in q.items() if k!='call_seconds'}!={k:z for k,z in qq.items() if k!='call_seconds'}:raise ValueError('raw_reply_equivalence')
            first=doc
        for sample in result['samples']:
            ref=sample['arms']['control']
            for arm in ['full','compact']:
                for field in ['api_calls','input_samples','unique_inputs','query_sha256']:
                    if ref[field]!=sample['arms'][arm][field]:raise ValueError('timing_sample_behavior')
        for arm in ['control','full','compact']:
            for metric in ['search_seconds','capture_wall_seconds']:
                check_close(np.median([s['arms'][arm][metric] for s in result['samples']]),result['medians'][arm][metric],'median')
            if result['memory'][arm]['query_sha256']!=ctrl['costs']['query_sha256'] or result['memory'][arm]['result']!=ctrl['result']:raise ValueError('memory_sample_behavior')
        med=result['medians'];ratios=dict(compact_search_control=med['compact']['search_seconds']/med['control']['search_seconds'],
            compact_capture_control=med['compact']['capture_wall_seconds']/med['control']['capture_wall_seconds'],
            compact_capture_full=med['compact']['capture_wall_seconds']/med['full']['capture_wall_seconds'],
            full_search_control=med['full']['search_seconds']/med['control']['search_seconds'])
        for key,value in ratios.items():check_close(value,result['ratios'][key],'ratio')
        if result['search_target_passed']!=(ratios['compact_search_control']<=1.15) or result['capture_target_passed']!=(ratios['compact_capture_control']<=1.15):raise ValueError('target_flag')
        import gzip
        for arm in ['full','compact']:
            payload=(out/(arm+'_encoding_probe.json')).read_bytes();packed=(out/(arm+'_encoding_probe.json.gz')).read_bytes()
            if gzip.decompress(packed)!=payload or len(payload)!=result['phases'][arm]['bytes'] or len(packed)!=result['phases'][arm]['gzip_bytes']:raise ValueError('encoding_roundtrip')
        rows.append(dict(case=c['case']['key'],verification=v,raw_event_equivalence=True))
    summary=read(root/'summary.json')
    if summary['main_solves']!=26*27 or len(rows)!=26:raise ValueError('case_or_solve_count')
    if summary['main_api_calls']!=sum(c['api_calls_per_solve']*27 for c in summary['cases']):raise ValueError('call_count')
    result=dict(accepted=True,cases=rows,saved_recorded_solves=len(rows)*2,raw_event_and_score_equivalence=True,
                source_and_inputs_passed=True,new_model_predictions=0,pooled_calls=0,truth_loaded=False)
    save(root/'independent_audit.json',result);print('AUDIT PASSED',len(rows),'cases;',len(rows)*2,'recorded solves')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);audit(ap.parse_args().root.resolve())
