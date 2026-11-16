"""Audit saved calibration/endpoint panels using independent Fraction reference."""
import argparse
import copy
from pathlib import Path
from contribution_analysis.custom_realized_gain_v1.schema import read,save,sha,key
from .reference import verify,polynomial

WORK=Path(__file__).resolve().parents[2]


def audit(root):
    for rel,h in read(root/'source_lock.json').items():
        if sha(WORK/rel)!=h or sha(root/'source_snapshot'/rel)!=h:raise ValueError('source_hash')
    if sha(root/'fixtures.json')!=read(root/'registration.json')['fixtures_sha256']:raise ValueError('fixture_hash')
    for p,h in read(root/'historical_receipts.json').items():
        if sha(p)!=h:raise ValueError('receipt_hash')
        for rel,oldh in read(p)['file_sha256'].items():
            if sha(Path(p).parent/rel)!=oldh:raise ValueError('old_artifact_hash')
    sources=read(root/'source_cases.json');spec=read(root/'fixtures.json');checks=mutations=reply_checks=0;maxerr=0.;available=undefined=0
    for name in spec['archived_cases']:
        d=read(root/'cases'/name);path=Path(d['source_path'])
        if sha(path)!=sources[str(path)]:raise ValueError('case_hash')
        old=read(path);p=d['panel']
        if d['endpoint_lock']!=old['lock']or d['fixture']!=old['fixture']or d['raw_score']!=old['score']:raise ValueError('old_data_link')
        if p['weights']!=old['model_weights']or p['full']!=old['score']['losses_full']or p['minus']!=old['score']['losses_minus']or p['target']!=old['fixture']['target']:raise ValueError('panel_endpoint_link')
        for arm,label in [('full','full'),('minus','minus')]:
            if old['lock'][arm]is None:
                if p[label]is not None:raise ValueError('missing_endpoint')
                continue
            for j,v in enumerate(p[label]):
                es=[e for i,e in enumerate(old['events'])if i>=old['lock']['event_index']and e['stage']=='common'and e['model']==j and key(e['x'])==key(old['lock'][arm])]
                if not es or any(abs(e['reply'])!=v for e in es):raise ValueError('raw_endpoint_reply')
        for model,values in zip(d['fixture']['models'],p['probes']):
            if len(values)!=len(spec['grid']):raise ValueError('probe_coverage')
            for z,v in zip(spec['grid'],values):
                if abs(float(abs(polynomial(model,z)))-v)>1e-10:raise ValueError('polynomial_probe')
                reply_checks+=1
        v=verify(p,d['baseline']);maxerr=max(maxerr,v['max_error']);checks+=1
        if len(d['variants'])!=2*len(spec['factors']):raise ValueError('variant_coverage')
        for row in d['variants']:
            panel=copy.deepcopy(p);j=row['model'];a=row['factor'];panel['probes'][j]=[a*v for v in panel['probes'][j]];panel['full'][j]*=a
            if panel['minus']is not None:panel['minus'][j]*=a
            if panel!=row['panel']:raise ValueError('transformation')
            v=verify(panel,row['score']);maxerr=max(maxerr,v['max_error']);checks+=1
            if row['score']['status']!=d['baseline']['status']:raise ValueError('status_invariance')
            for field in ('C','own','others'):
                got=row['score'][field];base=d['baseline'][field]
                if base is None:
                    if got is not None or row['absolute_errors'][field]is not None:raise ValueError('undefined_invariance')
                else:
                    err=abs(got-base)
                    if row['absolute_errors'][field]!=err or err>1e-10+1e-9*abs(base):raise ValueError('numeric_invariance')
            if row['score']['status']=='available':available+=1
            else:undefined+=1
        control=d['inconsistent_control']
        if control:
            q=copy.deepcopy(p);j=p['target'][0];q['full'][j]*=10;q['minus'][j]*=10
            if q!=control['panel']:raise ValueError('control_panel')
            verify(q,control['score']);checks+=1
            if control['delta_C']!=control['score']['C']-d['baseline']['C']:raise ValueError('control_delta')
            if d['case_id']=='useful_information_0'and abs(control['delta_C'])<=1e-8:raise ValueError('negative_control_missing')
        for field in ('C','scale'):
            bad=copy.deepcopy(d['baseline'])
            if field=='C':bad['C']=0. if bad['C']is None else bad['C']+1
            else:bad['scales'][0]+=1
            try:verify(p,bad)
            except ValueError:mutations+=1
            else:raise ValueError('mutation_not_detected')
    out=dict(passed=True,reference_panels=checks,available_scale_variants=available,undefined_scale_variants=undefined,
             analytic_probe_replies=reply_checks,mutation_detections=mutations,max_reference_absolute_error=maxerr,new_model_calls=0,new_searches=0)
    save(root/'independent_audit.json',out);return out


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();print(audit(a.root))
