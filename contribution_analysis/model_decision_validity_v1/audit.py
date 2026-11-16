"""Independent Decimal solver / Fraction score / raw reply audit and mutations."""
import argparse
import copy
import datetime
from fractions import Fraction as F
from pathlib import Path
from contribution_analysis.custom_realized_gain_v1.schema import read,save,sha,key
from contribution_analysis.three_pool_t3_model_v1.reference import verify as branch_verify
from contribution_analysis.model_scale_invariance_v1.reference import verify as score_verify,polynomial

WORK=Path(__file__).resolve().parents[2]


def close(a,b,tol=1e-10):
    if abs(float(a)-float(b))>tol:raise ValueError('rational_raw_value')


def diagnostic(doc):
    c=doc['fixture'];d=doc['decision'];want=polynomial(c['reference'],d['lock']['minus'])-polynomial(c['reference'],d['lock']['full'])
    close(want,doc['D']);close(want,d['score']['D_analytic'])


def audit(root):
    for rel,h in read(root/'source_lock.json').items():
        if sha(WORK/rel)!=h or sha(root/'source_snapshot'/rel)!=h:raise ValueError('source_hash')
    for p,h in read(root/'historical_receipts.json').items():
        if sha(p)!=h:raise ValueError('old_receipt')
        for rel,v in read(p)['file_sha256'].items():
            if sha(Path(p).parent/rel)!=v:raise ValueError('old_artifact')
    spec=read(root/'fixtures.json');reg=read(root/'registration.json');cal=read(root/'calibration_lock.json');summary=read(root/'summary.json')
    if reg['fixtures_sha256']!=sha(root/'fixtures.json')or cal['fixtures_sha256']!=sha(root/'fixtures.json'):raise ValueError('fixture_hash')
    if cal['solver_calls_so_far']!=0 or read(root/'technical_tests.json')['returncode']:raise ValueError('precondition')
    docs={};mutations=panels=replies=probes=0;maxerr=0.;stages=dict(control=0,full=0,minus=0,common=0)
    sgn=lambda x:None if x is None else 1 if x>1e-10 else -1 if x < -1e-10 else 0
    for c in spec['cases']:
        doc=read(root/'cases'/(c['id']+'.json'));docs[c['id']]=doc;d=doc['decision'];p=doc['panel'];v=doc['normalized']
        if c!=doc['fixture']or c!=d['fixture']:raise ValueError('fixture_link')
        if not cal['locked_at']<=doc['started_at']<=doc['ended_at']:raise ValueError('probe_after_search')
        if sha(root/'calibration_lock.json')!=doc['calibration_sha256']:raise ValueError('probe_drift')
        check=branch_verify(c,d);maxerr=max(maxerr,check['max_absolute_error'])
        for stage in stages:stages[stage]+=sum(e['stage']==stage for e in d['events'])
        strip=lambda stage:[{k:v for k,v in e.items()if k!='stage'}for e in d['events']if e['stage']==stage]
        if strip('control')!=strip('full')or d['control']!=d['full_result']:raise ValueError('control_mismatch')
        for i,e in enumerate(d['events']):
            close(abs(polynomial(c['models'][e['model']],e['x'])),abs(e['reply']));replies+=1
            if e['stage']=='minus'and e['model']in c['target']:raise ValueError('removed_model_called')
            if e['stage']=='common'and i<d['lock']['event_index']:raise ValueError('unlocked_common_score')
        for arm,label in [('full','losses_full'),('minus','losses_minus')]:
            if p[arm]!=d['score'][label]:raise ValueError('endpoint_loss_link')
            for j,val in enumerate(p[arm]):
                es=[e for i,e in enumerate(d['events'])if i>=d['lock']['event_index']and e['stage']=='common'and e['model']==j and key(e['x'])==key(d['lock'][arm])]
                if not es or any(abs(e['reply'])!=val for e in es):raise ValueError('common_reply_link')
        if p['probes']!=cal['probes'][c['id']]or p['weights']!=d['model_weights']or p['target']!=c['target']:raise ValueError('panel_link')
        for model,vs in zip(c['models'],p['probes']):
            if len(vs)!=25:raise ValueError('probe_count')
            for z,val in zip(spec['grid'],vs):close(abs(polynomial(model,z)),val);probes+=1
        check=score_verify(p,v);panels+=1;maxerr=max(maxerr,check['max_error']);diagnostic(doc)
        if len(doc['variants'])!=6 or {(r['model'],r['factor'])for r in doc['variants']}!={(j,a)for j in range(2)for a in spec['factors']}:raise ValueError('variant_coverage')
        for row in doc['variants']:
            q=copy.deepcopy(p);j=row['model'];factor=row['factor'];q['probes'][j]=[factor*a for a in q['probes'][j]];q['full'][j]*=factor;q['minus'][j]*=factor
            if q!=row['panel']:raise ValueError('variant_transformation')
            check=score_verify(q,row['score']);panels+=1;maxerr=max(maxerr,check['max_error'])
            for k in ['C','own','others']:
                err=abs(row['score'][k]-v[k])
                if row['errors'][k]!=err or err>1e-10+1e-9*abs(v[k]):raise ValueError('scale_invariance')
        # Independent checkers must reject four distinct corruptions per setting.
        for kind in ['C','scale','endpoint','D']:
            bad=copy.deepcopy(doc)
            if kind=='C':bad['normalized']['C']+=1
            elif kind=='scale':bad['normalized']['scales'][0]+=1
            elif kind=='endpoint':bad['decision']['full']['post']['S_current'][0]+=.01
            else:bad['D']=-bad['D']if abs(bad['D'])>1e-10 else 1
            try:
                if kind in ['C','scale']:score_verify(bad['panel'],bad['normalized'])
                elif kind=='endpoint':branch_verify(c,bad['decision'])
                else:diagnostic(bad)
            except ValueError:mutations+=1
            else:raise ValueError('mutation_not_detected:'+kind)
        row=next(r for r in summary['rows']if r['id']==c['id'])
        for field,want in [('raw_C',d['score']['C']),('C',v['C']),('own',v['own']),('others',v['others']),('D',doc['D'])]:
            if row[field]!=want:raise ValueError('summary_link')
        den=sum(d['model_weights'][i]for i in d['remaining_model_indices'])
        if row['peer']!=v['others']/den:raise ValueError('peer_link')
    # Count confusion / validity without importing production analysis.
    for split,groups in summary['analysis']['groups'].items():
        for cat,methods in groups.items():
            rr=[r for r in summary['rows']if(split=='all'or r['split']==split)and(cat=='all'or r['category']==cat)]
            for field,got in methods.items():
                usable=[r for r in rr if r[field]is not None and r['D']is not None]
                conf={f'{a},{b}':sum(sgn(r['D'])==a and sgn(r[field])==b for r in usable)for a in [-1,0,1]for b in [-1,0,1]}
                n=sum(sgn(r['D'])==sgn(r[field])for r in usable)
                if got['matches']!=n or got['available']!=len(usable)or got['total']!=len(rr)or got['confusion_D_then_score']!=conf:raise ValueError('summary_statistics')
                if got['undefined']!=len(rr)-len(usable)or got['agreement']!=(None if not usable else n/len(usable)):raise ValueError('summary_denominator')
    passed=all(r['C']is not None and sgn(r['C'])==sgn(r['D'])for r in summary['rows'])
    if passed!=summary['analysis']['quality_gate_passed']:raise ValueError('quality_gate')
    pairs=read(root/'matched_pairs.json')
    if len(pairs)!=12:raise ValueError('pair_count')
    for p in pairs:
        a,b=[docs[i]for i in p['ids']]
        for field in ['models','weights','target','anchors','anchor_weights','start','eta']:
            if a['fixture'][field]!=b['fixture'][field]:raise ValueError('pair_inputs')
        if a['decision']['events']!=b['decision']['events']or a['decision']['lock']!=b['decision']['lock']or a['panel']!=b['panel']or a['normalized']!=b['normalized']:raise ValueError('pair_observables')
        if p['C']!=a['normalized']['C']or p['D']!=[a['D'],b['D']]or sgn(a['D'])!=-sgn(b['D']):raise ValueError('pair_diagnostics')
    execution=read(root/'execution_order.json');dev=read(root/'development_checkpoint.json');held=read(root/'heldout_parameter_checkpoint.json')
    if [r['id']for r in execution]!=[c['id']for split in ['development','heldout_parameter']for c in spec['cases']if c['split']==split]:raise ValueError('execution_order')
    heldstart=min(d['started_at']for d in docs.values()if d['fixture']['split']=='heldout_parameter')
    if dev['completed_at']>heldstart:raise ValueError('heldout_before_development_locked')
    for cp in [dev,held]:
        if cp['rows']!=[r for r in summary['rows']if r['split']==cp['split']]:raise ValueError('checkpoint_rows')
    if stages!=summary['stage_api']or sum(stages.values())!=summary['synthetic_search_and_common_api']:raise ValueError('api_cost')
    result=dict(passed=True,branch_decisions=60,control_replay_comparisons=30,score_panels=panels,raw_reply_checks=replies,
                probe_reply_checks=probes,matched_pairs=12,mutations_detected=mutations,max_reference_error=maxerr,
                quality_gate_passed=passed,new_model_calls=0,independent_of_production_score=True)
    save(root/'independent_audit.json',result);return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);print(audit(ap.parse_args().root.resolve()))
