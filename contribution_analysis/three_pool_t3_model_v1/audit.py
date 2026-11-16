"""Independent persisted evidence checks; does not import the new core adapter."""
import argparse
import copy
from decimal import localcontext
from fractions import Fraction
from pathlib import Path
import numpy as np
from contribution_analysis.custom_realized_gain_v1.schema import read,save,sha
from .reference import verify,value,dec
from .historical import reanalyze,interpreter,server,same

WORK=Path(__file__).resolve().parents[2]


def audit(root):
    for rel,h in read(root/'source_lock.json').items():
        if sha(WORK/rel)!=h or sha(root/'source_snapshot'/rel)!=h:raise ValueError('source_hash')
    for p,h in read(root/'historical_inputs.json').items():
        if sha(p)!=h:raise ValueError('historical_input_hash')
    reg=read(root/'registration.json');fixed=read(root/'fixtures.json')
    if sha(root/'fixtures.json')!=reg['fixtures_sha256']:raise ValueError('fixture_hash')
    historic,hv,_=reanalyze()
    if historic!=read(root/'historical_reanalysis.json'):raise ValueError('historical_reconstruction_changed')
    reports=[];corruptions=rawchecks=0
    for c in fixed:
        doc=read(root/'cases'/(c['id']+'.json'));v=verify(c,doc);target=set(c['target']);lock=doc['lock'];events=doc['events']
        if doc['anchor_direction_weights']!=[float(Fraction(z)) for z in c['anchor_weights']]:raise ValueError('anchor_weight_changed')
        for j,e in enumerate(events):
            if e['stage']=='minus' and e['model'] in target:raise ValueError('excluded_model_in_decision')
            if (j<lock['event_index'])==(e['stage']=='common'):raise ValueError('common_before_lock_or_decision_after_lock')
            with localcontext() as ctx:
                ctx.prec=60;point=[dec(str(z)) for z in e['x']];expected=float(value(c['models'][e['model']],point))
            if abs(expected-e['reply'])>1e-8:raise ValueError('raw_polynomial_reply')
            rawchecks+=1
        clean=lambda es:[{k:z for k,z in e.items() if k!='stage'} for e in es]
        if clean([e for e in events if e['stage']=='control'])!=clean([e for e in events if e['stage']=='full']):raise ValueError('control_sequence')
        if doc['control']!=doc['full_result']:raise ValueError('control_result')
        s=server(dict(candidates=[[float(Fraction(z)) for z in a] for a in c['anchors']],alphas=doc['anchor_direction_weights']))
        table={(e['model'],tuple(e['x'])):e['reply'] for e in events};weights=np.array(doc['model_weights'])
        def objective(ids):
            w=weights[ids]/sum(weights[ids]);return lambda x:float(np.dot(w,[abs(table[i,tuple(x)]) for i in ids]))
        allids=list(range(len(c['models'])));same(interpreter()(s,doc['full']['pre'],objective(allids)),doc['full'])
        if doc['minus'] is not None:same(interpreter()(s,doc['minus']['pre'],objective(doc['remaining_model_indices'])),doc['minus'])
        for field in ['C','own','peer']:
            bad=copy.deepcopy(doc)
            if bad['score'][field] is None:bad['score'][field]=0.
            else:bad['score'][field]+=1.
            try:verify(c,bad)
            except ValueError:corruptions+=1
            else:raise ValueError('score_corruption_not_detected')
        reports.append(dict(id=c['id'],verification=v,excluded_model_guard_passed=True,common_scoring_after_lock=True))
    summary=read(root/'summary.json')
    if summary['new_synthetic_api_calls']!=rawchecks or len(reports)!=60 or corruptions!=180:raise ValueError('coverage_or_cost')
    scale=sorted([read(root/'cases'/(c['id']+'.json')) for c in fixed if c['category']=='individual_scale'],key=lambda d:float(Fraction(d['fixture']['models'][0]['linear'][0])))
    if not all(d['lock']['full']==scale[0]['lock']['full'] and d['lock']['minus']==scale[0]['lock']['minus'] for d in scale):raise ValueError('scale_endpoint_assumption_failed')
    if not all(a['score']['C']<b['score']['C'] for a,b in zip(scale,scale[1:])):raise ValueError('scale_internal_increase_not_observed')
    if not all(d['score']['C']>0 and d['score']['D_analytic']<0 for d in scale):raise ValueError('scale_counterexample_not_observed')
    save(root/'independent_audit.json',dict(accepted=True,analytic_cases=reports,historical=hv,raw_reply_checks=rawchecks,corruption_detections=corruptions,
        scale_same_endpoints_increasing_C_with_negative_diagnostic=True,new_core_imported=False,new_model_calls=0,actual_payments=0,
        technical_gate_passed=True,payment_validity_gate_passed=False))
    print('AUDIT PASSED',len(reports),'analytic;',hv['branches'],'historical;',corruptions,'corruptions')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);audit(ap.parse_args().root.resolve())
