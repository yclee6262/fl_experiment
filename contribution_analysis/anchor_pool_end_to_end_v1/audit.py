"""Offline verification from saved raw replies; no new model inference."""
import argparse
import copy
from fractions import Fraction as F
from pathlib import Path
from contribution_analysis.custom_realized_gain_v1.schema import read,save,sha
from contribution_analysis.custom_realized_gain_v1.verify import verify_run
from .reference import verify_branches,verify_settlement

WORK=Path(__file__).resolve().parents[2]


def audit(root):
    for rel,h in read(root/'source_lock.json').items():
        if sha(WORK/rel)!=h or sha(root/'source_snapshot'/rel)!=h:raise ValueError('source_hash')
    reg=read(root/'registration.json')
    if sha(root/'cases.json')!=reg['cases_sha256']:raise ValueError('case_hash')
    for p,h in read(root/'model_hashes.json').items():
        if sha(p)!=h:raise ValueError('model_hash')
    for receipt,h in read(root/'historical_receipts.json').items():
        if sha(receipt)!=h:raise ValueError('receipt_hash')
        for rel,oldh in read(receipt)['file_sha256'].items():
            if sha(Path(receipt).parent/rel)!=oldh:raise ValueError('old_artifact_hash')
    output=[];mutations=0;stress_checks=0;prefix_checks=0;ledger_checks=0
    for c in read(root/'cases.json'):
        folder=root/c['key'];base=read(folder/'base.json');mainowners=[str(x)for x in c['ids']];os=list(dict.fromkeys(base['manifest']['anchor_owners']))
        for variant in ('base','repeat','split_adjacent','split_separated','budget60'):
            d=read(folder/(variant+'.json'));m=d['manifest'];r=d['run'];ts=[it['t']for it in r['iterations']]
            scalar=verify_run(m,r,d['accounts']);branch=verify_branches(m,r,d['branches'],d['branch_trace']);output.append(dict(case=c['key'],variant=variant,main=scalar,branch=branch))
            # Verify actual input duplication, model list, ownership, and source weight conservation.
            if m['model_ids']!=base['manifest']['model_ids']or m['model_weights_raw']!=base['manifest']['model_weights_raw']:raise ValueError('duplicated_model')
            for k in range(len(c['candidates'])):
                slots=[j for j,i in enumerate(m['source_anchor_indices'])if i==k]
                if len(slots)!=(2 if variant.startswith('split_')else 1):raise ValueError('source_multiplicity')
                if any(m['anchors'][j]!=c['candidates'][k]or m['anchor_owners'][j]!=mainowners[k]for j in slots):raise ValueError('source_identity')
                a=sum((F(str(m['direction_weights_raw'][j]))for j in slots),F(0));b=F(str(base['manifest']['direction_weights_raw'][k]))
                if abs(a-b)>F(1,10**15):raise ValueError('source_weight_not_conserved')
            events=d['events'];expected={(b['owner'],b['t']):(str(b['C']),str(b['rho']))for b in d['branches']}
            if len(events)!=len(expected)or{(e['owner'],e['t']):(e['C'],e['rho'])for e in events}!=expected:raise ValueError('event_score_link')
            verify_settlement(os,ts,events,d['allocation'])
            for field in ('C','shares','payments'):
                if field=='C':
                    bad=copy.deepcopy(d['branches']);bad[0]['C']+=1
                    try:verify_branches(m,r,bad,d['branch_trace'])
                    except ValueError:mutations+=1
                    else:raise ValueError('branch_mutation_not_detected')
                else:
                    bad=copy.deepcopy(d['allocation']);bad[field][os[0]]=str(float(bad[field][os[0]]or 0)+.1)if field=='shares'else(bad[field][os[0]]or 0)+1
                    try:verify_settlement(os,ts,events,bad)
                    except ValueError:mutations+=1
                    else:raise ValueError('allocation_mutation_not_detected')
            for p in d['prefixes']['rows']:
                pts=ts[:p['n']];es=[e for e in events if e['t']in pts];verify_settlement(os,pts,es,p['settlement']);prefix_checks+=1
            for name,s in d['numerical'].items():
                constant=s['per_event_error'];maxdelta=0.;flips=payments=0
                for scenario in s['scenarios']:
                    changed=[]
                    for e in events:
                        bump=F(e['rho'])if constant is None else F(constant);value=F(e['C'])+scenario['signs'][e['owner']]*bump
                        # finite decimal here: denominator is a power of ten
                        from decimal import Decimal as D,localcontext
                        with localcontext()as ctx:ctx.prec=80;strvalue=str(D(value.numerator)/D(value.denominator))
                        changed.append(dict(e,C=strvalue))
                    v=scenario['settlement'];verify_settlement(os,ts,changed,v);stress_checks+=1
                    basealloc=d['allocation'];delta=max(abs(float(v['shares'][o])-float(basealloc['shares'][o]))for o in os)if v['status']==basealloc['status']=='settled'else None
                    f=sum((F(v['weights'][o])>0)!=(F(basealloc['weights'][o])>0)for o in os)
                    if scenario['delta_share']!=delta or scenario['qualification_flips']!=f:raise ValueError('stress_summary')
                    maxdelta=max(maxdelta,delta if delta is not None else 1.);flips+=f;payments+=v['payments']!=basealloc['payments']
                if s['max_delta_share']!=maxdelta or s['qualification_flips']!=flips or s['payment_changes']!=payments:raise ValueError('stress_extrema')
                if name=='arithmetic'and s['primary_passed']!=(maxdelta<=1e-6 and flips==0 and payments==0):raise ValueError('stress_gate')
            for l in d['ledger_checks']:
                if 'settlement'in l:verify_settlement(os,ts,l['events'],l['settlement']);ledger_checks+=1
        if base['allocation']!=read(folder/'repeat.json')['allocation']or base['events']!=read(folder/'repeat.json')['events']:raise ValueError('repeat')
    result=dict(passed=True,observed_runs=len(output),raw_run_and_branch_checks=output,
                main_iterations=sum(x['main']['iterations']for x in output),owner_rounds=sum(x['branch']['owner_rounds']for x in output),
                branch_api_calls=sum(x['branch']['branch_api_calls']for x in output),max_branch_error=max(x['branch']['max_branch_error']for x in output),
                max_saved_reply_aggregation_error=max(x['branch']['max_objective_arithmetic_error']for x in output),
                exact_reply_acceptance_flips=sum(x['branch']['exact_reply_acceptance_flips']for x in output),
                near_acceptance_margins=sum(x['branch']['near_acceptance_margins']for x in output),
                prefix_settlements=prefix_checks,stress_settlements=stress_checks,ledger_variant_checks=ledger_checks,
                mutation_detections=mutations,new_model_inference=0,actual_payments=0)
    save(root/'independent_audit.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=audit(a.root);print({k:v for k,v in r.items()if k!='raw_run_and_branch_checks'})
