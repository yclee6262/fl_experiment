"""Fixed score-only numerical stresses and path prefixes."""
import copy
import itertools
from decimal import Decimal as D
from .settlement import settle
from .reference import verify_settlement


def stress(owners,rounds,events,base):
    out={}
    for label,constant in [('arithmetic',None),('stress_1e-8','1e-8'),('stress_1e-6','1e-6')]:
        rows=[];max_delta=0.;qualification_flips=0;payment_changes=0
        for signs in itertools.product((-1,1),repeat=len(owners)):
            sign=dict(zip(owners,signs));changed=[]
            for e in events:
                error=D(str(e['rho']))if constant is None else D(constant)
                changed.append(dict(e,C=str(D(str(e['C']))+sign[e['owner']]*error)))
            v=settle(owners,rounds,changed);verify_settlement(owners,rounds,changed,v)
            delta=max((abs(float(v['shares'][o])-float(base['shares'][o]))for o in owners),default=0.)if v['status']==base['status']=='settled'else None
            flips=sum((D(v['weights'][o])>0)!=(D(base['weights'][o])>0)for o in owners)
            qualification_flips+=flips;payment_changes+=v['payments']!=base['payments']
            max_delta=max(max_delta,delta if delta is not None else 1.)
            rows.append(dict(signs=sign,settlement=v,delta_share=delta,qualification_flips=flips))
        out[label]=dict(per_event_error=constant,scenarios=rows,max_delta_share=max_delta,
                        qualification_flips=qualification_flips,payment_changes=payment_changes,
                        primary_passed=(max_delta<=1e-6 and qualification_flips==0 and payment_changes==0)if constant is None else None)
    return out


def prefixes(owners,rounds,events):
    rows=[]
    for n in range(1,len(rounds)+1):
        ts=rounds[:n];es=[e for e in events if e['t']in ts];v=settle(owners,ts,es);verify_settlement(owners,ts,es,v)
        rows.append(dict(n=n,settlement=v,accepted=None,
                         near_zero_events=sum(abs(float(e['C']))<=1e-8 for e in es)))
    tail_n=max(1,len(rounds)//4);cut=len(rounds)-tail_n
    tail_events=[e for e in events if e['t']in rounds[cut:]]
    base=rows[cut-1]['settlement']if cut else None;end=rows[-1]['settlement']if rows else None
    delta=max(abs(float(end['shares'][o])-float(base['shares'][o]))for o in owners)if base and base['status']==end['status']=='settled'else None
    return dict(rows=rows,last_quarter_rounds=tail_n,last_quarter_abs_C=sum(abs(float(e['C']))for e in tail_events),
                last_quarter_near_zero=sum(abs(float(e['C']))<=1e-8 for e in tail_events),last_quarter_delta_share=delta)


def ledger_checks(owners,rounds,events,base):
    results=[]
    for name,es in [('reverse',list(reversed(events))),('replay',events+events),
                    ('missing',events[1:]),('null',[dict(events[0],C=None)]+events[1:])]:
        v=settle(owners,rounds,es);verify_settlement(owners,rounds,es,v)
        expected='held'if name in ('missing','null')else base['status']
        if v['status']!=expected or(name in ('reverse','replay')and(v['shares']!=base['shares']or v['payments']!=base['payments'])):raise ValueError('ledger_invariance')
        results.append(dict(name=name,events=es,settlement=v))
    for name,es in [('conflict',events+[dict(events[0],C=str(D(events[0]['C'])+1))]),
                    ('unknown_owner',events+[dict(events[0],owner='not_admitted')])]:
        try:settle(owners,rounds,es)
        except ValueError as exc:results.append(dict(name=name,rejected=str(exc)))
        else:raise ValueError('ledger_rejection_missing')
    return results
