"""Independent rational settlement reference; no engine import."""
from fractions import Fraction as F


def calculate(c, events, rule):
    os=c['owners']; ts=c['rounds']; slots={}
    for e in events:
        if e['reference_id'] != c['reference_id'] or e['owner'] not in os or e['round'] not in ts:
            raise ValueError('bad_reference_input')
        k=(e['owner'],e['round']); v=None if e['C'] is None else F(str(e['C']))
        if k in slots and slots[k]!=v:raise ValueError('conflict')
        slots[k]=v
    rows={}; incomplete=not ts
    for o in os:
        vals=[slots.get((o,t)) for t in ts]; complete=bool(ts) and all(v is not None for v in vals)
        incomplete |= not complete
        rows[o]=dict(net=sum(vals,F(0)) if complete else None,
                     positive=sum((v for v in vals if v>0),F(0)) if complete else None,
                     negative=sum((v for v in vals if v<0),F(0)) if complete else None,
                     defined=sum(v is not None for v in vals),expected=len(ts),
                     coverage=F(sum(v is not None for v in vals),len(ts)) if ts else None)
    if incomplete:
        return dict(status='held',rows=rows,released=0,paid_total=0,reserved=c['budget'],payments={o:None for o in os},weights=None)
    epsilon=F(c['epsilon']); net={o:max(F(0),rows[o]['net']) if abs(rows[o]['net'])>epsilon else F(0) for o in os}
    release=c['budget'] if sum(net.values())>=F(c['min_release']) else 0
    ws=net if rule=='net' else {o:rows[o]['positive'] for o in os} if rule=='positive_rounds' else {o:F(1) for o in os}
    ps={o:0 for o in os}
    if release:
        quotas={o:F(release)*ws[o]/sum(ws.values()) for o in os}
        ps={o:quotas[o].numerator//quotas[o].denominator for o in os}
        priority=sorted(os,key=lambda o:(-(quotas[o]-ps[o]),o))
        for o in priority[:release-sum(ps.values())]:ps[o]+=1
    return dict(status='settled' if release else 'reserved',rows=rows,released=release,paid_total=sum(ps.values()),
                reserved=c['budget']-release,payments=ps,weights=ws)


def verify(c,events,rule,got):
    want=calculate(c,events,rule)
    for k in ('status','released','paid_total','reserved','payments'):
        if got[k]!=want[k]:raise ValueError('reference_mismatch:'+k)
    for o,values in want['rows'].items():
        for k,v in values.items():
            x=got['rows'][o][k]
            if v is None:
                if x is not None:raise ValueError('reference_null:'+k)
            elif x is None or abs(F(str(x))-v)>F(1,10**60):raise ValueError('reference_row:'+k)
    if want['weights'] is None:
        if got['weights'] is not None:raise ValueError('reference_weights_null')
    else:
        for o,v in want['weights'].items():
            if F(got['weights'][o])!=v:raise ValueError('reference_weight')
    missing=[dict(owner=o,round=t) for o in c['owners'] for t in c['rounds'] if not any(e['owner']==o and e['round']==t for e in events)]
    undefined=[dict(owner=o,round=t) for o in c['owners'] for t in c['rounds'] if any(e['owner']==o and e['round']==t and e['C'] is None for e in events)]
    if got['missing']!=missing or got['undefined']!=undefined:raise ValueError('reference_missing')
    if got['status']!='held' and (min(got['payments'].values())<0 or sum(got['payments'].values())+got['reserved']!=c['budget']):raise ValueError('budget')
    return True
