"""Frozen-endpoint peer scoring. No reference or solver inputs."""
from fractions import Fraction as F
from contribution_analysis.model_consensus_pool_v1.core import validate,quantile,settle,_json_numbers

METHODS=('peer','raw','full_normalized','min')
def _peer_score(manifest,slots):
    errors=validate(manifest,slots)
    if errors:return {k:dict(status='held',reasons=errors,scores=None,q=None,amounts=None,reserve=manifest.get('budget') if isinstance(manifest,dict) and type(manifest.get('budget')) is int and manifest['budget']>=0 else None) for k in METHODS}
    ms=[m for m in manifest['models'] if m['v']>0];sc={m['id']:quantile(m['probe'],.9)-quantile(m['probe'],.1) for m in ms}
    totals={k:dict.fromkeys(sorted(manifest['owners']),F(0)) for k in METHODS};details=[]
    for s in sorted(slots,key=lambda x:(x['round'],x['owner'])):
        o=s['owner'];delta={m['id']:F(str(s['minus']['losses'][m['id']]))-F(str(s['full']['losses'][m['id']])) for m in ms}
        d={k:v/sc[k] for k,v in delta.items()};den=sum(F(str(m['v'])) for m in ms);other=[m for m in ms if m['owner']!=o]
        peer=sum(F(str(m['v']))*d[m['id']] for m in other)/sum(F(str(m['v'])) for m in other)
        ownermeans=[sum(F(str(m['v']))*d[m['id']] for m in ms if m['owner']==who)/sum(F(str(m['v'])) for m in ms if m['owner']==who) for who in manifest['owners']]
        vals=dict(peer=peer,raw=sum(F(str(m['v']))*delta[m['id']] for m in ms)/den,full_normalized=sum(F(str(m['v']))*d[m['id']] for m in ms)/den,min=min(ownermeans))
        for k,v in vals.items():totals[k][o]+=v
        details.append(dict(owner=o,round=s['round'],values=vals))
    return _json_numbers({k:dict(**settle(v,manifest['budget'],len(manifest['rounds'])),scores=v,rounds=[dict(owner=d['owner'],round=d['round'],P=d['values'][k]) for d in details],tau=max(F(1,10**8),len(manifest['rounds'])*F(1,10**10)),reasons=[]) for k,v in totals.items()})


def peer_score(manifest,slots):
    try:return _peer_score(manifest,slots)
    except (OverflowError,ZeroDivisionError,ValueError) as exc:
        b=manifest.get('budget') if isinstance(manifest,dict) else None
        return {k:dict(status='held',reasons=['numeric representation failure: '+str(exc)],scores=None,q=None,amounts=None,reserve=b if type(b) is int and b>=0 else None) for k in METHODS}
