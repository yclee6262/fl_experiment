"""Conservative model-pool scoring; no solver, truth data, or payment side effects."""
import hashlib
import json
import math
from fractions import Fraction

DEADZONE = Fraction(1, 10**10)
SCALE_EPS = Fraction(1, 10**12)

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def number(x):
    try:
        return type(x) in (int, float) and math.isfinite(x) and x >= 0
    except OverflowError:
        return False

def quantile(values, p):
    a = sorted(Fraction(str(v)) for v in values)
    h = (len(a)-1)*Fraction(str(p))
    lo = int(h)
    return a[lo] + (a[min(lo+1,len(a)-1)]-a[lo])*(h-lo)

def validate(manifest, slots):
    errors = []
    if not isinstance(manifest, dict) or not isinstance(slots, list):
        return ['manifest must be object and slots must be array']
    if set(manifest) != {'schema_version','owners','models','rounds','budget','probe_id'}: errors.append('manifest fields mismatch')
    owners, models, rounds = (manifest.get(k) for k in ('owners','models','rounds'))
    def ids(x):
        return isinstance(x,list) and bool(x) and all(isinstance(i,str) and i for i in x) and len(set(x)) == len(x)
    if not ids(owners): errors.append('owners must be nonempty unique strings')
    if not ids(rounds): errors.append('rounds must be nonempty unique strings')
    if type(manifest.get('budget')) is not int or manifest['budget'] < 0: errors.append('budget must be nonnegative integer')
    if manifest.get('schema_version') != '1': errors.append('schema_version must be 1')
    if not isinstance(manifest.get('probe_id'),str) or not manifest['probe_id']: errors.append('probe_id required')
    if not isinstance(models,list) or not models: errors.append('models required')
    if errors: return errors
    seen, totals, lengths = set(), dict.fromkeys(owners,0), set()
    for m in models:
        if not isinstance(m,dict): errors.append('model must be object'); continue
        if set(m) != {'id','owner','v','version','probe'}: errors.append('model fields mismatch')
        mid = m.get('id')
        if not isinstance(mid,str) or not mid or mid in seen: errors.append('invalid or duplicate model id')
        else: seen.add(mid)
        if not isinstance(m.get('owner'),str) or m.get('owner') not in totals: errors.append('unknown model owner')
        if not number(m.get('v')): errors.append('invalid v')
        elif isinstance(m.get('owner'),str) and m.get('owner') in totals: totals[m['owner']] += Fraction(str(m['v']))
        if not isinstance(m.get('version'),str) or not m['version']: errors.append('model version required')
        p = m.get('probe')
        if not isinstance(p,list) or len(p)<2 or not all(number(x) for x in p): errors.append('invalid probe')
        else:
            lengths.add(len(p))
            if number(m.get('v')) and m['v']>0 and quantile(p,.9)-quantile(p,.1)<=SCALE_EPS: errors.append('zero or near-zero scale')
    if len(lengths)>1: errors.append('probe lengths differ')
    if any(v<=0 for v in totals.values()) or len(owners)<2: errors.append('each owner requires positive weight and at least two owners')
    if errors: return errors
    digest = canonical_hash(manifest)
    versions = {m['id']:m['version'] for m in models}
    expected = {(o,r) for o in owners for r in rounds}
    received, fulls, endpoints = set(), {}, {}
    for slot in slots:
        if not isinstance(slot,dict): errors.append('slot must be object'); continue
        if not isinstance(slot.get('owner'),str) or not isinstance(slot.get('round'),str):
            errors.append('slot owner and round must be strings'); continue
        if set(slot) != {'owner','round','manifest_hash','versions','full','minus'}: errors.append('slot fields mismatch')
        key = (slot.get('owner'),slot.get('round'))
        if key not in expected: errors.append('unknown owner or round')
        if key in received: errors.append('duplicate slot')
        received.add(key)
        if slot.get('manifest_hash') != digest: errors.append('manifest hash mismatch')
        if slot.get('versions') != versions: errors.append('version mismatch')
        valid_full = False
        for endpoint in ('full','minus'):
            e = slot.get(endpoint)
            if not isinstance(e,dict) or not isinstance(e.get('id'),str) or not e['id']:
                errors.append('undefined '+endpoint); continue
            if set(e) != {'id','losses'}: errors.append('endpoint fields mismatch')
            losses = e.get('losses')
            if not isinstance(losses,dict) or set(losses)!=seen or not all(number(x) for x in losses.values()):
                errors.append('invalid '+endpoint+' losses'); continue
            if e['id'] in endpoints and endpoints[e['id']] != losses: errors.append('inconsistent endpoint losses')
            endpoints[e['id']] = losses
            if endpoint=='full': valid_full=True
        if valid_full:
            r = slot.get('round')
            if r in fulls and fulls[r] != slot['full']: errors.append('inconsistent full endpoint or losses')
            fulls[r] = slot['full']
    if received != expected: errors.append('missing expected slots')
    return sorted(set(errors))

def settle(scores, budget, rounds):
    if type(budget) is not int or budget<0: raise ValueError('budget must be nonnegative integer')
    if type(rounds) is not int or rounds<1: raise ValueError('round count must be positive integer')
    weights = {o:s if s>max(Fraction(1,10**8),rounds*DEADZONE) else 0 for o,s in scores.items()}
    total = sum(weights.values())
    if total<=0:
        return dict(status='reserved',q=None,amounts=dict.fromkeys(scores,0),reserve=budget)
    q = {o:float(w/total) for o,w in weights.items()}
    exact = {o:Fraction(str(w)) for o,w in weights.items()}
    denom = sum(exact.values())
    raw = {o:budget*v/denom for o,v in exact.items()}
    amounts = {o:math.floor(v) for o,v in raw.items()}
    for o in sorted(scores,key=lambda o:(-(raw[o]-amounts[o]),o))[:budget-sum(amounts.values())]: amounts[o]+=1
    return dict(status='allocated',q=q,amounts=amounts,reserve=0)

def _score(manifest, slots):
    errors=validate(manifest, slots)
    if errors:
        b=manifest.get('budget') if isinstance(manifest,dict) else None
        return dict(status='held',reasons=errors,q=None,amounts=None,reserve=b if type(b) is int and b>=0 else None, scores=None, rounds=None)
    owners=sorted(manifest['owners']); models=manifest['models']
    scales={m['id']:quantile(m['probe'],.9)-quantile(m['probe'],.1) for m in models}
    totals={o:sum(Fraction(str(m['v'])) for m in models if m['owner']==o) for o in owners}
    cumulative=dict.fromkeys(owners,Fraction(0)); upper=dict(cumulative); center=dict(cumulative); details=[]
    for s in sorted(slots,key=lambda s:(s['round'],s['owner'])):
        d={m['id']:(Fraction(str(s['minus']['losses'][m['id']]))-Fraction(str(s['full']['losses'][m['id']])))/scales[m['id']] for m in models if m['v']>0}
        z={o:sum(Fraction(str(m['v']))*d[m['id']] for m in models if m['owner']==o and m['v']>0)/totals[o] for o in owners}
        if not all(math.isfinite(v) for v in z.values()): raise OverflowError('nonfinite normalized score')
        l,u=min(z.values()),max(z.values()); i=s['owner']
        c=sum(totals[o]*z[o] for o in owners)/sum(totals.values())
        others=sum(totals[o]*z[o] for o in owners if o!=i)/sum(totals.values())
        own=totals[i]*z[i]/sum(totals.values())
        peer=others/(sum(totals[o] for o in owners if o!=i)/sum(totals.values()))
        category='positive' if l>DEADZONE else 'negative' if u < -DEADZONE else 'near_zero' if l>=-DEADZONE and u<=DEADZONE else 'uncertain'
        details.append(dict(owner=i,round=s['round'],z=z,l=l,u=u,C=c,own=own,others=others,peer=peer,classification=category))
        cumulative[i]+=l; upper[i]+=u; center[i]+=c
    return dict(**settle(cumulative,manifest['budget'],len(manifest['rounds'])), reasons=[], scores=cumulative, upper_scores=upper, center_scores=center, scales=scales, rounds=details)


def score(manifest, slots):
    try:
        return _json_numbers(_score(manifest, slots))
    except (OverflowError, ZeroDivisionError, ValueError) as exc:
        b=manifest.get('budget') if isinstance(manifest,dict) else None
        return dict(status='held',reasons=['numeric representation failure: '+str(exc)],q=None,amounts=None,reserve=b if type(b) is int and b>=0 else None,scores=None,rounds=None)


def _json_numbers(value):
    if isinstance(value,Fraction): return float(value)
    if isinstance(value,dict): return {k:_json_numbers(v) for k,v in value.items()}
    if isinstance(value,list): return [_json_numbers(v) for v in value]
    return value
