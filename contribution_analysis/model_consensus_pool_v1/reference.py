"""Independent exact rational oracle for valid fixtures; imports no scoring code."""
from fractions import Fraction as F

def reference(manifest, slots):
    def f(x): return F(str(x))
    def scale(xs):
        a=sorted(map(f,xs))
        def q(p):
            h=(len(a)-1)*p; j=h.numerator//h.denominator
            return a[j]+(a[min(j+1,len(a)-1)]-a[j])*(h-j)
        return q(F(9,10))-q(F(1,10))
    owners=sorted(manifest['owners']); models=manifest['models']
    scales={m['id']:scale(m['probe']) for m in models}
    sums={o:F(0) for o in owners}; upper=sums.copy(); centers=sums.copy(); rows=[]
    for s in sorted(slots,key=lambda x:(x['round'],x['owner'])):
        z={}
        for o in owners:
            ms=[m for m in models if m['owner']==o and f(m['v'])>0]
            z[o]=sum((f(m['v'])*(f(s['minus']['losses'][m['id']])-f(s['full']['losses'][m['id']]))/scales[m['id']] for m in ms),F(0))/sum(f(m['v']) for m in ms)
        l=min(z.values()); u=max(z.values())
        c=sum((f(m['v'])*z[m['owner']] for m in models),F(0))/sum(f(m['v']) for m in models)
        sums[s['owner']]+=l; upper[s['owner']]+=u; centers[s['owner']]+=c
        total_v=sum(f(m['v']) for m in models)
        own_v=sum(f(m['v']) for m in models if m['owner']==s['owner'])
        own=own_v*z[s['owner']]/total_v
        others=c-own
        peer=others/(1-own_v/total_v)
        rows.append({'owner':s['owner'],'round':s['round'],'l':l,'u':u,'C':c,'z':z,'own':own,'others':others,'peer':peer})
    threshold=max(F(1,10**8),len(manifest['rounds'])*F(1,10**10))
    weights={o:v if v>threshold else F(0) for o,v in sums.items()}; total=sum(weights.values())
    if not total: return dict(status='reserved',scores=sums,upper_scores=upper,center_scores=centers,rounds=rows,q=None,amounts=dict.fromkeys(owners,0),reserve=manifest['budget'])
    q={o:w/total for o,w in weights.items()}; raw={o:manifest['budget']*v for o,v in q.items()}
    amounts={o:v.numerator//v.denominator for o,v in raw.items()}
    for o in sorted(owners,key=lambda o:(-(raw[o]-amounts[o]),o))[:manifest['budget']-sum(amounts.values())]: amounts[o]+=1
    return dict(status='allocated',scores=sums,upper_scores=upper,center_scores=centers,rounds=rows,q=q,amounts=amounts,reserve=0)

def audit(manifest, slots, result, tolerance=1e-9):
    """Check recorded C, signed sums, bounds, q and exact integer allocation."""
    expected=reference(manifest,slots); errors=[]
    def check(a,b,path):
        if isinstance(b,dict):
            if not isinstance(a,dict) or set(a)!=set(b): errors.append(path); return
            for k in b: check(a[k],b[k],path+'.'+k)
        elif isinstance(b,list):
            if not isinstance(a,list) or len(a)!=len(b): errors.append(path); return
            for j,v in enumerate(b):
                if not isinstance(a[j],dict): errors.append(path); continue
                for k in v: check(a[j].get(k),v[k],path+f'[{j}].'+k)
        elif isinstance(b,F):
            if type(a) not in (int,float) or abs(finite_fraction(a)-b)>F(str(tolerance))*max(1,abs(b)): errors.append(path)
        elif type(b) is int:
            if type(a) is not int or a!=b: errors.append(path)
        elif a!=b: errors.append(path)
    for k,v in expected.items(): check(result.get(k),v,k)
    return errors

def finite_fraction(x):
    try: return F(str(x))
    except (ValueError,OverflowError): return F(10**1000)
