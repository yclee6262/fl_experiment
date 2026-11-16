"""Independent rational endpoint scorer and 1D polynomial branch interpreter."""
from fractions import Fraction as F
import math

def poly(p,x):
    z=F(str(x[0]));return F(p['offset'])+F(p['linear'][0])*z+F(p['quadratic'][0][0])*z*z

def branch(c,ids):
    assert list(map(F,c['start']))==[F(0),F(0)]
    assert [[F(z) for z in row] for row in c['anchors']]==[[F(1),F(0)],[F(-1,4),F(0)]]
    assert list(map(F,c['anchor_weights']))==[F(1,2),F(1,2)]
    assert all(F(p['linear'][1])==0 and all(F(p['quadratic'][i][j])==0 for i,j in [(0,1),(1,0),(1,1)]) for p in c['models'])
    w=[F(c['weights'][i]) for i in ids];den=sum(w)
    L=sum(v*F(c['models'][i]['linear'][0]) for i,v in zip(ids,w))/den
    Q=sum(v*F(c['models'][i]['quadratic'][0][0]) for i,v in zip(ids,w))/den
    g=L+F(3,8)*Q;x=F(0);eta=F(c['eta']);tr=[]
    if abs(g)>F(1,10**8):
        for _ in range(10):
            z=-eta if g>0 else eta;loss=sum(v*poly(c['models'][i],[z,0]) for i,v in zip(ids,w))/den
            base=sum(v*poly(c['models'][i],[0,0]) for i,v in zip(ids,w))/den;ok=loss<base
            tr.append(dict(x=[float(z),0.],loss=float(loss),accepted=ok))
            if ok:x=z;break
            eta/=2
    return dict(x=[float(x),0.],g=[float(g),0.],trials=tr)

def verify_branch(c,d):
    errors=[]
    for arm,ids in [('full',list(range(len(c['models'])))),('minus',[i for i in range(len(c['models'])) if i not in c['target']])]:
        ref=branch(c,ids);got=d[arm]
        for a,b in zip(ref['x']+ref['g'],got['post']['S_current']+got['g']):errors.append(abs(a-b))
        assert len(ref['trials'])==len(got['trials'])
        for a,b in zip(ref['trials'],got['trials']):
            assert a['accepted']==b['accepted'];errors.extend([abs(a['loss']-b['loss']),max(abs(x-y) for x,y in zip(a['x'],b['x']))])
    assert max(errors,default=0)<1e-8
    return max(errors,default=0)

def score(m,slots):
    """No production scoring, validation, quantile, or settlement functions used."""
    scales={}
    for a in m['models']:
        vals=sorted(F(str(z)) for z in a['probe'])
        def q(p):
            h=(len(vals)-1)*p;j=int(h);return vals[j]+(vals[min(j+1,len(vals)-1)]-vals[j])*(h-j)
        scales[a['id']]=q(F(9,10))-q(F(1,10))
    out={k:{o:F(0) for o in m['owners']} for k in ['peer','raw','full_normalized','min']}
    for s in slots:
        delta={a['id']:F(str(s['minus']['losses'][a['id']]))-F(str(s['full']['losses'][a['id']])) for a in m['models']};active=[a for a in m['models'] if a['v']>0]
        def avg(rows,norm):return sum(F(str(a['v']))*delta[a['id']]/(scales[a['id']] if norm else 1) for a in rows)/sum(F(str(a['v'])) for a in rows)
        o=s['owner'];out['peer'][o]+=avg([a for a in active if a['owner']!=o],True);out['raw'][o]+=avg(active,False);out['full_normalized'][o]+=avg(active,True);out['min'][o]+=min(avg([a for a in active if a['owner']==who],True) for who in m['owners'])
    result={}
    for k,ss in out.items():
        w={o:max(F(0),v) if v>max(F(1,10**8),len(m['rounds'])*F(1,10**10)) else F(0) for o,v in ss.items()};total=sum(w.values());amounts=dict.fromkeys(ss,0);q=None
        if total:
            raw={o:m['budget']*v/total for o,v in w.items()};amounts={o:int(z) for o,z in raw.items()}
            for o in sorted(ss,key=lambda o:(-(raw[o]-amounts[o]),o))[:m['budget']-sum(amounts.values())]:amounts[o]+=1
            q={o:float(v/total) for o,v in w.items()}
        result[k]=dict(scores={o:float(v) for o,v in ss.items()},q=q,amounts=amounts,status='allocated' if total else 'reserved')
    return result
