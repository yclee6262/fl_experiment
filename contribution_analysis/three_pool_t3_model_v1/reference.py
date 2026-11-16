"""Independent 60-digit Decimal objective and one-round interpreter."""
from decimal import Decimal as D,localcontext
from fractions import Fraction
import math


def dec(z):
    r=Fraction(str(z));return D(r.numerator)/D(r.denominator)


def value(p,x):return dec(p['offset'])+sum(dec(a)*b for a,b in zip(p['linear'],x))+sum(x[i]*dec(p['quadratic'][i][j])*x[j] for i in range(2) for j in range(2))


def oracle(c):
    with localcontext() as ctx:
        ctx.prec=60;v=[dec(z) for z in c['weights']];total=sum(v);v=[z/total for z in v]
        x=[dec(z) for z in c['start']];anchors=[[dec(z) for z in row] for row in c['anchors']];a=[dec(z) for z in c['anchor_weights']];a=[z/sum(a) for z in a]
        allids=list(range(len(v)));ids=[i for i in allids if i not in c['target']]
        def objective(panel):
            den=sum(v[i] for i in panel)
            return lambda z:sum(v[i]*abs(value(c['models'][i],z)) for i in panel)/den
        def solve(panel):
            f=objective(panel);before=f(x);g=[D(0),D(0)]
            for ak,anchor in zip(a,anchors):
                delta=[z-y for z,y in zip(anchor,x)];dist=sum(z*z for z in delta).sqrt()
                if dist<D('1e-8'):continue
                for j in range(2):g[j]+=ak*(f(anchor)-before)*delta[j]/(dist*dist)
            norm=sum(z*z for z in g).sqrt();eta=dec(c['eta']);trials=[];mode='secant';halted=False;end=x;loss=before
            if norm<=D('1e-8'):halted=True
            else:
                q=[z/norm for z in g]
                for j in range(10):
                    step=eta
                    for z,d in zip(x,[-z for z in q]):
                        if d>0:step=min(step,(D(1)-z)/d)
                        elif d<0:step=min(step,(D(-1)-z)/d)
                    point=[z-max(D(0),step)*d for z,d in zip(x,q)];trialloss=f(point);ok=trialloss<before
                    trials.append(dict(x=point,loss=trialloss,accepted=ok))
                    if ok:end=point;loss=trialloss;eta=min(D('.5'),eta*D('1.5'));break
                    eta/=2
                if not any(t['accepted'] for t in trials):mode='dynamic';eta=D('.5')
            return dict(x=end,loss=loss,g=g,eta=eta,mode=mode,halted=halted,trials=trials)
        full=solve(allids);available=sum(v[i] for i in ids)>0;minus=solve(ids) if available else None
        full_loss=[abs(value(p,full['x'])) for p in c['models']];initial=[abs(value(p,x)) for p in c['models']]
        E=[vv*(before-after) for vv,before,after in zip(v,initial,full_loss)];C=own=others=peer=diagnostic=None
        if available:
            mloss=[abs(value(p,minus['x'])) for p in c['models']];delta=[aa-bb for aa,bb in zip(mloss,full_loss)]
            C=sum(w*z for w,z in zip(v,delta));own=sum(v[i]*delta[i] for i in c['target']);others=sum(v[i]*delta[i] for i in ids);peer=others/sum(v[i] for i in ids)
            diagnostic=value(c['reference'],minus['x'])-value(c['reference'],full['x'])
        sign=lambda z:None if z is None else 1 if z>D('1e-8') else -1 if z<D('-1e-8') else 0
        if sign(C)!=c['expected_C_sign'] or sign(diagnostic)!=c['expected_D_sign']:raise ValueError('predeclared_fixture_sign_mismatch:'+c['id'])
        grid=[[D(-1)/2+D(j)/20,D(0)] for j in range(21)]
        rmse=[(sum((value(p,z)-value(c['reference'],z))**2 for z in grid)/21).sqrt() for p in c['models']]
        return dict(full=full,minus=minus,C=C,own=own,others=others,peer=peer,D_analytic=diagnostic,E_model=E,E_target=sum(E[i] for i in c['target']),RMSE=rmse)


def verify(c,doc):
    expected=oracle(c);errors=[]
    def close(a,b,tol=1e-8):
        if a is None or b is None:
            if a is not b:raise ValueError('undefined_mismatch')
        elif isinstance(a,list):
            if len(a)!=len(b):raise ValueError('shape_mismatch')
            for aa,bb in zip(a,b):close(aa,bb,tol)
        else:
            err=abs(float(a)-float(b));errors.append(err)
            if not math.isfinite(err) or err>tol:raise ValueError('decimal_numeric_mismatch')
    for arm in ['full','minus']:
        ref=expected[arm];got=doc[arm]
        if ref is None:
            if got is not None:raise ValueError('undefined_branch_exists')
            continue
        close(ref['x'],got['post']['S_current'],1e-10);close(ref['g'],got['g'],1e-10)
        close(ref['loss'],got['post']['best_loss']);close(ref['eta'],got['post']['eta'])
        if ref['mode']!=got['post']['current_method'] or ref['halted']!=got['post']['halted'] or len(ref['trials'])!=len(got['trials']):raise ValueError('transition_mismatch')
        for a,b in zip(ref['trials'],got['trials']):
            close(a['x'],b['x'],1e-10);close(a['loss'],b['loss'])
            if a['accepted']!=b['accepted']:raise ValueError('acceptance_mismatch')
    for key in ['C','own','others','peer','D_analytic','E_model','E_target','RMSE']:close(expected[key],doc['score'][key])
    if doc['score']['status']!=('available' if expected['minus'] is not None else 'undefined_no_remaining_weight'):raise ValueError('status_mismatch')
    return dict(passed=True,max_absolute_error=max(errors,default=0),independent_decimal=True,expected_C_sign=c['expected_C_sign'],expected_D_sign=c['expected_D_sign'])
