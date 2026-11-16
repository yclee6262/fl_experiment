"""Independent Decimal polynomial, directions and decisions. No core import."""
from decimal import Decimal as D,localcontext
from fractions import Fraction


def decimal(x):
    f=Fraction(str(x));return D(f.numerator)/D(f.denominator)


def oracle(case):
    with localcontext() as ctx:
        ctx.prec=60
        x=[decimal(z) for z in case['start']];l=[decimal(z) for z in case['linear']];Q=[[decimal(z) for z in row] for row in case['quadratic']]
        def value(z):return decimal(case['offset'])+sum(a*b for a,b in zip(l,z))+sum(z[i]*Q[i][j]*z[j] for i in range(2) for j in range(2))
        before=value(x);H=[]
        for weight,anchor in zip(case['direction_weights'],case['anchors']):
            z=[decimal(v) for v in anchor];delta=[a-b for a,b in zip(z,x)];dist=sum(v*v for v in delta).sqrt()
            H.append([decimal(weight)*(value(z)-before)/dist*d/dist for d in delta] if dist>=D('1e-8') else [D(0),D(0)])
        g=[sum(h[j] for h in H) for j in range(2)];i=case['target_owner']
        def solve(gg):
            n=sum(a*a for a in gg).sqrt();eta=decimal(case['eta']);trials=[];q=None;mode='secant';end=x;loss=before;stop=None
            if n<=D('1e-8'):stop='zero_direction'
            else:
                q=[a/n for a in gg]
                for j in range(10):
                    step=eta
                    for z,d in zip(x,[-v for v in q]):
                        if d>0:step=min(step,(D(1)-z)/d)
                        elif d<0:step=min(step,(D(-1)-z)/d)
                    step=max(D(0),step);point=[z-step*d for z,d in zip(x,q)];v=value(point);ok=v<before
                    trials.append(dict(x=point,loss=v,requested_eta=eta,accepted=ok))
                    if ok:end=point;loss=v;eta=min(D('.5'),eta*D('1.5'));break
                    eta/=2
                if not any(t['accepted'] for t in trials):mode='dynamic';eta=D('.5')
            return dict(x=end,loss=loss,trials=trials,q=q,g_norm=n,mode_after=mode,eta_after=eta,stop=stop)
        full=solve(g);without=solve([g[j]-H[i][j] for j in range(2)]);p=B=None
        if full['q'] is not None:
            ps=[sum(a*b for a,b in zip(h,full['q'])) for h in H];p=ps[i];B=max(-p,D(0))/sum(abs(v) for v in ps)
        C=without['loss']-full['loss'];sign=1 if C>D('1e-8') else -1 if C<D('-1e-8') else 0
        plabel='undefined' if p is None else 'positive' if p>0 else 'negative' if p<0 else 'zero'
        if sign!=case['expected_C_sign'] or plabel!=case['expected_p']:raise ValueError('predeclared_fixture_label_mismatch:'+case['id'])
        return dict(H=H,g=g,p=p,B=B,C=C,full=full,without=without,C_sign=sign,p_label=plabel)


def floats(x):
    if isinstance(x,D):return float(x)
    if isinstance(x,dict):return {k:floats(v) for k,v in x.items()}
    if isinstance(x,list):return [floats(v) for v in x]
    return x


def verify(case,result):
    expected=floats(oracle(case));errors=[]
    def close(a,b,tol=1e-9):
        if a is None or b is None:
            if a is not b:raise ValueError('undefined_mismatch')
        elif isinstance(a,list):
            if len(a)!=len(b):raise ValueError('length_mismatch')
            for aa,bb in zip(a,b):close(aa,bb,tol)
        else:
            error=abs(float(a)-float(b));errors.append(error)
            if error>tol:raise ValueError('decimal_numeric_mismatch')
    for key in ['H','g']:close(expected[key],result[key],1e-10)
    for key in ['p','B','C']:close(expected[key],result[key])
    for arm in ['full','without']:
        ref=expected[arm];got=result[arm]
        close(ref['x'],got['x'],1e-10);close(ref['loss'],got['loss'])
        close(ref['eta_after'],got['eta_after']);close(ref['q'],got['q'],1e-10)
        if ref['mode_after']!=got['mode_after'] or ref['stop']!=got['stop'] or len(ref['trials'])!=len(got['trials']):raise ValueError('decimal_transition_mismatch')
        for a,b in zip(ref['trials'],got['trials']):
            close(a['x'],b['x'],1e-10);close(a['loss'],b['loss']);close(a['requested_eta'],b['requested_eta'])
            if a['accepted']!=b['accepted']:raise ValueError('decimal_acceptance_mismatch')
    if case['category']=='opposing_helpful' and (result['A']!=0 or result['C']<=0):raise ValueError('missed_opposing_benefit')
    if case['category'] in ['collinear_no_effect','rejected_equal'] and abs(result['C'])>1e-8:raise ValueError('false_result_value')
    return dict(passed=True,max_absolute_error=max(errors,default=0),expected_C_sign=expected['C_sign'],expected_p=expected['p_label'],independent_decimal=True)
