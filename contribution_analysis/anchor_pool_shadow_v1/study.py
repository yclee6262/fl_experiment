"""New local ablations on frozen T2 analytic states; no new main search."""
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
import math
import numpy as np
from contribution_analysis.a_e_b_validation.run import server
from contribution_analysis.three_pool_t2_anchor_v1.core import owner_vectors, decide
from contribution_analysis.three_pool_t2_anchor_v1.reference import oracle


def execute(doc):
    c=doc['fixture']; m=doc['manifest']; r=doc['run']; it=r['iterations'][0]
    l=np.array([float(F(x)) for x in c['linear']]); Q=np.array([[float(F(x)) for x in row] for row in c['quadratic']]); offset=float(F(c['offset']))
    def polynomial(x):return float(offset+l@x+x@Q@x)
    s=server([polynomial],m['anchors']); queries=[]
    phase=''
    def objective(x):
        y=float(s.trusted_agents[0].api_predict(x)[0]); value=abs(y)
        queries.append(dict(phase=phase,x=np.asarray(x).tolist(),reply=y,loss=value))
        return value
    owners,H=owner_vectors(m,r,it); g=np.array(it['g_solver']); x=np.array(it['x_before']); rows=[]
    for factor in (.5,1.,2.):
        eta=it['eta_before']*factor; phase=f'{factor}:full'
        full=decide(s,objective,x,g,it['L_before'],eta,it['mode_before'])
        arms={}
        for i,o in enumerate(owners):
            phase=f'{factor}:without:{o}'
            alt=decide(s,objective,x,g-H[i],it['L_before'],eta,it['mode_before'])
            # Model an exact weight-preserving duplicate of each source direction.
            split_H=np.vstack((H[i]/2,H[i]/2)); grouped=split_H.sum(axis=0)
            phase=f'{factor}:split:{o}'
            split=decide(s,objective,x,g-grouped,it['L_before'],eta,it['mode_before'])
            if not np.array_equal(H[i],grouped) or full['loss']-alt['loss']!=full['loss']-split['loss'] or not np.array_equal(alt['x'],split['x']):
                raise ValueError('owner_split_changed_effect')
            arms[o]=dict(without=alt,split_without=split,C=alt['loss']-full['loss'])
        phase=f'{factor}:joint';joint=decide(s,objective,x,g-H.sum(axis=0),it['L_before'],eta,it['mode_before'])
        J=joint['loss']-full['loss']
        rows.append(dict(factor=factor,full=full,owners=arms,joint=joint,J=J,
                         sum_C=math.fsum(a['C'] for a in arms.values()),interaction=J-math.fsum(a['C'] for a in arms.values())))
    original=next(row for row in rows if row['factor']==1.)
    if abs(original['owners'][doc['result']['owner']]['C']-doc['result']['C'])>1e-9 or not np.allclose(original['full']['x'],it['x_after'],atol=1e-10,rtol=0):
        raise ValueError('old_target_changed')
    return dict(fixture=c,manifest=m,H=H,g=g,rows=rows,queries=queries,synthetic_api_calls=len(queries),
                new_main_searches=0,scope='old_analytic_fixed_state_new_ablation_not_independent_tasks')


def independent(doc):
    """60-digit Decimal ablation and every trial; no production decide invocation."""
    c=doc['fixture']; ref=oracle(c); maxerr=0.; trials=0
    with localcontext() as ctx:
        ctx.prec=60
        def dec(x):
            f=F(str(x));return D(f.numerator)/D(f.denominator)
        start=[dec(z) for z in c['start']]; linear=[dec(z) for z in c['linear']];Q=[[dec(v) for v in row] for row in c['quadratic']]
        def objective(z):return abs(dec(c['offset'])+sum(a*b for a,b in zip(linear,z))+sum(z[i]*Q[i][j]*z[j] for i in range(2) for j in range(2)))
        def close(want,got,tol=1e-9):
            nonlocal maxerr
            if want is None:
                if got is not None:raise ValueError('analytic_null')
            elif isinstance(want,list):
                if len(want)!=len(got):raise ValueError('analytic_length')
                for a,b in zip(want,got):close(a,b,tol)
            else:
                error=abs(float(want)-float(got));maxerr=max(maxerr,error)
                if not math.isfinite(float(got)) or error>tol:raise ValueError('analytic_numeric')
        def branch(g,eta,got):
            nonlocal trials
            norm=sum(v*v for v in g).sqrt();end=start;value=objective(start);q=None;ts=[];stop=None;mode='secant'
            if norm<=D('1e-8'):stop='zero_direction'
            else:
                q=[v/norm for v in g]
                for attempt in range(10):
                    step=eta
                    for z,v in zip(start,[-v for v in q]):
                        if v>0:step=min(step,(D(1)-z)/v)
                        if v<0:step=min(step,(D(-1)-z)/v)
                    step=max(D(0),step);point=[z-step*v for z,v in zip(start,q)];loss=objective(point);ok=loss<objective(start)
                    ts.append((point,loss,eta,ok))
                    if ok:end=point;value=loss;eta=min(D('.5'),eta*D('1.5'));break
                    eta/=2
                if not any(t[3] for t in ts):mode='dynamic';eta=D('.5')
            close(end,got['x'],1e-10);close(value,got['loss']);close(q,got['q'],1e-10);close(eta,got['eta_after'])
            if got['mode_after']!=mode or got['stop']!=stop or len(ts)!=len(got['trials']) or got['accepted']!=any(t[3] for t in ts):raise ValueError('analytic_transition')
            for (point,loss,eta,ok),t in zip(ts,got['trials']):
                close(point,t['x'],1e-10);close(loss,t['loss']);close(eta,t['requested_eta'])
                if ok!=t['accepted']:raise ValueError('analytic_acceptance')
                trials+=1
            return value
        close(ref['H'],doc['H'],1e-10);close(ref['g'],doc['g'],1e-10)
        for row in doc['rows']:
            eta=dec(c['eta'])*dec(row['factor']);full=branch(ref['g'],eta,row['full']);cs=[]
            for i,(o,a) in enumerate(row['owners'].items()):
                altg=[g-h for g,h in zip(ref['g'],ref['H'][i])]
                alt=branch(altg,eta,a['without']);branch(altg,eta,a['split_without']);C=alt-full;close(C,a['C']);cs.append(C)
            joint=branch([D(0),D(0)],eta,row['joint']);close(joint-full,row['J']);close(sum(cs),row['sum_C']);close(joint-full-sum(cs),row['interaction'])
        for q in doc['queries']:
            value=objective([dec(v) for v in q['x']]);close(value,q['reply']);close(value,q['loss'])
    return dict(passed=True,trial_checks=trials,reply_checks=len(doc['queries']),max_error=maxerr)
