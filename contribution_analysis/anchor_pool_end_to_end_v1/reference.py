"""Independent Fraction settlement and scalar raw-reply branch interpreter."""
from fractions import Fraction as F
import math
import numpy as np
from contribution_analysis.custom_realized_gain_v1.schema import key


def settlement(owners,rounds,events,budget=500):
    slots={}
    for e in events:
        if e['owner']not in owners or type(e['t'])is not int or e['t']not in rounds:raise ValueError('unauthorized')
        k=e['owner'],e['t'];v=None if e['C']is None else F(str(e['C']));p=(v,F(str(e['rho'])))
        if k in slots and slots[k]!=p:raise ValueError('conflicting')
        slots[k]=p
    missing=[dict(owner=o,t=t)for o in owners for t in rounds if(o,t)not in slots]
    undefined=[dict(owner=o,t=t)for o in owners for t in rounds if(o,t)in slots and slots[o,t][0]is None]
    stats={};ws={}
    for o in owners:
        vals=[slots[o,t][0]for t in rounds if(o,t)in slots and slots[o,t][0]is not None];complete=len(vals)==len(rounds)
        net=sum(vals,F(0));err=sum((slots[o,t][1]for t in rounds if(o,t)in slots),F(0));tau=max(F('1e-8'),err)
        stats[o]=dict(net=net if complete else None,positive=sum((v for v in vals if v>0),F(0))if complete else None,
                      negative=sum((v for v in vals if v<0),F(0))if complete else None,tau=tau,error_sum=err,
                      defined=len(vals),expected=len(rounds),coverage=F(len(vals),len(rounds))if rounds else None)
        ws[o]=net if complete and net>tau else F(0)
    total=sum(ws.values());status='held'if missing or undefined else 'settled'if total>0 else 'reserved'
    ps={o:None if status=='held'else 0 for o in owners};qs={o:None for o in owners}
    if status=='settled':
        qs={o:ws[o]/total for o in owners};quota={o:budget*qs[o]for o in owners};ps={o:quota[o].numerator//quota[o].denominator for o in owners}
        for o in sorted(owners,key=lambda o:(-(quota[o]-ps[o]),o))[:budget-sum(ps.values())]:ps[o]+=1
    return dict(status=status,stats=stats,weights=None if status=='held'else ws,total_positive_net=None if status=='held'else total,
                shares=qs,payments=ps,reserved=budget if status!='settled'else 0,missing=missing,undefined=undefined)


def verify_settlement(owners,rounds,events,got):
    want=settlement(owners,rounds,events,got['budget'])
    def equal(a,b,path):
        if a is None:
            if b is not None:raise ValueError('null:'+path)
        elif isinstance(a,dict):
            if set(a)!=set(b):raise ValueError('keys:'+path)
            for k in a:equal(a[k],b[k],path+'.'+k)
        elif isinstance(a,F):
            if b is None or abs(F(str(b))-a)>F(1,10**65):raise ValueError('numeric:'+path)
        elif a!=b:raise ValueError('value:'+path)
    for k in want:equal(want[k],got[k],k)
    if got['owners']!=owners or got['rounds']!=rounds or got['actual_payment']:raise ValueError('contract')
    return True


def loss_fraction(m,e,queries):
    weights=[F(str(v))for v in m['model_weights_raw']];total=sum(weights)
    vals=[]
    for j,qid in enumerate(e['model_query_ids']):
        q=queries[qid]
        if q['model_index']!=j or q['model_id']!=m['model_ids'][j]or not q['success']:raise ValueError('raw_model_link')
        if key(q['x'])!=key(e['x']):raise ValueError('raw_input_link')
        vals.append(abs(F(str(q['output'][0]))-F(str(m['target']))))
    return sum((w*v for w,v in zip(weights,vals)),F(0))/total


def verify_branches(m,r,branches,trace):
    owners=list(dict.fromkeys(m['anchor_owners']));D=len(m['initial_solution']);eps=np.finfo(float).eps
    if len(branches)!=len(r['iterations'])*len(owners):raise ValueError('branch_coverage')
    max_error=0.;objective_errors=[];flips=0;near_margins=0;used=[]
    def near(a,b,tol=1e-10):
        nonlocal max_error
        if isinstance(a,list):
            if len(a)!=len(b):raise ValueError('branch_length')
            for x,y in zip(a,b):near(x,y,tol)
        elif not math.isfinite(float(b))or abs(float(a)-float(b))>tol:raise ValueError('branch_numeric')
        else:max_error=max(max_error,abs(float(a)-float(b)))
    main_prec=[]
    for e in r['evaluations']:
        z=loss_fraction(m,e,r['queries']);main_prec.append(z);objective_errors.append(abs(float(z)-e['objective_value']))
    branch_prec=[]
    for e in trace['evaluations']:
        z=loss_fraction(m,e,trace['queries']);branch_prec.append(z);objective_errors.append(abs(float(z)-e['objective_value']))
    for it in r['iterations']:
        before=r['evaluations'][it['current_eval_id']];g=it['g_solver'];x=it['x_before']
        # Independently combine scalar raw objective sources, preserving original contract weights.
        H=[]
        for o in owners:
            terms=[[]for _ in range(D)]
            for d in it['directions']:
                k=d['index']
                if d['skip_reason']or m['anchor_owners'][k]!=o:continue
                src=r['evaluations'][d['source_eval_id']]
                # Exact saved objective arithmetic; producer reconstructs from weighted model replies.
                slope=(src['objective_value']-before['objective_value'])/d['actual_h']
                for j in range(D):terms[j].append(m['direction_weights'][k]*slope*d['u'][j])
            H.append([math.fsum(v)for v in terms])
        near([math.fsum(h[j]for h in H)for j in range(D)],g)
        for tr in it['trials']:
            delta=main_prec[tr['eval_id']]-main_prec[it['current_eval_id']]
            flips+=bool(delta<0)!=tr['accepted']
        for i,o in enumerate(owners):
            b=next(z for z in branches if z['owner']==o and z['t']==it['t']);near(H[i],b['H']);gg=[v-h for v,h in zip(g,H[i])];norm=math.sqrt(math.fsum(v*v for v in gg));near(norm,b['decision']['g_norm'])
            ts=b['decision']['trials'];ids=b['eval_ids'];used.extend(ids)
            if len(ts)!=len(ids):raise ValueError('branch_trial_ids')
            if norm<=1e-8:
                if ts or b['decision']['stop']!='zero_direction':raise ValueError('branch_zero')
                point=x;end_loss=it['L_before'];mode=it['mode_before'];eta_after=it['eta_before'];accepted=False
            else:
                q=[v/norm for v in gg];near(q,b['decision']['q']);accepted=False
                for attempt,(tr,eid)in enumerate(zip(ts,ids)):
                    requested=it['eta_before']/2**attempt
                    step=max(0.,min([requested]+[(m['box'][1]-z)/v if v>0 else (m['box'][0]-z)/v for z,v in zip(x,[-z for z in q])if v]))
                    expected=[z-step*v for z,v in zip(x,q)];e=trace['evaluations'][eid]
                    near(expected,tr['x']);near(e['x'],tr['x'],1e-12);near(requested,tr['requested_eta']);near(e['objective_value'],tr['loss'],1e-12)
                    ok=e['objective_value']<it['L_before']
                    if ok!=tr['accepted']or(ok and attempt!=len(ts)-1):raise ValueError('branch_acceptance')
                    exact_delta=branch_prec[eid]-main_prec[it['current_eval_id']];flips+=bool(exact_delta<0)!=ok
                    bound=max(1e-12,32*eps*(1+abs(e['objective_value'])+abs(it['L_before'])))
                    near_margins+=abs(e['objective_value']-it['L_before'])<=bound
                    accepted=ok
                if not ts or(not accepted and len(ts)!=10):raise ValueError('branch_backtracking')
                point=ts[-1]['x']if accepted else x;end_loss=ts[-1]['loss']if accepted else it['L_before']
                mode=it['mode_before']if accepted else 'dynamic'if it['mode_before']=='secant'else it['mode_before']
                eta_after=min(.5,ts[-1]['requested_eta']*1.5)if accepted else .5 if mode!=it['mode_before']else it['eta_before']
            near(point,b['decision']['x']);near(end_loss,b['decision']['loss'],1e-12);near(eta_after,b['decision']['eta_after'])
            if accepted!=b['decision']['accepted']or mode!=b['decision']['mode_after']:raise ValueError('branch_transition')
            near(end_loss-it['L_after'],b['C'],1e-12)
            rho=max(1e-12,32*eps*(1+abs(end_loss)+abs(it['L_after'])))
            near(rho,b['rho'],1e-20)
    if used!=list(range(len(trace['evaluations']))):raise ValueError('branch_eval_coverage')
    if sum(len(e['model_query_ids'])for e in trace['evaluations'])!=len(trace['queries']):raise ValueError('branch_query_coverage')
    return dict(passed=True,owner_rounds=len(branches),branch_evaluations=len(trace['evaluations']),branch_api_calls=len(trace['queries']),
                max_branch_error=max_error,max_objective_arithmetic_error=max(objective_errors,default=0),
                exact_reply_acceptance_flips=flips,near_acceptance_margins=near_margins)
