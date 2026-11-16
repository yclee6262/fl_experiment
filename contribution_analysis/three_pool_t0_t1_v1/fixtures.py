"""Manual expected outcomes and an exhaustive small state space, not engine output."""
import copy
import itertools


def contract():
    owners=[]
    for oid,policy,rs in [('A','atomic',['anchor','model']),('B','staged',['anchor','model']),('C','atomic',['anchor']),('D','atomic',['model'])]:
        roles={}
        if 'anchor' in rs:roles['anchor']=dict(fee=100,slot='slot_'+oid,admitted_quality=True)
        if 'model' in rs:roles['model']=dict(fee=200,version='frozen_'+oid,requests=[dict(id=f'{oid}_q{j}',x=[j/10,0.],issued=10,deadline=200) for j in [1,2]])
        owners.append(dict(owner=oid,aliases=[oid+'_alias',oid+'_alias2'],policy=policy,roles=roles))
    return dict(version='three_pool_contract_v1',total=2000,pools=dict(basic=1000,anchor=500,model=500),open_tick=0,anchor_deadline=100,close_tick=300,
                dimension=2,box=[-1.,1.],owners=owners,contribution_pool_status='not_established',actual_payment_enabled=False,
                reference_measure='frozen_full_team_internal_loss_research_only',deployment_acceptance=None)


def event(eid,owner,kind,tick,payload):return dict(event_id=eid,participant=owner,kind=kind,received_tick=tick,payload=payload)


def base_events():
    es=[]
    for o in contract()['owners']:
        oid=o['owner']
        if 'anchor' in o['roles']:es.append(event(oid+'_a',oid,'anchor_delivery',20,dict(slot='slot_'+oid,x=[.2,-.1])))
        if 'model' in o['roles']:
            for j in [1,2]:es.append(event(f'{oid}_r{j}',oid,'score_response',40+j,dict(request_id=f'{oid}_q{j}',version='frozen_'+oid,x=[j/10,0.],value=1.+j)))
    return es


def cases():
    output=[];default_amounts=dict(A=300,B=300,C=100,D=200)
    full={o:{r:True for r in s['roles']} for o,s in zip(['A','B','C','D'],contract()['owners'])}
    def add(name,change=None,amounts=None,roles=None,reject=None):
        c=contract();es=base_events()
        if change:change(c,es)
        aa=dict(default_amounts);aa.update(amounts or {});rr=copy.deepcopy(full)
        for owner,roleflags in (roles or {}).items():rr[owner].update(roleflags)
        output.append(dict(name=name,contract=c,events=es,expected_rejection=reject,expected_amounts=aa,expected_roles=rr))
    def at(es,eid):return next(e for e in es if e['event_id']==eid)
    def payload(eid,key,val):return lambda c,es:at(es,eid)['payload'].__setitem__(key,val)
    def tick(eid,t):return lambda c,es:at(es,eid).__setitem__('received_tick',t)
    def remove(ids):return lambda c,es:es.__setitem__(slice(None),[e for e in es if e['event_id'] not in ids])
    add('complete_all')
    for label,diagnostic in [('negative_contribution',dict(A=-100,E=-10,B=.4)),('no_improvement',dict(G=0)),('opposing_anchor',dict(p=-10,B=.3)),('poor_outcome_qualified_service',dict(outcome='poor'))]:
        add(label,lambda c,es,d=diagnostic:[e.update(diagnostics=d) for e in es])
    add('atomic_anchor_only',remove(['A_r1','A_r2']),dict(A=0),dict(A=dict(model=False)))
    add('atomic_model_only',remove(['A_a']),dict(A=0),dict(A=dict(anchor=False)))
    add('staged_anchor_only',remove(['B_r1','B_r2']),dict(B=100),dict(B=dict(model=False)))
    add('staged_model_only',remove(['B_a']),dict(B=200),dict(B=dict(anchor=False)))
    add('single_anchor_missing',remove(['C_a']),dict(C=0),dict(C=dict(anchor=False)))
    add('single_model_incomplete',remove(['D_r2']),dict(D=0),dict(D=dict(model=False)))
    add('missing_all_events',lambda c,es:es.clear(),dict(A=0,B=0,C=0,D=0),{o:{r:False for r in fs} for o,fs in full.items()})
    for name,x in [('anchor_nan',['NaN',0]),('anchor_infinity',['Infinity',0]),('anchor_bool',[True,0]),('anchor_wrong_dimension',[.1]),('anchor_outside_box',[1.001,0])]:
        add(name,payload('C_a','x',x),dict(C=0),dict(C=dict(anchor=False)))
    add('anchor_wrong_slot',payload('C_a','slot','other'),dict(C=0),dict(C=dict(anchor=False)))
    add('anchor_late',tick('C_a',101),dict(C=0),dict(C=dict(anchor=False)))
    add('anchor_deadline_inclusive',tick('C_a',100))
    add('anchor_box_boundary',payload('C_a','x',[-1.,1.]))
    add('anchor_not_admitted_quality',lambda c,es:c['owners'][2]['roles']['anchor'].__setitem__('admitted_quality',False),dict(C=0),dict(C=dict(anchor=False)))
    for name,key,val in [('response_nan','value','NaN'),('response_bool','value',True),('wrong_query_input','x',[.123,0]),('unauthorized_query_only','request_id','not_authorized')]:
        add(name,payload('D_r2',key,val),dict(D=0),dict(D=dict(model=False)))
    add('response_before_issue',tick('D_r2',9),dict(D=0),dict(D=dict(model=False)))
    add('response_late',tick('D_r2',201),dict(D=0),dict(D=dict(model=False)))
    add('response_deadline_inclusive',tick('D_r2',200))
    add('response_issue_inclusive',tick('D_r2',10))
    add('version_mismatch',payload('D_r2','version','changed'),dict(D=0),dict(D=dict(model=False)))
    def extra(eid,newid,**updates):
        def mutate(c,es):
            e=copy.deepcopy(at(es,eid));e['event_id']=newid
            for k,v in updates.items():
                if k=='value':e['payload']['value']=v
                else:e[k]=v
            es.append(e)
        return mutate
    add('conflicting_valid_response',extra('D_r1','D_conflict',received_tick=80,value=999.),dict(D=0),dict(D=dict(model=False)))
    def mismatch_after(c,es):
        e=copy.deepcopy(at(es,'D_r1'));e.update(event_id='D_changed_after',received_tick=80);e['payload']['version']='changed';es.append(e)
    add('version_changed_after_completion',mismatch_after,dict(D=0),dict(D=dict(model=False)))
    def retry(c,es):
        bad=copy.deepcopy(at(es,'D_r2'));bad['event_id']='D_bad_first';bad['received_tick']=30;bad['payload']['value']='NaN';es.append(bad)
    add('invalid_then_valid_retry',retry)
    add('same_id_exact_replay',lambda c,es:es.extend(copy.deepcopy(es)))
    add('different_id_duplicate_response',extra('D_r1','D_duplicate',received_tick=90))
    add('additional_anchor_no_fee',extra('C_a','C_extra',received_tick=90))
    add('same_owner_alias',lambda c,es:[e.__setitem__('participant','A_alias2') for e in es if e['participant']=='A'])
    add('unknown_owner',lambda c,es:es.append(event('unknown','not_enrolled','anchor_delivery',20,dict(slot='slot_X',x=[0.,0.]))))
    add('extra_unauthorized_query',lambda c,es:es.append(event('extra_q','D','score_response',60,dict(request_id='unknown',version='frozen_D',x=[0.,0.],value=1.))))
    add('wrong_role_event',lambda c,es:es.append(event('C_score','C','score_response',60,dict(request_id='unknown',version='frozen_C',x=[0.,0.],value=1.))))
    def withdraw(owner,t):return lambda c,es:es.append(event(owner+'_withdraw',owner,'withdrawal',t,{}))
    add('atomic_withdraw_before_all',withdraw('A',0),dict(A=0),dict(A=dict(anchor=False,model=False)))
    add('atomic_withdraw_after_anchor',withdraw('A',30),dict(A=0),dict(A=dict(model=False)))
    add('staged_withdraw_after_anchor',withdraw('B',30),dict(B=100),dict(B=dict(model=False)))
    add('withdraw_after_completion',withdraw('A',90))
    add('withdraw_delivery_same_tick',withdraw('C',20),dict(C=0),dict(C=dict(anchor=False)))
    add('withdraw_score_same_tick',withdraw('D',42),dict(D=0),dict(D=dict(model=False)))
    add('event_after_close',extra('D_r1','D_late_change',received_tick=301,value=999.))
    def no_anchor(c,es):
        es[:]=[e for e in es if e['event_id']!='B_a'];es.append(event('B_fail','B','failure_notice',80,dict(reason='no_feasible_anchor')))
    add('no_feasible_anchor_notice',no_anchor,dict(B=200),dict(B=dict(anchor=False)))
    add('failure_notice_after_fulfillment',lambda c,es:es.append(event('C_notice','C','failure_notice',90,dict(reason='diagnostic_only'))))
    add('event_id_conflict',extra('D_r1','D_r1',value=99.),reject='event:id_conflict')
    add('unknown_event_kind',lambda c,es:es[0].__setitem__('kind','award'),reject='event:kind')
    invalids=[('overcommitted',lambda c:c['pools'].update(basic=800,anchor=700)),
              ('wrong_total',lambda c:c.__setitem__('total',2001)),
              ('negative_fee',lambda c:c['owners'][0]['roles']['anchor'].__setitem__('fee',-1)),
              ('float_fee',lambda c:c['owners'][0]['roles']['anchor'].__setitem__('fee',100.5)),
              ('bool_fee',lambda c:c['owners'][0]['roles']['anchor'].__setitem__('fee',True)),
              ('duplicate_owner',lambda c:c['owners'].append(copy.deepcopy(c['owners'][0]))),
              ('duplicate_alias',lambda c:c['owners'][1]['aliases'].append('A_alias')),
              ('missing_version',lambda c:c['owners'][3]['roles']['model'].__setitem__('version','')),
              ('missing_version_field',lambda c:c['owners'][3]['roles']['model'].pop('version')),
              ('duplicate_request',lambda c:c['owners'][3]['roles']['model']['requests'].append(copy.deepcopy(c['owners'][3]['roles']['model']['requests'][0]))),
              ('unknown_policy',lambda c:c['owners'][0].__setitem__('policy','by_calls')),
              ('invalid_request_time',lambda c:c['owners'][3]['roles']['model']['requests'][0].__setitem__('issued',250)),
              ('enable_actual_payment',lambda c:c.__setitem__('actual_payment_enabled',True))]
    for name,mutate in invalids:add('contract_'+name,lambda c,es,f=mutate:f(c),reject='contract:')
    for owner,atomic in [('A',True),('B',False)]:
        for a,q1,q2,w in itertools.product([False,True],repeat=4):
            def mutate(c,es,owner=owner,a=a,q1=q1,q2=q2,w=w):
                drop=[owner+'_a'] if not a else []
                if not q1:drop.append(owner+'_r1')
                if not q2:drop.append(owner+'_r2')
                es[:]=[e for e in es if e['event_id'] not in drop]
                if w:es.append(event(owner+'_withdraw_early',owner,'withdrawal',0,{}))
            anchor=a and not w;model=q1 and q2 and not w
            amount=300 if atomic and anchor and model else 0 if atomic else 100*anchor+200*model
            add('exhaustive_'+owner+''.join(str(int(v)) for v in (a,q1,q2,w)),mutate,{owner:amount},{owner:dict(anchor=anchor,model=model)})
    return output
