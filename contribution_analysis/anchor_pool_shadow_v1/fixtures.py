"""Fixed hand-labelled and generated ledger cases; not empirical quality data."""
import copy


def contract(owners=('A','B'), rounds=(0,1,2), budget=500):
    return dict(owners=list(owners),rounds=list(rounds),budget=budget,epsilon='0.00000001',min_release='0.001',reference_id='frozen_internal_v1')


def events(c, values):
    return [dict(event_id=f'{o}:{t}',owner=o,round=t,C=None if v is None else str(v),reference_id=c['reference_id'])
            for o,vs in values.items() for t,v in zip(c['rounds'],vs)]


def cases():
    out=[]
    def add(name,values,expected,split='development',edit=None,c=None,error=None):
        c=copy.deepcopy(c or contract()); es=events(c,values)
        if edit:edit(c,es)
        out.append(dict(id=name,split=split,contract=c,events=es,expected_net=expected,error=error))
    add('signed_example',{'A':[3,-2,1],'B':[1,1,1]},dict(A=200,B=300))
    add('all_zero',{'A':[0,0,0],'B':[0,0,0]},dict(A=0,B=0))
    add('all_negative',{'A':[-1,-2,0],'B':[-1,-1,-1]},dict(A=0,B=0))
    add('mixed_net',{'A':[3,-4,0],'B':[1,0,0]},dict(A=0,B=500))
    add('cycle_positive_part',{'A':[2,-2,0],'B':[1,0,0]},dict(A=0,B=500))
    add('gate_exact',{'A':['0.001',0,0],'B':[0,0,0]},dict(A=500,B=0))
    add('below_gate',{'A':['0.000999',0,0],'B':[0,0,0]},dict(A=0,B=0))
    add('deadzone',{'A':['0.00000001',0,0],'B':[1,0,0]},dict(A=0,B=500))
    add('near_round_fragments',{'A':['0.000000009','0.000000009',0],'B':[1,0,0]},dict(A=0,B=500))
    add('missing',{'A':[1,1,1],'B':[1,1]},None)
    add('undefined',{'A':[1,None,1],'B':[1,1,1]},None)
    add('no_rounds',{'A':[],'B':[]},None,c=contract(rounds=()))
    add('conflict',{'A':[1,1,1],'B':[1,1,1]},None,edit=lambda c,e:e.append(dict(e[0],C='2')),error='conflicting_slot')
    add('unknown_owner',{'A':[1,1,1],'B':[1,1,1]},None,edit=lambda c,e:e.append(dict(e[0],owner='alias')),error='unauthorized_slot')
    add('nan',{'A':['NaN',1,1],'B':[1,1,1]},None,error='nonfinite')
    add('wrong_reference',{'A':[1,1,1],'B':[1,1,1]},None,edit=lambda c,e:e[0].update(reference_id='changed'),error='reference_mismatch')
    add('negative_budget',{'A':[1,1,1],'B':[1,1,1]},None,edit=lambda c,e:c.update(budget=-1),error='invalid_budget')
    add('bool_C',{'A':[1,1,1],'B':[1,1,1]},None,edit=lambda c,e:e[0].update(C=True),error='invalid_number')
    add('heldout_signed',{'A':[4,-3,1],'B':[2,1,0]},dict(A=200,B=300),'heldout')
    add('heldout_tie',{'A':[1,0,0],'B':[1,0,0],'C':[1,0,0]},dict(A=167,B=167,C=166),'heldout',c=contract(('A','B','C')))
    add('heldout_negative',{'A':[-2,1,0],'B':[-3,0,2]},dict(A=0,B=0),'heldout')
    add('heldout_missing',{'A':[1],'B':[1,2,3]},None,'heldout')
    add('heldout_null',{'A':[1,2,3],'B':[None,2,3]},None,'heldout')
    add('heldout_tiny',{'A':['0.0001',0,0],'B':['0.0001',0,0]},dict(A=0,B=0),'heldout')
    add('heldout_cycle',{'A':[5,-5,0],'B':[2,0,0]},dict(A=0,B=500),'heldout')
    add('heldout_conflict',{'A':[2,1,1],'B':[1,1,1]},None,'heldout',edit=lambda c,e:e.append(dict(e[0],C=None)),error='conflicting_slot')
    # Exhaustive bounded rational patterns, a separate deterministic account-validation set.
    for i in range(48):
        a=i%5-2; b=(i//5)%5-2; d=(i//25)%3-1
        add(f'generated_{i:02}',{'A':[a,b,-a],'B':[b,d,1],'C':[d,a,0]},'reference',
            'development' if i<24 else 'heldout',c=contract(('A','B','C')))
    return out
