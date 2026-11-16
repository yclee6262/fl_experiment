"""Fixed budget shares from complete owner/round signed effects."""
from decimal import Decimal as D, localcontext, ROUND_FLOOR


def num(v):
    if isinstance(v,bool) or not isinstance(v,(str,int,float)):
        raise ValueError('invalid_number')
    try:x=D(str(v))
    except Exception as exc:raise ValueError('invalid_number') from exc
    if not x.is_finite():raise ValueError('nonfinite')
    return x


def settle(owners,rounds,events,budget=500):
    with localcontext() as ctx:
        ctx.prec=80
        if not owners or len(set(owners))!=len(owners) or any(not isinstance(o,str) or not o for o in owners):raise ValueError('invalid_roster')
        if len(set(rounds))!=len(rounds) or any(type(t)is not int or t<0 for t in rounds):raise ValueError('invalid_rounds')
        if type(budget)is not int or budget<0:raise ValueError('invalid_budget')
        slots={}
        for e in events:
            if e['owner'] not in owners or type(e['t'])is not int or e['t'] not in rounds:raise ValueError('unauthorized_slot')
            v=None if e['C'] is None else num(e['C']);error=num(e['rho'])
            if error<0:raise ValueError('negative_error')
            k=e['owner'],e['t'];pair=v,error
            if k in slots and slots[k]!=pair:raise ValueError('conflicting_slot')
            slots[k]=pair
        missing=[dict(owner=o,t=t) for o in owners for t in rounds if (o,t)not in slots]
        undefined=[dict(owner=o,t=t) for o in owners for t in rounds if (o,t)in slots and slots[o,t][0]is None]
        held=bool(missing or undefined);stats={};weights={}
        for o in owners:
            defined=[slots[o,t][0] for t in rounds if (o,t)in slots and slots[o,t][0]is not None]
            complete=len(defined)==len(rounds);net=sum(defined,D(0));error=sum((slots[o,t][1] for t in rounds if (o,t)in slots),D(0));tau=max(D('1e-8'),error)
            pos=sum((v for v in defined if v>0),D(0));neg=sum((v for v in defined if v<0),D(0))
            stats[o]=dict(net=str(net)if complete else None,positive=str(pos)if complete else None,negative=str(neg)if complete else None,
                          tau=str(tau),error_sum=str(error),defined=len(defined),expected=len(rounds),coverage=str(D(len(defined))/len(rounds))if rounds else None)
            weights[o]=net if complete and net>tau else D(0)
        total=sum(weights.values(),D(0));status='held'if held else 'settled'if total>0 else 'reserved'
        shares={o:None for o in owners};payments={o:None if held else 0 for o in owners}
        if status=='settled':
            shares={o:str(weights[o]/total)for o in owners};quota={o:D(budget)*weights[o]/total for o in owners}
            payments={o:int(quota[o].to_integral_value(rounding=ROUND_FLOOR))for o in owners}
            priority=sorted(owners,key=lambda o:(-(quota[o]-payments[o]),o))
            for o in priority[:budget-sum(payments.values())]:payments[o]+=1
        return dict(status=status,owners=owners,rounds=rounds,stats=stats,weights=None if held else {o:str(w)for o,w in weights.items()},
                    total_positive_net=None if held else str(total),shares=shares,payments=payments,
                    reserved=budget if status!='settled'else 0,budget=budget,missing=missing,undefined=undefined,actual_payment=False)
