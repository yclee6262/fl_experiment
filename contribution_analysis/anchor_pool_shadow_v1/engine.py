"""Decimal ledger, unique owner/round slots, fixed-contract release gate."""
from decimal import Decimal, localcontext, ROUND_FLOOR


def amount(x):
    if isinstance(x, bool) or not isinstance(x, (str, int, float)):
        raise ValueError('invalid_number')
    try:
        v = Decimal(str(x))
    except Exception as exc:
        raise ValueError('invalid_number') from exc
    if not v.is_finite():
        raise ValueError('nonfinite')
    return v


def settle(contract, events, rule='net'):
    with localcontext() as ctx:
        ctx.prec = 80
        owners = contract['owners']; rounds = contract['rounds']
        if (not owners or len(set(owners)) != len(owners) or
            any(not isinstance(o, str) or not o for o in owners) or
            len(set(rounds)) != len(rounds) or
            any(type(t) is not int or t < 0 for t in rounds)):
            raise ValueError('invalid_roster')
        budget = contract['budget']
        if type(budget) is not int or budget < 0:
            raise ValueError('invalid_budget')
        epsilon = amount(contract['epsilon']); gate = amount(contract['min_release'])
        if epsilon < 0 or gate <= epsilon or not contract['reference_id']:
            raise ValueError('invalid_contract')
        if rule not in ('net', 'positive_rounds', 'equal'):
            raise ValueError('invalid_rule')
        slots = {}
        for e in events:
            if e['reference_id'] != contract['reference_id']:
                raise ValueError('reference_mismatch')
            if e['owner'] not in owners or type(e['round']) is not int or e['round'] not in rounds:
                raise ValueError('unauthorized_slot')
            k = (e['owner'], e['round'])
            v = None if e['C'] is None else amount(e['C'])
            if k in slots and slots[k] != v:
                raise ValueError('conflicting_slot')
            slots[k] = v
        missing = [{'owner': o, 'round': t} for o in owners for t in rounds if (o, t) not in slots]
        undefined = [{'owner': o, 'round': t} for o in owners for t in rounds if (o, t) in slots and slots[o, t] is None]
        rows = {}
        for o in owners:
            vals = [slots[o, t] for t in rounds if (o, t) in slots and slots[o, t] is not None]
            complete = len(vals) == len(rounds) and bool(rounds)
            net = sum(vals, Decimal(0)); pos = sum((v for v in vals if v > 0), Decimal(0)); neg = sum((v for v in vals if v < 0), Decimal(0))
            rows[o] = dict(net=str(net) if complete else None, positive=str(pos) if complete else None,
                           negative=str(neg) if complete else None, defined=len(vals), expected=len(rounds),
                           coverage=str(Decimal(len(vals))/len(rounds)) if rounds else None)
        held = bool(missing or undefined or not rounds)
        if held:
            return dict(status='held', rule=rule, rows=rows, missing=missing, undefined=undefined,
                        released=0, paid_total=0, reserved=budget, payments={o: None for o in owners}, weights=None)
        net_weights = {o: max(amount(rows[o]['net']), Decimal(0)) if abs(amount(rows[o]['net'])) > epsilon else Decimal(0) for o in owners}
        total_net = sum(net_weights.values(), Decimal(0))
        released = budget if total_net >= gate else 0
        weights = net_weights if rule == 'net' else {o: amount(rows[o]['positive']) for o in owners} if rule == 'positive_rounds' else {o: Decimal(1) for o in owners}
        total = sum(weights.values(), Decimal(0)); payments = {o: 0 for o in owners}
        if released:
            # Decimal shares for production; independent Fraction implementation audits exact rounding.
            raw = {o: released*weights[o]/total for o in owners}
            payments = {o: int(raw[o].to_integral_value(rounding=ROUND_FLOOR)) for o in owners}
            rank = sorted(owners, key=lambda o: (-(raw[o]-payments[o]), o))
            for o in rank[:released-sum(payments.values())]:
                payments[o] += 1
        return dict(status='settled' if released else 'reserved', rule=rule, rows=rows, missing=[], undefined=[],
                    released=released, paid_total=sum(payments.values()), reserved=budget-released,
                    payments=payments, weights={o: str(w) for o, w in weights.items()})
