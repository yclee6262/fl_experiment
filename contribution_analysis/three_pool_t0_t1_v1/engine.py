"""No model, contribution-score, payment-provider or search dependency."""
import copy
import json
import math


def integer(x):return type(x) is int and x>=0
def number(x):return type(x) in (int,float) and math.isfinite(x)
def vector(x,d):return isinstance(x,list) and len(x)==d and all(number(v) for v in x)


def _validate_contract(c):
    def require(ok,reason):
        if not ok:raise ValueError('contract:'+reason)
    require(c.get('version')=='three_pool_contract_v1','version')
    p=c['pools'];require(set(p)=={'basic','anchor','model'} and all(integer(v) for v in p.values()),'pools')
    require(integer(c['total']) and sum(p.values())==c['total'],'total')
    require(integer(c['open_tick']) and integer(c['anchor_deadline']) and integer(c['close_tick']) and c['open_tick']<=c['anchor_deadline']<=c['close_tick'],'time')
    require(type(c['dimension']) is int and c['dimension']>0 and len(c['box'])==2 and all(number(v) for v in c['box']) and c['box'][0]<c['box'][1],'box')
    require(c['contribution_pool_status']=='not_established' and c['actual_payment_enabled'] is False,'research_only')
    require(c['reference_measure']=='frozen_full_team_internal_loss_research_only' and c['deployment_acceptance'] is None,'reference_scope')
    require(isinstance(c['owners'],list) and bool(c['owners']),'owners')
    aliases={};ownerids=set();requestids=set();reserved=0
    for o in c['owners']:
        oid=o['owner'];require(isinstance(oid,str) and bool(oid) and oid not in ownerids,'duplicate_owner');ownerids.add(oid)
        require(o['policy'] in ('atomic','staged'),'policy')
        require(isinstance(o['aliases'],list) and all(isinstance(a,str) and a for a in o['aliases']),'aliases')
        for a in [oid]+o['aliases']:
            require(a not in aliases,'duplicate_alias');aliases[a]=oid
        roles=o['roles'];require(bool(roles) and set(roles)<= {'anchor','model'},'roles')
        for role,spec in roles.items():
            require(integer(spec['fee']),'fee');reserved+=spec['fee']
            if role=='anchor':
                require(isinstance(spec['slot'],str) and bool(spec['slot']) and type(spec['admitted_quality']) is bool,'anchor')
            else:
                require(isinstance(spec['version'],str) and bool(spec['version']),'model_version')
                require(isinstance(spec['requests'],list) and bool(spec['requests']),'requests')
                for q in spec['requests']:
                    require(isinstance(q['id'],str) and bool(q['id']) and q['id'] not in requestids,'duplicate_request');requestids.add(q['id'])
                    require(vector(q['x'],c['dimension']),'request_input')
                    require(integer(q['issued']) and integer(q['deadline']) and c['open_tick']<=q['issued']<=q['deadline']<=c['close_tick'],'request_time')
    require(reserved<=p['basic'],'basic_overcommitted')
    return aliases,reserved


def validate_contract(c):
    try:return _validate_contract(c)
    except (KeyError,TypeError,IndexError,AttributeError) as exc:raise ValueError('contract:missing_or_malformed_field') from exc


def settle(contract,events):
    c=copy.deepcopy(contract);aliases,reserved=validate_contract(c)
    if not isinstance(events,list):raise ValueError('events:not_list')
    unique={};replayed=0
    for e in events:
        if not isinstance(e,dict) or not isinstance(e.get('event_id'),str) or not e['event_id']:raise ValueError('event:identity')
        if e.get('kind') not in ('anchor_delivery','score_response','withdrawal','failure_notice'):raise ValueError('event:kind')
        if not integer(e.get('received_tick')) or not isinstance(e.get('participant'),str) or not isinstance(e.get('payload'),dict):raise ValueError('event:format')
        canonical=json.dumps(e,sort_keys=True,allow_nan=False,separators=(',',':'))
        if e['event_id'] in unique:
            if unique[e['event_id']][0]!=canonical:raise ValueError('event:id_conflict')
            replayed+=1
        else:unique[e['event_id']]=(canonical,e)
    ordered=sorted([v[1] for v in unique.values()],key=lambda e:(e['received_tick'],e['event_id']))
    owners={o['owner']:o for o in c['owners']};withdraw={o:None for o in owners}
    for e in ordered:
        oid=aliases.get(e['participant']);tick=e['received_tick']
        if oid is not None and e['kind']=='withdrawal' and c['open_tick']<=tick<=c['close_tick']:
            withdraw[oid]=tick if withdraw[oid] is None else min(tick,withdraw[oid])
    state={o:dict(anchor=False,requests={},violations=set()) for o in owners};decisions=[]
    for e in ordered:
        tick=e['received_tick'];oid=aliases.get(e['participant']);kind=e['kind'];p=e['payload'];reason=None
        if oid is None:reason='unknown_owner'
        elif tick<c['open_tick']:reason='before_task'
        elif tick>c['close_tick']:reason='after_close'
        elif kind=='withdrawal':reason='withdrawal_recorded'
        elif withdraw[oid] is not None and tick>=withdraw[oid]:reason='after_withdrawal'
        elif kind=='failure_notice':reason='failure_notice_recorded'
        else:
            roles=owners[oid]['roles'];s=state[oid]
            if kind=='anchor_delivery':
                if 'anchor' not in roles:reason='role_not_contracted'
                elif tick>c['anchor_deadline']:reason='anchor_late'
                elif p.get('slot')!=roles['anchor']['slot']:reason='wrong_anchor_slot'
                elif not roles['anchor']['admitted_quality']:reason='anchor_admission_not_qualified'
                elif not vector(p.get('x'),c['dimension']):reason='invalid_anchor_value'
                elif not all(c['box'][0]<=v<=c['box'][1] for v in p['x']):reason='anchor_outside_box'
                elif s['anchor']:reason='extra_anchor_no_extra_fee'
                else:s['anchor']=True;reason='anchor_accepted'
            else:
                if 'model' not in roles:reason='role_not_contracted'
                else:
                    spec=roles['model'];q=next((q for q in spec['requests'] if q['id']==p.get('request_id')),None)
                    if q is None:reason='request_not_authorized'
                    elif tick<q['issued']:reason='response_before_issue'
                    elif tick>q['deadline']:reason='response_late'
                    elif p.get('version')!=spec['version']:s['violations'].add('model_version_violation');reason='model_version_violation'
                    elif not vector(p.get('x'),c['dimension']) or p['x']!=q['x']:reason='request_input_mismatch'
                    elif not number(p.get('value')):reason='invalid_response_value'
                    elif q['id'] in s['requests'] and s['requests'][q['id']]!=p['value']:s['violations'].add('determinism_violation');reason='determinism_violation'
                    elif q['id'] in s['requests']:reason='duplicate_response_no_extra_fee'
                    else:s['requests'][q['id']]=p['value'];reason='response_accepted'
        decisions.append(dict(event_id=e['event_id'],owner=oid,received_tick=tick,reason=reason))
    results=[]
    for oid,o in owners.items():
        s=state[oid];roles={};complete={}
        for role,spec in o['roles'].items():
            if role=='anchor':
                ok=s['anchor'];status='fulfilled' if ok else 'not_qualified' if not spec['admitted_quality'] else 'missing_valid_delivery'
                detail=dict(accepted_slot=spec['slot'] if ok else None)
            else:
                expected=[q['id'] for q in spec['requests']];missing=[q for q in expected if q not in s['requests']]
                ok=not missing and not s['violations'];status='fulfilled' if ok else 'service_violation' if s['violations'] else 'missing_valid_responses'
                detail=dict(completed_requests=sorted(s['requests']),missing_requests=missing,violations=sorted(s['violations']))
            complete[role]=ok;roles[role]=dict(fulfilled=ok,status=status,**detail)
        fee=sum(o['roles'][r]['fee'] for r,ok in complete.items() if ok) if o['policy']=='staged' else sum(s['fee'] for s in o['roles'].values()) if all(complete.values()) else 0
        results.append(dict(owner=oid,policy=o['policy'],roles=roles,all_roles_fulfilled=all(complete.values()),withdraw_tick=withdraw[oid],
                            basic_shadow_amount=fee,anchor_contribution_amount=None,model_contribution_amount=None,
                            contribution_status='not_established',actual_transfer=False))
    spent=sum(r['basic_shadow_amount'] for r in results)
    if spent>reserved:raise AssertionError('budget_invariant')
    return dict(version='three_pool_t0_t1_v1',owners=results,events=decisions,replayed_events_ignored=replayed,
                basic=dict(budget=c['pools']['basic'],maximum_committed=reserved,spent_shadow=spent,remaining=c['pools']['basic']-spent),
                contribution_pools=dict(anchor=dict(budget=c['pools']['anchor'],status='reserved_not_established',amount=None),
                                        model=dict(budget=c['pools']['model'],status='reserved_not_established',amount=None)),
                total_budget=c['total'],actual_transfers=0,trust_assumption='trusted_registry_and_receipt_log')
