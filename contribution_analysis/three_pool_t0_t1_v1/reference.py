"""Independent scan of trusted raw events. Never imports settlement engine."""
import math


def reference(c,events):
    unique={e['event_id']:e for e in events}
    amounts={};flags={};details={};withdrawals={}
    def finite(z):return type(z) in [float,int] and math.isfinite(z)
    def matches(x):return type(x) is list and len(x)==c['dimension'] and all(finite(v) for v in x)
    for owner in c['owners']:
        oid=owner['owner'];names=[oid,*owner['aliases']]
        raw=sorted([e for e in unique.values() if e['participant'] in names and c['open_tick']<=e['received_tick']<=c['close_tick']],key=lambda e:(e['received_tick'],e['event_id']))
        exits=[e['received_tick'] for e in raw if e['kind']=='withdrawal'];end=min(exits) if exits else None;withdrawals[oid]=end
        active=[e for e in raw if end is None or e['received_tick']<end]
        fs={};ds={}
        for role,spec in owner['roles'].items():
            if role=='anchor':
                delivered=[e for e in active if e['kind']=='anchor_delivery' and e['received_tick']<=c['anchor_deadline'] and e['payload'].get('slot')==spec['slot'] and matches(e['payload'].get('x')) and all(c['box'][0]<=z<=c['box'][1] for z in e['payload']['x'])]
                ok=bool(delivered) and spec['admitted_quality'];fs[role]=ok
                ds[role]=dict(fulfilled=ok,status='fulfilled' if ok else 'not_qualified' if not spec['admitted_quality'] else 'missing_valid_delivery',accepted_slot=spec['slot'] if ok else None)
            else:
                done=[];violations=[]
                for request in spec['requests']:
                    replies=[e['payload'] for e in active if e['kind']=='score_response' and e['payload'].get('request_id')==request['id'] and request['issued']<=e['received_tick']<=request['deadline']]
                    if any(p.get('version')!=spec['version'] for p in replies):violations.append('model_version_violation')
                    values=[p['value'] for p in replies if p.get('version')==spec['version'] and matches(p.get('x')) and p['x']==request['x'] and finite(p.get('value'))]
                    if values:done.append(request['id'])
                    if len(set(values))>1:violations.append('determinism_violation')
                missing=[q['id'] for q in spec['requests'] if q['id'] not in done];violations=sorted(set(violations));ok=not missing and not violations
                fs[role]=ok;ds[role]=dict(fulfilled=ok,status='fulfilled' if ok else 'service_violation' if violations else 'missing_valid_responses',completed_requests=sorted(done),missing_requests=missing,violations=violations)
        flags[oid]=fs;details[oid]=ds
        if owner['policy']=='atomic':amounts[oid]=sum(s['fee'] for s in owner['roles'].values()) if all(fs.values()) else 0
        else:amounts[oid]=sum(owner['roles'][r]['fee'] for r in fs if fs[r])
    return dict(amounts=amounts,flags=flags,details=details,withdrawals=withdrawals)


def verify(case,output):
    r=reference(case['contract'],case['events']);c=case['contract']
    if r['amounts']!=case['expected_amounts'] or r['flags']!=case['expected_roles']:raise ValueError('reference_disagrees_with_golden')
    actual={o['owner']:o for o in output['owners']}
    if set(actual)!=set(r['amounts']):raise ValueError('owner_coverage')
    for oid,amount in r['amounts'].items():
        got=actual[oid]
        if got['basic_shadow_amount']!=amount or got['roles']!=r['details'][oid] or got['withdraw_tick']!=r['withdrawals'][oid] or got['all_roles_fulfilled']!=all(r['flags'][oid].values()):raise ValueError('shadow_amount_or_role_mismatch')
        if got['anchor_contribution_amount'] is not None or got['model_contribution_amount'] is not None or got['actual_transfer'] is not False:raise ValueError('unapproved_contribution_payment')
    reserved=sum(s['fee'] for o in c['owners'] for s in o['roles'].values());spent=sum(r['amounts'].values())
    expected=dict(budget=c['pools']['basic'],maximum_committed=reserved,spent_shadow=spent,remaining=c['pools']['basic']-spent)
    if output['basic']!=expected or output['actual_transfers']!=0 or not 0<=spent<=reserved<=c['pools']['basic']:raise ValueError('budget_invariant')
    for role in ['anchor','model']:
        if output['contribution_pools'][role]!=dict(budget=c['pools'][role],status='reserved_not_established',amount=None):raise ValueError('contribution_pool_not_reserved')
    ids={e['event_id'] for e in case['events']}
    if len(output['events'])!=len(ids) or {e['event_id'] for e in output['events']}!=ids or any(not e['reason'] for e in output['events']):raise ValueError('event_decision_coverage')
    if output['replayed_events_ignored']!=len(case['events'])-len(ids):raise ValueError('replay_count')
    return True
