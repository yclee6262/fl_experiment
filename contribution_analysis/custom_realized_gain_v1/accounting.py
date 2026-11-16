"""Pure offline accounting. Deliberately has no server/model/evaluator dependency."""
import numpy as np
from .schema import check_close, validate_manifest, key


def compute_accounts(manifest, evaluations, iterations, initial_eval_id, final_eval_id):
    a, v = validate_manifest(manifest)
    K, M = len(a), len(v)
    owners = list(dict.fromkeys(manifest['anchor_owners']))
    all_owners = list(dict.fromkeys(manifest['anchor_owners']+manifest['model_owners']))
    owner_index = {o:j for j,o in enumerate(owners)}
    ev = {r['eval_id']:r for r in evaluations}
    if len(ev)!=len(evaluations) or set(ev)!=set(range(len(ev))): raise ValueError('duplicate_or_missing_eval')
    if [r['t'] for r in iterations] != list(range(len(iterations))): raise ValueError('duplicate_or_missing_iteration')
    if len(iterations)>manifest['solver_config']['num_iterations']: raise ValueError('iteration_budget')
    initial, final = ev[initial_eval_id], ev[final_eval_id]
    if initial['purpose']!='initialization_current' or final['purpose']!='final_return': raise ValueError('wrong_endpoint_event')
    if manifest['solver_config']['num_iterations']>0 and not iterations: raise ValueError('missing_trace')
    if iterations and iterations[-1]['stop_reason'] not in ['zero_direction','no_improvement','iteration_budget']: raise ValueError('incomplete_trace')
    for r in evaluations:
        if len(r['individual_losses'])!=M: raise ValueError('missing_model_loss')
        losses = np.asarray(r['individual_losses'], float)
        if (losses<0).any(): raise ValueError('negative_absolute_loss')
        check_close(np.dot(v, losses), r['objective_value'], 'objective_rebuild')
    A=np.zeros(len(owners)); E=np.zeros(M); total=0.; details=[]
    current_eid=initial_eval_id; max_g_error=max_motion_error=0.
    sensitive_count=accepted_count=0; sensitive_gain=0.
    for it in iterations:
        if not it.get('closed') or it['current_eval_id']!=current_eid: raise ValueError('missing_trace_or_state_chain')
        before=ev[current_eid]; x=np.asarray(it['x_before'],float)
        check_close(x,before['x'],'current_point'); check_close(it['L_before'],before['objective_value'],'current_loss')
        if len(it['directions'])!=K or [d['index'] for d in it['directions']]!=list(range(K)): raise ValueError('missing_direction')
        V=np.zeros((K,M,len(x)))
        for d in it['directions']:
            k=d['index']; distance=np.linalg.norm(np.asarray(manifest['anchors'][k])-x)
            check_close(distance,d['distance'],'distance')
            if d['skip_reason']:
                if d['skip_reason']=='coincident_anchor' and distance>=1e-8: raise ValueError('false_skip')
                if d['skip_reason']=='bounded_probe_zero' and (distance<1e-8 or d['actual_h']>1e-12 or it['mode_before']=='secant'): raise ValueError('false_probe_skip')
                if d['skip_reason'] not in ['coincident_anchor','bounded_probe_zero']: raise ValueError('unknown_skip')
                continue
            h=d['actual_h']; u=np.asarray(d['u'],float); source=ev[d['source_eval_id']]
            if h<=0: raise ValueError('invalid_h')
            check_close(u,(np.asarray(manifest['anchors'][k])-x)/distance,'unit')
            if it['mode_before']=='secant':
                check_close(h,distance,'secant_width'); check_close(source['x'],manifest['anchors'][k],'anchor_ref')
                if source['purpose']!='initialization_anchor' or source['anchor_id']!=manifest['anchor_ids'][k]: raise ValueError('wrong_anchor_event')
            else:
                check_close(source['x'],x+h*u,'probe_ref')
                if source['purpose']!='dynamic_probe' or source['iteration']!=it['t'] or source['anchor_id']!=manifest['anchor_ids'][k]: raise ValueError('wrong_probe_event')
            slopes=(np.asarray(source['individual_losses'])-before['individual_losses'])/h
            derivative=(source['objective_value']-before['objective_value'])/h
            check_close(derivative,d['original_derivative'],'derivative')
            check_close(a[k],d['direction_weight'],'direction_weight')
            V[k]=(a[k]*v*slopes)[:,None]*u
        H=V.sum(axis=1); F=V.sum(axis=0); g=np.asarray(it['g_solver'],float); norm=float(np.linalg.norm(g))
        max_g_error=max(max_g_error,check_close(H.sum(axis=0),g,'g_rebuild'))
        check_close(norm,it['g_norm'],'g_norm')
        accepted=[r for r in it['trials'] if r['accepted']]
        if len(accepted)>1: raise ValueError('multiple_accepted')
        for j,r in enumerate(it['trials']):
            e=ev[r['eval_id']]
            if e['purpose']!='trial' or e['iteration']!=it['t'] or e['attempt']!=j: raise ValueError('wrong_trial_event')
            if r['accepted']!=(e['objective_value']<before['objective_value']): raise ValueError('acceptance_changed')
            if r['accepted'] and j!=len(it['trials'])-1: raise ValueError('trial_after_acceptance')
        accepted_id=accepted[0]['eval_id'] if accepted else None
        if accepted_id!=it['accepted_eval_id']: raise ValueError('accepted_reference')
        gain=0.; inc=np.zeros(len(owners)); einc=np.zeros(M); share=None; P=None; p=None; po=None
        ownerH=np.zeros((len(owners),len(x)))
        for k,o in enumerate(manifest['anchor_owners']): ownerH[owner_index[o]]+=H[k]
        rho={}
        for label, vectors in [('anchor',H),('owner',ownerH),('model',F)]:
            denom=float(np.linalg.norm(vectors,axis=1).sum()); rho[label]=norm/denom if denom else None
        sensitive=any(z is not None and z<1e-6 for z in rho.values())
        if sensitive: sensitive_count+=1
        if norm>1e-8:
            q=g/norm; P=V@q; p=H@q; po=ownerH@q
            check_close(P.sum(axis=1),p,'P_rows'); check_close(P.sum(),norm,'P_total')
        elif accepted or it['trials']: raise ValueError('zero_direction_with_trials')
        if accepted:
            after=ev[accepted_id]; dx=np.asarray(after['x'])-x; length=float(np.linalg.norm(dx))
            gain=float(before['objective_value']-after['objective_value'])
            if gain<=0 or norm<=1e-8 or length<=0: raise ValueError('invalid_accepted_motion')
            max_motion_error=max(max_motion_error,check_close(dx,-length*g/norm,'motion'))
            check_close(length,accepted[0]['actual_bounded_step'],'actual_step')
            positive=np.maximum(po,0); support=float(positive.sum())
            if support<=0: raise ValueError('no_positive_support')
            share=positive/support; inc=gain*share
            einc=v*(np.asarray(before['individual_losses'])-after['individual_losses'])
            check_close(inc.sum(),gain,'A_conservation'); check_close(einc.sum(),gain,'E_conservation')
            if (inc<0).any(): raise ValueError('negative_support_allocation')
            A+=inc; E+=einc; total+=gain; accepted_count+=1
            if sensitive: sensitive_gain+=gain
            current_eid=accepted_id
        after_state=ev[current_eid]
        check_close(it['x_after'],after_state['x'],'post_point'); check_close(it['L_after'],after_state['objective_value'],'post_loss')
        details.append(dict(t=it['t'],accepted=bool(accepted),G=gain,P=P,p_anchor=p,p_owner=po,
                            allocation_share=share,A_increment_owner=inc,E_increment_model=einc,rho=rho,
                            supporting_projection=None if po is None else np.maximum(po,0),
                            opposing_projection=None if po is None else np.minimum(po,0),
                            gain_below_reporting_tolerance=bool(accepted and gain<1e-10),numerically_sensitive=sensitive))
    check_close(ev[current_eid]['x'],final['x'],'delivered_point')
    endpoint=float(initial['objective_value']-final['objective_value'])
    check_close(total,endpoint,'G_endpoint'); check_close(A.sum(),total,'A_total'); check_close(E.sum(),total,'E_total')
    check_close(E,v*(np.asarray(initial['individual_losses'])-final['individual_losses']),'E_endpoint')
    E_owner={o:sum(float(E[j]) for j,mo in enumerate(manifest['model_owners']) if mo==o) for o in all_owners}
    status='no_realized_gain' if accepted_count==0 else 'numerically_sensitive' if sensitive_count else 'available'
    return dict(method_version=manifest['method_version'],attribution_status=status,verification_status='not_run',
                anchor_owners=owners,model_ids=manifest['model_ids'],A_support_owner=A,E_model=E,E_owner=E_owner,
                owner_share=A/total if total>0 else None,G_total=total,G_endpoint=endpoint,
                residual=float(total-A.sum()),accepted_count=accepted_count,nonaccepted_count=len(iterations)-accepted_count,
                sensitive_rounds=sensitive_count,sensitive_gain=sensitive_gain,sensitive_gain_fraction=sensitive_gain/total if total else None,
                max_g_error=max_g_error,max_motion_error=max_motion_error,steps=details)
