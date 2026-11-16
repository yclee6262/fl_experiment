"""One-round direction ablation using the original acceptance/box rules."""
import math
import numpy as np


def owner_vectors(manifest,run,it):
    owners=list(dict.fromkeys(manifest['anchor_owners']));D=len(it['x_before']);H=np.zeros((len(owners),D))
    if len(it['directions'])!=len(manifest['anchors']):raise ValueError('missing_direction')
    before=run['evaluations'][it['current_eval_id']]['individual_losses']
    for d in it['directions']:
        if d['skip_reason']:continue
        k=d['index'];source=run['evaluations'][d['source_eval_id']]['individual_losses']
        slope=math.fsum(w*(b-a)/d['actual_h'] for w,a,b in zip(manifest['model_weights'],before,source))
        H[owners.index(manifest['anchor_owners'][k])]+=manifest['direction_weights'][k]*slope*np.asarray(d['u'])
    if not np.allclose(H.sum(axis=0),it['g_solver'],atol=1e-10,rtol=1e-10):raise ValueError('direction_reconstruction')
    return owners,H


def decide(server,objective,x,g,before,eta,mode='secant'):
    x=np.asarray(x,float);g=np.asarray(g,float)
    if x.shape!=g.shape or x.ndim!=1 or not np.isfinite(x).all() or not np.isfinite(g).all() or not np.isfinite(before) or not np.isfinite(eta) or eta<0 or mode not in ('secant','dynamic'):raise ValueError('invalid_state')
    norm=float(np.linalg.norm(g));mode_after=mode;eta_after=eta;stop=None;trials=[];end=x.copy();loss=before
    q=None
    if norm<=1e-8:stop='zero_direction'
    else:
        q=g/norm;current_eta=eta
        for attempt in range(10):
            step=server._bounded_step(x,-q,current_eta);z=x-step*q;value=float(objective(z))
            if not np.isfinite(value):raise ValueError('nonfinite_objective')
            accepted=value<before;trials.append(dict(attempt=attempt,x=z,loss=value,requested_eta=current_eta,actual_step=float(np.linalg.norm(z-x)),accepted=accepted))
            if accepted:end=z;loss=value;eta_after=min(.5,current_eta*1.5);break
            current_eta/=2
        if not any(r['accepted'] for r in trials):
            if mode=='secant':mode_after='dynamic';eta_after=.5
            else:stop='no_improvement'
    return dict(x=end,loss=float(loss),accepted=any(t['accepted'] for t in trials),trials=trials,q=q,g_norm=norm,
                mode_after=mode_after,eta_after=float(eta_after),stop=stop,mode_switch=mode_after!=mode)


def diagnose(server,objective,manifest,run,accounts,target):
    it=run['iterations'][0];owners,H=owner_vectors(manifest,run,it);i=owners.index(target)
    x=np.asarray(it['x_before']);g=np.asarray(it['g_solver']);before=it['L_before'];eta=it['eta_before']
    full=decide(server,objective,x,g,before,eta,it['mode_before'])
    if not np.allclose(full['x'],it['x_after'],atol=1e-10,rtol=0) or abs(full['loss']-it['L_after'])>1e-9:raise ValueError('original_round_mismatch')
    if full['mode_after']!=it['mode_after'] or abs(full['eta_after']-it['eta_after'])>1e-12:raise ValueError('original_transition_mismatch')
    alt=decide(server,objective,x,g-H[i],before,eta,it['mode_before'])
    norm=np.linalg.norm(g);p=None;B=None
    if norm>1e-8:
        ps=H@(g/norm);p=float(ps[i]);B=float(max(-p,0)/sum(abs(ps)))
    return dict(owner=target,H=H,g=g,p=p,B=B,A=float(accounts['steps'][0]['A_increment_owner'][i]),
                A_share=None if accounts['steps'][0]['allocation_share'] is None else float(accounts['steps'][0]['allocation_share'][i]),
                full=full,without=alt,C=float(alt['loss']-full['loss']),trial_difference=len(alt['trials'])-len(full['trials']),
                result_scope='conditional_one_round_direction_component_effect',payment=None)
