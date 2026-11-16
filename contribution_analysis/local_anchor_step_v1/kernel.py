"""Fixed-trajectory single-step anchor ablation. No pooled/LOO inputs."""
import time
import numpy as np

def key(x):return np.asarray(x,dtype=np.float32).tobytes()

def alternative(x,actual_next,g,component,anchor_weight):
    if not 0<=anchor_weight<1:return dict(status='no_remaining_anchor',x=None)
    minus=(np.asarray(g)-component)/(1-anchor_weight)
    norm=float(np.linalg.norm(minus));x=np.asarray(x,dtype=float)
    if norm<=1e-8:return dict(status='no_motion',x=x.copy(),g_minus=minus)
    length=float(np.linalg.norm(np.asarray(actual_next)-x))
    trial=x-length*minus/norm
    status='infeasible_equal_step' if np.max(np.abs(trial))>1+1e-8 else 'trial'
    return dict(status=status,x=trial,g_minus=minus)

def cache_for(it,m):
    cache={}
    def put(x,losses):
        if losses is None:raise ValueError('missing_existing_replies')
        v=np.asarray(losses,dtype=float)
        if v.shape!=(m,) or not np.isfinite(v).all():raise ValueError('invalid_existing_replies')
        k=key(x)
        if k in cache and np.max(abs(cache[k]-v))>1e-10:raise ValueError('conflicting_float32_cache')
        cache[k]=v
    put(it['x'],it['current_losses'])
    for d in it['directions']:put(d['point'],d['reply_losses'])
    for t in it['trials']:put(t['x'],t['reply_losses'])
    return cache

def score(case,baseline,agents):
    begin=time.perf_counter();m=len(case['ids']);w=np.asarray(case['alphas'],dtype=float);w/=w.sum()
    if len(agents)!=m:raise ValueError('agent_count')
    raw=np.zeros(m);positive=np.zeros(m);negative=np.zeros(m);missing=np.zeros(m,dtype=int)
    records=[];counter=dict(calls=0,samples=0,cache_hits=0,no_motion=0);query_seconds=0.;accepted_steps=0
    for it in baseline['iterations']:
        accepted=[t for t in it['trials'] if t['accepted']]
        if not accepted:continue
        if len(accepted)!=1:raise ValueError('accepted_branch_count')
        accepted_steps+=1;cache=cache_for(it,m);before=float(it['loss']);after=float(accepted[0]['loss'])
        if abs(float(w@np.asarray(it['current_losses']))-before)>1e-10:raise ValueError('current_loss_reconstruction')
        if abs(float(w@np.asarray(accepted[0]['reply_losses']))-after)>1e-10:raise ValueError('accepted_loss_reconstruction')
        if not after<before:raise ValueError('original_step_not_improving')
        components={d['k']:d['weight']*d['derivative']*np.asarray(d['u']) for d in it['directions']}
        for k,agent_id in enumerate(case['ids']):
            alt=alternative(it['x'],accepted[0]['x'],it['g'],components.get(k,np.zeros(len(it['g']))),float(w[k]))
            row=dict(iteration=it['iteration'],mode=it['mode'],anchor_index=k,agent_id=agent_id,
                     current_x=it['x'],actual_next=accepted[0]['x'],before_loss=before,actual_after_loss=after,
                     status=alt['status'],alternative_x=alt['x'],g_minus=alt.get('g_minus'))
            if alt['status'] not in {'trial','no_motion'}:
                missing[k]+=1;row.update(J=None);records.append(row);continue
            if alt['status']=='no_motion':
                losses=cache[key(it['x'])];source='no_motion';counter['no_motion']+=1;responses=None
            elif key(alt['x']) in cache:
                losses=cache[key(alt['x'])];source='cache';counter['cache_hits']+=1;responses=None
            else:
                source='new';start=time.perf_counter();values=[]
                for agent in agents:
                    y=np.asarray(agent.api_predict(alt['x']))
                    if y.shape!=(1,) or not np.isfinite(y).all():raise ValueError('invalid_new_reply')
                    values.append(float(y[0]));counter['calls']+=1;counter['samples']+=1
                query_seconds+=time.perf_counter()-start;responses=np.array(values);losses=abs(responses)
                cache[key(alt['x'])]=losses
            trial=float(w@losses);effective=min(before,trial);J=effective-after
            if J>before-after+1e-10:raise ValueError('single_step_upper_bound')
            raw[k]+=J;positive[k]+=max(J,0.);negative[k]+=min(J,0.)
            row.update(source=source,responses=responses,reply_losses=losses,trial_loss=trial,
                       alternative_accepted=bool(trial<before),effective_loss=effective,J=J)
            records.append(row)
    complete=bool(accepted_steps>0 and not missing.any())
    return dict(A=raw if complete else None,A_partial=raw,A_positive=positive,A_negative=negative,
                missing_steps=missing,status='available' if complete else 'no_accepted_steps' if not accepted_steps else 'incomplete_steps',
                n_accepted_steps=accepted_steps,comparisons=records,counter=counter,query_seconds=query_seconds,
                core_seconds=time.perf_counter()-begin,evaluator_loaded=False,LOO_labels_loaded=False)
