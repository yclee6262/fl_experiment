"""Observe existing Stage-3 API replies and accepted states; never add a query."""
import types
import time
import numpy as np

FEATURES=['relative_error','disagreement','own_error_change','candidate_alignment',
          'candidate_difference','coalition_size','final_loss_ratio','path_length']


def solve(server,record=False,history_mode='objective',initial_solution=None,
          max_calls=None,max_samples=None,max_seconds=None):
    """Same original best_of for control/record; no pooled model or labels here."""
    counter={'calls':0,'samples':0}
    started=time.perf_counter()
    current={'values':None}
    originals=[]
    for agent in server.trusted_agents:
        original=agent.api_predict
        originals.append((agent,original))
        def counted(x,_original=original):
            size=len(x) if np.asarray(x).ndim>1 else 1
            if max_calls is not None and counter['calls']>=max_calls:
                raise RuntimeError('model_call_budget')
            if max_samples is not None and counter['samples']+size>max_samples:
                raise RuntimeError('model_input_sample_budget')
            if max_seconds is not None and time.perf_counter()-started>=max_seconds:
                raise RuntimeError('solver_wall_time_budget')
            counter['calls']+=1
            counter['samples']+=size
            y=_original(x)
            if record and current['values'] is not None:
                current['values'].append(float(y[0]))
            return y
        agent.api_predict=counted
    engine_originals={}
    traces={}
    if record:
        for engine,name in [('bfgs','_solve_subspace_bfgs'),('custom','_solve_subspace_custom')]:
            original=getattr(server,name);engine_originals[name]=original
            def observed(self,candidates,objective,*a,_original=original,_engine=engine,**kw):
                states=[];cache={};infeasible_callbacks=0
                def add(x):
                    nonlocal infeasible_callbacks
                    x=np.asarray(x,dtype=float)
                    if self.box_violation(x)>1e-8:
                        infeasible_callbacks+=1;return
                    if states and np.array_equal(states[-1],x):return
                    states.append(x.copy())
                def key(x):return np.asarray(x,dtype=np.float32).tobytes()
                def observed_objective(x):
                    current['values']=[]
                    value=objective(x)
                    if len(current['values'])!=len(self.trusted_agents):
                        raise RuntimeError('incomplete_existing_API_replies')
                    cache[key(x)]=np.array(current['values'])
                    if not states:add(x)
                    return value
                previous_metric=kw.get('history_metric')
                def metric(x):
                    value=previous_metric(x) if previous_metric is not None else observed_objective(x)
                    add(x)
                    return value
                kw['history_metric']=metric
                try:
                    result=_original(candidates,observed_objective,*a,**kw)
                    # Both historical engines evaluate objective(final X) at return.
                    add(result['solution'])
                    if any(key(x) not in cache for x in states):
                        raise RuntimeError('accepted_state_missing_existing_API_reply')
                    traces[_engine]={'states':np.array(states),'responses':np.array([cache[key(x)] for x in states]),
                                     'cached_objective_points':len(cache),
                                     'infeasible_callbacks':infeasible_callbacks}
                    return result
                finally:
                    current['values']=None
            setattr(server,name,types.MethodType(observed,server))
    start=time.perf_counter()
    try:
        # Preserve the saved X0 exactly, including float32-vs-float64 provenance.
        x0=(np.mean(np.asarray(server.I_list),axis=0) if initial_solution is None
            else np.asarray(initial_solution,dtype=float).copy())
        result=server.optimize_candidate_subspace(strategy='best_of',custom_iterations=30,
                                                  mode='optimization',initial_solution=x0,
                                                  history_metric=server._true_target_error if history_mode=='diagnostic' else None)
        elapsed=time.perf_counter()-start
    finally:
        for agent,original in originals:agent.api_predict=original
        for name,original in engine_originals.items():setattr(server,name,original)
    return dict(result=result,counter=counter,seconds=elapsed,
                trace=traces.get(result['chosen_engine']) if record else None,
                fixed_start=x0)


def features(trace,candidates,alpha,lower=-1.,upper=1.,target=0.):
    x=np.asarray(trace['states'],dtype=float)
    r=np.asarray(trace['responses'],dtype=float)-target
    candidates=np.asarray(candidates,dtype=float);alpha=np.asarray(alpha,dtype=float)
    if len(x)==0 or r.shape!=(len(x),len(candidates)):
        raise ValueError('invalid_trajectory_shape')
    if not np.isfinite(x).all() or not np.isfinite(r).all():raise ValueError('nonfinite_trace')
    loss=np.abs(r);L=loss@alpha
    scale=1e-8+float(loss.mean())
    B=float(np.linalg.norm(np.full(x.shape[1],upper-lower)))
    relative=(loss-L[:,None]).mean(axis=0)/scale
    disagreement=np.abs(r-np.median(r,axis=1)[:,None]).mean(axis=0)/scale
    progress=(loss[0]-loss[-1])/scale
    alignment=(candidates-x[0])@(x[-1]-x[0])/B**2
    pair=np.linalg.norm(candidates[:,None,:]-candidates[None,:,:],axis=2)
    np.fill_diagonal(pair,np.inf)
    difference=pair.min(axis=1)/B
    m=len(candidates)
    shared=np.tile([m,L[-1]/scale,np.linalg.norm(np.diff(x,axis=0),axis=1).sum()/B],(m,1))
    z=np.column_stack([relative,disagreement,progress,alignment,difference,shared])
    if not np.isfinite(z).all():raise ValueError('nonfinite_features')
    return z
