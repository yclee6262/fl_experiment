"""Fixed full-panel evaluation and candidate-only interventions."""
import hashlib
import time
import numpy as np

def fixed_evaluate(agents,alpha,x):
    w=np.asarray(alpha,dtype=float);w=w/w.sum()
    if len(w)!=len(agents):raise ValueError('panel_weight_count')
    values=np.array([float(agent.api_predict(x)[0]) for agent in agents])
    if not np.isfinite(values).all():raise ValueError('nonfinite_full_panel_reply')
    return dict(predictions=values,individual_losses=abs(values),loss=float(w@abs(values)),calls=len(agents),samples=len(agents))

def anchor_only(server,case,k):
    start=time.perf_counter();ids=[a.agent_id for a in server.trusted_agents];w=np.array(server.alphas)
    if len(ids)!=len(case['ids']) or ids!=case['ids']:raise ValueError('full_panel_required')
    if not np.allclose(w,np.ones(len(w))/len(w),atol=1e-15,rtol=0):raise ValueError('nonuniform_direction_semantics_not_supported')
    candidates=np.delete(np.asarray(case['candidates'],dtype=float),k,axis=0)
    original=[(agent,agent.api_predict) for agent in server.trusted_agents];last={};counter=dict(calls=0,samples=0);digest=hashlib.sha256()
    for j,(agent,predict) in enumerate(original):
        def counted(x,_predict=predict,_j=j):
            n=len(x) if np.asarray(x).ndim>1 else 1
            if counter['samples']+n>20000 or time.perf_counter()-start>60:raise RuntimeError('case_resource_budget')
            counter['calls']+=1;counter['samples']+=n;digest.update(_j.to_bytes(4,'little'));digest.update(np.asarray(x,dtype=np.float64).tobytes())
            y=_predict(x);last[_j]=(np.asarray(x,dtype=float).copy(),float(y[0]));return y
        agent.api_predict=counted
    try:
        result=server.optimize_candidate_subspace(candidates=candidates,strategy='custom',mode='optimization',
            custom_iterations=30,initial_solution=np.asarray(case['fixed_start'],dtype=float))
    finally:
        for agent,predict in original:agent.api_predict=predict
    elapsed=time.perf_counter()-start
    if [a.agent_id for a in server.trusted_agents]!=ids or not np.array_equal(server.alphas,w):raise ValueError('panel_changed')
    if len(last)!=len(ids) or any(not np.array_equal(last[j][0],result['solution']) for j in last):raise ValueError('missing_final_existing_replies')
    values=np.array([last[j][1] for j in range(len(ids))]);loss=float((w/w.sum())@abs(values))
    if abs(loss-result['loss'])>1e-10:raise ValueError('fixed_objective_mismatch')
    return dict(result=result,counter=counter,seconds=elapsed,query_order_sha256=digest.hexdigest(),
        remaining_anchor_ids=[v for j,v in enumerate(ids) if j!=k],scoring_model_ids=ids,scoring_alphas=w,
        remaining_direction_weights=np.ones(len(ids)-1)/(len(ids)-1),fixed_start=case['fixed_start'],candidates=candidates,
        final_predictions=values,final_fixed_loss=loss,evaluator_loaded=False,truth_loaded=False)
