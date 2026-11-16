"""Independent scalar reconstruction and original-solver replay of saved events.

Does not import accounting.py or use its intermediate vectors as reference.
"""
import math
import numpy as np
from .schema import key


def verify_run(manifest, run, accounts):
    from host_server import HostServer
    def near(x,y,label):
        aa,bb=np.asarray(x,float),np.asarray(y,float)
        if aa.shape!=bb.shape or not np.isfinite(aa).all() or not np.isfinite(bb).all(): raise ValueError('verify_'+label)
        error=float(np.max(abs(aa-bb))) if aa.size else 0.
        if not np.all(abs(aa-bb)<=1e-10+1e-10*np.maximum(abs(aa),abs(bb))): raise ValueError('verify_'+label)
        errors.append(error)
    errors=[]; ev=run['evaluations']; queries=run['queries']; iterations=run['iterations']
    if [e['eval_id'] for e in ev]!=list(range(len(ev))): raise ValueError('verify_eval_ids')
    if [q['query_id'] for q in queries]!=list(range(len(queries))): raise ValueError('verify_query_ids')
    if [it['t'] for it in iterations]!=list(range(len(iterations))): raise ValueError('verify_iteration_ids')
    v=manifest['model_weights']; a=manifest['direction_weights']; M=len(v); K=len(a); D=len(manifest['initial_solution'])
    used=[]; seen={}; sample_count=0
    for e in ev:
        if len(e['model_query_ids'])!=M: raise ValueError('verify_missing_query')
        raw=[]
        for m,qid in enumerate(e['model_query_ids']):
            q=queries[qid]; used.append(qid)
            if q['eval_id']!=e['eval_id'] or q['model_index']!=m or q['model_id']!=manifest['model_ids'][m] or not q['success']: raise ValueError('verify_query_link')
            if q['input_hex']!=key(q['x']) or q['input_shape']!=list(np.asarray(q['x']).shape): raise ValueError('verify_input_encoding')
            near(q['x'],e['x'],'query_point'); raw.append(abs(float(q['output'][0])-manifest['target']))
            ident=(m,q['input_hex']); reply=tuple(q['output'])
            if ident in seen and seen[ident]!=reply: raise ValueError('verify_stochastic')
            seen[ident]=reply;sample_count+=q['input_samples']
        near(raw,e['individual_losses'],'raw_losses')
        near(math.fsum(v[m]*raw[m] for m in range(M)),e['objective_value'],'objective')
    if used!=list(range(len(queries))) or sample_count!=run['costs']['input_samples']: raise ValueError('verify_query_coverage')
    owners=list(dict.fromkeys(manifest['anchor_owners'])); A=[0.]*len(owners); E=[0.]*M; G=[]
    current=run['initial_eval_id']
    for it,sc in zip(iterations,accounts['steps']):
        if it['current_eval_id']!=current or not it['closed']: raise ValueError('verify_chain')
        before=ev[current];x=it['x_before'];g=it['g_solver'];norm=math.sqrt(math.fsum(z*z for z in g))
        hvec=[[0.]*D for _ in range(K)]; fvec=[[0.]*D for _ in range(M)]
        if len(it['directions'])!=K: raise ValueError('verify_directions')
        cells=[[[0.]*D for _ in range(M)] for _ in range(K)]
        for k,d in enumerate(it['directions']):
            if d['index']!=k: raise ValueError('verify_direction_order')
            if d['skip_reason']: continue
            source=ev[d['source_eval_id']]
            # Source references, not most-recent-coordinate lookups.
            if it['mode_before']=='secant':
                if source['purpose']!='initialization_anchor' or source['anchor_id']!=manifest['anchor_ids'][k]: raise ValueError('verify_anchor_event')
            elif source['purpose']!='dynamic_probe' or source['iteration']!=it['t'] or source['anchor_id']!=manifest['anchor_ids'][k]: raise ValueError('verify_probe_event')
            for m in range(M):
                slope=(source['individual_losses'][m]-before['individual_losses'][m])/d['actual_h']
                for j in range(D): cells[k][m][j]=a[k]*v[m]*slope*d['u'][j]
            for j in range(D): hvec[k][j]=math.fsum(cells[k][m][j] for m in range(M))
        for m in range(M):
            for j in range(D): fvec[m][j]=math.fsum(cells[k][m][j] for k in range(K))
        near([math.fsum(hvec[k][j] for k in range(K)) for j in range(D)],g,'direction')
        accepted=[tr for tr in it['trials'] if tr['accepted']]
        if len(accepted)>1: raise ValueError('verify_multiple_accepted')
        for idx,tr in enumerate(it['trials']):
            e=ev[tr['eval_id']]
            if e['purpose']!='trial' or e['attempt']!=idx or e['iteration']!=it['t'] or tr['accepted']!=(e['objective_value']<before['objective_value']): raise ValueError('verify_trial')
        if norm>1e-8:
            proj=[math.fsum(hvec[k][j]*g[j]/norm for j in range(D)) for k in range(K)]
            op=[math.fsum(proj[k] for k in range(K) if manifest['anchor_owners'][k]==o) for o in owners]
            P=[[math.fsum(cells[k][m][j]*g[j]/norm for j in range(D)) for m in range(M)] for k in range(K)]
            near(proj,sc['p_anchor'],'projection');near(op,sc['p_owner'],'owner_projection');near(P,sc['P'],'joint_projection')
            for label,vec in [('anchor',hvec),('owner',[[math.fsum(hvec[k][j] for k in range(K) if manifest['anchor_owners'][k]==o) for j in range(D)] for o in owners]),('model',fvec)]:
                den=math.fsum(math.sqrt(math.fsum(z*z for z in row)) for row in vec)
                if den: near(norm/den,sc['rho'][label],'rho')
        if accepted:
            after=ev[accepted[0]['eval_id']]; gain=before['objective_value']-after['objective_value'];G.append(gain)
            if gain<=0: raise ValueError('verify_gain')
            den=math.fsum(max(z,0) for z in op); share=[max(z,0)/den for z in op]
            ai=[gain*z for z in share]; ei=[v[m]*(before['individual_losses'][m]-after['individual_losses'][m]) for m in range(M)]
            near(share,sc['allocation_share'],'share'); near(ai,sc['A_increment_owner'],'A_step');near(ei,sc['E_increment_model'],'E_step')
            near(sum(ai),gain,'A_step_conservation');near(sum(ei),gain,'E_step_conservation')
            A=[z+dz for z,dz in zip(A,ai)];E=[z+dz for z,dz in zip(E,ei)];current=after['eval_id']
        else:
            if sc['allocation_share'] is not None or sc['G']!=0: raise ValueError('verify_nonaccepted_allocation')
            near(sc['A_increment_owner'],[0.]*len(owners),'zero_A');near(sc['E_increment_model'],[0.]*M,'zero_E')
    if len(accounts['steps'])!=len(iterations): raise ValueError('verify_steps')
    near(A,accounts['A_support_owner'],'A_total');near(E,accounts['E_model'],'E_total')
    total=math.fsum(G);near(total,accounts['G_total'],'G_total')
    first,last=ev[run['initial_eval_id']],ev[run['final_eval_id']]
    near(total,first['objective_value']-last['objective_value'],'telescoping')
    near(E,[v[m]*(first['individual_losses'][m]-last['individual_losses'][m]) for m in range(M)],'E_endpoint')
    # Replay the UNINSTRUMENTED original solver using the exact recorded objective order.
    server=HostServer.__new__(HostServer);server.constraint_mode=manifest['constraint_mode']
    server.feasible_lower,server.feasible_upper=manifest['box'];server.feasibility_tolerance=1e-8
    pos=0; metpos=0
    def objective(x):
        nonlocal pos
        if pos>=len(ev):raise ValueError('verify_replay_extra_query')
        e=ev[pos];pos+=1
        if key(x)!=key(e['x']):raise ValueError('verify_replay_query_sequence')
        return e['objective_value']
    def metric(x):
        nonlocal metpos
        row=run['metric_events'][metpos];metpos+=1
        if key(x)!=key(row['x']):raise ValueError('verify_metric_sequence')
        return row['value']
    result=HostServer._solve_subspace_custom(server,np.asarray(manifest['anchors']),objective,
                 direction_weights=np.asarray(manifest['direction_weights_raw']),initial_solution=np.asarray(manifest['initial_solution']),
                 history_metric=metric if manifest['history_metric_kind']!='objective' else None,**manifest['solver_config'])
    if pos!=len(ev):raise ValueError('verify_replay_unused_queries')
    from .schema import plain
    if plain(result)!=run['result']:raise ValueError('verify_replay_result')
    # Recompute query digest independently, including original dtype/shape.
    import hashlib
    digest=hashlib.sha256()
    for q in queries:
        arr=np.asarray(q['x'],dtype=q['input_dtype']); yy=np.asarray(q['output'],dtype=np.float64)
        payload=f"{q['model_index']}:{arr.dtype}:{arr.shape}:".encode()+arr.tobytes()+yy.tobytes()
        # Original API may return float32; preserve its explicit output dtype.
        yy=np.asarray(q['output'],dtype=q.get('output_dtype','float64'))
        payload=f"{q['model_index']}:{arr.dtype}:{arr.shape}:".encode()+arr.tobytes()+yy.tobytes()
        digest.update(len(payload).to_bytes(8,'little'));digest.update(payload)
    if digest.hexdigest()!=run['costs']['query_sha256']: raise ValueError('verify_digest')
    return dict(accepted=True,verification_status='passed',max_absolute_error=max(errors,default=0),
                evaluations=len(ev),queries=len(queries),iterations=len(iterations),accepted_steps=len(G),
                independent_scalar_reconstruction=True,original_solver_replay=True,new_model_predictions=0,
                pooled_loaded=False,truth_loaded=False)
