"""Read-only attribution; no evaluator, removal label, or truth input."""
import hashlib
import time
import numpy as np
from direction_observer import Collector, instrument, point_key


def quality(mse, variance):
    mse = np.asarray(mse, float)
    if not np.isfinite(variance) or variance <= 0 or not np.isfinite(mse).all() or (mse < 0).any():
        return None
    return variance / (variance + mse)


def attribute(iterations, alpha, q):
    w = np.asarray(alpha, float); w = w / w.sum(); m = len(w)
    A = np.zeros(m); B = np.zeros(m); Ap = A.copy(); An = A.copy(); Bp = A.copy(); Bn = A.copy()
    path_model = A.copy(); path_anchor = A.copy(); path = 0.; total = 0.; sensitive = False; steps = []
    max_g = 0.; max_motion = 0.
    for it in iterations:
        g = np.asarray(it['g'], float); norm = float(np.linalg.norm(g))
        rows = np.zeros((m, len(g))); cols = rows.copy()
        for d in it['directions']:
            slopes = (np.asarray(d['reply_losses']) - it['current_losses']) / d['width']
            v = (w * d['weight'] * slopes)[:, None] * np.asarray(d['u'])
            rows += v; cols[d['k']] += v.sum(axis=0)
        error = float(np.linalg.norm(rows.sum(axis=0) - g)); max_g = max(max_g, error)
        if error > 1e-10 * max(1., norm): raise ValueError('direction_reconstruction')
        accepted = [x for x in it['trials'] if x['accepted']]
        if len(accepted) > 1: raise ValueError('multiple_accepted')
        if not accepted: continue
        trial = accepted[0]; gain = float(it['loss'] - trial['loss'])
        dx = np.asarray(trial['x']) - it['x']; length = float(np.linalg.norm(dx))
        if gain <= 0 or norm <= 1e-8 or length <= 0: raise ValueError('invalid_accepted_step')
        motion = float(np.linalg.norm(dx + length * g / norm)); max_motion = max(max_motion, motion)
        if motion > 1e-10 * max(1., length): raise ValueError('noncollinear_motion')
        den_m = float(np.linalg.norm(rows, axis=1).sum()); den_a = float(np.linalg.norm(cols, axis=1).sum())
        rho_m = norm / den_m if den_m else 0.; rho_a = norm / den_a if den_a else 0.
        sensitive |= min(rho_m, rho_a) < 1e-6
        rm = rows @ g / norm**2; ra = cols @ g / norm**2
        a = gain * ra; b = gain * rm
        A += a; B += b; Ap += np.maximum(a, 0); An += np.minimum(a, 0)
        Bp += np.maximum(b, 0); Bn += np.minimum(b, 0)
        path_model += length * rm; path_anchor += length * ra; path += length; total += gain
        steps.append(dict(iteration=it['iteration'],gain=gain,anchor_increment=a,model_increment=b,
                          rho_anchor=rho_a,rho_model=rho_m))
    for vector in [A, B]:
        if abs(vector.sum() - total) > 1e-10 * max(1., abs(total)): raise ValueError('gain_conservation')
    status = 'no_accepted_motion' if not path else 'numerically_sensitive' if sensitive else 'available'
    return dict(A=A,B=B,M=None if q is None else np.asarray(q)*B,q=q,
                model_status='quality_unavailable' if q is None else status,
                status=status,A_plus=Ap,A_minus=An,B_plus=Bp,B_minus=Bn,
                G=total,path_length=path,old_model_S=path_model/path if path else None,
                old_anchor_S=path_anchor/path if path else None,
                max_g_error=max_g,max_motion_error=max_motion,steps=steps)


def run(server, start, record=False, q=None, role='baseline', excluded=None):
    """Keep candidate and objective weights separate, even for model-only removal."""
    from host_server import HostServer
    agents = list(server.trusted_agents); weights = np.asarray(server.alphas, float)
    anchors = np.asarray(server.I_list, float).copy(); a = weights / weights.sum()
    indices = list(range(len(agents)))
    if role == 'model': indices.remove(excluded)
    if role == 'anchor': anchors = np.delete(anchors, excluded, axis=0); a = np.delete(a, excluded); a /= a.sum()
    if role not in ['baseline','anchor','model']: raise ValueError('unknown_role')
    panel = [agents[k] for k in indices]; w = weights[indices]; w /= w.sum()
    counts = dict(calls=0,samples=0); per_agent = {str(x.agent_id):0 for x in panel}
    digest = hashlib.sha256(); originals = []; responses = None; collector = Collector() if record else None
    t0 = time.perf_counter()
    for j,agent in enumerate(panel):
        old = agent.api_predict; originals.append((agent,old))
        def counted(x,_old=old,_j=j,_id=str(agent.agent_id)):
            n = len(x) if np.asarray(x).ndim > 1 else 1
            if counts['samples'] + n > 20000 or time.perf_counter()-t0 > 60: raise RuntimeError('per_solve_budget')
            counts['calls'] += 1; counts['samples'] += n; per_agent[_id] += n
            digest.update(_j.to_bytes(4,'little')); digest.update(np.asarray(x,dtype=np.float64).tobytes())
            y = _old(x)
            if responses is not None: responses.append(float(y[0]))
            return y
        agent.api_predict = counted
    def objective(x):
        nonlocal responses
        if record: responses = []
        # Same normalized weighted loss implementation, explicit original panel.
        value = server._consensus_loss(x,agents=panel,mode='fixed',weights=w)
        if record:
            if len(responses) != len(panel): raise ValueError('missing_reply')
            collector.cache[point_key(x)] = np.abs(np.asarray(responses)-server.target_T)
            responses = None
        return value
    try:
        method = instrument(HostServer._solve_subspace_custom) if record else HostServer._solve_subspace_custom
        result = method(server,anchors,objective,num_iterations=30,use_annealing=True,allow_tangent=True,
                        direction_weights=a,initial_solution=np.asarray(start,float),
                        **({'_observer':collector} if record else {}))
        elapsed_solver = time.perf_counter()-t0
        t1=time.perf_counter()
        if record:
            # Add already available raw replies without recomputing predictions.
            for it in collector.iterations:
                it['current_losses']=collector.cache[point_key(it['x'])]
                for d in it['directions']: d['reply_losses']=collector.cache[point_key(d['point'])]
            scores=attribute(collector.iterations,w,q)
        else: scores=None
        elapsed_attribute=time.perf_counter()-t1
    finally:
        for agent,old in originals: agent.api_predict=old
    return dict(result=result,role=role,excluded_index=excluded,
                model_ids=[x.agent_id for x in panel],anchor_ids=[x.agent_id for k,x in enumerate(agents) if role!='anchor' or k!=excluded],
                model_weights=w,direction_weights=a,candidates=anchors,fixed_start=start,
                counter=counts,per_agent_samples=per_agent,query_order_sha256=digest.hexdigest(),
                solver_seconds=elapsed_solver,attribution_seconds=elapsed_attribute,
                total_seconds=elapsed_solver+elapsed_attribute,scores=scores,
                iterations=collector.iterations if record else None,evaluator_loaded=False,truth_loaded=False)
