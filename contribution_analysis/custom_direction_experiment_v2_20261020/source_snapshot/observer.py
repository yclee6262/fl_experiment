"""Inject read-only hooks; original solver statements retain their order."""
import ast
import functools
import hashlib
import inspect
import textwrap
import time
import types
import numpy as np


@functools.lru_cache(None)
def instrument(original):
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    function=tree.body[0]
    function.args.kwonlyargs.append(ast.arg(arg='_observer'))
    function.args.kw_defaults.append(None)
    added=[]
    def hook(code):return ast.parse('_observer.'+code).body[0]
    class Hooks(ast.NodeTransformer):
        def visit_For(self,node):
            node=self.generic_visit(node)
            if isinstance(node.target,ast.Name) and node.target.id=='iteration':
                node.body.insert(0,hook('begin(iteration, S_current, current_method, best_loss)'));added.append('begin')
            return node
        def visit_AugAssign(self,node):
            if isinstance(node.target,ast.Name) and node.target.id=='gradient':
                added.append('direction')
                return [node,hook("direction(index, unit_direction, distance if current_method=='secant' else step, anchor if current_method=='secant' else S_current+step*unit_direction, derivative, direction_weights[index])")]
            return node
        def visit_Assign(self,node):
            if len(node.targets)!=1 or not isinstance(node.targets[0],ast.Name):return node
            name=node.targets[0].id
            if name=='gradient_norm':
                added.append('combined');return [node,hook('combined(gradient, gradient_norm)')]
            if name=='candidate_loss':
                added.append('trial');return [node,hook('trial(candidate_S, candidate_loss, best_loss)')]
            return node
    Hooks().visit(tree)
    if sorted(added)!=sorted(['begin','direction','combined','trial','trial']):raise ValueError('solver_hook_layout_changed')
    ast.fix_missing_locations(tree);namespace=dict(original.__globals__)
    exec(compile(tree,'<read_only_custom_observer>','exec'),namespace)
    return namespace[function.name]


def point_key(x):return np.asarray(x,dtype=np.float64).tobytes()


class Collector:
    def __init__(self):self.iterations=[];self.cache={};self.current=None
    def begin(self,number,x,mode,loss):
        self.current=dict(iteration=number,x=np.array(x).copy(),mode=mode,loss=float(loss),directions=[],trials=[])
        self.iterations.append(self.current)
    def direction(self,k,u,width,point,derivative,weight):
        self.current['directions'].append(dict(k=k,u=np.array(u).copy(),width=float(width),point=np.array(point).copy(),
                                             derivative=float(derivative),weight=float(weight)))
    def combined(self,g,norm):self.current.update(g=np.array(g).copy(),g_norm=float(norm))
    def trial(self,x,loss,previous):
        self.current['trials'].append(dict(x=np.array(x).copy(),loss=float(loss),accepted=bool(loss<previous),
                                          reply_losses=self.cache.get(point_key(x))))


def calculate(collector,alpha):
    w=np.array(alpha,dtype=float);w=w/w.sum();m=len(w)
    D=np.zeros(m);plus=np.zeros(m);minus=np.zeros(m);candidate_D=np.zeros(m)
    P=0.;sensitive=False;max_g_error=0.;max_dx_error=0.;max_projection_error=0.
    for it in collector.iterations:
        current=collector.cache[point_key(it['x'])];g=np.array(it['g']);norm=it['g_norm'];dim=len(g)
        joint=np.zeros((m,m,dim))
        for direction in it['directions']:
            reply=collector.cache[point_key(direction['point'])]
            slopes=(reply-current)/direction['width']
            joint[:,direction['k'],:]=(w*direction['weight']*slopes)[:,None]*direction['u']
            direction['individual_slopes']=slopes
            direction['reply_losses']=reply
        models=joint.sum(axis=1);candidates=joint.sum(axis=0)
        error=float(np.linalg.norm(models.sum(axis=0)-g));max_g_error=max(max_g_error,error)
        if error>1e-10*max(1.,norm):raise ValueError('direction_reconstruction')
        denominator=float(np.linalg.norm(models,axis=1).sum());rho=norm/denominator if denominator else None
        accepted=[a for a in it['trials'] if a['accepted']]
        if len(accepted)>1:raise ValueError('multiple_accepted_trials')
        it.update(current_losses=current,model_vectors=models,candidate_vectors=candidates,rho=rho,
                  reconstruction_error=error,accepted=bool(accepted))
        if not accepted:
            it.update(d=np.zeros(m),candidate_d=np.zeros(m),step_length=0.);continue
        dx=np.array(accepted[0]['x'])-it['x'];step=float(np.linalg.norm(dx))
        if norm<=1e-8 or step==0:raise ValueError('invalid_accepted_motion')
        parts=-step*models/norm;d=step*(models@g)/(norm*norm);cd=step*(candidates@g)/(norm*norm)
        dx_error=float(np.linalg.norm(parts.sum(axis=0)-dx));projection_error=abs(float(d.sum())-step)
        max_dx_error=max(max_dx_error,dx_error);max_projection_error=max(max_projection_error,projection_error)
        unstable=rho is None or rho<1e-6
        if (dx_error>1e-10*max(1.,step) or projection_error>1e-10*max(1.,step)) and not unstable:raise ValueError('motion_reconstruction')
        if unstable:sensitive=True
        D+=d;plus+=np.maximum(d,0);minus+=np.minimum(d,0);candidate_D+=cd;P+=step
        it.update(delta_x=dx,model_delta_x=parts,d=d,candidate_d=cd,step_length=step,
                  cancellation_amplification=float(abs(d).sum()/step),dx_reconstruction_error=dx_error)
    status='no_accepted_motion' if P==0 else 'numerically_sensitive' if sensitive else 'available'
    return dict(D=D,D_plus=plus,D_minus=minus,path_length=P,S_raw=D/P if P else None,
                S=D/P if status=='available' else None,candidate_S=candidate_D/P if status=='available' else None,
                status=status,n_iterations=len(collector.iterations),n_accepted=sum(i['accepted'] for i in collector.iterations),
                max_g_error=max_g_error,max_dx_error=max_dx_error,max_projection_error=max_projection_error)


def solve(server,record=False,initial_solution=None,max_samples=20000,max_seconds=60):
    from host_server import HostServer
    original=HostServer._solve_subspace_custom
    compiled=instrument(original) if record else None
    collector=Collector() if record else None
    totals=dict(calls=0,samples=0);digest=hashlib.sha256();replies=None
    originals=[];begin=time.perf_counter()
    for idx,agent in enumerate(server.trusted_agents):
        prior=agent.api_predict;originals.append((agent,prior))
        def prediction(x,_prior=prior,_idx=idx):
            nonlocal replies
            size=len(x) if np.asarray(x).ndim>1 else 1
            if totals['samples']+size>max_samples or time.perf_counter()-begin>max_seconds:raise RuntimeError('solve_budget')
            totals['calls']+=1;totals['samples']+=size
            digest.update(_idx.to_bytes(4,'little'));digest.update(np.asarray(x,dtype=np.float64).tobytes())
            y=_prior(x)
            if replies is not None:replies.append(float(y[0]))
            return y
        agent.api_predict=prediction
    previous=server._solve_subspace_custom
    if record:
        def observed(self,candidates,objective,*a,**kw):
            def objective_record(x):
                nonlocal replies
                replies=[]
                value=objective(x)
                if len(replies)!=len(self.trusted_agents):raise ValueError('missing_existing_replies')
                collector.cache[point_key(x)]=np.abs(np.array(replies)-self.target_T)
                replies=None
                return value
            return compiled(self,candidates,objective_record,*a,**kw,_observer=collector)
        server._solve_subspace_custom=types.MethodType(observed,server)
    try:
        result=server.optimize_candidate_subspace(strategy='custom',mode='optimization',custom_iterations=30,
                                                  initial_solution=np.array(initial_solution,dtype=float))
        solver_seconds=time.perf_counter()-begin
        start=time.perf_counter();scores=calculate(collector,server.alphas) if record else None
        attribution_seconds=time.perf_counter()-start
    finally:
        for agent,prior in originals:agent.api_predict=prior
        server._solve_subspace_custom=previous
    return dict(result=result,counter=totals,query_order_sha256=digest.hexdigest(),solver_seconds=solver_seconds,
                attribution_seconds=attribution_seconds,total_seconds=solver_seconds+attribution_seconds,
                scores=scores,iterations=collector.iterations if record else None,
                evaluator_loaded=False,removal_labels_loaded=False)
