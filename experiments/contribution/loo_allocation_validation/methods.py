"""Read-only streaming hooks and frozen numerical solver adapters."""
import ast
import copy
import inspect
import textwrap
import importlib.util
from pathlib import Path
import numpy as np
from .allocation import panel_weights

FIELDS = ('S_current', 'best_loss', 'global_best_S', 'global_best_loss', 'eta',
          'current_method', 'delta', 'history', 'states', 'anchor_losses')


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class Observer:
    """O(nd) method arrays, one compact resume state; no all-round trace."""
    def __init__(self, n, d, mode):
        self.mode = mode; self.h = np.zeros((n, d)); self.total = np.zeros((n, d))
        self.last = None; self.accepted = 0; self.max_error = 0.; self.array_bytes = self.h.nbytes+self.total.nbytes

    def begin(self, v):
        self.x = v['S_current'].copy(); self.h.fill(0)
        if self.mode == 'LAST_J':
            # branch_state resets history, states and anchor losses; only current numerical state persists.
            self.pre = {k: copy.deepcopy(v[k]) for k in FIELDS if k not in ('history','states')}
            self.pre.update(history=[], states=[], next_iteration=v['iteration'], halted=False)

    def direction(self, v):
        self.h[v['index']] = v['direction_weights'][v['index']]*v['derivative']*v['unit_direction']

    def gradient(self, v):
        self.norm = v['gradient_norm']
        np.testing.assert_allclose(self.h.sum(0), v['gradient'], atol=1e-9, rtol=0)

    def trial(self, v):
        if v['candidate_loss'] < v['best_loss']:
            self.accepted += 1
            if self.mode == 'LAST_J':
                self.last = self.pre
            else:
                dx = v['candidate_S']-self.x
                parts = -np.linalg.norm(dx)*self.h/self.norm
                error = float(np.linalg.norm(parts.sum(0)-dx)); self.max_error=max(self.max_error,error)
                if error > 1e-9: raise ValueError('Accepted displacement reconstruction failed')
                self.total += parts


def observed_solver(original):
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    fn = tree.body[0]; fn.args.kwonlyargs.append(ast.arg(arg='_observer')); fn.args.kw_defaults.append(ast.Constant(None))
    def hook(name): return ast.parse(f'_observer.{name}(locals())').body[0]
    class Hooks(ast.NodeTransformer):
        def visit_For(self, node):
            node = self.generic_visit(node)
            if isinstance(node.target, ast.Name) and node.target.id == 'iteration': node.body.insert(0, hook('begin'))
            return node
        def visit_AugAssign(self, node):
            return [node, hook('direction')] if isinstance(node.target, ast.Name) and node.target.id=='gradient' else node
        def visit_Assign(self, node):
            names=[t.id for t in node.targets if isinstance(t,ast.Name)]
            if 'gradient_norm' in names: return [node,hook('gradient')]
            if 'candidate_loss' in names: return [node,hook('trial')]
            return node
    tree=Hooks().visit(tree);ast.fix_missing_locations(tree);ns=dict(original.__globals__)
    exec(compile(tree,'<loo-allocation-streaming>','exec'),ns)
    return ns[fn.name]


class Queries:
    def __init__(self, server, budget, version):
        self.s=server;self.budget=budget;self.version=version;self.cache={};self.events=[];self.stage='baseline';self.owner=None
    def reply(self,i,x):
        key=f'{self.version}:{i}:'+np.asarray(x,np.float64).tobytes().hex()
        hit=self.stage!='baseline' and key in self.cache
        if not hit:
            self.budget.charge()
            self.cache[key]=float(self.s.trusted_agents[i].api_predict(np.asarray(x))[0])
        y=self.cache[key]
        self.events.append(dict(key=key,stage=self.stage,owner=self.owner,model=i,hit=hit,y=y))
        return abs(y-self.s.target_T)
    def objective(self, owner=None):
        w=panel_weights(self.s.alphas,owner);ids=np.flatnonzero(w)
        return lambda x:float(np.dot(w[ids], [self.reply(int(i),x) for i in ids]))


def execute(server, case, method, budget, version, solver, observed, engine, iterations=30):
    import time
    tic=time.perf_counter();q=Queries(server,budget,version);n=len(case['ids']);obs=None
    kwargs=dict(num_iterations=iterations,direction_weights=server.alphas,initial_solution=np.array(case['fixed_start']))
    if method=='FULL_LOO':base=solver(server,server.I_list,q.objective(),**kwargs)
    else:
        obs=Observer(n,server.n_features,method)
        base=observed(server,server.I_list,q.objective(),**kwargs,_observer=obs)
    q.stage='generation';raw=[];removal_history_lengths=[]
    if method=='CUM_A':
        if np.linalg.norm(obs.total.sum(0)-(base['solution']-case['fixed_start']))>1e-9:raise ValueError('Total reconstruction')
        raw=np.asarray(base['solution'])-obs.total
    elif method=='LAST_J' and obs.last is None:raw=None
    else:
        for i in range(n):
            q.owner=i;panel=[j for j in range(n) if j!=i];b=copy.copy(server)
            b.I_list=server.I_list[panel];b.alphas=panel_weights(server.alphas,i)[panel];f=q.objective(i)
            if method=='FULL_LOO':r=solver(b,b.I_list,f,num_iterations=iterations,direction_weights=b.alphas,initial_solution=np.array(case['fixed_start']))
            else:
                state=engine.branch_state(obs.last,f,b.I_list);tr=engine.Trace()
                r=engine.compile_engine(solver)(b,b.I_list,f,num_iterations=1,direction_weights=b.alphas,_resume=state,_trace=tr)
            raw.append(r['solution']);removal_history_lengths.append(len(r['history']))
        raw=np.asarray(raw)
    x=None if raw is None else np.clip(raw,-1,1)
    # Lock all endpoints before common scoring; removed owners never participate in generation.
    q.stage='peer';peer=[]
    if x is not None:
        for i in range(n):
            q.owner=i;f=q.objective(i);peer.append(f(x[i])-f(base['solution']))
    elapsed=time.perf_counter()-tic;primary_events=len(q.events)
    q.stage='full_supplement';full=[];tic2=time.perf_counter()
    if x is not None:
        for i in range(n):
            q.owner=i;f=q.objective();full.append(f(x[i])-f(base['solution']))
    supplementary_seconds=time.perf_counter()-tic2
    return dict(method=method,base=base,removal_history_lengths=removal_history_lengths,raw=raw,x=x,peer=None if x is None else peer,full=None if x is None else full,
                seconds=elapsed,supplementary_seconds=supplementary_seconds,events=q.events,
                primary_events=primary_events,observer_array_bytes=0 if obs is None else obs.array_bytes,
                max_reconstruction_error=None if obs is None else obs.max_error,
                last_iteration=None if obs is None or obs.last is None else obs.last['next_iteration'],
                repairs=None if x is None else np.linalg.norm(x-raw,axis=1))
