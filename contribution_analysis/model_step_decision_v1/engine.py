"""Frozen custom AST instrumentation: no numerical statements replaced."""
import ast
import copy
import functools
import inspect
import textwrap
import numpy as np

FIELDS = ('S_current','best_loss','global_best_S','global_best_loss','eta',
          'current_method','delta','history','states','anchor_losses')

class Trace:
    def __init__(self):
        self.pre=[]; self.post=[]; self.trials=[]; self.gradients=[]; self.final=None
    def capture(self, v, iteration, halted=False):
        return dict(**{k:copy.deepcopy(v[k]) for k in FIELDS},
                    next_iteration=int(iteration),halted=halted)
    def begin(self,v): self.pre.append(self.capture(v,v['iteration']))
    def finish(self,v): self.post.append(self.capture(v,v['iteration']+1))
    def end(self,v): self.final=self.capture(v,v['_next_iteration'],v['_halted'])
    def trial(self,v):
        self.trials.append(dict(iteration=v['iteration'],x=v['candidate_S'].copy(),
            loss=v['candidate_loss'],accepted=bool(v['candidate_loss']<v['best_loss'])))
    def gradient(self,v):
        self.gradients.append(dict(iteration=v['iteration'],g=v['gradient'].copy()))

@functools.lru_cache(None)
def compile_engine(original):
    tree=ast.parse(textwrap.dedent(inspect.getsource(original))); fn=tree.body[0]
    for name in ('_resume','_trace'):
        fn.args.kwonlyargs.append(ast.arg(arg=name));fn.args.kw_defaults.append(ast.Constant(None))
    idx=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='iteration')
    loop=fn.body[idx]; init=fn.body[:idx]; tail=fn.body[idx+1:]
    init+=ast.parse('_start_iteration=0\n_halted=False').body
    restore=ast.parse('candidates=np.asarray(candidates,dtype=float)\nn_candidates=len(candidates)\n'
        'direction_weights=np.asarray(direction_weights,dtype=float) / np.sum(direction_weights)\nmetric=history_metric or objective').body
    restore+=ast.parse('\n'.join(f'{name}=_resume[{name!r}]' for name in FIELDS)).body
    restore+=ast.parse('_start_iteration=_resume["next_iteration"]\n_halted=_resume["halted"]').body
    class Hooks(ast.NodeTransformer):
        def visit_Assign(self,node):
            names=[t.id for t in node.targets if isinstance(t,ast.Name)]
            if 'candidate_loss' in names:return [node,*ast.parse('_trace.trial(locals())').body]
            if 'gradient_norm' in names:return [node,*ast.parse('_trace.gradient(locals())').body]
            return node
        def visit_For(self,node):
            # Trial hooks apply inside the annealing loop; its break is not terminal.
            return self.generic_visit(node)
    loop=Hooks().visit(loop)
    class Halt(ast.NodeTransformer):
        def visit_For(self,node):return node
        def visit_Break(self,node):return [*ast.parse('_halted=True').body,node]
    loop.body=[n for item in loop.body for n in ((lambda x:x if isinstance(x,list) else [x])(Halt().visit(item)))]
    loop.iter=ast.parse('range(_start_iteration,_start_iteration+num_iterations)',mode='eval').body
    loop.body[:0]=ast.parse('_next_iteration=iteration+1\n_trace.begin(locals())').body
    loop.body+=ast.parse('_trace.finish(locals())').body
    fn.body=[ast.If(test=ast.Compare(left=ast.Name('_resume',ast.Load()),ops=[ast.Is()],comparators=[ast.Constant(None)]),body=init,orelse=restore)]
    fn.body+=ast.parse('_next_iteration=_start_iteration').body
    fn.body+=[ast.If(test=ast.UnaryOp(op=ast.Not(),operand=ast.Name('_halted',ast.Load())),body=[loop],orelse=[])]
    fn.body+=ast.parse('_trace.end(locals())').body+tail
    ast.fix_missing_locations(tree);ns=dict(original.__globals__)
    exec(compile(tree,'<model_step_frozen_custom>','exec'),ns)
    return ns[fn.name]

def run(server,objective,iterations=30,initial=None,resume=None):
    from host_server import HostServer
    tr=Trace()
    r=compile_engine(HostServer._solve_subspace_custom)(server,server.I_list,objective,
        num_iterations=iterations,direction_weights=server.alphas,initial_solution=initial,
        _resume=copy.deepcopy(resume),_trace=tr)
    return r,tr

def branch_state(pre,objective,anchors):
    new=copy.deepcopy(pre); x=np.asarray(pre['S_current']).copy()
    loss=float(objective(x))
    new.update(S_current=x,best_loss=loss,global_best_S=x.copy(),global_best_loss=loss,
               history=[loss],states=['Local Start'],anchor_losses=[float(objective(a)) for a in anchors],halted=False)
    return new
