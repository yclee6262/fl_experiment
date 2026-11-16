"""Resume the frozen custom engine with AST hooks; never edit production source."""
import ast
import copy
import functools
import inspect
import textwrap
import numpy as np

FIELDS = ('S_current','best_loss','global_best_S','global_best_loss','eta',
          'current_method','delta','history','states','anchor_losses')


class Trace:
    def __init__(self, count=None):
        self.pre = []
        self.post = []
        self.final = None
        self.count = count or (lambda:None)

    def capture(self, values, next_iteration, halted=False):
        return dict(**{k:copy.deepcopy(values[k]) for k in FIELDS},
                    next_iteration=int(next_iteration), halted=bool(halted),call_index=self.count())

    def begin(self, values):
        self.pre.append(self.capture(values, values['iteration']))

    def finish(self, values):
        self.post.append(self.capture(values, values['iteration']+1))

    def end(self, values):
        # An early break does not run finish; the halted state is terminal.
        self.final = self.capture(values, values['_next_iteration'], values['_halted'])


@functools.lru_cache(None)
def compile_engine(original):
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    fn = tree.body[0]
    for name in ('_resume','_trace'):
        fn.args.kwonlyargs.append(ast.arg(arg=name))
        fn.args.kw_defaults.append(ast.Constant(None))
    idx = next(i for i,n in enumerate(fn.body) if isinstance(n,ast.For)
               and isinstance(n.target,ast.Name) and n.target.id=='iteration')
    loop = fn.body[idx]
    # All pre-loop numerical initialization is untouched for a normal solve.
    init = fn.body[:idx]
    restore = ast.parse('\n'.join(f'{name} = _resume[{name!r}]' for name in FIELDS)).body
    restore += ast.parse('_start_iteration = _resume["next_iteration"]\n_halted = _resume["halted"]').body
    init += ast.parse('_start_iteration = 0\n_halted = False').body
    prefix = [ast.If(test=ast.Compare(left=ast.Name('_resume',ast.Load()),ops=[ast.Is()],comparators=[ast.Constant(None)]),
                     body=init,orelse=restore)]
    # Candidate/weight setup is needed for either path but does not call models.
    restore[:0] = ast.parse('candidates = np.asarray(candidates, dtype=float)\nn_candidates = len(candidates)\n'
        'direction_weights = np.ones(n_candidates, dtype=float) / n_candidates if direction_weights is None else '
        'np.asarray(direction_weights, dtype=float) / np.sum(direction_weights)\nmetric = history_metric or objective').body
    loop.iter = ast.parse('range(_start_iteration, _start_iteration + num_iterations)',mode='eval').body
    class Halt(ast.NodeTransformer):
        def visit_For(self,node): return node  # leave the inner annealing loop unchanged
        def visit_Break(self,node):
            return ast.parse('_halted = True').body + [node]
    loop.body = [n for item in loop.body for n in ((lambda x:x if isinstance(x,list) else [x])(Halt().visit(item)))]
    loop.body[:0] = ast.parse('_next_iteration = iteration + 1\n_trace.begin(locals())').body
    loop.body += ast.parse('_trace.finish(locals())').body
    prefix += ast.parse('_next_iteration = _start_iteration').body
    prefix.append(ast.If(test=ast.UnaryOp(op=ast.Not(),operand=ast.Name('_halted',ast.Load())),body=[loop],orelse=[]))
    # The exact original return remains, including its objective evaluation.
    prefix += ast.parse('_trace.end(locals())').body
    fn.body = prefix + fn.body[idx+1:]
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    exec(compile(tree,'<frozen_custom_resume>','exec'),namespace)
    return namespace[fn.name]


def run(server, objective, iterations, initial=None, resume=None, count=None):
    from host_server import HostServer
    trace=Trace(count)
    result=compile_engine(HostServer._solve_subspace_custom)(server,server.I_list,objective,
        num_iterations=iterations,direction_weights=server.alphas,
        initial_solution=initial,_resume=copy.deepcopy(resume),_trace=trace)
    return result,trace


def relocate(state, point, loss):
    new=copy.deepcopy(state)
    new['S_current']=np.asarray(point,dtype=float).copy()
    new['best_loss']=float(loss)
    new['global_best_S']=np.asarray(point,dtype=float).copy()
    new['global_best_loss']=float(loss)
    new['history'][-1]=float(loss)
    # This study only branches after an accepted (nonterminal) step.
    assert not new['halted']
    return new
