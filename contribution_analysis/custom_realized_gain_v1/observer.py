"""Read-only AST hooks. Strip hooks and assert the original AST is restored."""
import ast
import copy
import functools
import inspect
import textwrap
import time
import numpy as np
from .schema import plain, validate_manifest
from .objective_trace import ObjectiveTrace


def hook(code): return ast.parse('_observer.'+code).body[0]


@functools.lru_cache(None)
def instrument(original):
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    original_dump = ast.dump(tree, include_attributes=False)
    fn = tree.body[0]
    fn.args.kwonlyargs.append(ast.arg(arg='_observer')); fn.args.kw_defaults.append(None)
    layout = []
    class Transform(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id in ('objective', 'metric'):
                # Preserve the argument expression and its exact evaluation count.
                layout.append(node.func.id)
                return ast.Call(func=ast.Attribute(ast.Name('_observer', ast.Load()), 'evaluate', ast.Load()),
                                args=[ast.Constant(node.func.id), node.func, *node.args,
                                      ast.Call(ast.Name('locals', ast.Load()), [], [])], keywords=[])
            return node
        def visit_For(self, node):
            node = self.generic_visit(node)
            if isinstance(node.target, ast.Name) and node.target.id == 'iteration':
                node.body.insert(0, hook('begin(iteration, S_current, current_method, best_loss, eta, delta)'))
                node.body.append(hook('end(S_current, best_loss, current_method, eta, False)'))
            return node
        def visit_AugAssign(self, node):
            if isinstance(node.target, ast.Name) and node.target.id == 'gradient':
                return [node, hook('direction(index, unit_direction, distance, step if current_method != "secant" else distance, derivative, direction_weights[index])')]
            return node
        def visit_Assign(self, node):
            node = self.generic_visit(node)
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name): return node
            if node.targets[0].id == 'gradient_norm': return [node, hook('combined(gradient, gradient_norm)')]
            if node.targets[0].id == 'candidate_loss': return [node, hook('trial(candidate_S, candidate_loss, best_loss, current_eta)')]
            return node
        def visit_Continue(self, node): return [hook('skip(index, distance, step if current_method != "secant" and distance >= 1e-8 else None)'), node]
        def visit_Break(self, node): return [hook('end(S_current, best_loss, current_method, eta, True)'), node]
    Transform().visit(tree)
    if layout.count('objective') != 6 or layout.count('metric') != 2: raise ValueError('solver_hook_layout_changed:'+repr(layout))
    class Strip(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == '_observer': return None
            return self.generic_visit(node)
        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == '_observer':
                return ast.Call(func=node.args[1], args=[node.args[2]], keywords=[])
            return self.generic_visit(node)
    stripped = copy.deepcopy(tree)
    stripped.body[0].args.kwonlyargs.pop(); stripped.body[0].args.kw_defaults.pop()
    Strip().visit(stripped)
    if ast.dump(stripped, include_attributes=False) != original_dump: raise ValueError('non_hook_ast_changed')
    ast.fix_missing_locations(tree)
    env = dict(original.__globals__)
    exec(compile(tree, '<read_only_realized_gain_observer>', 'exec'), env)
    return env[fn.name]


class Observer:
    def __init__(self, trace, manifest):
        self.trace, self.manifest = trace, manifest
        self.iterations = []; self.current = None; self.anchor_evals = []
        self.initial_eval = self.current_eval = self.final_eval = None
        self.last_probe = None
        self.metric_events = []

    def evaluate(self, kind, fn, x, env):
        if kind == 'metric': purpose = 'history_metric'
        elif self.current is None:
            purpose = 'initialization_anchor' if 'candidate' in env else 'initialization_current'
        elif 'global_best_S' in env and env.get('gradient_norm', 0) is not None:
            # Identify objective calls using whether the trial/probe expression is
            # the original local object, not a coordinate cache.
            if x is env.get('candidate_S'): purpose = 'trial'
            elif self.current.get('closed'): purpose = 'final_return'
            else: purpose = 'dynamic_probe'
        else: purpose = 'final_return'
        if kind != 'metric' and self.current is None and len(self.anchor_evals) >= len(self.manifest['anchors']) and self.initial_eval is not None:
            purpose = 'final_return'  # zero-iteration solver
        context = dict(purpose=purpose, iteration=None if self.current is None else self.current['t'],
                       anchor_id=None, attempt=None)
        if purpose == 'initialization_anchor': context['anchor_id'] = self.manifest['anchor_ids'][len(self.anchor_evals)]
        if purpose == 'dynamic_probe': context['anchor_id'] = self.manifest['anchor_ids'][env['index']]
        if purpose == 'trial': context['attempt'] = len(self.current['trials'])
        old = self.trace.context; self.trace.context = context
        try: value = fn(x)
        finally: self.trace.context = old
        eid = self.trace.last_eval_id
        if kind == 'metric':
            self.metric_events.append(dict(x=np.array(x).copy(), value=float(value), objective_eval_id=eid if fn is self.trace else None))
        elif purpose == 'initialization_anchor': self.anchor_evals.append(eid)
        elif purpose == 'initialization_current': self.initial_eval = self.current_eval = eid
        elif purpose == 'dynamic_probe': self.last_probe = eid
        elif purpose == 'final_return': self.final_eval = eid
        return value

    def begin(self, t, x, mode, loss, eta, delta):
        if self.current is not None and not self.current.get('closed'): raise ValueError('unclosed_iteration')
        self.current = dict(t=t, mode_before=mode, x_before=np.array(x).copy(), current_eval_id=self.current_eval,
                            L_before=float(loss), eta_before=float(eta), delta=float(delta), directions=[], trials=[])
        self.iterations.append(self.current)

    def skip(self, index, distance, step):
        self.current['directions'].append(dict(index=index, skip_reason='coincident_anchor' if distance<1e-8 else 'bounded_probe_zero', distance=float(distance), actual_h=None if step is None else float(step)))

    def direction(self, index, u, distance, h, derivative, weight):
        eid = self.anchor_evals[index] if self.current['mode_before'] == 'secant' else self.last_probe
        self.current['directions'].append(dict(index=index, skip_reason=None, distance=float(distance), u=np.array(u).copy(), actual_h=float(h),
                                              source_eval_id=eid, original_derivative=float(derivative), direction_weight=float(weight)))

    def combined(self, g, norm): self.current.update(g_solver=np.array(g).copy(), g_norm=float(norm))

    def trial(self, x, loss, previous, eta):
        self.current['trials'].append(dict(eval_id=self.trace.last_eval_id, requested_eta=float(eta),
                                          actual_bounded_step=float(np.linalg.norm(np.asarray(x)-self.current['x_before'])),
                                          accepted=bool(loss<previous)))

    def end(self, x, loss, mode, eta, broken):
        c = self.current
        accepted = [a for a in c['trials'] if a['accepted']]
        if len(accepted)>1: raise ValueError('multiple_accepted')
        if accepted: self.current_eval = accepted[0]['eval_id']
        reason = ('zero_direction' if c['g_norm']<=1e-8 else 'no_improvement') if broken else None
        if not broken and c['t']+1 == self.manifest['solver_config']['num_iterations']: reason='iteration_budget'
        c.update(closed=True, mode_after=mode, eta_after=float(eta), x_after=np.array(x).copy(), L_after=float(loss),
                 accepted_eval_id=accepted[0]['eval_id'] if accepted else None, mode_switch=mode!=c['mode_before'], stop_reason=reason)


def record_custom_run(server, initial_solution, manifest, record=True, history_metric=None):
    validate_manifest(manifest)
    from host_server import HostServer
    trace = ObjectiveTrace(server, record, manifest['model_versions'])
    obs = Observer(trace, manifest) if record else None
    method = instrument(HostServer._solve_subspace_custom) if record else HostServer._solve_subspace_custom
    t = time.perf_counter()
    with trace:
        result = method(server, np.asarray(manifest['anchors']), trace,
                        direction_weights=np.asarray(manifest['direction_weights_raw']), initial_solution=np.asarray(initial_solution),
                        history_metric=history_metric, **manifest['solver_config'], **({'_observer':obs} if record else {}))
    elapsed = time.perf_counter()-t
    if record and (obs.initial_eval is None or obs.final_eval is None): raise ValueError('missing_endpoint_eval')
    return dict(result=plain(result), costs=dict(**trace.costs(), search_seconds=elapsed),
                evaluations=trace.evaluations, queries=trace.queries,
                iterations=[] if not record else obs.iterations,
                initial_eval_id=None if not record else obs.initial_eval,
                final_eval_id=None if not record else obs.final_eval,
                metric_events=[] if not record else obs.metric_events,
                execution_status='complete', instrumentation_ast_restored=True if record else None)
