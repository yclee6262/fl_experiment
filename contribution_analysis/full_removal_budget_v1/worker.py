"""Isolated prepare/solve/score workers; historical algorithms are untouched."""
import argparse
import hashlib
import inspect
import json
import signal
import sys
import time
import types
from pathlib import Path

import numpy as np


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def write(path, data):
    with Path(path).open('x') as f:
        json.dump(plain(data), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class BudgetStop(RuntimeError):
    pass


def classify_custom_stop(trace_data, outer_breaks):
    """An inner line-search break is not an outer solver termination."""
    line=trace_data.get('break_line')
    if line in outer_breaks:
        return outer_breaks[line]
    if trace_data.get('iteration',-1)+1 == trace_data.get('num_iterations'):
        return 'iteration_budget_exhausted'
    return 'unclassified_stop'


def alarm_handler(*_):
    raise BudgetStop('wall_time_limit')


def prepare(args):
    from p3_contribution_validation import build_parser, _base_args
    from alpha_weight_experiments import build_base_server, configure_controlled_coalition
    from research_evaluation import save_scenario_artifacts
    a = build_parser().parse_args(['--output-dir', str(args.directory)])
    a.scenario = args.scenario
    a.poison_ratio = 0.0 if args.scenario == 'clean' else 0.3
    a.constraint_mode = 'box_constrained'
    start = time.perf_counter()
    server = build_base_server(_base_args(a), args.seed, 2, a.poison_ratio, 0.0)
    if args.scenario != 'clean':
        configure_controlled_coalition(server, set(range(15, 21)), 5, 1)
    size = 3 if args.scenario == 'clean' else 5
    if len(server.trusted_agents) != size:
        raise ValueError(f'qualification_coalition_size:{len(server.trusted_agents)}!={size}')
    preparation_seconds = time.perf_counter() - start
    server.alphas = [1.0 / size] * size
    state = dict(seed=args.seed, scenario=args.scenario, target=0.0,
                 ids=[x.agent_id for x in server.trusted_agents],
                 alphas=server.alphas, candidates=server.I_list,
                 fixed_start=np.mean(server.I_list, axis=0),
                 preparation_costs=server.preparation_costs,
                 preparation_seconds=preparation_seconds,
                 phase1_report=server.phase1_report)
    write(args.directory / 'coalition.json', state)
    start = time.perf_counter()
    _, artifact, metrics = save_scenario_artifacts(
        args.directory, dict(seed=args.seed, dimension=2, poison_ratio=a.poison_ratio, target=0.0),
        server.all_agents, server.study_records, evaluator_epochs=100)
    state.update(artifact=str(artifact.relative_to(args.directory)),
                 evaluator_seconds=time.perf_counter()-start, evaluator_metrics=metrics)
    state['artifact_sha256'] = {str(p.relative_to(args.directory)): digest(p)
                                for p in artifact.iterdir() if p.is_file()}
    # Semantic data digest excludes ZIP timestamps, unlike raw NPZ file hashes.
    h = hashlib.sha256()
    for r in server.study_records:
        for name in ['X', 'clean_y', 'observed_y']:
            h.update(np.asarray(r[name]).tobytes())
    state['semantic_data_sha256'] = h.hexdigest()
    write(args.output, dict(status='complete', **state))


def load_server(args):
    import torch
    from host_server import HostServer
    from agent_client import AgentNode
    state = json.loads((args.directory / 'preparation.json').read_text())
    for path, expected in state['artifact_sha256'].items():
        # Solve workers intentionally never open the pooled evaluator file.
        if not path.endswith('local_models.pt'):
            continue
        if digest(args.directory/path) != expected:
            raise ValueError('local_model_hash_mismatch')
    artifact = args.directory/state['artifact']
    models = torch.load(artifact/'local_models.pt', map_location='cpu', weights_only=True)
    server = HostServer(0.0, 2, constraint_mode='box_constrained')
    for agent_id in state['ids']:
        agent = AgentNode(agent_id, None, n_features=2)
        agent.model.load_state_dict(models['agents'][agent_id])
        agent.model.eval()
        server.trusted_agents.append(agent)
    server.I_list = [np.array(v) for v in state['candidates']]
    server.alphas = list(state['alphas'])
    if args.excluded != -1:
        server.trusted_agents.pop(args.excluded)
        server.I_list.pop(args.excluded)
        server.alphas.pop(args.excluded)
        total = sum(server.alphas)
        server.alphas = [x / total for x in server.alphas]
    return server, state


def solve(args):
    import host_server
    server, state = load_server(args)
    start_x = np.array(state['fixed_start'])
    matrix = np.array(server.I_list).T
    beta = np.linalg.lstsq(matrix, start_x, rcond=None)[0]
    projection = matrix@beta
    geometry = dict(rank=int(np.linalg.matrix_rank(matrix)),
                    singular_values=np.linalg.svd(matrix, compute_uv=False),
                    beta0=beta, mapped_X0=projection, custom_X0=start_x,
                    projection_residual=float(np.linalg.norm(projection-start_x)))
    if geometry['projection_residual'] > 1e-8 or server.box_violation(projection)>1e-8:
        write(args.output, dict(status='semantic_qualification_failure', geometry=geometry))
        return
    calls = samples = 0
    begin = time.perf_counter()
    diagnostics = {}
    for agent in server.trusted_agents:
        original = agent.api_predict
        def counted(x, _original=original):
            nonlocal calls, samples
            n = len(x) if np.asarray(x).ndim > 1 else 1
            if calls >= args.calls or samples+n > args.samples:
                raise BudgetStop('api_budget_limit')
            if time.perf_counter()-begin >= args.seconds:
                raise BudgetStop('wall_time_limit')
            calls += 1
            samples += n
            return _original(x)
        agent.api_predict = counted
    # Observe scipy's actual start and return metadata without altering arguments.
    original_minimize = host_server.minimize
    def observed_minimize(fun, x0, *a, **kw):
        result = original_minimize(fun, x0, *a, **kw)
        diagnostics['scipy'] = dict(beta0=np.array(x0), actual_X0=matrix@x0,
                                    method=kw.get('method'), options=kw.get('options'),
                                    **{k: result.get(k) for k in ['nit','nfev','njev','status','success','message']})
        return result
    host_server.minimize = observed_minimize
    for engine, method in [('bfgs','_solve_subspace_bfgs'),('custom','_solve_subspace_custom')]:
        original = getattr(server, method)
        def observed(self, *a, _original=original, _engine=engine, **kw):
            before_calls, before_samples = calls, samples
            started = time.perf_counter()
            trace_data = {}
            source, line0 = inspect.getsourcelines(_original.__func__)
            lines = {line0+i: line.strip() for i,line in enumerate(source)}
            def trace(frame, event, arg):
                if frame.f_code is not _original.__func__.__code__:
                    return None
                if event == 'line' and lines.get(frame.f_lineno) == 'break':
                    trace_data['break_line'] = frame.f_lineno
                if event == 'return':
                    loc = frame.f_locals
                    for name in ['iteration','gradient_norm','current_method','success','num_iterations']:
                        if name in loc:
                            trace_data[name] = loc[name]
                return trace
            # Trace only the original engine frame, with no extra objective calls.
            sys.settrace(trace)
            try:
                result = _original(*a, **kw)
            finally:
                sys.settrace(None)
            metadata = dict(result)
            metadata.update(calls=calls-before_calls, input_samples=samples-before_samples,
                            seconds=time.perf_counter()-started)
            if _engine == 'custom':
                outer_lines=[line0+i for i,line in enumerate(source)
                             if line.strip()=='break' and len(line)-len(line.lstrip())<=20]
                outer_breaks=dict(zip(outer_lines,['direction_gradient_below_threshold','no_improving_step_after_search']))
                reason=classify_custom_stop(trace_data,outer_breaks)
                metadata.update(observed_trace=trace_data, stop_reason=reason,
                                iterations_attempted=trace_data.get('iteration', -1)+1)
            diagnostics[_engine] = metadata
            return result
        setattr(server, method, types.MethodType(observed, server))
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, args.seconds)
    try:
        result = server.optimize_candidate_subspace(strategy='best_of', mode='optimization',
                    custom_iterations=args.iterations, initial_solution=start_x)
        status = 'valid_delivery'
        error = None
    except Exception as exc:
        result = None
        status = 'budget_stop' if isinstance(exc, BudgetStop) else 'solver_failure'
        error = f'{type(exc).__name__}:{exc}'
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        host_server.minimize = original_minimize
    write(args.output, dict(status=status,error=error, seed=state['seed'],scenario=state['scenario'],
          excluded_index=args.excluded, excluded_agent_id=None if args.excluded==-1 else state['ids'][args.excluded],
          remaining_ids=[a.agent_id for a in server.trusted_agents], alphas=server.alphas,
          candidates=server.I_list, fixed_start=start_x, geometry=geometry,
          iterations=args.iterations, calls=calls,input_samples=samples,
          seconds=time.perf_counter()-begin, result=result, engines=diagnostics,
          coalition_sha256=digest(args.directory/'coalition.json'),
          local_models_sha256=digest(args.directory/state['artifact']/'local_models.pt'),
          evaluator_loaded=False))


def score(args):
    from research_evaluation import load_evaluator_artifact, score_solution
    from run_experiments import true_function
    state = json.loads((args.directory/'preparation.json').read_text())
    artifact = args.directory/state['artifact']
    for p,h in state['artifact_sha256'].items():
        if digest(args.directory/p) != h:
            raise ValueError(f'artifact_hash_mismatch:{p}')
    evaluator, _ = load_evaluator_artifact(artifact)
    rows = []
    for p in sorted((args.directory/'solves').glob('*.json')):
        data = json.loads(p.read_text())
        if data['status'] != 'valid_delivery':
            rows.append(dict(file=p.name,status=data['status']))
            continue
        x = np.array(data['result']['solution'])
        begin = time.perf_counter()
        pred, error = score_solution(evaluator,x,0.0)
        rows.append(dict(file=p.name,status='scored',pooled_prediction=pred,pooled_error=error,
                         truth_error=abs(true_function(x)),seconds=time.perf_counter()-begin))
    write(args.output,dict(status='complete',rows=rows,evaluator_sha256=digest(artifact/'global_evaluator.pt')))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['prepare','solve','score'])
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int)
    p.add_argument('--scenario',choices=['clean','controlled_poison'])
    p.add_argument('--excluded',type=int,default=-1)
    p.add_argument('--iterations',type=int,default=30)
    p.add_argument('--calls',type=int,default=20000)
    p.add_argument('--samples',type=int,default=1000000)
    p.add_argument('--seconds',type=float,default=60)
    args=p.parse_args()
    sys.path.insert(0,str(args.snapshot.resolve()))
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    try:
        globals()[args.mode](args)
    except Exception as exc:
        if not args.output.exists():
            write(args.output,dict(status='worker_failure',error=f'{type(exc).__name__}:{exc}'))
        raise


if __name__=='__main__':
    main()
