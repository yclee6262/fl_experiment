"""Controlled output shifts and fixed objective/direction alpha ablations."""
import argparse
import copy
import gzip
import hashlib
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path
import time

import numpy as np
import scipy
import torch
from project_support.legacy import prepare
prepare()
from experiments.alpha.replay import load_base
from experiments.alpha.compare_initializations import METHODS
from experiments.alpha.methods.four_alpha_initialization import initializer
from experiments.alpha.methods.pure_es_initialization import initializer as pure_initializer
from experiments.alpha.methods.se_qu_initialization import initializer as pair_initializer
from experiments.baselines.paired_benchmark.training import invert_model
from experiments.baselines.paired_benchmark.search import PanelOracle, BudgetExhausted, run_search
from experiments.shared.solver_policy import apply_solver_policy

ROOT = Path(__file__).resolve().parents[3]
PLAN = Path(__file__).with_name('protocol.json')
REGISTRATION = ROOT / 'alpha_weight_outputs/se_qu_extension_v1_20261107/registration.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def dump(path, value):
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + '\n')


def jsonlines(path, values):
    path.write_text(''.join(json.dumps(plain(v), allow_nan=False) + '\n' for v in values))


def truth(point, target):
    x = np.asarray(point)
    return abs(float(x.sum() + x[0] * x[1]) - target)


class ShiftedModel(torch.nn.Module):
    """A differentiable float32 output shift, preserving the original model."""
    def __init__(self, base, offset):
        super().__init__()
        self.base = copy.deepcopy(base).cpu().eval()
        self.offset = float(offset)

    @property
    def fc1(self):
        return self.base.fc1

    def forward(self, x):
        return self.base(x) + self.offset


def custom_search(server, objective_weights, direction_weights, budget=200):
    local = copy.copy(server)
    local.alphas = np.asarray(objective_weights, float).tolist()
    direction = np.asarray(direction_weights, float)
    if direction.shape != (len(server.I_list),) or not np.isfinite(direction).all() or min(direction) < 0 or direction.sum() <= 0:
        raise ValueError('Invalid direction weights')
    oracle = PanelOracle(local, budget, 'custom')
    reason = 'solver_completed'
    message = ''
    success = False
    start = time.perf_counter()
    try:
        result = local._solve_subspace_custom(np.asarray(local.I_list), oracle, num_iterations=30,
            direction_weights=direction, history_metric=oracle)
        success = bool(result['success'])
        message = str(result['message'])
    except BudgetExhausted as exc:
        reason, message = 'budget_exhausted', str(exc)
    except Exception as exc:
        reason, message = 'solver_or_model_error', repr(exc)
    best = oracle.best
    return dict(method='custom', X=None if best is None else best['X'],
        loss=None if best is None else best['loss'], status='valid' if best else 'no_valid_endpoint',
        panel_budget=budget, panel_evaluations=oracle.panel_evaluations, model_queries=oracle.model_queries,
        success=success, stop_reason=reason, message=message, seconds=time.perf_counter()-start,
        events=oracle.events, delivery_rule='best_observed_finite_box_feasible_Lopt')


def enrich(row, events, target, quality):
    row = copy.deepcopy(row)
    row.update(truth_error=None, uniform_loss=None, fixed_Q_loss=None, endpoint_disagreement=None)
    if row['X'] is not None:
        match = next(e for e in events if e.get('counted') and e.get('loss') == row['loss'] and e['X'] == row['X'])
        residuals = abs(np.asarray(match['predictions']) - target)
        row.update(truth_error=truth(row['X'], target), uniform_loss=float(residuals.mean()),
            fixed_Q_loss=float(np.asarray(quality) @ residuals), endpoint_disagreement=float(np.std(match['predictions'])))
    return row


def run_condition(server, seed, target, regime, owner, source_artifact, baseline, blind_cost, output):
    output.mkdir()
    counter = Counter()
    for agent in server.trusted_agents:
        original = agent.api_predict
        def predict(x, original=original):
            v = np.asarray(x)
            counter['requests'] += 1
            counter['samples'] += 1 if v.ndim == 1 else len(v)
            return original(x)
        agent.api_predict = predict
    server.target_T = target
    inverses = [invert_model(a.model, target, seed * 100 + a.agent_id, starts=1, steps=500) for a in server.trusted_agents]
    if any(v['X'] is None for v in inverses):
        dump(output/'failed_inversions.json', inverses)
        raise ValueError('No finite inversion; failure preserved without replacement')
    server.I_list = [np.array(v['X']) for v in inverses]
    apply_solver_policy(server, 'float32_slsqp_v1')
    predictions = server.weight_roles_blind_predictions
    mse = ((predictions-server.test_y[:, None])**2).mean(0)
    peer = np.full((3, 3), np.nan)
    counter.clear()
    for j, agent in enumerate(server.trusted_agents):
        for i, x in enumerate(server.I_list):
            if i != j:
                peer[j, i] = agent.api_predict(x)[0]
    support_cost = dict(counter)
    alphas, audits, initcosts = {}, {}, {}
    for label, method in METHODS.items():
        factory = pure_initializer if label in ['E', 'S'] else pair_initializer if label in ['SE', 'QU'] else initializer
        counter.clear()
        alpha, audit = factory(peer)(method, server.trusted_agents, mse, server.test_X, server.test_y,
            server.I_list, target, cached_blind_predictions=predictions, bootstrap_seed=seed, bootstrap_replicates=1000)
        alphas[label], audits[label], initcosts[label] = alpha.tolist(), audit, dict(counter)
    meta = dict(seed=seed, target=target, regime=regime, owner_index=owner,
        owner_id=server.trusted_agents[owner].agent_id, source_artifact=str(source_artifact.relative_to(ROOT)),
        offsets=[a.model.offset for a in server.trusted_agents], blind_X=server.test_X, blind_y=server.test_y,
        baseline_blind_predictions=baseline, blind_predictions=predictions, mse=mse, peer_predictions=peer,
        anchors=server.I_list, inverses=inverses, alpha=alphas, alpha_audit=audits, initialization_costs=initcosts,
        support_cost=support_cost, blind_cost_charged=blind_cost if target == -.5 else dict(requests=0, samples=0),
        blind_cost_scope='same regime blind panel shared across two targets',
        blind_disagreement=float(np.std(predictions, axis=1).mean()),
        blind_mean_signed_residual=float(np.mean(predictions-server.test_y[:, None])),
        linear_rank=int(np.linalg.matrix_rank(server.I_list)),
        affine_rank=int(np.linalg.matrix_rank(np.asarray(server.I_list)[1:]-server.I_list[0])))
    dump(output/'case.json', meta)
    searches, traces = {}, {}
    for label in METHODS:
        key = 'coupled_' + label
        server.alphas = alphas[label]
        counter.clear()
        result = run_search(server, 'best_of', 400)
        assert result['model_queries'] == counter['requests'] == counter['samples']
        result.update(search_id=key, label=label, objective_alpha=alphas[label], direction_alpha=alphas[label])
        traces[key] = result.pop('events')
        searches[key] = result
    for label in ['Q', 'EU']:
        for cell, obj, direction in [('AU', label, 'U'), ('UA', 'U', label)]:
            key = label + '_' + cell
            counter.clear()
            result = custom_search(server, alphas[obj], alphas[direction])
            assert result['model_queries'] == counter['requests'] == counter['samples']
            result.update(search_id=key, label=label, objective_alpha=alphas[obj], direction_alpha=alphas[direction])
            traces[key] = result.pop('events')
            searches[key] = result
    for key, row in searches.items():
        path = output/(key + '.json.gz')
        with gzip.open(path, 'wt') as f:
            json.dump(plain(traces[key]), f, allow_nan=False)
        row.update(trace_path=path.name, trace_sha256=sha(path), truth_error=None)
    jsonlines(output/'decisions.jsonl', searches.values())
    # All 14 physical searches are sealed before any truth quality is computed.
    results = [enrich(row, traces[key], target, alphas['Q']) for key, row in searches.items()]
    jsonlines(output/'searches.jsonl', results)
    factorial = []
    for label in ['Q', 'EU']:
        for cell in ['UU', 'AU', 'UA', 'AA']:
            key = 'coupled_U' if cell == 'UU' else 'coupled_'+label if cell == 'AA' else label+'_'+cell
            if cell in ['UU', 'AA']:
                result = copy.deepcopy(next(r for r in searches[key]['engine_details'] if r['method'] == 'custom'))
            else:
                result = copy.deepcopy(searches[key])
            obj = label if cell[0] == 'A' else 'U'
            direction = label if cell[1] == 'A' else 'U'
            result.update(label=label, cell=cell, source_search_id=key, objective_alpha=alphas[obj],
                          direction_alpha=alphas[direction], derived_from_physical_search=True)
            ee = [e for e in traces[key] if e['engine'] == 'custom']
            factorial.append(enrich(result, ee, target, alphas['Q']))
    jsonlines(output/'factorial.jsonl', factorial)
    dump(output/'completion.json', dict(status='complete', physical_searches=14, factorial_rows=8,
        files_sha256={f:sha(output/f) for f in ['case.json', 'decisions.jsonl', 'searches.jsonl', 'factorial.jsonl']}))


def source_cases(smoke):
    cases = []
    registration = json.loads(REGISTRATION.read_text())
    for batch, b in enumerate(registration['batches'][:3], 1):
        source = ROOT/f'alpha_weight_outputs/four_alpha_v1_20261014/formal_batch_{batch}/source'
        shared = source/'reference_clean/shared_costs.jsonl'
        index = {r['seed']:r for r in map(json.loads, shared.read_text().splitlines())}
        for seed in b['seeds']:
            artifact = source/'reference_clean'/index[seed]['artifact_path']
            cases.append((seed, source, artifact, shared))
    assert len(cases) == 90 and len({r[0] for r in cases}) == 90
    return cases[:1] if smoke else cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    cases = source_cases(args.smoke)
    runtime = [Path(__file__), PLAN, ROOT/'experiments/baselines/paired_benchmark/search.py',
        ROOT/'experiments/baselines/paired_benchmark/training.py', ROOT/'experiments/alpha/replay.py',
        ROOT/'experiments/alpha/compare_initializations.py', ROOT/'experiments/shared/solver_policy.py',
        ROOT/'experiments/shared/solver_policy_v1.json', ROOT/'core/models.py', ROOT/'core/agent_client.py',
        ROOT/'core/host_server.py', ROOT/'project_support/legacy.py', ROOT/'project_support/legacy_modules.json']
    runtime += sorted((ROOT/'experiments/alpha/methods').glob('*.py'))
    inputs = {REGISTRATION} | {f for _, _, artifact, shared in cases for f in [shared, artifact/'local_models.pt']}
    manifest = dict(status='running', protocol=json.loads(PLAN.read_text()), smoke=args.smoke,
        runtime_sha256={str(f.relative_to(ROOT)):sha(f) for f in runtime},
        input_sha256={str(f.relative_to(ROOT)):sha(f) for f in sorted(inputs)}, completed=0,
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        versions=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__, scipy=scipy.__version__))
    dump(args.output_dir/'manifest.json', manifest)
    try:
        for position, (seed, source, artifact, _) in enumerate(cases):
            base, _, loaded_artifact = load_base(source, 'reference_clean', seed)
            assert loaded_artifact == artifact and [a.agent_id for a in base.trusted_agents] == [1, 2, 20]
            baseline = np.column_stack([a.api_predict(base.test_X) for a in base.trusted_agents])
            for regime in manifest['protocol']['regimes']:
                server = copy.deepcopy(base)
                offsets = [0.5 if regime == 'common_shift' or regime == 'single_shift' and j == position % 3 else 0. for j in range(3)]
                for agent, offset in zip(server.trusted_agents, offsets):
                    agent.model = ShiftedModel(agent.model, offset)
                predictions = np.column_stack([a.api_predict(server.test_X) for a in server.trusted_agents])
                assert np.max(abs(predictions - baseline - offsets)) <= 2e-6
                if regime == 'common_shift':
                    assert np.max(abs(np.diff(predictions, axis=1) - np.diff(baseline, axis=1))) <= 2e-6
                server.weight_roles_blind_predictions = predictions
                for target in [-.5, .5]:
                    directory = args.output_dir/f'seed{seed}_{regime}_T{target:g}'
                    run_condition(copy.deepcopy(server), seed, target, regime, position % 3,
                        artifact, baseline, dict(requests=3, samples=3*len(server.test_X)), directory)
                    manifest['completed'] += 1
                print(seed, regime, 'complete', flush=True)
        # Baseline fixed-input measurements precede each regime; counted separately from regime panels.
        manifest['baseline_diagnostic_queries'] = dict(requests=3*len(cases), samples=3*len(cases)*20)
        for name, value in {**manifest['runtime_sha256'], **manifest['input_sha256']}.items():
            assert sha(ROOT/name) == value
        manifest['status'] = 'complete'
    except BaseException as exc:
        manifest.update(status='interrupted', exception=repr(exc))
        raise
    finally:
        dump(args.output_dir/'manifest.json', manifest)


if __name__ == '__main__':
    main()
