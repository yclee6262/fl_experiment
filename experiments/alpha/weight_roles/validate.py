"""Independent model reload, sealed decisions, target, factorial and cost audit."""
import argparse
import csv
import gzip
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from core.models import PretrainedModel

ROOT = Path(__file__).resolve().parents[3]
LABELS = ['U', 'Q', 'E', 'S', 'EU', 'EQ', 'SU', 'SQ', 'SE', 'QU']


def read(path): return json.loads(path.read_text())
def lines(path): return [json.loads(s) for s in path.read_text().splitlines()]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(nets, point, offsets):
    x = torch.tensor(np.asarray(point), dtype=torch.float32)
    if x.ndim == 1:
        x = x[None, :]
    with torch.no_grad():
        return np.column_stack([(net(x) + float(offset)).numpy().reshape(-1)
                                for net, offset in zip(nets, offsets)])


def truth(point, target):
    x = np.asarray(point, float)
    return abs(float(x[0] + x[1] + x[0]*x[1]) - target)


def validate(source, smoke=False):
    output = source/'acceptance.json'
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    manifest = read(source/'manifest.json')
    assert manifest['status'] == 'complete' and manifest['smoke'] == smoke
    for file, expected in {**manifest['runtime_sha256'], **manifest['input_sha256']}.items():
        assert sha(ROOT/file) == expected, file
    identities, source_models, starts, baseline_cache = set(), {}, {}, {}
    endpoints = inverse_checks = trace_events = study_queries = 0
    inverse_samples = init_samples = blind_samples = support_samples = 0
    errors, branch_stops = Counter(), Counter()
    for completed in sorted(source.glob('seed*/completion.json')):
        directory = completed.parent
        done, case = read(completed), read(directory/'case.json')
        for name, expected in done['files_sha256'].items():
            assert sha(directory/name) == expected
        seed, regime, target = case['seed'], case['regime'], case['target']
        identity = seed, regime, target
        assert identity not in identities
        identities.add(identity)
        assert case['linear_rank'] == np.linalg.matrix_rank(case['anchors'])
        assert case['affine_rank'] == np.linalg.matrix_rank(np.array(case['anchors'])[1:]-case['anchors'][0])
        artifact = case['source_artifact']
        if artifact not in source_models:
            payload = torch.load(ROOT/artifact/'local_models.pt', map_location='cpu', weights_only=True)
            nets = []
            for ident in [1, 2, 20]:
                net = PretrainedModel(2)
                net.load_state_dict(payload['agents'][ident])
                nets.append(net.eval())
            source_models[artifact] = nets
        nets = source_models[artifact]
        if seed not in baseline_cache:
            baseline_cache[seed] = evaluate(nets, case['blind_X'], [0, 0, 0])
        np.testing.assert_allclose(baseline_cache[seed], case['baseline_blind_predictions'], atol=1e-7, rtol=0)
        expected_owner = sorted({r[0] for r in source_seed_list()}).index(seed) % 3
        assert case['owner_index'] == expected_owner
        offsets = [.5 if regime == 'common_shift' or regime == 'single_shift' and j == expected_owner else 0. for j in range(3)]
        assert case['offsets'] == offsets
        predictions = evaluate(nets, case['blind_X'], offsets)
        np.testing.assert_allclose(predictions, case['blind_predictions'], atol=1e-7, rtol=0)
        assert np.max(abs(predictions - baseline_cache[seed] - offsets)) <= 2e-6
        if regime == 'common_shift':
            assert np.max(abs(np.diff(predictions, axis=1)-np.diff(baseline_cache[seed], axis=1))) <= 2e-6
        mse = ((predictions-np.array(case['blind_y'])[:, None])**2).mean(0)
        np.testing.assert_allclose(mse, case['mse'], atol=1e-12, rtol=0)
        q = 1/(mse+1e-5)
        q /= q.sum()
        np.testing.assert_allclose(case['alpha']['Q'], q, atol=1e-12, rtol=0)
        np.testing.assert_allclose(case['alpha']['U'], [1/3]*3, atol=1e-12, rtol=0)
        for alpha in case['alpha'].values():
            assert min(alpha) >= 0 and abs(sum(alpha)-1) < 1e-10
        assert set(case['alpha']) == set(LABELS)
        for j, inversion in enumerate(case['inverses']):
            np.testing.assert_array_equal(case['anchors'][j], inversion['X'])
            first = inversion['start_results'][0]['initial_X']
            key = seed, j
            if key in starts:
                assert first == starts[key]
            else:
                starts[key] = first
            rng = np.random.default_rng(seed*100+[1, 2, 20][j])
            np.testing.assert_array_equal(first, np.clip(rng.normal(size=(1, 2)), -1, 1).astype(np.float32).reshape(-1))
            prediction = evaluate([nets[j]], inversion['X'], [offsets[j]])[0, 0]
            assert abs(abs(float(prediction)-target)-inversion['loss']) < 1e-10
            inverse_checks += 1
            inverse_samples += inversion['forward_samples']
        peer = np.array(case['peer_predictions'], float)
        for i, x in enumerate(case['anchors']):
            preds = evaluate(nets, x, offsets)[0]
            for j in range(3):
                if i != j:
                    assert abs(preds[j]-peer[j, i]) < 1e-10
        init_samples += sum(c.get('samples', 0) for c in case['initialization_costs'].values())
        blind_samples += case['blind_cost_charged'].get('samples', 0)
        support_samples += case['support_cost'].get('samples', 0)
        decisions = {r['search_id']: r for r in lines(directory/'decisions.jsonl')}
        searches = {r['search_id']: r for r in lines(directory/'searches.jsonl')}
        factorial = lines(directory/'factorial.jsonl')
        expected_ids = {'coupled_'+label for label in LABELS} | {label+'_'+cell for label in ['Q', 'EU'] for cell in ['AU', 'UA']}
        assert set(searches) == set(decisions) == expected_ids
        assert len(factorial) == 8 and {(r['label'], r['cell']) for r in factorial} == set(itertools.product(['Q', 'EU'], ['UU', 'AU', 'UA', 'AA']))
        traces = {}
        for key, result in searches.items():
            if key.startswith('coupled_'):
                label = key.removeprefix('coupled_')
                np.testing.assert_array_equal(result['objective_alpha'], case['alpha'][label])
                np.testing.assert_array_equal(result['direction_alpha'], case['alpha'][label])
            assert decisions[key]['truth_error'] is None
            for field in decisions[key]:
                if field != 'truth_error':
                    assert decisions[key][field] == result[field]
            trace_path = directory/result['trace_path']
            assert sha(trace_path) == result['trace_sha256']
            with gzip.open(trace_path, 'rt') as f:
                events = json.load(f)
            traces[key] = events
            counted = [e for e in events if e['counted']]
            assert len(counted) == result['panel_evaluations']
            assert sum(e['model_queries'] for e in counted) == result['model_queries']
            assert all(sum(e['engine'] == engine for e in counted) <= 200 for engine in ['coefficient', 'custom'])
            a = np.array(result['objective_alpha'])
            a /= a.sum()
            for event in counted:
                if event['loss'] is not None:
                    assert abs(a@abs(np.array(event['predictions'])-target)-event['loss']) < 1e-12
            if result.get('engine_details'):
                for branch in result['engine_details']:
                    branch_stops[branch['stop_reason']] += 1
                    vv = [e for e in counted if e['engine'] == branch['method'] and e['loss'] is not None]
                    if branch['X'] is not None:
                        assert branch['loss'] == min(e['loss'] for e in vv)
            check_endpoint(result, counted, nets, offsets, target, case['alpha']['Q'])
            endpoints += int(result['X'] is not None)
            study_queries += result['model_queries']
            trace_events += len(events)
            errors[result['stop_reason']] += 1
        for cell in factorial:
            label, name = cell['label'], cell['cell']
            key = 'coupled_U' if name == 'UU' else 'coupled_'+label if name == 'AA' else label+'_'+name
            assert cell['source_search_id'] == key and cell['method'] == 'custom'
            np.testing.assert_array_equal(cell['objective_alpha'], case['alpha'][label if name[0] == 'A' else 'U'])
            np.testing.assert_array_equal(cell['direction_alpha'], case['alpha'][label if name[1] == 'A' else 'U'])
            original = next(r for r in searches[key]['engine_details'] if r['method'] == 'custom') if name in ['UU', 'AA'] else searches[key]
            for field in ['X', 'loss', 'status', 'model_queries', 'panel_evaluations']:
                assert cell[field] == original[field]
            events = [e for e in traces[key] if e.get('counted') and e['engine'] == 'custom']
            check_endpoint(cell, events, nets, offsets, target, case['alpha']['Q'])
            endpoints += int(cell['X'] is not None)
    seeds = [r[0] for r in source_seed_list()]
    if smoke:
        seeds = seeds[:1]
    assert identities == set(itertools.product(seeds, ['control', 'single_shift', 'common_shift'], [-.5, .5]))
    assert len(source_models) == len(seeds) and manifest['completed'] == len(identities)
    assert errors['solver_or_model_error'] == branch_stops['solver_or_model_error'] == 0
    report = dict(passed=True, smoke=smoke, conditions=len(identities), physical_searches=14*len(identities),
        factorial_rows=8*len(identities), independently_recomputed_endpoint_panels=endpoints,
        inverse_endpoint_checks=inverse_checks, paired_initial_points=len(starts), trace_events=trace_events,
        study_model_queries=study_queries, inversion_forward_samples=inverse_samples,
        initialization_samples=init_samples, shared_blind_samples=blind_samples, support_samples=support_samples,
        fixed_baseline_diagnostic_samples=manifest['baseline_diagnostic_queries']['samples'],
        stop_reasons=dict(errors), coupled_branch_stop_reasons=dict(branch_stops), validator_sha256=sha(Path(__file__)))
    dump = json.dumps(report, indent=2)+'\n'
    output.write_text(dump)
    print(dump)


def source_seed_list():
    reg = read(ROOT/'alpha_weight_outputs/se_qu_extension_v1_20261107/registration.json')
    return [(s, batch) for batch, b in enumerate(reg['batches'][:3], 1) for s in b['seeds']]


def check_endpoint(result, events, nets, offsets, target, quality):
    if result['X'] is None:
        assert result['status'] != 'valid'
        return
    assert result['status'] == 'valid' and max(abs(np.array(result['X']))) <= 1
    valid = [e for e in events if e['loss'] is not None]
    assert result['loss'] == min(e['loss'] for e in valid)
    prediction = evaluate(nets, result['X'], offsets)[0].astype(float)
    a = np.array(result['objective_alpha'])
    a /= a.sum()
    assert abs(a@abs(prediction-target)-result['loss']) < 1e-10
    assert abs(truth(result['X'], target)-result['truth_error']) < 1e-12
    assert abs(abs(prediction-target).mean()-result['uniform_loss']) < 1e-10
    assert abs(np.array(quality)@abs(prediction-target)-result['fixed_Q_loss']) < 1e-10


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    validate(args.source, args.smoke)
