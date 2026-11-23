"""Seed-paired controlled-bias analysis with separate 36 and 54 families."""
import argparse
import csv
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
METHODS = ['U', 'Q', 'E', 'S', 'EU', 'EQ', 'SU', 'SQ', 'SE', 'QU']
REGIMES = ['control', 'single_shift', 'common_shift']
TARGETS = [-.5, .5]
CELLS = ['UU', 'AU', 'UA', 'AA']
REPS = 200000
RNG = 20261124


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def csvout(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def valid(row):
    return bool(row['status'] == 'valid' and row['truth_error'] is not None
                and np.isfinite(row['truth_error']) and row['X'] is not None)


def has_fallback(value):
    return isinstance(value, dict) and (bool(value.get('fallback_reason'))
        or any(has_fallback(v) for v in value.values()))


@lru_cache(maxsize=16)
def weights(n):
    rng = np.random.default_rng(RNG)
    return np.concatenate([rng.multinomial(n, np.full(n, 1 / n), size=2000)
                           for _ in range(REPS // 2000)])


def stats(values, family=1):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return dict(n=0, mean=None, sd=None, ci95_low=None, ci95_high=None,
                    family_low=None, family_high=None)
    assert np.isfinite(x).all()
    bs = weights(len(x)) @ x / len(x)
    q = np.quantile(bs, [.025, .975, .05 / (2 * family), 1 - .05 / (2 * family)])
    return dict(n=len(x), mean=float(x.mean()), sd=float(x.std(ddof=1)) if len(x) > 1 else None,
                ci95_low=float(q[0]), ci95_high=float(q[1]), family_low=float(q[2]), family_high=float(q[3]))


def mean_field(rows, field):
    values = [r[field] for r in rows if r.get(field) is not None]
    return float(np.mean(values)) if values else None


def analyze(source, output):
    if output.exists():
        raise FileExistsError(output)
    manifest = read(source / 'manifest.json')
    assert manifest['status'] == 'complete' and not manifest['smoke']
    spec = manifest['protocol']
    assert spec['analysis']['primary_family_size'] == 36
    assert spec['analysis']['secondary_family_size'] == 54
    for name, digest in {**manifest['runtime_sha256'], **manifest['input_sha256']}.items():
        assert sha(ROOT / name) == digest, name
    coupled, factorial, cases, searches = {}, {}, {}, {}
    hashes = {str((source / 'manifest.json').resolve()): sha(source / 'manifest.json')}
    for complete in sorted(source.glob('seed*/completion.json')):
        path = complete.parent
        done = read(complete)
        assert done['status'] == 'complete' and done['physical_searches'] == 14 and done['factorial_rows'] == 8
        for name, digest in done['files_sha256'].items():
            assert sha(path / name) == digest, path / name
            hashes[str((path / name).resolve())] = digest
        case = read(path / 'case.json')
        key = case['seed'], case['regime'], case['target']
        assert key not in cases
        cases[key] = case
        physical = lines(path / 'searches.jsonl')
        ff = lines(path / 'factorial.jsonl')
        assert len(physical) == 14 and len(ff) == 8
        searches[key] = physical
        for row in physical:
            if row['search_id'].startswith('coupled_'):
                identity = key + (row['label'],)
                assert identity not in coupled
                coupled[identity] = row
        for row in ff:
            identity = key + (row['label'], row['cell'])
            assert identity not in factorial
            factorial[identity] = row
        for label in ['Q', 'EU']:
            assert factorial[key + (label, 'UU')]['source_search_id'] == 'coupled_U'
            assert factorial[key + (label, 'AA')]['source_search_id'] == 'coupled_' + label
        assert factorial[key + ('Q', 'UU')]['X'] == factorial[key + ('EU', 'UU')]['X']
    seeds = sorted({key[0] for key in cases})
    assert len(seeds) == 90 and set(cases) == set(itertools.product(seeds, REGIMES, TARGETS))
    assert len(coupled) == 5400 and len(factorial) == 4320
    coupled_groups, factorial_groups, effects, comparisons, weight_changes, diagnostics = [], [], [], [], [], []
    for regime, target in itertools.product(REGIMES, TARGETS):
        for label in METHODS:
            rows = [coupled[s, regime, target, label] for s in seeds]
            good = [r for r in rows if valid(r)]
            summary = dict(regime=regime, target=target, method=label, engine='best_of', panel_budget=400,
                planned=90, failed=90-len(good), **stats([r['truth_error'] for r in good]),
                mean_Lopt=mean_field(good, 'loss'), mean_uniform_loss=mean_field(good, 'uniform_loss'),
                mean_fixed_Q_loss=mean_field(good, 'fixed_Q_loss'),
                mean_endpoint_disagreement=mean_field(good, 'endpoint_disagreement'),
                mean_model_queries=mean_field(rows, 'model_queries'),
                mean_panel_evaluations=mean_field(rows, 'panel_evaluations'),
                mean_owner_alpha=float(np.mean([cases[s, regime, target]['alpha'][label][cases[s, regime, target]['owner_index']] for s in seeds])),
                fallbacks=sum(has_fallback(cases[s, regime, target]['alpha_audit'][label]) for s in seeds))
            coupled_groups.append(summary)
            if label != 'U':
                pairs = [(coupled[s, regime, target, label], coupled[s, regime, target, 'U']) for s in seeds]
                values = [a['truth_error']-b['truth_error'] for a, b in pairs if valid(a) and valid(b)]
                comparisons.append(dict(regime=regime, target=target, contrast=label+' - U', method=label,
                    family=54, planned=90, excluded=90-len(values), **stats(values, 54)))
        for label in ['Q', 'EU']:
            for cell in CELLS:
                rows = [factorial[s, regime, target, label, cell] for s in seeds]
                good = [r for r in rows if valid(r)]
                factorial_groups.append(dict(regime=regime, target=target, method=label, cell=cell,
                    engine='custom', panel_budget=200, planned=90, failed=90-len(good),
                    **stats([r['truth_error'] for r in good]), mean_Lopt=mean_field(good, 'loss'),
                    mean_uniform_loss=mean_field(good, 'uniform_loss'), mean_fixed_Q_loss=mean_field(good, 'fixed_Q_loss'),
                    mean_endpoint_disagreement=mean_field(good, 'endpoint_disagreement'),
                    mean_branch_model_queries=mean_field(rows, 'model_queries'),
                    query_scope='branch diagnostic; UU and AA already charged to physical searches'))
            eligible = [s for s in seeds if all(valid(factorial[s, regime, target, label, c]) for c in CELLS)]
            matrix = np.array([[factorial[s, regime, target, label, c]['truth_error'] for c in CELLS] for s in eligible])
            for effect, coefficients in [('objective', [-.5, .5, -.5, .5]),
                                         ('direction', [-.5, -.5, .5, .5]),
                                         ('interaction', [1., -1., -1., 1.])]:
                values = matrix @ coefficients if len(matrix) else []
                effects.append(dict(regime=regime, target=target, method=label, effect=effect, family=36,
                    planned=90, excluded=90-len(eligible), **stats(values, 36)))
    for target, label in itertools.product(TARGETS, METHODS):
        baseline, shifted, delta, delivered_delta = [], [], [], []
        for seed in seeds:
            c, b = cases[seed, 'control', target], cases[seed, 'single_shift', target]
            assert c['owner_index'] == b['owner_index']
            owner = c['owner_index']
            baseline.append(c['alpha'][label][owner])
            shifted.append(b['alpha'][label][owner])
            delta.append(shifted[-1]-baseline[-1])
            r0, r1 = coupled[seed, 'control', target, label], coupled[seed, 'single_shift', target, label]
            if valid(r0) and valid(r1):
                delivered_delta.append(r1['truth_error']-r0['truth_error'])
        summary = stats(delta)
        summary.pop('family_low'); summary.pop('family_high')
        weight_changes.append(dict(target=target, method=label, contrast='single_shift - control', planned=90,
            mean_control_owner_alpha=float(np.mean(baseline)), mean_shifted_owner_alpha=float(np.mean(shifted)),
            **summary, weight_decreased_count=sum(v < -1e-12 for v in delta),
            valid_delivery_pairs=len(delivered_delta),
            mean_delivery_truth_change=float(np.mean(delivered_delta)) if delivered_delta else None,
            delivery_improved_count=sum(v < -1e-12 for v in delivered_delta),
            scope='descriptive paired weights; cross-regime delivered quality also changes anchors'))
    for (seed, regime, target), case in sorted(cases.items()):
        raw = searches[seed, regime, target]
        residuals = [v['loss'] for v in case['inverses']]
        original = np.asarray(case['baseline_blind_predictions'])
        shifted = np.asarray(case['blind_predictions'])
        diagnostics.append(dict(seed=seed, regime=regime, target=target, owner_index=case['owner_index'],
            owner_id=case['owner_id'], linear_rank=case['linear_rank'], affine_rank=case['affine_rank'],
            blind_disagreement=case['blind_disagreement'], baseline_blind_disagreement=float(np.std(original, axis=1).mean()),
            blind_mean_signed_residual=case['blind_mean_signed_residual'], mean_blind_mse=float(np.mean(case['mse'])),
            owner_blind_mse=case['mse'][case['owner_index']],
            max_shift_identity_error=float(np.max(abs(shifted-original-case['offsets']))),
            max_pairwise_shift_change=float(np.max(abs(np.diff(shifted, axis=1)-np.diff(original, axis=1)))),
            inverse_bad=sum(v is None or v > .1 for v in residuals),
            inversion_forward_samples=sum(v['forward_samples'] for v in case['inverses']),
            inversion_backward_steps=sum(v['backward_steps'] for v in case['inverses']),
            physical_searches=len(raw), physical_valid=sum(valid(r) for r in raw),
            search_model_queries=sum(r['model_queries'] for r in raw),
            search_panel_evaluations=sum(r['panel_evaluations'] for r in raw),
            blind_requests=case['blind_cost_charged'].get('requests', 0), blind_samples=case['blind_cost_charged'].get('samples', 0),
            support_requests=case['support_cost'].get('requests', 0), support_samples=case['support_cost'].get('samples', 0),
            initialization_requests=sum(v.get('requests', 0) for v in case['initialization_costs'].values()),
            initialization_samples=sum(v.get('samples', 0) for v in case['initialization_costs'].values())))
    output.mkdir(parents=True)
    for name, rows in [('coupled_groups', coupled_groups), ('factorial_groups', factorial_groups),
                       ('primary_effects', effects), ('coupled_comparisons', comparisons),
                       ('bias_weight_changes', weight_changes), ('condition_diagnostics', diagnostics)]:
        csvout(output / (name+'.csv'), rows)
    audit = dict(passed=True, conditions=len(cases), source_seeds=len(seeds), physical_searches=7560,
        physical_valid=sum(valid(r) for rows in searches.values() for r in rows),
        physical_failed=sum(not valid(r) for rows in searches.values() for r in rows),
        coupled_rows=len(coupled), factorial_rows=len(factorial), primary_family=36, secondary_family=54,
        bootstrap_replicates=REPS, rng=RNG, source_sha256=hashes, analysis_sha256=sha(Path(__file__)),
        scope='Reused clean models with controlled post-training shifts; custom200 factorial holds anchors fixed within condition; no common-loss comparison across own Lopt',
        costs=dict(search_model_queries=sum(r['search_model_queries'] for r in diagnostics),
            inversion_forward_samples=sum(r['inversion_forward_samples'] for r in diagnostics),
            shared_regime_blind_samples=sum(r['blind_samples'] for r in diagnostics),
            baseline_diagnostic_queries=manifest['baseline_diagnostic_queries'],
            support_samples=sum(r['support_samples'] for r in diagnostics),
            initialization_samples=sum(r['initialization_samples'] for r in diagnostics)))
    (output/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps({k: v for k, v in audit.items() if k != 'source_sha256'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    analyze(args.source, args.output_dir)
