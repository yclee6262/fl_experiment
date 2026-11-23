"""Independent raw-record reconstruction of the 36/54 paired families."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def table(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def is_valid(row):
    return row['status'] == 'valid' and row['X'] is not None and row['truth_error'] is not None and np.isfinite(row['truth_error'])


def validate(source, analysis):
    output = analysis / 'statistics_acceptance.json'
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((source / 'manifest.json').read_text())
    assert manifest['status'] == 'complete' and not manifest['smoke']
    coupled, factorial, cases = {}, {}, {}
    physical_queries = 0
    checked_hashes = 0
    for path in sorted(source.glob('seed*/completion.json')):
        parent = path.parent
        complete = json.loads(path.read_text())
        assert complete['status'] == 'complete'
        for filename, h in complete['files_sha256'].items():
            assert digest(parent / filename) == h
            checked_hashes += 1
        c = json.loads((parent/'case.json').read_text())
        key = c['seed'], c['regime'], c['target']
        assert key not in cases
        cases[key] = c
        physical = records(parent/'searches.jsonl')
        assert len(physical) == 14
        for r in physical:
            physical_queries += r['model_queries']
            if r['search_id'].startswith('coupled_'):
                k = key + (r['label'],)
                assert k not in coupled
                coupled[k] = r
        ff = records(parent/'factorial.jsonl')
        assert len(ff) == 8
        for r in ff:
            k = key + (r['label'], r['cell'])
            assert k not in factorial
            factorial[k] = r
    regimes, targets = ['control', 'single_shift', 'common_shift'], [-.5, .5]
    labels = ['U', 'Q', 'E', 'S', 'EU', 'EQ', 'SU', 'SQ', 'SE', 'QU']
    seeds = sorted({key[0] for key in cases})
    assert len(seeds) == 90 and set(cases) == set(itertools.product(seeds, regimes, targets))
    assert len(coupled) == 5400 and len(factorial) == 4320
    cache = {}
    def draws_for(n):
        if n not in cache:
            generator = np.random.default_rng(20261124)
            cache[n] = np.concatenate([generator.multinomial(n, [1/n]*n, size=1000) for _ in range(200)])
        return cache[n]
    def compare(row, values, family):
        n = len(values)
        assert n == int(row['n'])
        assert int(row['excluded']) == 90-n
        assert int(row['family']) == family
        if n == 0:
            assert row['mean'] == ''
            return
        assert np.isclose(np.mean(values), float(row['mean']), atol=1e-12, rtol=0)
        draws = np.einsum('ij,j->i', draws_for(n), values) / n
        quantiles = np.quantile(draws, [.025, .975, .05/(2*family), 1-.05/(2*family)])
        assert np.allclose(quantiles, [float(row[field]) for field in ['ci95_low', 'ci95_high', 'family_low', 'family_high']], atol=1e-12, rtol=0)
    primary = table(analysis/'primary_effects.csv')
    secondary = table(analysis/'coupled_comparisons.csv')
    assert len(primary) == 36 and len(secondary) == 54
    seen = set()
    for row in primary:
        regime, target, label = row['regime'], float(row['target']), row['method']
        identity = regime, target, label, row['effect']
        assert identity not in seen
        seen.add(identity)
        values = []
        for seed in seeds:
            cells = {cell: factorial[seed, regime, target, label, cell] for cell in ['UU', 'AU', 'UA', 'AA']}
            if not all(is_valid(r) for r in cells.values()):
                continue
            y00, y10, y01, y11 = [cells[c]['truth_error'] for c in ['UU', 'AU', 'UA', 'AA']]
            if row['effect'] == 'objective':
                values.append(((y10-y00)+(y11-y01))/2)
            elif row['effect'] == 'direction':
                values.append(((y01-y00)+(y11-y10))/2)
            else:
                assert row['effect'] == 'interaction'
                values.append(y11-y10-y01+y00)
        compare(row, values, 36)
    assert seen == set(itertools.product(regimes, targets, ['Q', 'EU'], ['objective', 'direction', 'interaction']))
    seen = set()
    for row in secondary:
        regime, target, label = row['regime'], float(row['target']), row['method']
        assert row['contrast'] == label+' - U'
        identity = regime, target, label
        assert identity not in seen
        seen.add(identity)
        values = []
        for seed in seeds:
            a, u = coupled[seed, regime, target, label], coupled[seed, regime, target, 'U']
            if is_valid(a) and is_valid(u):
                values.append(a['truth_error']-u['truth_error'])
        compare(row, values, 54)
    assert seen == set(itertools.product(regimes, targets, labels[1:]))
    group_checks = 0
    for filename, mapping in [('coupled_groups.csv', coupled), ('factorial_groups.csv', factorial)]:
        rows = table(analysis/filename)
        assert len(rows) == (60 if filename == 'coupled_groups.csv' else 48)
        identities = set()
        for row in rows:
            identity = row['regime'], float(row['target']), row['method']
            if 'cell' in row:
                identity += (row['cell'],)
            assert identity not in identities
            identities.add(identity)
            selected = [mapping[(seed,)+identity] for seed in seeds]
            good = [r for r in selected if is_valid(r)]
            assert len(good) == int(row['n']) and int(row['failed']) == 90-len(good)
            for field, column in [('truth_error', 'mean'), ('loss', 'mean_Lopt'), ('uniform_loss', 'mean_uniform_loss'), ('fixed_Q_loss', 'mean_fixed_Q_loss'), ('endpoint_disagreement', 'mean_endpoint_disagreement')]:
                if good:
                    assert np.isclose(np.mean([r[field] for r in good]), float(row[column]), atol=1e-12, rtol=0)
                else:
                    assert row[column] == ''
            query_column = 'mean_model_queries' if 'mean_model_queries' in row else 'mean_branch_model_queries'
            assert np.isclose(np.mean([r['model_queries'] for r in selected]), float(row[query_column]), atol=1e-12, rtol=0)
            group_checks += 1
        expected = set(itertools.product(regimes, targets, labels)) if filename == 'coupled_groups.csv' else set(itertools.product(regimes, targets, ['Q', 'EU'], ['UU', 'AU', 'UA', 'AA']))
        assert identities == expected
    weight_rows = table(analysis/'bias_weight_changes.csv')
    assert len(weight_rows) == 20
    assert {(float(r['target']), r['method']) for r in weight_rows} == set(itertools.product(targets, labels))
    for row in weight_rows:
        target, label = float(row['target']), row['method']
        changes = []
        for seed in seeds:
            a, b = cases[seed, 'control', target], cases[seed, 'single_shift', target]
            assert a['owner_index'] == b['owner_index']
            owner = a['owner_index']
            changes.append(b['alpha'][label][owner]-a['alpha'][label][owner])
        assert len(changes) == int(row['n']) == 90
        assert np.isclose(np.mean(changes), float(row['mean']), atol=1e-12, rtol=0)
        draws = np.einsum('ij,j->i', draws_for(90), changes)/90
        assert np.allclose(np.quantile(draws, [.025, .975]), [float(row['ci95_low']), float(row['ci95_high'])], atol=1e-12, rtol=0)
    audit = json.loads((analysis/'audit.json').read_text())
    assert audit['passed'] and audit['costs']['search_model_queries'] == physical_queries
    result = dict(passed=True, source_cases=540, raw_physical_searches=7560, factorial_rows=4320,
        raw_hashes_checked=checked_hashes, group_means_and_common_scores=group_checks,
        primary_intervals=36, secondary_intervals=54, paired_weight_intervals=20,
        replicates=200000, rng=20261124, physical_query_sum=physical_queries,
        validator_sha256=digest(Path(__file__)),
        analysis_files_sha256={p.name:digest(p) for p in analysis.glob('*.csv')})
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--analysis', type=Path)
    args = p.parse_args()
    validate(args.source, args.analysis or args.source/'analysis')
