"""Rebuild all pair metrics and intervals from raw runs without analyzer imports."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def read(path): return json.loads(path.read_text())
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def table(path):
    with path.open() as f: return list(csv.DictReader(f))


def okay(row): return row.get('status') == 'valid'


def close(actual, expected):
    if expected is None:
        assert actual == '', actual
    else:
        assert np.isclose(float(actual), expected, atol=1e-10, rtol=0), (actual, expected)


def symbols(values):
    return np.array([1 if v > 1e-10 else -1 if v < -1e-10 else 0 for v in values])


def proportions(values, thresholds):
    x = np.array(values, float)
    cutoff = np.broadcast_to(thresholds, x.shape)
    selected = np.array([v if v > t else 0. for v, t in zip(x, cutoff)])
    if np.sum(selected) == 0: return np.r_[np.zeros(len(x)), 1.]
    return np.r_[selected / np.sum(selected), 0.]


def correlation(x, y):
    if len(x) < 3 or max(x)-min(x) <= 1e-10 or max(y)-min(y) <= 1e-10:
        return None
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def mean_defined(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(sum(values)/len(values)) if values else None


def role_comparison(a, b, judge, role):
    x = [p['scores'][judge][role] for p in a['participants']]
    y = [p['scores'][judge][role] for p in b['participants']]
    tx = [p['tauA'] if role == 'A' else 1e-10 for p in a['participants']]
    ty = [p['tauA'] if role == 'A' else 1e-10 for p in b['participants']]
    sx, sy = symbols(x), symbols(y)
    return dict(sign_change=sum(int(v != w) for v, w in zip(sx, sy))/len(x),
        strict_sign_flip=sum(int(v*w == -1) for v, w in zip(sx, sy))/len(x),
        eligibility_change=sum(int((v > t) != (w > u)) for v, t, w, u in zip(x, tx, y, ty))/len(x),
        allocation_TV=float(np.linalg.norm(proportions(x, tx)-proportions(y, ty), 1)/2),
        spearman=correlation(x, y))


def payment_comparison(a, b, judge):
    x, y = a['payments'][judge], b['payments'][judge]
    assert x['budget'] == y['budget']
    differences = [abs(v-w) for v, w in zip(x['total'], y['total'])]
    nx = np.r_[x['total'], x['reserved']]/x['budget']
    ny = np.r_[y['total'], y['reserved']]/y['budget']
    np.testing.assert_allclose(nx, x['normalized_total'], atol=1e-12, rtol=0)
    np.testing.assert_allclose(ny, y['normalized_total'], atol=1e-12, rtol=0)
    assert sum(x['total'])+x['reserved'] == x['budget']
    assert sum(y['total'])+y['reserved'] == y['budget']
    return dict(basic_eligibility_change=sum(int((v > 0) != (w > 0)) for v, w in zip(x['basic'], y['basic']))/len(differences),
        total_payment_TV=float(np.linalg.norm(nx-ny, 1)/2),
        mean_owner_amount_change=float(sum(differences)/len(differences)), max_owner_amount_change=float(max(differences)))


def validate(source, analysis):
    output = analysis/'statistics_acceptance.json'
    if output.exists(): raise FileExistsError(output)
    manifest = read(source/'manifest.json')
    assert manifest['status'] == 'complete' and not manifest['smoke']
    for filename, expected in {**manifest['runtime_sha256'], **manifest['input_sha256']}.items():
        assert digest(Path(filename)) == expected
    jobs = {}
    for item in read(source/'index.json'):
        path = source/item['file']
        assert digest(path) == item['sha256']
        with gzip.open(path, 'rt') as f: raw = json.load(f)
        identity = raw['key'], raw['target'], raw['r'], raw['w']
        assert identity not in jobs
        jobs[identity] = {k:v for k, v in raw.items() if k not in ['events', 'rows', 'full', 'LOO']}
    teams = sorted({k[0] for k in jobs})
    assert len(teams) == 20 and set(jobs) == set(itertools.product(teams, [-.5, .5], range(5), range(3)))
    role_expected, payment_expected, matched_expected = {}, {}, {}
    metric_checks = pair_checks = 0
    for key, target in itertools.product(teams, [-.5, .5]):
        for axis in ['inversion', 'solver']:
            coordinates = list(itertools.product(range(5), range(3)))
            pairs = [(a, b) for a, b in itertools.combinations(coordinates, 2)
                     if (a[1] == b[1] if axis == 'inversion' else a[0] == b[0])]
            assert len(pairs) == (30 if axis == 'inversion' else 15)
            valid_pairs = [(jobs[(key, target)+a], jobs[(key, target)+b]) for a, b in pairs
                           if okay(jobs[(key, target)+a]) and okay(jobs[(key, target)+b])]
            pair_checks += len(valid_pairs)
            for judge in ['F', 'Fminus']:
                for role in ['A', 'M', 'J', 'AM', 'LOO']:
                    values = [role_comparison(a, b, judge, role) for a, b in valid_pairs]
                    entry = {metric:mean_defined([v[metric] for v in values]) for metric in
                             ['sign_change', 'strict_sign_flip', 'eligibility_change', 'allocation_TV', 'spearman']}
                    entry.update(planned_pairs=len(pairs), valid_pairs=len(valid_pairs),
                                 defined_rank_pairs=sum(v['spearman'] is not None for v in values))
                    role_expected[key, target, axis, judge, role] = entry
                values = [payment_comparison(a, b, judge) for a, b in valid_pairs]
                entry = {metric:mean_defined([v[metric] for v in values]) for metric in
                         ['basic_eligibility_change', 'total_payment_TV', 'mean_owner_amount_change', 'max_owner_amount_change']}
                entry.update(planned_pairs=len(pairs), valid_pairs=len(valid_pairs),
                             largest_observed_owner_change=max((v['max_owner_amount_change'] for v in values), default=None))
                payment_expected[key, target, axis, judge] = entry
        raw = jobs[key, target, 0, 0]
        if okay(raw):
            game = raw['game']
            assert game is not None
            n = raw['n']
            for field, phi_field in [('vF', 'phi_F'), ('vTruth', 'phi_truth')]:
                v = {c['mask']:c[field] for c in game['coalitions']}
                assert set(v) == set(range(2**n)) and abs(v[0]) < 1e-12
                phi = np.zeros(n)
                perms = list(itertools.permutations(range(n)))
                for permutation in perms:
                    previous = 0
                    for owner in permutation:
                        current = previous | (1 << owner)
                        phi[owner] += (v[current]-v[previous])/len(perms)
                        previous = current
                np.testing.assert_allclose(phi, game[phi_field], atol=1e-10, rtol=0)
                for role in ['J', 'LOO']:
                    x = np.array([p['scores']['F'][role] for p in raw['participants']])
                    matched_expected[key, target, role, phi_field] = dict(spearman=correlation(x, phi),
                        sign_agreement=float(np.mean(symbols(x) == symbols(phi))),
                        allocation_TV=float(np.linalg.norm(proportions(x, 1e-10)-proportions(phi, 1e-10), 1)/2),
                        sum_gap=float(x.sum()-phi.sum()))
    for filename, expected, fields in [
            ('team_role_stability.csv', role_expected, ['key', 'target', 'axis', 'judge', 'role']),
            ('team_payment_stability.csv', payment_expected, ['key', 'target', 'axis', 'judge']),
            ('matched_reference.csv', matched_expected, ['key', 'target', 'role', 'reference'])]:
        seen = set()
        for row in table(analysis/filename):
            identity = tuple(float(row[f]) if f == 'target' else row[f] for f in fields)
            assert identity in expected and identity not in seen
            if filename == 'matched_reference.csv':
                assert row['source_judge'] == 'F'
                assert row['alignment'] == ('same_fixed_F_utility' if row['reference'] == 'phi_F' else 'offline_cross_scorer_truth')
            seen.add(identity)
            for metric, value in expected[identity].items():
                close(row[metric], value)
                metric_checks += 1
        assert seen == set(expected)
    weight_cache = {}
    intervals = 0
    def check_summary(row, values):
        nonlocal intervals
        values = [float(v) for v in values if v is not None and np.isfinite(v)]
        n = len(values)
        assert int(row['n']) == n and int(row['missing']) == 10-n
        if n == 0:
            assert row['mean'] == row['ci95_low'] == row['ci95_high'] == ''
        else:
            close(row['mean'], sum(values)/n)
            if n not in weight_cache:
                rng = np.random.default_rng(20261124)
                weight_cache[n] = np.concatenate([rng.multinomial(n, [1/n]*n, size=1000) for _ in range(200)])
            draws = np.einsum('ij,j->i', weight_cache[n], values)/n
            lo, hi = np.quantile(draws, [.025, .975])
            close(row['ci95_low'], lo)
            close(row['ci95_high'], hi)
        intervals += 1
    scenarios = {key:jobs[key, -.5, 0, 0]['scenario'] for key in teams}
    for filename, expected in [('role_summary.csv', role_expected), ('payment_summary.csv', payment_expected), ('matched_summary.csv', matched_expected)]:
        rows = table(analysis/filename)
        assert len(rows) == {'role_summary.csv':400, 'payment_summary.csv':64, 'matched_summary.csv':64}[filename]
        seen = set()
        for row in rows:
            scenario, target = row['scenario'], float(row['target'])
            metric = row['metric']
            if filename == 'role_summary.csv':
                suffix = (target, row['axis'], row['judge'], row['role'])
            elif filename == 'payment_summary.csv':
                suffix = (target, row['axis'], row['judge'])
            else:
                suffix = (target, row['role'], row['reference'])
            identity = (scenario,)+suffix+(metric,)
            assert identity not in seen
            seen.add(identity)
            selected = [(key,)+suffix for key in teams if scenarios[key] == scenario]
            values = [expected.get(k, {}).get(metric) for k in selected]
            check_summary(row, values)
    ranges = table(analysis/'owner_payment_ranges.csv')
    assert len(ranges) == 320
    for row in ranges:
        key, target, judge, owner = row['key'], float(row['target']), row['judge'], int(row['owner_index'])
        vv = [jobs[key, target, r, w] for r in range(5) for w in range(3) if okay(jobs[key, target, r, w])]
        totals = [v['payments'][judge]['total'][owner] for v in vv]
        basics = [v['payments'][judge]['basic'][owner] for v in vv]
        assert int(row['valid_runs']) == len(vv)
        for field, value in [('total_min', min(totals) if totals else None), ('total_max', max(totals) if totals else None),
                             ('total_range', max(totals)-min(totals) if totals else None)]: close(row[field], value)
        if basics:
            assert row['basic_eligibility_varies'] == str(min(basics) != max(basics))
            assert row['basic_ever_canceled'] == str(min(basics) == 0)
            assert row['basic_always_canceled'] == str(max(basics) == 0)
    preps = {}
    for entry in read(source/'preparations.json'):
        path = source/entry['file']
        assert digest(path) == entry['sha256']
        raw = read(path)
        preps[raw['key'], raw['target'], raw['r']] = raw
    assert len(preps) == 200
    prep_rows = table(analysis/'preparation_summary.csv')
    assert len(prep_rows) == 200
    for row in prep_rows:
        p = preps[row['key'], float(row['target']), int(row['r'])]
        assert int(row['inversion_forward_samples']) == p['inversion_forward_samples']
        assert int(row['inversion_backward_steps']) == p['inversion_backward_steps']
        assert int(row['inverse_bad']) == sum(v['loss'] is None or v['loss'] > .1 for v in p['inversions'])
    run_rows = table(analysis/'run_summary.csv')
    assert len(run_rows) == 600
    cost_fields = ['cache_miss_calls', 'requests', 'role_cache_misses', 'matched_game_incremental_misses',
                   'observer_equivalence_cache_misses', 'observer_equivalence_requests']
    for row in run_rows:
        raw = jobs[row['key'], float(row['target']), int(row['r']), int(row['w'])]
        assert row['status'] == raw['status']
        for field in cost_fields:
            close(row[field], raw.get(field))
    audit = read(analysis/'audit.json')
    assert audit['passed']
    for field in cost_fields:
        assert audit['costs'][field] == sum(r.get(field, 0) or 0 for r in jobs.values())
    for field in ['inversion_forward_samples', 'inversion_backward_steps']:
        assert audit['costs'][field] == sum(p[field] for p in preps.values())
    result = dict(passed=True, jobs=len(jobs), within_team_pairs_recomputed=pair_checks,
        team_metrics_recomputed=metric_checks, summary_intervals=intervals, owner_ranges=len(ranges),
        preparation_costs=len(preps), run_cost_rows=len(run_rows), replicates=200000, rng=20261124,
        matched_games_permutation_checked=len(matched_expected)//4,
        validator_sha256=digest(Path(__file__)),
        analysis_files_sha256={p.name:digest(p) for p in analysis.glob('*.csv')},
        scope='Independent raw reconstruction; no analyzer import or model calls; descriptive team-cluster intervals')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--analysis', type=Path)
    args = p.parse_args()
    validate(args.source, args.analysis or args.source/'analysis')
