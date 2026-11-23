"""Team-level crossed-start role/payment stability and matched game diagnostics."""
import argparse
from collections import Counter, defaultdict
import csv
from functools import lru_cache
import gzip
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

TOL = 1e-10
REPS = 200000
RNG = 20261124
ROLES = ['A', 'M', 'J', 'AM', 'LOO']
JUDGES = ['F', 'Fminus']
ROLE_METRICS = ['sign_change', 'strict_sign_flip', 'eligibility_change', 'allocation_TV', 'spearman']
PAY_METRICS = ['basic_eligibility_change', 'total_payment_TV', 'mean_owner_amount_change', 'max_owner_amount_change']


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def csvout(path, rows, fieldnames=None):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def valid(row): return row.get('status') == 'valid'
def sign(x): return np.where(abs(np.asarray(x)) > TOL, np.sign(x), 0)


def share(x, threshold=TOL):
    positive = np.where(np.asarray(x) > threshold, x, 0.)
    return np.r_[positive/positive.sum(), 0.] if positive.sum() else np.r_[positive, 1.]


def rank(x, y):
    if len(x) < 3 or np.ptp(x) <= TOL or np.ptp(y) <= TOL:
        return None
    return float(spearmanr(x, y).statistic)


def average(values):
    good = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(good)) if good else None


@lru_cache(maxsize=16)
def bootstrap_weights(n):
    rng = np.random.default_rng(RNG)
    return np.concatenate([rng.multinomial(n, np.full(n, 1/n), size=2000) for _ in range(REPS//2000)])


def stats(values):
    good = [float(v) for v in values if v is not None and np.isfinite(v)]
    n = len(good)
    if not n:
        return dict(n=0, missing=len(values), mean=None, ci95_low=None, ci95_high=None)
    draws = bootstrap_weights(n)@np.asarray(good)/n
    return dict(n=n, missing=len(values)-n, mean=float(np.mean(good)),
                ci95_low=float(np.quantile(draws, .025)), ci95_high=float(np.quantile(draws, .975)))


def pair_indices(axis):
    if axis == 'inversion':
        return [((r1, w), (r2, w)) for w in range(3) for r1, r2 in itertools.combinations(range(5), 2)]
    return [((r, w1), (r, w2)) for r in range(5) for w1, w2 in itertools.combinations(range(3), 2)]


def role_pair(left, right, judge, role):
    x = np.array([p['scores'][judge][role] for p in left['participants']])
    y = np.array([p['scores'][judge][role] for p in right['participants']])
    tx = np.array([p['tauA'] for p in left['participants']]) if role == 'A' else TOL
    ty = np.array([p['tauA'] for p in right['participants']]) if role == 'A' else TOL
    sx, sy = sign(x), sign(y)
    return dict(sign_change=float(np.mean(sx != sy)), strict_sign_flip=float(np.mean(sx*sy == -1)),
        eligibility_change=float(np.mean((x > tx) != (y > ty))),
        allocation_TV=float(abs(share(x, tx)-share(y, ty)).sum()/2), spearman=rank(x, y))


def payment_pair(left, right, judge):
    x, y = left['payments'][judge], right['payments'][judge]
    amounts = abs(np.array(x['total'])-y['total'])
    assert x['budget'] == y['budget']
    return dict(basic_eligibility_change=float(np.mean((np.array(x['basic']) > 0) != (np.array(y['basic']) > 0))),
        total_payment_TV=float(abs(np.array(x['normalized_total'])-y['normalized_total']).sum()/2),
        mean_owner_amount_change=float(amounts.mean()), max_owner_amount_change=float(amounts.max()))


def analyze(source, output):
    if output.exists(): raise FileExistsError(output)
    manifest = read(source/'manifest.json')
    assert manifest['status'] == 'complete' and not manifest['smoke']
    for file, digest in {**manifest['runtime_sha256'], **manifest['input_sha256']}.items():
        assert sha(Path(file)) == digest, file
    index = read(source/'index.json')
    assert len(index) == 600
    hashes = {str((source/'manifest.json').resolve()): sha(source/'manifest.json'),
              str((source/'index.json').resolve()): sha(source/'index.json'),
              str((source/'preparations.json').resolve()): sha(source/'preparations.json')}
    jobs = {}
    keep = ['status', 'key', 'scenario', 'seed', 'target', 'r', 'w', 'ids', 'participants', 'payments',
        'game', 'truth_error', 'actual_rounds', 'cache_miss_calls', 'requests', 'role_cache_misses',
        'role_requests', 'matched_game_incremental_misses', 'observer_equivalence_cache_misses',
        'observer_equivalence_requests', 'preparation_file', 'preparation_sha256']
    for item in index:
        file = source/item['file']
        assert sha(file) == item['sha256']
        hashes[str(file.resolve())] = item['sha256']
        with gzip.open(file, 'rt') as f: raw = json.load(f)
        identity = raw['key'], raw['target'], raw['r'], raw['w']
        assert identity not in jobs
        jobs[identity] = {name:raw[name] for name in keep if name in raw}
        assert raw['status'] == item['status']
    teams = sorted({k[0] for k in jobs})
    assert len(teams) == 20 and set(jobs) == set(itertools.product(teams, [-.5, .5], range(5), range(3)))
    scenarios = {key:jobs[key, -.5, 0, 0]['scenario'] for key in teams}
    assert Counter(scenarios.values()) == dict(clean=10, controlled_poison=10)
    preps = {}
    preparation_summary = []
    for entry in read(source/'preparations.json'):
        path = source/entry['file']
        assert sha(path) == entry['sha256']
        hashes[str(path.resolve())] = entry['sha256']
        p = read(path)
        identity = p['key'], p['target'], p['r']
        assert identity not in preps
        preps[identity] = p
        losses = [v['loss'] for v in p['inversions']]
        preparation_summary.append(dict(key=p['key'], scenario=p['scenario'], target=p['target'], r=p['r'],
            owners=len(p['ids']), valid_anchors=sum(v['X'] is not None for v in p['inversions']),
            inverse_bad=sum(v is None or v > .1 for v in losses),
            mean_inverse_residual=average(losses), max_inverse_residual=max((v for v in losses if v is not None), default=None),
            inversion_forward_samples=p['inversion_forward_samples'], inversion_backward_steps=p['inversion_backward_steps'],
            preparation_file=entry['file']))
    assert set(preps) == set(itertools.product(teams, [-.5, .5], range(5)))
    run_summary = []
    for key, row in sorted(jobs.items()):
        identity = dict(zip(['key', 'target', 'r', 'w'], key))
        identity.update(scenario=row['scenario'], status=row['status'])
        for field in ['truth_error', 'actual_rounds', 'cache_miss_calls', 'requests', 'role_cache_misses',
                      'role_requests', 'matched_game_incremental_misses', 'observer_equivalence_cache_misses',
                      'observer_equivalence_requests']:
            identity[field] = row.get(field)
        identity['preparation_file'] = row['preparation_file']
        run_summary.append(identity)
    role_rows, payment_rows, ranges, references = [], [], [], []
    for key, target in itertools.product(teams, [-.5, .5]):
        team = {(r, w):jobs[key, target, r, w] for r in range(5) for w in range(3)}
        n = len(preps[key, target, 0]['ids'])
        for axis in ['inversion', 'solver']:
            planned = pair_indices(axis)
            pairs = [(team[a], team[b]) for a, b in planned if valid(team[a]) and valid(team[b])]
            for judge in JUDGES:
                for role in ROLES:
                    metrics = [role_pair(a, b, judge, role) for a, b in pairs]
                    out = dict(key=key, scenario=scenarios[key], target=target, axis=axis, judge=judge, role=role,
                        owners=n, planned_pairs=len(planned), valid_pairs=len(pairs), missing_pairs=len(planned)-len(pairs),
                        owner_pair_comparisons=n*len(pairs), defined_rank_pairs=sum(m['spearman'] is not None for m in metrics))
                    out.update({metric:average([m[metric] for m in metrics]) for metric in ROLE_METRICS})
                    role_rows.append(out)
                metrics = [payment_pair(a, b, judge) for a, b in pairs]
                out = dict(key=key, scenario=scenarios[key], target=target, axis=axis, judge=judge,
                    owners=n, planned_pairs=len(planned), valid_pairs=len(pairs), missing_pairs=len(planned)-len(pairs))
                out.update({metric:average([m[metric] for m in metrics]) for metric in PAY_METRICS})
                out['largest_observed_owner_change'] = max((m['max_owner_amount_change'] for m in metrics), default=None)
                payment_rows.append(out)
        vv = [team[r, w] for r in range(5) for w in range(3) if valid(team[r, w])]
        for judge, owner in itertools.product(JUDGES, range(n)):
            totals = [r['payments'][judge]['total'][owner] for r in vv]
            basic = [r['payments'][judge]['basic'][owner] for r in vv]
            ranges.append(dict(key=key, scenario=scenarios[key], target=target, judge=judge,
                id=preps[key, target, 0]['ids'][owner], owner_index=owner, planned_runs=15, valid_runs=len(vv),
                total_min=min(totals) if totals else None, total_max=max(totals) if totals else None,
                total_range=max(totals)-min(totals) if totals else None,
                basic_ever_canceled=bool(any(v == 0 for v in basic)) if basic else None,
                basic_always_canceled=bool(all(v == 0 for v in basic)) if basic else None,
                basic_eligibility_varies=bool(len(set(basic)) > 1) if basic else None))
        baseline = team[0, 0]
        if valid(baseline):
            game = baseline['game']
            assert game is not None
            for role, reference in itertools.product(['J', 'LOO'], ['phi_F', 'phi_truth']):
                x = np.array([p['scores']['F'][role] for p in baseline['participants']])
                y = np.array(game[reference])
                references.append(dict(key=key, scenario=scenarios[key], target=target, role=role,
                    reference=reference, source_judge='F', owners=n,
                    alignment='same_fixed_F_utility' if reference == 'phi_F' else 'offline_cross_scorer_truth',
                    spearman=rank(x, y), sign_agreement=float(np.mean(sign(x) == sign(y))),
                    allocation_TV=float(abs(share(x)-share(y)).sum()/2), sum_gap=float(x.sum()-y.sum())))
    role_summary, payment_summary, matched_summary = [], [], []
    for scenario, target, axis, judge, role in itertools.product(['clean', 'controlled_poison'], [-.5, .5],
            ['inversion', 'solver'], JUDGES, ROLES):
        selected = [r for r in role_rows if (r['scenario'], r['target'], r['axis'], r['judge'], r['role']) == (scenario, target, axis, judge, role)]
        assert len(selected) == 10
        for metric in ROLE_METRICS:
            role_summary.append(dict(scenario=scenario, target=target, axis=axis, judge=judge, role=role,
                metric=metric, planned_teams=10, **stats([r[metric] for r in selected]),
                planned_pairs=sum(r['planned_pairs'] for r in selected), valid_pairs=sum(r['valid_pairs'] for r in selected),
                defined_rank_pairs=sum(r['defined_rank_pairs'] for r in selected)))
    for scenario, target, axis, judge in itertools.product(['clean', 'controlled_poison'], [-.5, .5], ['inversion', 'solver'], JUDGES):
        selected = [r for r in payment_rows if (r['scenario'], r['target'], r['axis'], r['judge']) == (scenario, target, axis, judge)]
        for metric in PAY_METRICS:
            payment_summary.append(dict(scenario=scenario, target=target, axis=axis, judge=judge, metric=metric,
                planned_teams=10, **stats([r[metric] for r in selected]),
                planned_pairs=sum(r['planned_pairs'] for r in selected), valid_pairs=sum(r['valid_pairs'] for r in selected)))
    for scenario, target, role, reference in itertools.product(['clean', 'controlled_poison'], [-.5, .5], ['J', 'LOO'], ['phi_F', 'phi_truth']):
        selected = [r for r in references if (r['scenario'], r['target'], r['role'], r['reference']) == (scenario, target, role, reference)]
        for metric in ['spearman', 'sign_agreement', 'allocation_TV', 'sum_gap']:
            values = [r[metric] for r in selected]+[None]*(10-len(selected))
            matched_summary.append(dict(scenario=scenario, target=target, role=role, reference=reference, metric=metric,
                alignment='same_fixed_F_utility' if reference == 'phi_F' else 'offline_cross_scorer_truth',
                planned_teams=10, **stats(values)))
    output.mkdir(parents=True)
    tables = dict(team_role_stability=role_rows, team_payment_stability=payment_rows,
        role_summary=role_summary, payment_summary=payment_summary, owner_payment_ranges=ranges,
        run_summary=run_summary, preparation_summary=preparation_summary,
        matched_reference=references, matched_summary=matched_summary)
    for name, rows in tables.items():
        fields = ['key', 'scenario', 'target', 'role', 'reference', 'source_judge', 'owners',
                  'alignment', 'spearman', 'sign_agreement', 'allocation_TV', 'sum_gap'] if name == 'matched_reference' else None
        csvout(output/(name+'.csv'), rows, fields)
    costs = {field:sum(r.get(field, 0) or 0 for r in jobs.values()) for field in
        ['cache_miss_calls', 'requests', 'role_cache_misses', 'matched_game_incremental_misses', 'observer_equivalence_cache_misses', 'observer_equivalence_requests']}
    costs['inversion_forward_samples'] = sum(p['inversion_forward_samples'] for p in preps.values())
    costs['inversion_backward_steps'] = sum(p['inversion_backward_steps'] for p in preps.values())
    audit = dict(passed=True, jobs=len(jobs), valid=sum(valid(r) for r in jobs.values()),
        failed=sum(not valid(r) for r in jobs.values()), teams=len(teams), preparations=len(preps),
        matched_games=sum(valid(jobs[k, t, 0, 0]) for k, t in itertools.product(teams, [-.5, .5])),
        bootstrap_replicates=REPS, rng=RNG, costs=costs, source_sha256=hashes,
        tables={name:len(rows) for name, rows in tables.items()}, analysis_sha256=sha(Path(__file__)),
        scope='Descriptive team-cluster intervals; crossed start configurations are dependent. Fixed-model extension; no fairness or stability-pass claim.')
    (output/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps({k:v for k, v in audit.items() if k != 'source_sha256'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.source, args.output_dir)
