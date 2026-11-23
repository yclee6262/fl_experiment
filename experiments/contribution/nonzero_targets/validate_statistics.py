"""Reconstruct team statistics directly from the saved target-specific runs."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def allocation(values, threshold):
    positive = np.where(values > threshold, values, 0.)
    total = positive.sum()
    return np.r_[positive / total, 0.] if total else np.r_[positive, 1.]


def validate(source):
    analysis = source / 'analysis_final'
    output = analysis / 'statistics_acceptance.json'
    if output.exists():
        raise FileExistsError(output)
    data = {}
    for item in json.loads((source / 'index.json').read_text()):
        with gzip.open(source / item['file'], 'rt') as f:
            raw = json.load(f)
        assert raw['status'] == 'valid'
        base = item['scenario'], item['target']
        for metric, field in [('truth_error', 'truth_error'), ('actual_rounds', 'actual_rounds'),
                              ('cache_miss_calls', 'cache_miss_calls'), ('inversion_forward_samples', 'inversion_forward_samples'),
                              ('observer_equivalence_cache_miss_calls', 'observer_equivalence_cache_miss_calls'),
                              ('inverse_bad', 'inverse_target_error_above_point1')]:
            data[base + ('all', 'none', metric, item['key'])] = raw[field]
        truth = np.array([p['truth_LOO'] for p in raw['participants']])
        for role in ['A', 'M', 'J']:
            scores = {judge: np.array([p['scores'][judge][role] for p in raw['participants']])
                      for judge in ['F', 'Fminus']}
            tau = np.array([p['tauA'] for p in raw['participants']]) if role == 'A' else 1e-10
            shares = {j: allocation(v, tau) for j, v in scores.items()}
            tv = abs(shares['F'] - shares['Fminus']).sum() / 2
            for judge, values in scores.items():
                rank = None if np.ptp(values) <= 1e-10 or np.ptp(truth) <= 1e-10 else float(np.corrcoef(rankdata(values), rankdata(truth))[0, 1])
                values_sign = np.where(abs(values) > 1e-10, np.sign(values), 0)
                truth_sign = np.where(abs(truth) > 1e-10, np.sign(truth), 0)
                for metric, value in dict(rank_truth_LOO=rank, sign_truth_LOO=float(np.mean(values_sign == truth_sign)), reserved=float(shares[judge][-1] > .5), TV_peer_full=tv).items():
                    data[base + (role, judge, metric, item['key'])] = value
    cache = {}
    def check(row, differences=False):
        identity = row['scenario'], float(row['target']), row['role'], row['judge'], row['metric']
        values = []
        for key, value in data.items():
            if key[:-1] != identity or value is None:
                continue
            if differences:
                zero = data[(key[0], 0.) + key[2:]]
                if zero is None:
                    continue
                value -= zero
            values.append(float(value))
        n = len(values)
        assert n == int(row['n'])
        if not n:
            assert row['mean'] == ''
            return
        if n not in cache:
            rng = np.random.default_rng(20261123)
            cache[n] = np.concatenate([rng.multinomial(n, [1 / n] * n, size=2000) for _ in range(100)])
        bs = np.einsum('ij,j->i', cache[n], values) / n
        assert np.isclose(np.mean(values), float(row['mean']), atol=1e-12, rtol=0)
        assert np.allclose(np.quantile(bs, [.025, .975]), [float(row['ci95_low']), float(row['ci95_high'])], atol=1e-12, rtol=0)
    summary = list(csv.DictReader((analysis / 'summary.csv').open()))
    paired = list(csv.DictReader((analysis / 'paired_target_changes.csv').open()))
    for row in summary:
        check(row)
    for row in paired:
        check(row, True)
    report = dict(passed=True, raw_tasks=80, summary_intervals=len(summary), paired_intervals=len(paired),
                  replicates=200000, validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    validate(p.parse_args().source)
