"""Reconstruct group means and the 24-family intervals from raw decisions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def validate(source):
    analysis = source / 'analysis_final'
    output = analysis / 'statistics_acceptance.json'
    if output.exists():
        raise FileExistsError(output)
    records = {}
    for path in source.glob('seed*/*_k*/results.jsonl'):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            key = tuple(r[k] for k in ['seed', 'dimension', 'rho', 'requested_k', 'target_label', 'method', 'panel_budget'])
            assert key not in records
            records[key] = r
    assert len(records) == 17280
    groups = list(csv.DictReader((analysis / 'groups.csv').open()))
    primary = list(csv.DictReader((analysis / 'primary_comparisons.csv').open()))
    assert len(groups) == 576 and len(primary) == 24
    for row in groups:
        key = (int(row['dimension']), float(row['rho']), int(row['requested_k']), row['target_label'], row['method'], int(row['panel_budget']))
        values = [r['truth_error'] for k, r in records.items() if k[1:] == key and r['status'] == 'valid']
        assert len(values) == int(row['n'])
        assert np.isclose(np.mean(values), float(row['mean']), atol=1e-12, rtol=0)
    # A single common bootstrap weight matrix applies to all complete 30-seed cells.
    rng = np.random.default_rng(20261123)
    weights = np.concatenate([rng.multinomial(30, [1 / 30] * 30, size=2000) for _ in range(100)])
    for row in primary:
        key = (int(row['dimension']), float(row['rho']), int(row['requested_k']), row['target_label'])
        delta = []
        for seed in range(30000, 30030):
            left = records[(seed,) + key + ('best_of', 400)]
            right = records[(seed,) + key + ('nelder_mead', 400)]
            assert left['status'] == right['status'] == 'valid'
            delta.append(left['truth_error'] - right['truth_error'])
        assert int(row['n']) == 30 and int(row['family_size']) == 24
        assert np.isclose(np.mean(delta), float(row['mean']), atol=1e-12, rtol=0)
        draws = np.einsum('ij,j->i', weights, delta) / 30
        quantiles = np.quantile(draws, [.025, .975, .05 / 48, 1 - .05 / 48])
        assert np.allclose(quantiles, [float(row[k]) for k in ['ci95_low', 'ci95_high', 'family_low', 'family_high']], atol=1e-12, rtol=0)
    report = dict(passed=True, raw_records=len(records), group_means=len(groups),
                  primary_paired_intervals=len(primary), replicates=200000,
                  validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    validate(p.parse_args().source)
