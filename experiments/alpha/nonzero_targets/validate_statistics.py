"""Independently reconstruct all group means and the 27 paired intervals."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def validate(source):
    analysis = source / 'analysis'
    output = analysis / 'statistics_acceptance.json'
    if output.exists():
        raise FileExistsError(output)
    records = {}
    for path in source.glob('*/results.jsonl'):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            records[row['seed'], row['scenario'], row['target'], row['label']] = row
    labels = ['U', 'Q', 'E', 'S', 'EU', 'EQ', 'SU', 'SQ', 'SE', 'QU']
    scenarios = ['reference_clean', 'low_quality_25pct', 'sample_imbalance_4x']
    seeds = sorted({key[0] for key in records})
    assert len(seeds) == 90 and len(records) == 10800
    groups = list(csv.DictReader((analysis / 'groups.csv').open()))
    primary = list(csv.DictReader((analysis / 'primary_comparisons.csv').open()))
    assert len(groups) == 160 and len(primary) == 27
    means = 0
    for row in groups:
        target = float(row['target'])
        selected = scenarios if row['scenario'] == 'mean_three_scenarios' else [row['scenario']]
        eligible = [seed for seed in seeds if all(
            records[seed, sc, target, label]['status'] == 'valid'
            for sc in selected for label in labels)]
        values = [np.mean([records[s, sc, target, row['method']]['truth_error']
                           for sc in selected]) for s in eligible]
        assert len(values) == int(row['n'])
        assert np.isclose(np.mean(values), float(row['mean']), atol=1e-13, rtol=0)
        means += 1
    interval_checks = 0
    for target in [-.5, .5, 2.7]:
        eligible = [s for s in seeds if all(records[s, sc, target, label]['status'] == 'valid'
                                           for sc in scenarios for label in labels)]
        data = np.array([[np.mean([records[s, sc, target, label]['truth_error']
                                  for sc in scenarios]) for label in labels] for s in eligible])
        rng = np.random.default_rng(20261123)
        draws = np.empty((200000, 10))
        for start in range(0, len(draws), 1000):
            multiplicities = rng.multinomial(len(data), [1 / len(data)] * len(data), size=1000)
            draws[start:start + 1000] = np.einsum('ij,jk->ik', multiplicities, data) / len(data)
        for j, label in enumerate(labels[1:], 1):
            row = next(r for r in primary if float(r['target']) == target and r['contrast'] == label + ' - U')
            assert int(row['family']) == 27 and int(row['n']) == len(data)
            assert np.isclose((data[:, j] - data[:, 0]).mean(), float(row['mean_delta']), atol=1e-13, rtol=0)
            q = np.quantile(draws[:, j] - draws[:, 0], [.025, .975, .05 / 54, 1 - .05 / 54])
            assert np.allclose(q, [float(row[k]) for k in ['ci95_low', 'ci95_high', 'family_low', 'family_high']], atol=1e-12, rtol=0)
            interval_checks += 1
    report = dict(passed=True, raw_records=len(records), group_means=means,
                  primary_paired_intervals=interval_checks, replicates=200000,
                  validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    validate(parser.parse_args().source)
