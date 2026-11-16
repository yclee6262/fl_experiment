"""Recompute saved means and CI quantiles without altering registered evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def audit(registration):
    reg = json.loads(Path(registration).read_text())
    root = Path(reg['output'])
    formal = json.loads((root / 'final_analysis.json').read_text())
    diagnostic = json.loads((root / 'qualified_case_diagnostic/analysis.json').read_text())
    rows = [json.loads(line) for batch in reg['batches']
            for line in (Path(batch['results']) / 'results.jsonl').read_text().splitlines()]
    index = {(r['scenario'], r['seed'], r['method']): r for r in rows}
    errors = []
    max_mean_difference = 0.0
    for group in formal['groups']:
        scenarios = reg['scenarios'] if group['endpoint'] == 'equal_scenario_mean' else [group['endpoint']]
        values = []
        for seed in reg['seeds']:
            selected = [index[s, seed, group['method']] for s in scenarios]
            if all(r['status'] == 'valid' for r in selected):
                values.append(np.mean([r['pooled_error'] for r in selected]))
        diff = abs(float(np.mean(values)) - group['mean'])
        max_mean_difference = max(max_mean_difference, diff)
        if len(values) != group['n'] or diff > 1e-12:
            errors.append('group mismatch: ' + str((group['endpoint'], group['method'])))
    for comparison in formal['comparisons']:
        scenarios = reg['scenarios'] if comparison['endpoint'] == 'equal_scenario_mean' else [comparison['endpoint']]
        values = []
        for seed in reg['seeds']:
            pairs = [(index[s, seed, comparison['method']], index[s, seed, comparison['baseline']]) for s in scenarios]
            if all(a['status'] == b['status'] == 'valid' for a, b in pairs):
                values.append(np.mean([a['pooled_error'] - b['pooled_error'] for a, b in pairs]))
        diff = abs(float(np.mean(values)) - comparison['mean_delta'])
        max_mean_difference = max(max_mean_difference, diff)
        if len(values) != comparison['n'] or diff > 1e-12:
            errors.append('paired mean mismatch')
    original = {(c['endpoint'], c['method'], c['baseline']): c for c in formal['comparisons']}
    selected_sets = [
        (root / 'bootstrap_means.npy', [c for c in formal['comparisons'] if c['CI_simultaneous'] is not None]),
        (root / 'qualified_case_diagnostic/bootstrap_means.npy',
         [c for c in diagnostic['comparisons'] if original[c['endpoint'], c['method'], c['baseline']]['CI_simultaneous'] is None]),
    ]
    tail = (1 - (1 - reg['family_error'] / reg['comparison_family'])) / 2
    max_interval_difference = 0.0
    checked_intervals = 0
    for path, comparisons in selected_sets:
        samples = np.load(path, mmap_mode='r')
        if samples.shape != (reg['bootstrap_replicates'], len(comparisons)):
            errors.append('bootstrap dimensions mismatch')
            continue
        for j, comparison in enumerate(comparisons):
            interval = np.quantile(samples[:, j], [tail, 1 - tail])
            diff = float(np.max(np.abs(interval - comparison['CI_simultaneous'])))
            max_interval_difference = max(max_interval_difference, diff)
            if diff > 1e-12:
                errors.append('saved bootstrap quantile mismatch')
            checked_intervals += 1
    source_unchanged = all(Path(p).exists() and hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
                           for p, h in reg['source_sha256'].items())
    original_unchanged = hashlib.sha256((root / 'final_analysis.json').read_bytes()).hexdigest() == diagnostic['original_analysis_sha256']
    if not source_unchanged or not original_unchanged:
        errors.append('frozen evidence changed')
    output = dict(passed=not errors, errors=errors, raw_rows=len(rows),
                  group_means_checked=len(formal['groups']), paired_means_checked=len(formal['comparisons']),
                  simultaneous_intervals_checked=checked_intervals,
                  max_mean_difference=max_mean_difference, max_interval_difference=max_interval_difference,
                  frozen_source_count=len(reg['source_sha256']), frozen_sources_unchanged=source_unchanged,
                  original_analysis_unchanged=original_unchanged,
                  interval_scope='Verifies saved resampling quantiles; does not establish population coverage or superiority')
    (root / 'final_statistics_audit.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    groups = {(g['endpoint'], g['method']): g for g in formal['groups']}
    diag = {(c['endpoint'], c['method'], c['baseline']): c for c in diagnostic['comparisons']}
    fields = ['method', 'label', 'information_role', 'pooled_mean', 'valid_n', 'shared_failure_cases',
              'formal_stable_vs_A_and_B', 'qualified_diagnostic_signal_vs_A_and_B',
              'qualified_delta_vs_B', 'qualified_CI_vs_B_low', 'qualified_CI_vs_B_high']
    with (root / 'all_method_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for method in sorted(reg['methods'], key=lambda m: groups['equal_scenario_mean', m]['mean']):
            group = groups['equal_scenario_mean', method]
            is_baseline = method in ['A_uniform', 'B_stage0', 'B_augmented29', 'B_augmented100']
            versus_b = diag.get(('equal_scenario_mean', method, 'B_stage0'))
            keys = [('equal_scenario_mean', method, b) for b in ['A_uniform', 'B_stage0']]
            writer.writerow(dict(method=method, label=group['label'], information_role='baseline_or_control' if is_baseline else 'candidate',
                                 pooled_mean=group['mean'], valid_n=group['n'], shared_failure_cases=group['failures'],
                                 formal_stable_vs_A_and_B='' if is_baseline else all(original[k]['stable_advantage'] for k in keys),
                                 qualified_diagnostic_signal_vs_A_and_B='' if is_baseline else all(diag[k]['qualified_case_replication_signal'] for k in keys),
                                 qualified_delta_vs_B='' if versus_b is None else versus_b['mean_delta'],
                                 qualified_CI_vs_B_low='' if versus_b is None else versus_b['CI_simultaneous'][0],
                                 qualified_CI_vs_B_high='' if versus_b is None else versus_b['CI_simultaneous'][1]))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('registration')
    result = audit(parser.parse_args().registration)
    print(json.dumps(result, ensure_ascii=False))
    if not result['passed']:
        raise SystemExit(1)
