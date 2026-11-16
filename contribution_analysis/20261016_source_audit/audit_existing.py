"""Read-only post-hoc audit; no training, optimization, or historical writes."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'alpha_weight_outputs/p3_external_validity_formal_v1_20261006'
OUT = Path(__file__).resolve().parent

def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def sign(x):
    return int(float(x) > 1e-7) - int(float(x) < -1e-7)

def save(name, obj):
    with (OUT / name).open('x') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write('\n')

result = {'status': 'posthoc_source_audit_not_formal_confirmation', 'scenarios': {}}
sources = {}
cases = []
for scenario in ['clean', 'controlled_poison']:
    base = SOURCE / scenario
    manifest = json.loads((base / 'run_manifest.json').read_text())
    rows = read(base / 'participant_effects.csv')
    summaries = read(base / 'scenario_summary.csv')
    drift = {}
    for name, expected in manifest['source_snapshot_sha256'].items():
        assert sha(base / 'source_snapshot' / name) == expected
        drift[name] = sha(ROOT / name) != expected
    for name, expected in manifest['artifact_sha256'].items():
        assert sha(base / name) == expected
    fields = ['operational_internal_contribution', 'operational_external_contribution',
              'full_removal_external_contribution']
    for row in rows:
        assert row['delivery_status'] == 'valid_delivery'
        assert abs(float(row[fields[1]]) - (float(row['operational_external_error_without']) - float(row['baseline_external_error']))) < 1e-12
        assert abs(float(row[fields[2]]) - (float(row['full_removal_external_error_without']) - float(row['baseline_external_error']))) < 1e-12
        assert abs(sum(json.loads(row['full_removal_alphas'])) - 1) < 1e-12
        assert int(row['agent_id']) not in json.loads(row['full_removal_agent_ids'])
    # Saved candidates also permit checking geometry without rerunning a solver.
    projection_residuals = []
    ranks = Counter()
    for row in rows:
        same = [r for r in rows if r['seed'] == row['seed'] and r['agent_id'] != row['agent_id']]
        matrix = np.array([json.loads(r['candidate_point']) for r in same]).T
        start = np.array(json.loads(row['fixed_initial_solution']))
        projection_residuals.append(float(np.linalg.norm(matrix @ np.linalg.lstsq(matrix, start, rcond=None)[0] - start)))
        ranks[int(np.linalg.matrix_rank(matrix))] += 1
    stat = {
        'n_seeds': len(summaries), 'n_effects': len(rows),
        'directions_positive_neutral_negative': {f: dict(Counter(sign(r[f]) for r in rows)) for f in fields},
        'current_source_differs_from_formal_snapshot': drift,
        'remaining_linear_ranks': dict(ranks), 'max_fixed_start_linear_projection_residual': max(projection_residuals),
        'solver_labels': {f: dict(Counter(r[f] for r in rows)) for f in ['baseline_solver','operational_solver','full_removal_engine']},
        'mean_costs_per_scenario': {f: sum(float(r[f]) for r in summaries)/len(summaries) for f in ['baseline_stage3_requests','candidate_exclusion_requests','full_removal_requests','candidate_exclusion_seconds','full_removal_seconds']},
        'saved_reevaluation_rows': len(read(base / 'reevaluation.csv')),
        'op_negative_full_positive': sum(sign(r[fields[0]]) == -1 and sign(r[fields[2]]) == 1 for r in rows),
    }
    for row in rows:
        summary = next(s for s in summaries if s['seed'] == row['seed'])
        assert abs(float(row[fields[0]]) - (float(row['operational_internal_loss_without']) - float(summary['baseline_internal_loss']))) < 1e-12
    ordered = sorted(rows, key=lambda r:(int(r['seed']), int(r['agent_id'])))
    predicates = {
        'op_negative_full_positive': lambda r:sign(r[fields[0]]) == -1 and sign(r[fields[2]]) == 1,
        'same_nonzero_direction': lambda r:sign(r[fields[0]]) == sign(r[fields[2]]) != 0,
        'op_neutral': lambda r:sign(r[fields[0]]) == 0,
        'pooled_truth_conflict': lambda r:sign(r[fields[2]]) != sign(r['full_removal_true_contribution']),
    }
    for category, predicate in predicates.items():
        matched = [r for r in ordered if predicate(r)]
        if not matched:
            cases.append({'scenario':scenario,'category':category,'available':False})
            continue
        r = dict(matched[0])
        b = np.array(json.loads(r['baseline_solution']))
        r.update(category=category, available=True,
                 operational_X_distance=float(np.linalg.norm(np.array(json.loads(r['operational_restricted_solution']))-b)),
                 full_X_distance=float(np.linalg.norm(np.array(json.loads(r['full_removal_solution']))-b)))
        cases.append(r)
    result['scenarios'][scenario] = stat
    for name in ['run_manifest.json','participant_effects.csv','scenario_summary.csv','reevaluation.csv']:
        sources[str((base/name).relative_to(ROOT))] = sha(base/name)
for path in (SOURCE/'analysis_v3').rglob('*'):
    if path.is_file(): sources[str(path.relative_to(ROOT))] = sha(path)
save('audit_summary.json', result)
save('representative_cases.json', cases)
save('source_hashes.json', sources)
print(json.dumps(result, indent=2, ensure_ascii=False))
print('CASES', [(r['scenario'],r['category'],r.get('seed'),r.get('agent_id')) for r in cases])
