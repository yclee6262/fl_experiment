"""Audit stored engine choices in the frozen ten-initializer cohort (no solves)."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[3]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(registration, output):
    reg = json.loads(registration.read_text())
    rows, hashes, seen = [], {}, set()
    for batch in reg['batches'][:3]:
        for kind in ('six_method_results', 'pure_results', 'results'):
            path = ROOT / batch[kind] / 'results.jsonl'
            hashes[str(path.relative_to(ROOT))] = digest(path)
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (row['scenario'], row['seed'], row['method'])
                if key in seen:
                    raise ValueError(f'Duplicate case: {key}')
                seen.add(key)
                rows.append(row)
    expected_hashes = reg.get('original_results_sha256', {})
    for name, expected in expected_hashes.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Frozen source changed: {name}')

    def summary(selected):
        choices = Counter()
        pairs = []
        missing = []
        for r in selected:
            solver = r.get('solver') or {}
            choices[solver.get('chosen_engine', 'missing')] += 1
            loss = solver.get('engine_losses') or {}
            a, b = loss.get('bfgs'), loss.get('custom')
            if a is None or b is None:
                missing.append([r['scenario'], r['seed'], r['method']])
            else:
                pairs.append((a, b))
        return dict(rows=len(selected), statuses=dict(Counter(r.get('status') for r in selected)),
                    choices=dict(choices), paired_losses=len(pairs), missing_losses=missing,
                    coefficient_mean_loss=statistics.mean(a for a,b in pairs) if pairs else None,
                    custom_mean_loss=statistics.mean(b for a,b in pairs) if pairs else None,
                    coefficient_strictly_better=sum(a < b for a,b in pairs),
                    custom_strictly_better=sum(b < a for a,b in pairs),
                    exact_ties=sum(a == b for a,b in pairs))

    result = {'overall': summary(rows)}
    for field in ('method', 'scenario'):
        groups = defaultdict(list)
        for row in rows:
            groups[row[field]].append(row)
        result['by_' + field] = {k: summary(v) for k,v in sorted(groups.items())}
    result['scope'] = ('Historical records, not new independent cases. The bfgs key denotes '
                       'the coefficient engine; these box-constrained cases use SLSQP. '
                       'Engine choice alone does not establish numerical failure. '
                       'Original records do not retain each losing engine endpoint or its start.')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    (output / 'manifest.json').write_text(json.dumps(dict(
        registration=str(registration.resolve()), registration_sha256=digest(registration),
        source_sha256=hashes, script_sha256=digest(Path(__file__)),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        new_model_queries=0), indent=2) + '\n')
    print(json.dumps(result['overall']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, default=ROOT / 'alpha_weight_outputs/se_qu_extension_v1_20261107/registration.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    audit(args.registration, args.output_dir)
