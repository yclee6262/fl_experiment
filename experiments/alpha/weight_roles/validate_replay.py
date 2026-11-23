"""Check all unshifted controls against task nine and smoke numerical replay."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text())
def lines(path): return [json.loads(s) for s in path.read_text().splitlines()]


def stripped(value):
    if isinstance(value, dict):
        return {k: stripped(v) for k, v in value.items() if k not in ['seconds', 'trace_sha256']}
    if isinstance(value, list):
        return [stripped(v) for v in value]
    return value


def validate(source, reference, smoke_reference=None):
    output = source/'replay_acceptance.json'
    if output.exists():
        raise FileExistsError(output)
    assert read(source/'manifest.json')['status'] == read(reference/'manifest.json')['status'] == 'complete'
    assert read(reference/'acceptance.json')['passed']
    conditions = records = smoke_conditions = traces = 0
    for file in sorted(source.glob('seed*_control_*/case.json')):
        case = read(file)
        previous = reference/f"reference_clean_{case['seed']}_T{case['target']:g}"
        oldcase = read(previous/'case.json')
        assert sha(previous/'results.jsonl') == read(previous/'completion.json')['results_sha256']
        for field in ['anchors', 'inverses', 'blind_X', 'blind_y', 'mse', 'blind_predictions', 'peer_predictions']:
            assert case[field] == oldcase[field], (file, field)
        old = {r['label']: r for r in lines(previous/'results.jsonl')}
        current = [r for r in lines(file.parent/'searches.jsonl') if r['search_id'].startswith('coupled_')]
        assert len(current) == 10
        for row in current:
            other = old[row['label']]
            for field in ['X', 'loss', 'status', 'model_queries', 'panel_evaluations', 'chosen_engine', 'truth_error']:
                assert row[field] == other[field], (file, row['label'], field)
            assert row['objective_alpha'] == other['alpha']
            assert sha(file.parent/row['trace_path']) == row['trace_sha256']
            assert sha(previous/other['trace_path']) == other['trace_sha256']
            with gzip.open(file.parent/row['trace_path'], 'rt') as f:
                current_trace = json.load(f)
            with gzip.open(previous/other['trace_path'], 'rt') as f:
                old_trace = json.load(f)
            assert current_trace == old_trace
            records += 1
            traces += 1
        conditions += 1
    assert conditions == 180 and records == 1800
    if smoke_reference:
        assert read(smoke_reference/'acceptance.json')['passed']
        for file in sorted(smoke_reference.glob('seed*/case.json')):
            formal = source/file.parent.name
            assert file.read_bytes() == (formal/'case.json').read_bytes()
            for name in ['decisions.jsonl', 'searches.jsonl', 'factorial.jsonl']:
                assert stripped(lines(file.parent/name)) == stripped(lines(formal/name))
            for path in file.parent.glob('*.json.gz'):
                with gzip.open(path, 'rt') as f:
                    smoke_trace = json.load(f)
                with gzip.open(formal/path.name, 'rt') as f:
                    formal_trace = json.load(f)
                assert smoke_trace == formal_trace
                traces += 1
            smoke_conditions += 1
        assert smoke_conditions == 6
    report = dict(passed=True, unshifted_conditions=conditions, task9_decisions_identical=records,
        smoke_conditions_identical=smoke_conditions, decoded_traces_identical=traces,
        excluded_replay_fields=['seconds', 'trace_sha256 (gzip creation metadata)'],
        reference_manifest_sha256=sha(reference/'manifest.json'),
        source_manifest_sha256=sha(source/'manifest.json'), validator_sha256=sha(Path(__file__)),
        scope='Regression identity, not independent confirmation; no new model calls')
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--smoke-reference', type=Path)
    a = p.parse_args()
    validate(a.source, a.reference, a.smoke_reference)
