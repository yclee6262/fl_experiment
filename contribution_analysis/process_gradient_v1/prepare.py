"""Freeze source and existing input artifacts before computing new scores."""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(root):
    workspace = Path(__file__).resolve().parents[2]
    previous = workspace/'contribution_analysis/trajectory_calibration_experiment_v2_20261017'
    config = json.loads((previous/'registration.json').read_text())
    historical = json.loads((workspace/'alpha_weight_outputs/p3_external_validity_formal_v1_20261006/clean/run_manifest.json').read_text())['source_snapshot_sha256']
    for name, expected in historical.items():
        if sha(workspace/name) != expected:
            raise ValueError('historical_source_drift:'+name)
    cases = json.loads((previous/'cases.json').read_text())
    inputs = {str(previous/name): sha(previous/name) for name in
              ['cases.json', 'labels.json', 'registration.json', 'extraction_audit.json']}
    for case in cases['pilot']+cases['development']:
        inputs[case['models']] = sha(Path(case['models']))
    for case in cases['development']:
        p = previous/'records'/(case['key']+'.json')
        inputs[str(p)] = sha(p)
    for name in list(historical)+['recorder.py', 'experiment.py']:
        if sha(previous/'source_snapshot'/name) != config['source_sha256'][name]:
            raise ValueError('previous_frozen_source_drift:'+name)
    root.mkdir(exist_ok=False)
    frozen = root/'source_snapshot'; frozen.mkdir()
    for name in list(historical)+['recorder.py']:
        shutil.copy2(previous/'source_snapshot'/name, frozen/name)
    shutil.copy2(previous/'source_snapshot/experiment.py', frozen/'legacy_experiment.py')
    for p in Path(__file__).parent.glob('*.py'):
        shutil.copy2(p, frozen/p.name)
    shutil.copy2(workspace/'docs/process_gradient_preregistration_20261018.md', root/'preregistration.md')
    shutil.copy2(previous/'cases.json', root/'cases.json')
    registration = dict(study='process_gradient_v1', frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        previous=str(previous), labels_path=str(previous/'labels.json'),
        method='left_endpoint_black_box_gradient_work', h=.001, target=0., delta=1e-4,
        no_fitting=True, independent_confirmation=False, source_sha256={p.name:sha(p) for p in frozen.glob('*.py')},
        historical_source_sha256=historical,
        preregistration_sha256=sha(root/'preregistration.md'), cases_sha256=sha(root/'cases.json'),
        python_executable=__import__('sys').executable)
    for name, obj in [('registration.json',registration), ('input_sha256.json',inputs)]:
        with (root/name).open('x') as f: json.dump(obj,f,indent=2); f.write('\n')
    print(root)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('root', type=Path); a=p.parse_args(); prepare(a.root.resolve())
