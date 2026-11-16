import argparse
import shutil
import sys
from datetime import datetime,timezone
from pathlib import Path
from common import read,sha,save


def prepare(root):
    workspace=Path(__file__).resolve().parents[2]
    prior=workspace/'contribution_analysis/trajectory_calibration_experiment_v2_20261017'
    historical=read(workspace/'alpha_weight_outputs/p3_external_validity_formal_v1_20261006/clean/run_manifest.json')['source_snapshot_sha256']
    for n,h in historical.items():
        if sha(workspace/n)!=h:raise ValueError('historical_source_drift:'+n)
    old=read(prior/'cases.json');cases={'pilot':old['pilot'],'development':[c for c in old['development'] if c['key'].startswith('budget_')]}
    if (len(cases['pilot']),len(cases['development']))!=(4,20):raise ValueError('case_coverage')
    inputs={str(prior/'cases.json'):sha(prior/'cases.json')}
    for case in cases['pilot']+cases['development']:
        for n in ['models','evaluator']:inputs[case[n]]=sha(case[n])
        # Best-of expected outcomes are not the custom-only reference.
        for n in list(case):
            if n.startswith('expected_'):case.pop(n)
        case['history_mode']='objective'
    root.mkdir(exist_ok=False);snapshot=root/'source_snapshot';snapshot.mkdir()
    for n in historical:shutil.copy2(workspace/n,snapshot/n)
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,snapshot/p.name)
    docs={}
    for n in ['custom_direction_contribution_design_20261020.md','custom_direction_preregistration_20261020.md']:
        shutil.copy2(workspace/'docs'/n,root/n);docs[n]=sha(root/n)
    save(root/'cases.json',cases);inputs[str(root/'cases.json')]=sha(root/'cases.json')
    save(root/'input_sha256.json',inputs)
    save(root/'registration.json',dict(study='custom_direction_v1',registered_at_utc=datetime.now(timezone.utc).isoformat(),
        user_authorization='實作實驗並跑跑看效果如何',python_executable=sys.executable,
        source_sha256={p.name:sha(p) for p in snapshot.glob('*.py')},historical_source_sha256=historical,
        document_sha256=docs,model_samples_budget=500000,compute_seconds_budget=300,solves_budget=244,
        bootstrap_seed=20261020,bootstrap_repeats=10000,near_flat_reference=1e-4,
        custom_iterations=30,score='accepted_step_projection_D_over_path_length',new_confirmation=False))
    print(root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();prepare(a.root.resolve())
