import sys
import shutil
from pathlib import Path
from datetime import datetime,timezone
from common import read,save,sha


def prepare(root):
    repo=Path(__file__).resolve().parents[2]
    prior=repo/'contribution_analysis/custom_direction_experiment_v2_20261020'
    diagnostic=repo/'contribution_analysis/contribution_diagnostic_experiment_v1_20261020'
    cfg=read(prior/'registration.json'); cases=read(prior/'cases.json')
    assert len(cases['pilot'])==4 and len(cases['development'])==20
    inputs={str(prior/'cases.json'):sha(prior/'cases.json')}
    for n,h in cfg['historical_source_sha256'].items():
        assert sha(repo/n)==h and sha(prior/'source_snapshot'/n)==h, n
    for c in cases['pilot']+cases['development']:
        for name in ['models','evaluator']: inputs[c[name]]=sha(c[name])
        prep=Path(c['models']).parents[2]/'coalition.json'
        inputs[str(prep)]=sha(prep)
        p=prior/'baselines'/(c['key']+'.json'); inputs[str(p)]=sha(p)
        if c in cases['development']:
            for i in c['ids']:
                for p in [diagnostic/'anchor_removals'/(c['key']+'_'+str(i)+'.json'),prior/'removals'/(c['key']+'_'+str(i)+'.json')]:
                    inputs[str(p)]=sha(p)
    root.mkdir(exist_ok=False); snapshot=root/'source_snapshot';snapshot.mkdir()
    for n in cfg['historical_source_sha256']:shutil.copy2(prior/'source_snapshot'/n,snapshot/n)
    shutil.copy2(prior/'source_snapshot/observer.py',snapshot/'direction_observer.py')
    shutil.copy2(prior/'source_snapshot/experiment.py',snapshot/'legacy_experiment.py')
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,snapshot/p.name)
    docs={}
    for n in ['contribution_dual_role_economic_design_v2_20261020.md','dual_role_validity_preregistration_20261020.md']:
        shutil.copy2(repo/'docs'/n,root/n); docs[n]=sha(root/n)
    save(root/'cases.json',cases);inputs[str(root/'cases.json')]=sha(root/'cases.json')
    save(root/'input_sha256.json',inputs)
    save(root/'registration.json',dict(study='dual_role_v2',registered_at_utc=datetime.now(timezone.utc).isoformat(),
        previous_custom=str(prior),previous_diagnostic=str(diagnostic),python_executable=sys.executable,
        source_sha256={p.name:sha(p) for p in snapshot.glob('*.py')},document_sha256=docs,
        historical_source_sha256=cfg['historical_source_sha256'],near_flat_reference=1e-4,sign_deadzone=1e-4,
        bootstrap_seed=20261020,bootstrap_repeats=10000,interval_percentiles=[.625,99.375],
        new_solver_budget=180,model_sample_budget=200000,solver_seconds_budget=300,
        new_confirmation=False,scoring_evaluator_allowed=False,quality_source='existing Stage1 20-point common qualification',
        sign_metrics='descriptive only; no payment decision gate established'))
    print(root)


if __name__=='__main__':prepare(Path(sys.argv[1]).resolve())
