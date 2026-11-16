import argparse
import shutil
import sys
from pathlib import Path
from datetime import datetime,timezone
from common import read,save,sha

def prepare(root):
    workspace=Path(__file__).resolve().parents[2];old=workspace/'contribution_analysis/custom_direction_experiment_v2_20261020';local=workspace/'contribution_analysis/local_anchor_step_experiment_v1_20261020'
    cfg=read(old/'registration.json');historical=cfg['historical_source_sha256']
    for n,h in historical.items():
        if sha(workspace/n)!=h or sha(old/'source_snapshot'/n)!=h:raise ValueError('historical_source_drift:'+n)
    for p in [old,local]:
        c=read(p/'registration.json')
        for n,h in c['source_sha256'].items():
            if sha(p/'source_snapshot'/n)!=h:raise ValueError('previous_source_drift:'+n)
    cases=read(old/'cases.json')['development']
    if len(cases)!=20 or sum(len(c['ids']) for c in cases)!=80:raise ValueError('coverage')
    inputs={}
    for p in [old/'cases.json',old/'participant_comparison.csv',local/'participant_comparison.csv',local/'scores_lock.json']:
        inputs[str(p)]=sha(p)
    for c in cases:
        for n in ['models','evaluator']:inputs[c[n]]=sha(c[n])
        for p in [old/'baselines'/(c['key']+'.json'),local/'records'/(c['key']+'.json')]:inputs[str(p)]=sha(p)
        for i in c['ids']:
            p=old/'removals'/(c['key']+'_'+str(i)+'.json');inputs[str(p)]=sha(p)
    root.mkdir(exist_ok=False);snapshot=root/'source_snapshot';snapshot.mkdir()
    for n in historical:shutil.copy2(old/'source_snapshot'/n,snapshot/n)
    shutil.copy2(old/'source_snapshot/experiment.py',snapshot/'legacy_experiment.py')
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,snapshot/p.name)
    docs={}
    for n in ['contribution_diagnostic_next_steps_20261020.md','contribution_diagnostic_preregistration_20261020.md']:
        shutil.copy2(workspace/'docs'/n,root/n);docs[n]=sha(root/n)
    save(root/'cases.json',cases);inputs[str(root/'cases.json')]=sha(root/'cases.json');save(root/'input_sha256.json',inputs)
    save(root/'registration.json',dict(study='contribution_diagnostic_v1',registered_at_utc=datetime.now(timezone.utc).isoformat(),
        previous_custom=str(old),previous_local=str(local),python_executable=sys.executable,
        source_sha256={p.name:sha(p) for p in snapshot.glob('*.py')},document_sha256=docs,historical_source_sha256=historical,
        full_revaluation_sample_budget=340,anchor_solves_budget=80,anchor_sample_budget=200000,anchor_solver_seconds_budget=300,
        bootstrap_seed=20261020,bootstrap_repeats=10000,near_flat_effect_range=1e-4,custom_iterations=30,new_confirmation=False))
    print(root)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();prepare(a.root.resolve())
