import argparse
import shutil
import sys
from pathlib import Path
from datetime import datetime,timezone
from common import read,save,sha

def prepare(root):
    workspace=Path(__file__).resolve().parents[2];old=workspace/'contribution_analysis/custom_direction_experiment_v2_20261020'
    oldcfg=read(old/'registration.json');historical=oldcfg['historical_source_sha256']
    for n,h in historical.items():
        if sha(workspace/n)!=h or sha(old/'source_snapshot'/n)!=h:raise ValueError('historical_source_drift:'+n)
    for n,h in oldcfg['source_sha256'].items():
        if sha(old/'source_snapshot'/n)!=h:raise ValueError('previous_snapshot_drift:'+n)
    inputs={str(old/'cases.json'):sha(old/'cases.json'),str(old/'participant_comparison.csv'):sha(old/'participant_comparison.csv'),
            str(old/'registration.json'):sha(old/'registration.json')}
    cases=read(old/'cases.json')
    for case in cases['pilot']+cases['development']:
        for name in ['models','evaluator']:inputs[case[name]]=sha(case[name])
        p=old/'baselines'/(case['key']+'.json');inputs[str(p)]=sha(p)
        for i in case['ids']:
            p=old/'removals'/(case['key']+'_'+str(i)+'.json');inputs[str(p)]=sha(p)
    root.mkdir(exist_ok=False);snapshot=root/'source_snapshot';snapshot.mkdir()
    for n in historical:shutil.copy2(old/'source_snapshot'/n,snapshot/n)
    for source,target in [('observer.py','direction_observer.py'),('experiment.py','legacy_experiment.py')]:
        shutil.copy2(old/'source_snapshot'/source,snapshot/target)
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,snapshot/p.name)
    docs={}
    for n in ['local_anchor_step_value_proposal_20261020.md','local_anchor_step_preregistration_20261020.md']:
        shutil.copy2(workspace/'docs'/n,root/n);docs[n]=sha(root/n)
    save(root/'cases.json',cases);inputs[str(root/'cases.json')]=sha(root/'cases.json');save(root/'input_sha256.json',inputs)
    save(root/'registration.json',dict(study='local_anchor_step_v1',registered_at_utc=datetime.now(timezone.utc).isoformat(),
        previous=str(old),labels_path=str(old/'participant_comparison.csv'),python_executable=sys.executable,
        user_authorization='根據實驗提案執行實驗做比較，再將實驗結果整理給我',
        source_sha256={p.name:sha(p) for p in snapshot.glob('*.py')},document_sha256=docs,historical_source_sha256=historical,
        sample_budget=500000,solves_budget=450,compute_seconds_budget=300,wall_seconds_budget=900,
        bootstrap_seed=20261020,bootstrap_repeats=10000,near_flat_reference=1e-4,new_confirmation=False,
        main_score='A=sum(min(original_loss,trial_without_anchor_loss)-original_accepted_loss)',
        benchmark_pairs=15,benchmark_warmups=2))
    print(root)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();prepare(a.root.resolve())
