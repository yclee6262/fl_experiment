"""Prepare a NEW immutable run when explicitly invoked; performs no solves."""
import argparse
import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime,timezone
from pathlib import Path


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open() as f:return list(csv.DictReader(f))


def prepare(root,confirmation_start):
    workspace=next(p for p in Path(__file__).resolve().parents
                   if (p/'docs/contribution_analysis_research.md').is_file())
    old=workspace/'alpha_weight_outputs/p3_external_validity_formal_v1_20261006'
    budget=workspace/'contribution_analysis/full_removal_budget_experiment_v1_20261016'
    sources=read(old/'clean/run_manifest.json')['source_snapshot_sha256']
    cases={'pilot':[],'development':[]};labels=[];inputs={}
    for scenario in ['clean','controlled_poison']:
        base=old/scenario
        if read(base/'run_manifest.json')['source_snapshot_sha256']!=sources:
            raise ValueError('historical_source_version_difference')
        for n in ['run_manifest.json','participant_effects.csv','scenario_summary.csv']:inputs[str(base/n)]=sha(base/n)
        effects=rows(base/'participant_effects.csv')
        for summary in rows(base/'scenario_summary.csv'):
            seed=int(summary['seed']);subset=[r for r in effects if int(r['seed'])==seed]
            ids=json.loads(summary['selected_agent_ids']);byid={int(r['agent_id']):r for r in subset}
            artifact=base/summary['scenario_artifact_dir'];key=f'p3_{scenario}_{seed}'
            case=dict(key=key,scenario=scenario,seed=seed,ids=ids,
                candidates=[json.loads(byid[i]['candidate_point']) for i in ids],alphas=[float(byid[i]['alpha']) for i in ids],
                models=str(artifact/'local_models.pt'),evaluator=str(artifact/'global_evaluator.pt'),
                expected_X=json.loads(summary['baseline_solution']),fixed_start=json.loads(summary['fixed_initial_solution']),
                expected_calls=int(summary['baseline_stage3_requests']),expected_pooled_error=float(summary['baseline_external_error']),
                history_mode='diagnostic')
            cases['development'].append(case)
            for r in subset:labels.append(dict(key=key,scenario=scenario,seed=seed,agent_id=int(r['agent_id']),C_full=float(r['full_removal_external_contribution'])))
    for phase,seeds in [('pilot',range(2026101600,2026101602)),('confirmation',range(2026101610,2026101620))]:
        for scenario in ['clean','controlled_poison']:
            for seed in seeds:
                d=budget/phase/scenario/str(seed);state=read(d/'preparation.json');sol=read(d/'solves/A_-1.json')
                artifact=d/state['artifact'];key=f'budget_{scenario}_{seed}'
                score=next(r for r in read(d/'scores.json')['rows'] if r['file']=='A_-1.json')
                case=dict(key=key,scenario=scenario,seed=seed,ids=state['ids'],candidates=state['candidates'],
                    alphas=state['alphas'],fixed_start=state['fixed_start'],models=str(artifact/'local_models.pt'),
                    evaluator=str(artifact/'global_evaluator.pt'),expected_X=sol['result']['solution'],
                    expected_calls=sol['calls'],expected_loss=sol['result']['loss'],expected_pooled_error=score['pooled_error'],
                    history_mode='objective')
                cases['pilot' if phase=='pilot' else 'development'].append(case)
                for p in [d/'preparation.json',d/'solves/A_-1.json',d/'scores.json']:inputs[str(p)]=sha(p)
    effect_file=budget/'recovery_analysis/participant_effects.csv';inputs[str(effect_file)]=sha(effect_file)
    for r in rows(effect_file):
        if r['phase']=='confirmation':
            labels.append(dict(key=f"budget_{r['scenario']}_{r['seed']}",scenario=r['scenario'],seed=int(r['seed']),agent_id=int(r['agent_id']),C_full=float(r['C_A'])))
    for c in cases['pilot']+cases['development']:
        for name in ['models','evaluator']:inputs[c[name]]=sha(Path(c[name]))
    if (len(cases['pilot']),len(cases['development']),len(labels))!=(4,80,320):raise ValueError('unexpected_source_coverage')
    seeds=list(range(confirmation_start,confirmation_start+20))
    known={c['seed'] for c in cases['pilot']+cases['development']}
    if known&set(seeds):raise ValueError('confirmation_seed_overlap')
    if min(seeds)<0 or max(seeds)+104729>=2**32:raise ValueError('confirmation_seed_range')
    # Inspect experiment metadata only, not historical source snapshots.
    conflicts=[]
    for base in [workspace/'contribution_analysis',workspace/'alpha_weight_outputs']:
        for pattern in ['**/run_manifest.json','**/registration.json']:
            for p in base.glob(pattern):
                if 'source_snapshot' in str(p):continue
                try:payload=read(p)
                except (ValueError,OSError):continue
                def integers(x):
                    if isinstance(x,bool):return []
                    if isinstance(x,int):return [x]
                    if isinstance(x,list):return [v for y in x for v in integers(y)]
                    if isinstance(x,dict):return [v for y in x.values() for v in integers(y)]
                    return []
                if set(seeds)&set(integers(payload)):conflicts.append(str(p))
    if conflicts:raise ValueError('confirmation_seeds_already_registered:'+repr(conflicts[:5]))
    root.mkdir(parents=True,exist_ok=False);snapshot=root/'source_snapshot';snapshot.mkdir()
    source_hashes={}
    for name,h in sources.items():
        if sha(workspace/name)!=h:raise ValueError('historical_algorithm_drift:'+name)
        shutil.copy2(workspace/name,snapshot/name);source_hashes[name]=h
    for p in Path(__file__).parent.glob('*.py'):
        shutil.copy2(p,snapshot/p.name);source_hashes[p.name]=sha(p)
    reference=workspace/'contribution_analysis/full_removal_budget_v1/worker.py'
    shutil.copy2(reference,snapshot/'reference_worker.py');source_hashes['reference_worker.py']=sha(reference)
    for name,obj in [('cases.json',cases),('labels.json',labels),('input_sha256.json',inputs)]:
        (root/name).write_text(json.dumps(obj,indent=2)+'\n')
    config=dict(study='trajectory_calibration_v2_saved_start',status='prepared_not_executed',
       registered_at_utc=datetime.now(timezone.utc).isoformat(),source_sha256=source_hashes,
       confirmation_seeds=seeds,confirmation_batches=[seeds[:10],seeds[10:]],confirmation_seed_conflicts=conflicts,
       group_folds=5,fold_seed=20261016,bootstrap_seed=20261016,bootstrap_replicates=10000,
       simultaneous_confidence=.9875,neutral_delta=1e-4,depth=2,min_leaf_distinct_seeds=5,
       pilot_pairs_per_case=15,pilot_cost_ratio_limit=1.10,development_calls_budget=500000,
       confirmation_calls_budget=500000,confirmation_solver_seconds_budget=1800,
       confirmation_preparation_seconds_budget=3600,
       original_stopped_run='contribution_analysis/trajectory_calibration_experiment_v1_20261016',
       saved_start_policy='read original stored X0; never recompute from serialized candidate means')
    (root/'registration.json').write_text(json.dumps(config,indent=2)+'\n')
    for name in ['trajectory_calibration_preregistration_20261016.md','trajectory_calibration_implementation_20261016.md']:
        source=workspace/'docs'/name
        if source.exists():shutil.copy2(source,root/name)
    # Freeze documentation hashes separately; preparing a run still executes no solver.
    (root/'documentation_sha256.json').write_text(json.dumps(
        {p.name:sha(p) for p in root.glob('*.md')},indent=2)+'\n')
    return root


if __name__=='__main__':
    p=argparse.ArgumentParser(description='Prepare files only; does not train or solve.')
    p.add_argument('root',type=Path);p.add_argument('--confirmation-start',type=int,required=True)
    args=p.parse_args();print(prepare(args.root.resolve(),args.confirmation_start))
