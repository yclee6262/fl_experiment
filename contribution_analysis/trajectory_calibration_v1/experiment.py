"""Frozen development experiment; fail closed before new-data confirmation."""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x
def save(p,obj):
    with Path(p).open('x') as f:json.dump(plain(obj),f,ensure_ascii=False,indent=2,allow_nan=False)
def csvsave(p,rows):
    with Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['status']);w.writeheader();w.writerows(rows)


def load_server(case):
    import torch
    from host_server import HostServer
    from agent_client import AgentNode
    payload=torch.load(case['models'],map_location='cpu',weights_only=True)
    s=HostServer(0.,2,constraint_mode='box_constrained')
    for i in case['ids']:
        agent=AgentNode(i,None,n_features=2);agent.model.load_state_dict(payload['agents'][i]);agent.model.eval()
        s.trusted_agents.append(agent)
    s.I_list=np.array(case['candidates']);s.alphas=list(case['alphas'])
    return s


def verify(case,result):
    x=np.asarray(result['result']['solution'])
    if not np.isfinite(x).all() or np.max(np.abs(x))>1+1e-8:raise ValueError('invalid_delivery')
    dx=float(np.max(np.abs(x-np.array(case['expected_X']))))
    if dx>1e-7:raise ValueError(f'historical_replay_X_mismatch:{dx}')
    if result['counter']['calls']!=case['expected_calls']:raise ValueError('historical_call_count_mismatch')
    if case.get('expected_loss') is not None and abs(result['result']['loss']-case['expected_loss'])>1e-7:
        raise ValueError('historical_loss_mismatch')
    return dx


def extract(root):
    import torch
    from recorder import solve,features,FEATURES
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    cases=read(root/'cases.json');config=read(root/'registration.json')
    for name,h in config['source_sha256'].items():
        if sha(root/'source_snapshot'/name)!=h:raise ValueError('source_hash_mismatch')
    outputs=root/'records';outputs.mkdir(exist_ok=False)
    pilot=[];totals=dict(calls=0,samples=0,solver_seconds=0.,solves=0)
    def count(r):
        totals['calls']+=r['counter']['calls'];totals['samples']+=r['counter']['samples'];totals['solver_seconds']+=r['seconds'];totals['solves']+=1
        if totals['calls']>500000 or totals['samples']>500000 or totals['solver_seconds']>300:
            raise RuntimeError('development_resource_budget')
    try:
        for case in cases['pilot']:
            print('PILOT',case['key'],flush=True)
            server=load_server(case)
            for _ in range(2):count(solve(server,False,case['history_mode'],initial_solution=case['fixed_start']))
            controls=[];records=[];feature_times=[]
            for repeat in range(15):
                pair={}
                for enabled in ([False,True] if repeat%2==0 else [True,False]):
                    r=solve(server,enabled,case['history_mode'],initial_solution=case['fixed_start'],max_calls=20000,max_samples=20000,max_seconds=60);count(r);verify(case,r)
                    if enabled:
                        start=time.perf_counter();z=features(r['trace'],server.I_list,server.alphas)
                        extra=time.perf_counter()-start;feature_times.append(extra);records.append(r['seconds']+extra)
                    else:controls.append(r['seconds'])
                    pair[enabled]=r
                if pair[False]['counter']!=pair[True]['counter']:raise ValueError('extra_model_queries')
                if pair[False]['result']['loss']!=pair[True]['result']['loss']:raise ValueError('recorder_changed_loss')
                if not np.array_equal(pair[False]['result']['solution'],pair[True]['result']['solution']):raise ValueError('recorder_changed_X')
            pilot.append(dict(key=case['key'],control_seconds=controls,record_and_features_seconds=records,
                    median_ratio=float(np.median(records)/np.median(controls)),calls_equal=True,
                    median_feature_seconds=float(np.median(feature_times)),features=z))
        ratio=float(sum(np.median(p['record_and_features_seconds']) for p in pilot)/sum(np.median(p['control_seconds']) for p in pilot))
        save(root/'pilot_results.json',dict(cases=pilot,aggregate_timing_ratio=ratio,
             technical_accepted=True,cost_target_met=ratio<=1.10,totals=totals,
             inference_cost_not_yet_included=True))
        # Ten-percent overhead is an explicit proposal target; do not hide a failed gate.
        if ratio>1.10:
            save(root/'extraction_execution.json',dict(status='stopped_pilot_cost',totals=totals,
                 reason='recording_and_features_exceed_10_percent_overhead'));return
        rows=[]
        for case in cases['development']:
            print('DEVELOPMENT',case['key'],flush=True)
            server=load_server(case);r=solve(server,True,case['history_mode'],initial_solution=case['fixed_start'],max_calls=20000,max_samples=20000,max_seconds=60);count(r);dx=verify(case,r)
            begin=time.perf_counter();z=features(r['trace'],server.I_list,server.alphas);feature_seconds=time.perf_counter()-begin
            for i,agent_id in enumerate(case['ids']):
                rows.append(dict(key=case['key'],scenario=case['scenario'],seed=case['seed'],agent_id=agent_id,
                                 **{name:float(z[i,j]) for j,name in enumerate(FEATURES)}))
            save(outputs/(case['key']+'.json'),dict(status='complete',case_key=case['key'],seed=case['seed'],
                 scenario=case['scenario'],ids=case['ids'],counter=r['counter'],seconds=r['seconds'],
                 feature_seconds=feature_seconds,chosen_engine=r['result']['chosen_engine'],
                 solution=r['result']['solution'],loss=r['result']['loss'],trace=r['trace'],features=z,
                 replay_X_max_delta=dx,models_sha256=sha(case['models']),
                 evaluator_loaded=False,labels_loaded=False))
        csvsave(root/'development_features.csv',rows)
        save(root/'extraction_execution.json',dict(status='complete',totals=totals,n_rows=len(rows)))
    except Exception as exc:
        save(root/'extraction_execution.json',dict(status='stopped_technical_failure',reason=f'{type(exc).__name__}:{exc}',totals=totals))
        raise


def evaluate(root):
    import rules
    from recorder import FEATURES
    state=read(root/'extraction_execution.json')
    if state['status']!='complete':
        save(root/'development_gate.json',dict(accepted=False,status=state['status'],independent_confirmation_started=False));return
    audit=read(root/'extraction_audit.json')
    if not audit['accepted']:
        save(root/'development_gate.json',dict(accepted=False,status='extraction_audit_failed',independent_confirmation_started=False));return
    rows=list(csv.DictReader((root/'development_features.csv').open()));labels=read(root/'labels.json')
    label_map={(r['key'],int(r['agent_id'])):r['C_full'] for r in labels}
    X=np.array([[float(r[f]) for f in FEATURES] for r in rows]);y=np.array([label_map[r['key'],int(r['agent_id'])] for r in rows])
    groups=np.array([int(r['seed']) for r in rows]);m=X[:,5].astype(int);weights=1/(2*m)
    unique=np.unique(groups);np.random.default_rng(20261016).shuffle(unique)
    fold_map={int(g):i%5 for i,g in enumerate(unique)};folds=np.array([fold_map[int(g)] for g in groups])
    names=['zero','constant','model_only','candidate_only','full']
    pred={n:np.empty(len(y)) for n in names};models=[]
    for fold in range(5):
        train=folds!=fold;test=~train
        if set(groups[train])&set(groups[test]):raise ValueError('seed_leakage')
        c=rules.constant_fit(y[train],m[train],weights[train]);pred['zero'][test]=0.;pred['constant'][test]=rules.constant_predict(c,m[test])
        model_bundle=dict(fold=fold,training_seeds=sorted(map(int,set(groups[train]))),validation_seeds=sorted(map(int,set(groups[test]))),constant=c)
        for name,cols in [('model_only',[0,1,2,5,6,7]),('candidate_only',[3,4,5,6,7]),('full',list(range(8)))]:
            tree=rules.fit(X[train],y[train],groups[train],weights[train],cols)
            pred[name][test]=rules.predict(tree,X[test]);model_bundle[name]=tree
        models.append(model_bundle)
    output=[];metrics=[];seed_metrics=[]
    def signs(a,d=1e-4):return (a>d).astype(int)-(a < -d).astype(int)
    for k,r in enumerate(rows):output.append(dict(**r,fold=int(folds[k]),C_full=float(y[k]),**{n:float(pred[n][k]) for n in names}))
    for scenario in ['clean','controlled_poison']:
        mask=np.array([r['scenario']==scenario for r in rows]);truth=signs(y[mask])
        for name in names:
            estimate=pred[name][mask];p=signs(estimate)
            true_negative=truth==-1;pred_negative=p==-1;tp=int(np.sum(true_negative&pred_negative))
            metrics.append(dict(scenario=scenario,method=name,n_effects=int(mask.sum()),n_seeds=len(set(groups[mask])),
              MAE=float(np.mean(abs(estimate-y[mask]))),direction_agreement=float(np.mean(p==truth)),
              true_negative=int(true_negative.sum()),predicted_negative=int(pred_negative.sum()),
              negative_precision=None if not pred_negative.any() else tp/int(pred_negative.sum()),
              negative_recall=None if not true_negative.any() else tp/int(true_negative.sum())))
        for seed in sorted(set(groups[mask])):
            take=mask&(groups==seed)
            seed_metrics.append(dict(scenario=scenario,seed=int(seed),constant_MAE=float(np.mean(abs(pred['constant'][take]-y[take]))),
                full_MAE=float(np.mean(abs(pred['full'][take]-y[take]))),
                MAE_improvement=float(np.mean(abs(pred['constant'][take]-y[take])-abs(pred['full'][take]-y[take]))),
                direction_agreement=float(np.mean(signs(pred['full'][take])==signs(y[take])))))
    by={(r['scenario'],r['method']):r for r in metrics}
    improvement={s:(by[s,'constant']['MAE']-by[s,'full']['MAE'])/by[s,'constant']['MAE'] for s in ['clean','controlled_poison']}
    improvement_pass=all(v>0 for v in improvement.values())
    for name,cols in [('model_only',[0,1,2,5,6,7]),('candidate_only',[3,4,5,6,7]),('full',list(range(8)))]:
        save(root/(name+'_development_model.json'),rules.fit(X,y,groups,weights,cols))
    save(root/'constant_development_model.json',rules.constant_fit(y,m,weights))
    final_tree=read(root/'full_development_model.json')
    pilot=read(root/'pilot_results.json');timing=[]
    for case in pilot['cases']:
        z=np.asarray(case['features']);times=[]
        for _ in range(15):
            begin=time.perf_counter();rules.predict(final_tree,z);times.append(time.perf_counter()-begin)
        timing.append(dict(key=case['key'],prediction_seconds_median=float(np.median(times)),
             recorded_seconds_median=float(np.median(case['record_and_features_seconds'])),
             control_seconds_median=float(np.median(case['control_seconds']))))
    cost_ratio=sum(r['prediction_seconds_median']+r['recorded_seconds_median'] for r in timing)/sum(r['control_seconds_median'] for r in timing)
    cost_pass=cost_ratio<=1.10
    passed=improvement_pass and cost_pass
    save(root/'complete_online_cost_gate.json',dict(accepted=cost_pass,aggregate_ratio=cost_ratio,rows=timing))
    csvsave(root/'development_predictions.csv',output);csvsave(root/'development_metrics.csv',metrics);csvsave(root/'development_seed_metrics.csv',seed_metrics)
    save(root/'development_folds.json',models)
    save(root/'development_gate.json',dict(accepted=passed,
         status='development_gate_passed' if passed else 'development_improvement_not_established' if not improvement_pass else 'online_cost_gate_failed',
         MAE_relative_improvement=improvement,independent_confirmation_started=False,
         interpretation='Grouped out-of-fold DEVELOPMENT evidence only; no independent validation claim.'))
    save(root/'fitted_models_manifest.json',dict(features=FEATURES,training_seeds=sorted(map(int,unique)),
         model_sha256={name:sha(root/name) for name in ['full_development_model.json','model_only_development_model.json',
                                                     'candidate_only_development_model.json','constant_development_model.json']},
         predictions_are_development_only=True))
    print(json.dumps(dict(metrics=metrics,gate=read(root/'development_gate.json')),indent=2))


def audit_extraction(root):
    """Offline replay scoring only; this never adds feature-extraction queries."""
    import torch
    from models import PretrainedModel
    torch.set_num_threads(1)
    cases=read(root/'cases.json');expected=read(root/'input_sha256.json');errors=[];checks=[]
    for p,h in expected.items():
        if sha(p)!=h:errors.append('input_hash:'+p)
    for name,h in read(root/'registration.json')['source_sha256'].items():
        if sha(root/'source_snapshot'/name)!=h:errors.append('source_hash:'+name)
    state=read(root/'extraction_execution.json')
    if state['status']!='complete':errors.append('extraction_incomplete:'+state['status'])
    for case in cases['development']:
        p=root/'records'/(case['key']+'.json')
        if not p.exists():errors.append('missing:'+case['key']);continue
        record=read(p)
        ep=torch.load(case['evaluator'],map_location='cpu',weights_only=True)
        model=PretrainedModel(2);model.load_state_dict(ep['state_dict']);model.eval()
        with torch.no_grad():
            score=abs(float(model(torch.tensor(record['solution'],dtype=torch.float32)[None,:]).item()))
        delta=abs(score-case['expected_pooled_error'])
        if delta>1e-7:errors.append('pooled_reload:'+case['key'])
        if record['models_sha256']!=sha(case['models']):errors.append('model_hash:'+case['key'])
        if record['counter']['calls']!=case['expected_calls']:errors.append('calls:'+case['key'])
        if record['evaluator_loaded'] or record['labels_loaded']:errors.append('feature_information_leakage:'+case['key'])
        if len(record['features'])!=len(case['ids']):errors.append('feature_row_count:'+case['key'])
        checks.append(dict(key=case['key'],pooled_delta=delta,replay_X_delta=record['replay_X_max_delta']))
    csvsave(root/'extraction_audit_rows.csv',checks)
    save(root/'extraction_audit.json',dict(accepted=not errors,errors=errors,n_cases=len(checks)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['extract','audit','evaluate']);parser.add_argument('root',type=Path)
    args=parser.parse_args();root=args.root.resolve();sys.path.insert(0,str(root/'source_snapshot'))
    if args.mode=='extract':extract(root)
    elif args.mode=='audit':audit_extraction(root)
    else:evaluate(root)
