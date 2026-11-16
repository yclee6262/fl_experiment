"""Independent artifact audit; no refitting, solving, or extra model calls."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvread(p):
    with p.open() as f:return list(csv.DictReader(f))
def sign(v):return int(v>1e-4)-int(v < -1e-4)


def audit(root):
    errors=[]
    config=read(root/'registration.json')
    for n,h in config['source_sha256'].items():
        if sha(root/'source_snapshot'/n)!=h:errors.append('source_hash:'+n)
    for p,h in read(root/'input_sha256.json').items():
        if sha(Path(p))!=h:errors.append('input_hash:'+p)
    labels={(r['key'],r['agent_id']):r['C_full'] for r in read(root/'labels.json')}
    feature_names=['relative_error','disagreement','own_error_change','candidate_alignment',
                   'candidate_difference','coalition_size','final_loss_ratio','path_length']
    rows=csvread(root/'development_predictions.csv');folds=read(root/'development_folds.json')
    groups={int(r['seed']) for r in rows}
    keys=[(r['key'],int(r['agent_id'])) for r in rows]
    if len(keys)!=320 or len(set(keys))!=320 or set(keys)!=set(labels):errors.append('effect_coverage')
    if len(groups)!=40:errors.append('seed_coverage')
    def predict(tree,x):
        while 'feature' in tree:tree=tree['left'] if x[tree['feature']]<=tree['threshold'] else tree['right']
        return tree['value']
    def check_leaves(tree,train):
        if tree['n_groups']!=len({int(r['seed']) for r in train}):errors.append('leaf_group_count')
        if 'feature' in tree:
            col=feature_names[tree['feature']];left=[r for r in train if float(r[col])<=tree['threshold']]
            right=[r for r in train if float(r[col])>tree['threshold']]
            check_leaves(tree['left'],left);check_leaves(tree['right'],right)
        else:
            if len({int(r['seed']) for r in train})<5:errors.append('leaf_support')
            w=np.array([1/(2*int(float(r['coalition_size']))) for r in train]);y=np.array([float(r['C_full']) for r in train])
            if abs(float(np.average(y,weights=w))-tree['value'])>1e-12:errors.append('leaf_value')
    for f in folds:
        train=set(f['training_seeds']);validation=set(f['validation_seeds'])
        if train&validation or train|validation!=groups or len(validation)!=8:errors.append('fold_leakage')
        tr=[r for r in rows if int(r['seed']) in train]
        va=[r for r in rows if int(r['fold'])==f['fold']]
        if {int(r['seed']) for r in va}!=validation:errors.append('validation_fold_assignment')
        for name in ['model_only','candidate_only','full']:
            check_leaves(f[name],tr)
            for r in va:
                x=[float(r[c]) for c in feature_names]
                if abs(predict(f[name],x)-float(r[name]))>1e-12:errors.append('OOF_prediction')
        for r in va:
            if float(r['constant'])!=f['constant'][str(int(float(r['coalition_size'])))]:errors.append('constant_prediction')
    for r in rows:
        if float(r['C_full'])!=labels[r['key'],int(r['agent_id'])]:errors.append('label_pairing')
    reported={(r['scenario'],r['method']):r for r in csvread(root/'development_metrics.csv')}
    examples=[];bootstrap=[]
    for scenario in ['clean','controlled_poison']:
        rs=[r for r in rows if r['scenario']==scenario]
        for method in ['zero','constant','model_only','candidate_only','full']:
            mae=float(np.mean([abs(float(r[method])-float(r['C_full'])) for r in rs]))
            accuracy=float(np.mean([sign(float(r[method]))==sign(float(r['C_full'])) for r in rs]))
            reference=reported[scenario,method]
            if abs(mae-float(reference['MAE']))>1e-12 or abs(accuracy-float(reference['direction_agreement']))>1e-12:errors.append('metric_recalculation')
        for kind in ['missed_negative','false_negative_prediction','correct_negative','largest_absolute_error']:
            subset=([r for r in rs if sign(float(r['C_full']))==-1 and sign(float(r['full']))!=-1] if kind=='missed_negative'
                  else [r for r in rs if sign(float(r['C_full']))!=-1 and sign(float(r['full']))==-1] if kind=='false_negative_prediction'
                  else [r for r in rs if sign(float(r['C_full']))==sign(float(r['full']))==-1] if kind=='correct_negative' else rs)
            if not subset:continue
            r=sorted(subset,key=lambda r:(-abs(float(r['full'])-float(r['C_full'])),int(r['seed']),int(r['agent_id'])))[0]
            examples.append(dict(scenario=scenario,category=kind,seed=int(r['seed']),agent_id=int(r['agent_id']),
                                  C_full=float(r['C_full']),prediction=float(r['full']),fold=int(r['fold'])))
        sr=[r for r in csvread(root/'development_seed_metrics.csv') if r['scenario']==scenario]
        values=np.array([float(r['MAE_improvement']) for r in sr])
        rng=np.random.default_rng(20261017);resampled=values[rng.integers(0,len(values),(10000,len(values)))].mean(axis=1)
        bootstrap.append(dict(scenario=scenario,n_seeds=len(values),mean_MAE_improvement=float(values.mean()),
              exploratory_95pct_CI=np.quantile(resampled,[.025,.975]).tolist(),
              status='posthoc_development_diagnostic_not_formal_confirmation'))
    gate=read(root/'development_gate.json')
    if gate['accepted'] or gate['independent_confirmation_started'] or (root/'confirmation').exists():errors.append('stop_gate_not_respected')
    result=dict(accepted=not errors,errors=errors,n_rows=len(rows),n_seeds=len(groups),n_folds=len(folds),
           source_and_inputs_unchanged=True if not any('hash' in x for x in errors) else False,
           OOF_predictions_reconstructed=True if 'OOF_prediction' not in errors else False,
           independent_confirmation_started=False,audit_source_sha256=sha(Path(__file__)))
    for name,obj in [('development_independent_audit.json',result),('development_representative_cases.json',examples),
                     ('development_exploratory_uncertainty.json',bootstrap)]:
        with (root/name).open('x') as f:json.dump(obj,f,indent=2)
    print(json.dumps(dict(audit=result,examples=examples,uncertainty=bootstrap),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();audit(a.root.resolve())
