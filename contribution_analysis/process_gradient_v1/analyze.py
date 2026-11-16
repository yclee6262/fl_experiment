"""Only after scores are locked: compare with existing offline full-removal."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from run import read,save,csvsave,sha,check_sources


def csvread(p):
    with p.open() as f:return list(csv.DictReader(f))
def signs(a):return (np.asarray(a)>1e-4).astype(int)-(np.asarray(a)<-1e-4).astype(int)
def correlation(a,b):
    a=rankdata(a);b=rankdata(b)
    if np.ptp(a)==0 or np.ptp(b)==0:return None
    return float(np.corrcoef(a,b)[0,1])
def metrics(y,pred):
    s=signs(y);p=signs(pred);neg=s==-1;pn=p==-1;tp=int(np.sum(neg&pn))
    return dict(MAE=float(np.mean(abs(pred-y))),RMSE=float(np.sqrt(np.mean((pred-y)**2))),
        direction_agreement=float(np.mean(s==p)),spearman=correlation(y,pred),
        true_negative=int(neg.sum()),predicted_negative=int(pn.sum()),true_positive_negative=tp,
        negative_precision=float(tp/pn.sum()) if pn.any() else None,
        negative_recall=float(tp/neg.sum()) if neg.any() else None)


def analyze(root):
    config=check_sources(root);lock=read(root/'scores_lock.json')
    if sha(root/'scores.csv')!=lock['scores_sha256']: raise ValueError('scores_unlocked')
    for n,h in lock['record_sha256'].items():
        if sha(root/'records'/n)!=h:raise ValueError('record_changed')
    labels=read(Path(config['labels_path']));labelmap={(r['key'],r['agent_id']):r['C_full'] for r in labels}
    scores=csvread(root/'scores.csv');keys=[(r['key'],int(r['agent_id'])) for r in scores]
    if len(set(keys))!=len(keys) or set(keys)!=set(labelmap):raise ValueError('label_coverage')
    rows=[dict(**r,C_full=labelmap[r['key'],int(r['agent_id'])]) for r in scores]
    output=[];within=[];uncertainty=[];examples=[];progress={}
    for scenario in ['clean','controlled_poison']:
        rs=sorted([r for r in rows if r['scenario']==scenario],key=lambda r:(int(r['seed']),int(r['agent_id'])))
        seeds=sorted({int(r['seed']) for r in rs});m=len(rs)//len(seeds)
        y=np.array([r['C_full'] for r in rs]);lookup={n:np.array([float(r[n]) for r in rs]) for n in ['gradient_work','endpoint_work']};lookup['zero']=np.zeros(len(y))
        rng=np.random.default_rng(20261018);idx=rng.integers(0,len(seeds),(10000,len(seeds)))
        groups=np.array([int(r['seed']) for r in rs])
        for method,pred in lookup.items():
            met=metrics(y,pred);output.append(dict(scenario=scenario,method=method,n_effects=len(y),n_seeds=len(seeds),**met))
            if method=='gradient_work':
                negseed=len(set(groups[signs(y)==-1]));progress[scenario]=dict(negative_seed_count=negseed,
                    effect_screen_pass=bool(met['MAE']<np.mean(abs(y)) and met['direction_agreement']>=.70
                        and met['negative_precision'] is not None and met['negative_precision']>=.70
                        and met['negative_recall'] is not None and met['negative_recall']>=.70
                        and met['true_negative']>=10 and negseed>=5))
            for seed in seeds:
                take=groups==seed
                within.append(dict(scenario=scenario,method=method,seed=seed,
                    spearman=correlation(y[take],pred[take]),MAE=float(abs(pred[take]-y[take]).mean())))
            mae=abs(pred-y).reshape(len(seeds),m).mean(axis=1)
            direction=(signs(pred)==signs(y)).reshape(len(seeds),m).mean(axis=1)
            improvement=abs(y).reshape(len(seeds),m).mean(axis=1)-mae
            uncertainty.append(dict(scenario=scenario,method=method,
                MAE_exploratory_95pct_CI=np.quantile(mae[idx].mean(axis=1),[.025,.975]),
                direction_exploratory_95pct_CI=np.quantile(direction[idx].mean(axis=1),[.025,.975]),
                MAE_improvement_vs_zero_exploratory_95pct_CI=np.quantile(improvement[idx].mean(axis=1),[.025,.975]),
                interpretation='seed bootstrap on reused exploratory data, not independent confirmation'))
        pred=lookup['gradient_work'];true=signs(y);estimated=signs(pred)
        subsets={'missed_negative':(true==-1)&(estimated!=-1),'false_negative_prediction':(true!=-1)&(estimated==-1),
            'correct_negative':(true==-1)&(estimated==-1),'largest_absolute_gap':np.ones(len(y),bool)}
        for name,mask in subsets.items():
            candidates=np.flatnonzero(mask)
            if len(candidates):
                k=min(candidates,key=lambda k:(-abs(pred[k]-y[k]),int(rs[k]['seed']),int(rs[k]['agent_id'])))
                examples.append(dict(scenario=scenario,category=name,key=rs[k]['key'],seed=int(rs[k]['seed']),
                    agent_id=int(rs[k]['agent_id']),C_full=float(y[k]),gradient_work=float(pred[k]),
                    endpoint_work=float(lookup['endpoint_work'][k]),gap=float(pred[k]-y[k])))
    pilot=read(root/'pilot_results.json')
    effect=all(v['effect_screen_pass'] for v in progress.values())
    gate=dict(accepted=bool(effect and pilot['cost_target_met'] and pilot['technical_accepted']),
        technical_accepted=pilot['technical_accepted'],cost_target_met=pilot['cost_target_met'],effect_screen_pass=effect,
        scenarios=progress,independent_confirmation_started=False,
        status='exploratory_screen_passed' if effect and pilot['cost_target_met'] else 'exploratory_screen_not_passed',
        not_a_formal_validity_result=True,scores_locked_before_labels=True)
    csvsave(root/'comparison_rows.csv',rows);csvsave(root/'metrics.csv',output);csvsave(root/'within_coalition.csv',within)
    save(root/'uncertainty.json',uncertainty);save(root/'representative_cases.json',examples);save(root/'gate.json',gate)
    diagnostics=[]
    for case in read(root/'cases.json')['development']:
        r=read(root/'records'/(case['key']+'.json'));gap=np.array(r['integration_gap'])
        diagnostics.append(dict(key=case['key'],n_steps=r['n_steps'],internal_loss_decrease=r['internal_loss_decrease'],
            gradient_sum=float(sum(r['gradient_work'])),sum_gap=float(gap.sum()),max_person_gap=float(abs(gap).max()),
            additional_calls=r['calls'],additional_samples=r['samples']))
    csvsave(root/'integration_diagnostics.csv',diagnostics)
    print(json.dumps(dict(metrics=output,gate=gate),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();analyze(a.root.resolve())
