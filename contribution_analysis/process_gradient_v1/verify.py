"""Reconstruct arithmetic from locked API replies without querying any model."""
import argparse
from pathlib import Path
import numpy as np
from run import read,sha,save
from analyze import csvread


def verify(root):
    errors=[];config=read(root/'registration.json');previous=Path(config['previous'])
    for n,h in config['source_sha256'].items():
        if sha(root/'source_snapshot'/n)!=h:errors.append('source:'+n)
    for p,h in read(root/'input_sha256.json').items():
        if sha(Path(p))!=h:errors.append('input:'+p)
    if sha(root/'preregistration.md')!=config['preregistration_sha256']:errors.append('preregistration')
    if sha(root/'cases.json')!=config['cases_sha256']:errors.append('cases')
    lock=read(root/'scores_lock.json')
    if sha(root/'scores.csv')!=lock['scores_sha256']:errors.append('score_lock')
    preds={(r['key'],int(r['agent_id'])):r for r in csvread(root/'scores.csv')}
    labels={(r['key'],r['agent_id']):r['C_full'] for r in read(Path(config['labels_path']))}
    if len(preds)!=320 or set(preds)!=set(labels):errors.append('coverage')
    calls=0;samples=0
    for case in read(root/'cases.json')['development']:
        name=case['key']+'.json';r=read(root/'records'/name)
        if sha(root/'records'/name)!=lock['record_sha256'][name]:errors.append('record_lock:'+name)
        old=read(previous/'records'/name)
        x=np.array(old['trace']['states']);y=np.array(old['trace']['responses']);w=np.array(case['alphas'])
        n=len(x)-1;d=x.shape[1];m=len(w)
        g=np.zeros((n,m,d));responses=np.array(r['fd_responses']);points=np.array(r['fd_points'])
        for t in range(n):
            for j in range(d):
                plus=x[t].astype(np.float32);minus=x[t].astype(np.float32)
                plus[j]=min(1.,x[t,j]+.001);minus[j]=max(-1.,x[t,j]-.001)
                if not np.array_equal(points[t,j,0],plus) or not np.array_equal(points[t,j,1],minus):errors.append('fd_point:'+name)
                width=float(plus[j])-float(minus[j])
                for i in range(m):g[t,i,j]=(abs(responses[i,t,j,0])-abs(responses[i,t,j,1]))/width
        a=np.zeros(m)
        for t in range(n):
            for i in range(m):a[i]-=w[i]*float(np.dot(g[t,i],x[t+1]-x[t]))
        b=w*(abs(y[0])-abs(y[-1]))
        if not np.allclose(a,r['gradient_work'],atol=1e-12,rtol=0):errors.append('gradient_work:'+name)
        if not np.allclose(b,r['endpoint_work'],atol=1e-12,rtol=0):errors.append('endpoint_work:'+name)
        if not np.isclose(b.sum(),(abs(y[0])@w)-(abs(y[-1])@w),atol=1e-12):errors.append('telescoping:'+name)
        if r['calls']!=(m if n else 0) or r['samples']!=2*d*m*n:errors.append('cost:'+name)
        if r['labels_loaded'] or r['evaluator_loaded']:errors.append('leakage:'+name)
        for i,agent in enumerate(case['ids']):
            p=preds[case['key'],agent]
            if abs(float(p['gradient_work'])-a[i])>1e-12 or abs(float(p['endpoint_work'])-b[i])>1e-12:errors.append('csv_score:'+name)
        calls+=r['calls'];samples+=r['samples']
    comparison=csvread(root/'comparison_rows.csv')
    for r in comparison:
        if float(r['C_full'])!=labels[r['key'],int(r['agent_id'])]:errors.append('label_pairing')
    for met in csvread(root/'metrics.csv'):
        rs=[r for r in comparison if r['scenario']==met['scenario']];method=met['method']
        y=[float(r['C_full']) for r in rs];p=[0. if method=='zero' else float(r[method]) for r in rs]
        mae=sum(abs(a-b) for a,b in zip(y,p))/len(y)
        s=lambda a:int(a>1e-4)-int(a < -1e-4)
        direction=sum(s(a)==s(b) for a,b in zip(y,p))/len(y)
        tp=sum(s(a)==s(b)==-1 for a,b in zip(y,p));pn=sum(s(a)==-1 for a in p);tn=sum(s(a)==-1 for a in y)
        if abs(mae-float(met['MAE']))>1e-12 or abs(direction-float(met['direction_agreement']))>1e-12:errors.append('metric')
        if pn!=int(met['predicted_negative']) or tn!=int(met['true_negative']) or tp!=int(met['true_positive_negative']):errors.append('negative_counts')
    pilot=read(root/'pilot_results.json')
    ratio=sum(np.median(r['augmented_seconds']) for r in pilot['cases'])/sum(np.median(r['control_seconds']) for r in pilot['cases'])
    if abs(ratio-pilot['aggregate_ratio'])>1e-12:errors.append('timing_ratio')
    if calls!=lock['totals']['calls'] or samples!=lock['totals']['samples']:errors.append('totals')
    result=dict(accepted=not errors,errors=errors,n_cases=80,n_effects=len(preds),n_seeds=len({r['seed'] for r in comparison}),
        gradient_scores_rebuilt_from_API_replies=True,endpoint_identity_checked=True,
        source_and_input_hashes_checked=True,new_LOO_solves=0,new_model_queries_in_verifier=0)
    save(root/'verification.json',result);print(result)
    if errors:raise ValueError('verification_failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();verify(a.root.resolve())
