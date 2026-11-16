"""Independent aggregation/ranks/bootstrap and final provenance/cost checks."""
import numpy as np
from common import *
def ranked(a):
    a=np.asarray(a);order=np.argsort(a,kind='stable');r=np.empty(len(a),float);k=0
    while k<len(a):
        end=k+1
        while end<len(a) and a[order[end]]==a[order[k]]:end+=1
        r[order[k:end]]=(k+1+end)/2.;k=end
    return r
def stats(v):
    a=np.array([x for x in v if x is not None]);rng=np.random.default_rng(20261021)
    if not len(a):return None,None
    dist=[]
    # Same predeclared draws, independent per-replicate aggregation.
    draws=rng.integers(0,len(a),(10000,len(a)))
    for row in draws:dist.append(sum(a[i] for i in row)/len(a))
    return float(a.sum()/len(a)),np.percentile(dist,[.1,99.9]).tolist()
def audit():
    check();rows=read(ROOT/'comparisons.json');teams=read(ROOT/'team_metrics.json');summary=read(ROOT/'summary.json');error=0.
    for team in teams:
        rr=[r for r in rows if all(r[k]==team[k] for k in ['key','method','role'])];rhos=[]
        for t in range(team['rounds']):
            rs=[r for r in rr if r['t']==t];x=np.array([r['predicted'] for r in rs]);y=np.array([r['exact'] for r in rs])
            if len(rs)<3 or max(x)-min(x)<=1e-4 or max(y)-min(y)<=1e-4:continue
            a=ranked(x);b=ranked(y);a-=a.mean();b-=b.mean()
            rhos.append(float(sum(a*b)/np.sqrt(sum(a*a)*sum(b*b))))
        if rhos:error=max(error,near(sum(rhos)/len(rhos),team['rho']))
        else:assert team['rho'] is None
        sign=lambda x:1 if x>1e-4 else -1 if x< -1e-4 else 0
        for name,pred in [('signs',lambda r:sign(r['exact'])!=0),('positive_recall',lambda r:sign(r['exact'])==1),('negative_recall',lambda r:sign(r['exact'])==-1)]:
            subset=[r for r in rr if pred(r)]
            if subset:error=max(error,near(sum(sign(r['predicted'])==sign(r['exact']) for r in subset)/len(subset),team[name]))
            else:assert team[name] is None
        den=sum(abs(r['exact']) for r in rr)
        if den>len(rr)*1e-4:error=max(error,near(sum(abs(r['predicted']-r['exact']) for r in rr)/den,team['nmae']))
    for row in summary:
        ts=[t for t in teams if all(t[k]==row[k] for k in ['scenario','method','role'])]
        for metric in row['metrics']:
            mean,ci=stats([t[metric] for t in ts]);ref=row['metrics'][metric]
            if mean is None:assert ref['mean'] is None
            else:error=max(error,near(mean,ref['mean']),near(ci,ref['ci']))
    # Byte-level prior inputs and both prediction phases stay unchanged.
    for p,h in read(ROOT/'predictions_locked.json')['hashes'].items():assert sha(ROOT/p)==h
    actual=sum(read(ROOT/(phase+'_completion.json'))['new_samples'] for phase in ['pilot','development'])
    verification=read(ROOT/'verification.json');timing=read(ROOT/'timing.json');failed=read(ROOT/'timing_initial_failure.json')['accounted_model_samples'] if (ROOT/'timing_initial_failure.json').exists() else 0
    total=actual+verification['baseline_reply_samples']+verification['new_endpoint_reply_samples']+timing['model_samples']+failed
    assert total<=620000
    size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file());assert size<=2_000_000_000
    save(ROOT/'final_audit.json',dict(accepted=True,independent_statistics_max_error=error,prior_inputs_unchanged=True,
        predictions_still_locked=True,actual_prediction_new_samples=actual,total_research_model_samples=total,
        baseline_verification_samples=verification['baseline_reply_samples'],new_endpoint_verification_samples=verification['new_endpoint_reply_samples'],
        timing_samples=timing['model_samples'],failed_timing_samples=failed,output_bytes=size,pooled_calls=0,truth_calls=0,
        independent_confirmation_started=False,automatic_payment_or_pruning=False))
if __name__=='__main__':audit()
