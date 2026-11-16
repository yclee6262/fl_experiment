"""Rebuild raw sums and metrics independently; no solver/model inference."""
import numpy as np
from scipy.stats import rankdata
from study import ROOT,ANCHOR,MODEL,DIAG,OLD,read,save,sha,csvread,check

def verify():
    check()
    for n,h in read(ROOT/'analysis_lock.json').items():assert sha(ROOT/n)==h
    assert sha(ROOT/'scores.json')==read(ROOT/'scores_lock.json')['sha256']
    labels={(r['key'],int(r['agent_id'])):r for r in csvread(OLD/'participant_comparison.csv')}
    rows=read(ROOT/'comparisons.json');max_error=0.;rebuild=[]
    for c in read(OLD/'cases.json')['development']:
        a=read(ANCHOR/'records'/(c['key']+'.json'));m=read(MODEL/'development'/(c['key']+'.json'))
        fr={r['agent_id']:r for r in read(DIAG/'full_revaluation'/(c['key']+'.json'))['rows']}
        for i in c['ids']:
            row=next(r for r in rows if r['key']==c['key'] and r['agent_id']==i)
            av=[p['J'] for p in a['comparisons'] if p['agent_id']==i]
            mv=[b['m'] for b in m['branches'] if c['ids'][b['i']]==i]
            assert len(av)==a['n_accepted_steps'] and len(mv)==len(m['full'])
            assert None not in av and None not in mv
            A=sum(av);M=sum(mv)
            for name,value in [('A',A),('M',M),('C',A+M)]:
                err=abs(row[name]-value);max_error=max(max_error,err);assert err<1e-10
            f=fr[i];w=np.asarray(c['alphas']);w=w/w.sum()
            dl=sum(float(w[j])*abs(float(f['predictions'][j])) for j in range(len(w)))-f['baseline_full_L']
            lab=labels[c['key'],i];de=float(lab['removal_error'])-float(lab['full_error'])
            assert abs(dl-row['D_L'])<1e-10 and abs(de-row['D_E'])<1e-12
    teams=read(ROOT/'team_metrics.json');summary=read(ROOT/'summary.json')
    def rho(x,y):
        if len(x)<3 or max(x)-min(x)<=1e-4 or max(y)-min(y)<=1e-4:return None
        a=rankdata(x);b=rankdata(y);a=a-a.mean();b=b-b.mean()
        return float(np.dot(a,b)/np.sqrt(np.dot(a,a)*np.dot(b,b)))
    def sign(x):return 1 if x>1e-4 else -1 if x< -1e-4 else 0
    for t in teams:
        rr=[r for r in rows if r['key']==t['key'] and r['available']]
        x=[r[t['score']] for r in rr];y=[r[t['reference']] for r in rr];v=rho(x,y)
        assert (v is None)==(t['rho'] is None)
        if v is not None:assert abs(v-t['rho'])<1e-12
        ds=[(a,b) for a,b in zip(x,y) if sign(a)]
        sg=None if not ds else sum(sign(a)==sign(b) for a,b in ds)/len(ds)
        assert sg==t['sign_agreement']
    for metric in summary['metrics']:
        tt=[t for t in teams if t['scenario']==metric['scenario'] and t['reference']==metric['reference'] and t['score']==metric['score']]
        for field,name in [('rho','rank'),('sign_agreement','sign')]:
            vals=[t[field] for t in tt if t[field] is not None]
            if not vals:continue
            rng=np.random.default_rng(20261021);means=[]
            for _ in range(10000):means.append(float(np.mean(np.asarray(vals)[rng.integers(0,len(vals),len(vals))])))
            assert abs(np.mean(vals)-metric[name]['mean'])<1e-12
            np.testing.assert_allclose(np.percentile(means,[.625,99.375]),metric[name]['interval'],rtol=0,atol=1e-12)
    for p in summary['paired']:
        vals=[]
        for pair in p['pairs']:
            ts={t['score']:t for t in teams if t['key']==pair['key'] and t['reference']==p['reference']}
            v=ts['C']['rho']-ts[p['comparison'][-1]]['rho'];assert abs(v-pair['delta'])<1e-12;vals.append(v)
        if vals:
            rng=np.random.default_rng(20261021);stats=[]
            for _ in range(10000):stats.append(float(np.mean(np.asarray(vals)[rng.integers(0,len(vals),len(vals))])))
            np.testing.assert_allclose(np.percentile(stats,[.625,99.375]),p['stat']['interval'],rtol=0,atol=1e-12)
    check();save(ROOT/'verification.json',dict(accepted=True,participants=len(rows),max_sum_error=max_error,
        scores_rebuilt_from_step_records=True,references_rebuilt_from_saved_outputs=True,
        statistics_independently_rebuilt=True,historical_sources_unchanged=True,
        new_model_samples=0,new_solver_runs=0,new_pooled_calls=0))
    print('INDEPENDENT_VERIFICATION_PASSED',len(rows),max_error)

if __name__=='__main__':verify()
