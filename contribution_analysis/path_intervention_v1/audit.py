"""Independent source-row, sign, rank, bootstrap, gate and cost audit."""
import math,time
import numpy as np
from common import ROOT,CA,read,save,sha,check
def sgn(x):return 1 if x>1e-4 else -1 if x< -1e-4 else 0
def mean(x):return sum(x)/len(x) if x else None
def rank(v):
    order=sorted(range(len(v)),key=lambda i:v[i]);out=[0.]*len(v);k=0
    while k<len(v):
        j=k+1
        while j<len(v) and v[order[j]]==v[order[k]]:j+=1
        for i in order[k:j]:out[i]=(k+j+1)/2
        k=j
    return out
def corr(x,y):
    if len(x)<3 or max(x)-min(x)<=1e-4 or max(y)-min(y)<=1e-4:return None
    a=rank(x);b=rank(y);aa=[v-mean(a) for v in a];bb=[v-mean(b) for v in b]
    return sum(x*y for x,y in zip(aa,bb))/math.sqrt(sum(x*x for x in aa)*sum(y*y for y in bb))
def close(a,b):
    if a is None or b is None:assert a is None and b is None;return 0.
    assert abs(a-b)<1e-10,(a,b);return abs(a-b)
def audit():
    check();tic=time.perf_counter();rows=read(ROOT/'internal_comparisons.json');teams=read(ROOT/'team_metrics.json');summary=read(ROOT/'internal_summary.json');maxerr=0.
    for c in read(ROOT/'cases.json')['development']:
        r=read(ROOT/'development'/(c['key']+'.json'))
        for g in r['groups']:
            for q in g['q_labels']:
                row=next(v for v in rows if v['key']==c['key'] and v['agent_id']==g['agent_id'] and v['q']==q)
                for field in ['j','tau','pi','u','tau_abs']:maxerr=max(maxerr,close(row[field],g[field]))
                for h in [1,3,5]:
                    for role,field in [('T','tau'),('P','pi')]:maxerr=max(maxerr,close(row[field+'_'+str(h)],g['observations'][str(h)][role]['difference']))
    for t in teams:
        rr=[r for r in rows if r['key']==t['key'] and r['q']==t['q']]
        assert len(rr)==t['n']
        if t['rq']=='RQ1':
            field={'1':'tau_1','3':'tau_3','5':'tau_5','final':'tau','abs':'tau_abs'}[t['h']]
            nz=[r for r in rr if sgn(r['j'])];assert len(nz)==t['nonzero']
            vals=dict(retained=mean([sgn(r[field])==sgn(r['j']) for r in nz]),reversed=mean([sgn(r[field])==-sgn(r['j']) for r in nz]),
                vanished=mean([sgn(r[field])==0 for r in nz]),attenuated=mean([abs(r[field])<=.5*abs(r['j']) for r in nz]),
                amplified=mean([abs(r[field])>=2*abs(r['j']) for r in nz]),rho=corr([r['j'] for r in rr],[r[field] for r in rr]))
        elif t['rq']=='RQ2':
            vals=dict(material=mean([abs(r['u'])>1e-4 for r in rr]),mean_abs_u=mean([abs(r['u']) for r in rr]),
                sign_change=mean([sgn(r['pi'])!=sgn(r['tau']) for r in rr]),rho=corr([r['pi'] for r in rr],[r['tau'] for r in rr]),
                restart=mean([r['restart'] for r in rr]),distance=mean([r['final_distance'] for r in rr]))
            for field in ['pi','tau','u']:
                for label,s in [('positive',1),('negative',-1),('nearzero',0)]:vals[field+'_'+label]=mean([sgn(r[field])==s for r in rr])
        else:
            decisive=[r for r in rr if sgn(r['D_L'])];pos=[r for r in rr if sgn(r['D_L'])==1];neg=[r for r in rr if sgn(r['D_L'])==-1];zero=[r for r in rr if not sgn(r['D_L'])]
            denom=sum(abs(r['D_L']) for r in rr)
            vals=dict(rho=corr([r['pi'] for r in rr],[r['D_L'] for r in rr]),sign=mean([sgn(r['pi'])==sgn(r['D_L']) for r in decisive]),
                positive_recall=mean([sgn(r['pi'])==1 for r in pos]),negative_recall=mean([sgn(r['pi'])==-1 for r in neg]),
                nearzero_false_positive=mean([sgn(r['pi'])!=0 for r in zero]),nmae=None if denom<=len(rr)*1e-4 else sum(abs(r['pi']-r['D_L']) for r in rr)/denom)
        for k,v in vals.items():maxerr=max(maxerr,close(v,t[k]))
    for s in summary['metrics']:
        ts=[t for t in teams if t['rq']==s['rq'] and t['scenario']==s['scenario'] and t['q']==s['q'] and (s['h'] is None or t['h']==s['h'])]
        for field,stat in s['metrics'].items():
            v=[t[field] for t in ts if t[field] is not None];assert len(v)==stat['n']
            if not v:continue
            rng=np.random.default_rng(20261024);indices=rng.integers(0,len(v),(10000,len(v)))
            means=np.asarray([sum(v[int(i)] for i in ix)/len(v) for ix in indices]);bounds=np.quantile(means,[.001,.999])
            maxerr=max(maxerr,close(mean(v),stat['mean']),close(float(bounds[0]),stat['interval'][0]),close(float(bounds[1]),stat['interval'][1]))
        m=s['metrics']
        if s['rq']=='RQ1':
            a=m['retained'];b=m['reversed'];passed=s['nonzero']>=20 and a['n']>=8 and a['mean']>=.8 and a['interval'][0]>=.7 and b['mean']<=.1 and b['interval'][1]<=.2
        elif s['rq']=='RQ2':a=m['material'];passed=a['n']>=8 and a['mean']>=.3 and a['interval'][0]>.1
        else:
            a=m['rho'];b=m['sign'];d=m['nmae'];eff=a['n']>=8 and b['n']>=8 and d['n']>=8 and a['mean']>=.8 and a['interval'][0]>.5 and b['mean']>=.85 and b['interval'][0]>.7 and d['mean']<=.25 and d['interval'][1]<.5
            assert eff==s['effect_pass'];cost=next(c for c in summary['costs'] if c['scenario']==s['scenario'] and c['q']==s['q']);passed=eff and cost['ratio']<=.7
        assert bool(passed)==s['pass_gate']
    research=sum(read(ROOT/(p+'_completion.json'))['physical_samples'] for p in ['pilot','development'])
    assert research==sum(c['physical_samples'] for c in read(ROOT/'verification.json')['cases'])
    total=research+read(ROOT/'verification.json')['verification_model_samples'];assert total<=2000000
    external=read(ROOT/'external_comparisons.json');ext=read(ROOT/'external_summary.json')
    for s in ext:
        x,y=s['comparison'].split(':')
        for t in s['teams']:
            rr=[r for r in external if r['key']==t['key'] and r['q']==s['q']]
            maxerr=max(maxerr,close(corr([r[x] for r in rr],[r[y] for r in rr]),t['rho']))
    size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file());assert size<1_000_000_000
    check();save(ROOT/'final_audit.json',dict(accepted=True,max_statistical_error=maxerr,rows=len(rows),team_metrics=len(teams),
        physical_decision_samples=research,verification_samples=read(ROOT/'verification.json')['verification_model_samples'],total_model_samples=total,
        pooled_samples=read(ROOT/'analysis_completion.json')['pooled_samples'],output_bytes=size,all_input_hashes_unchanged=True,
        independent_data_confirmation=False,wall_seconds=time.perf_counter()-tic))
    print(read(ROOT/'final_audit.json'))
if __name__=='__main__':audit()
