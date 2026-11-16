"""Independent math/rank audit. Never import the analysis implementation."""
import hashlib,json,math,itertools,time
from pathlib import Path
import numpy as np
WORK=Path(__file__).resolve().parents[2];CA=WORK/'contribution_analysis';ROOT=CA/'j_local_quality_analysis_v1_20261027';SOURCE=CA/'j_loo_start_stability_experiment_v1_20261027'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def avg(v):return math.fsum(v)/len(v) if v else None
def sign(x):return 1 if x>1e-4 else -1 if x< -1e-4 else 0
def key(x):return np.asarray(x,np.float64).tobytes().hex()
def ranks(x):
    return [1+sum(v<a for v in x)+(sum(v==a for v in x)-1)/2 for a in x]
def rank(x,y):
    if len(x)<3 or max(x)-min(x)<=1e-4 or max(y)-min(y)<=1e-4:return None
    a=ranks(x);b=ranks(y);ma=avg(a);mb=avg(b)
    return math.fsum((v-ma)*(w-mb) for v,w in zip(a,b))/(math.fsum((v-ma)**2 for v in a)*math.fsum((w-mb)**2 for w in b))**.5
def interval(v):
    if not v:return dict(mean=None,ci=None,n=0)
    rng=np.random.default_rng(20261027);values=[avg([v[i] for i in rng.integers(0,len(v),len(v))]) for _ in range(10000)]
    return dict(mean=avg(v),ci=np.quantile(values,[.025,.975]).tolist(),n=len(v))
error=0.
def match(a,b):
    global error
    if a is None or b is None:assert a is None and b is None;return
    a=np.array(a,float);b=np.array(b,float);e=float(np.max(abs(a-b)));assert np.allclose(a,b,atol=1e-10,rtol=1e-10),(a,b);error=max(error,e)
def main():
    tic=time.perf_counter();assert read(ROOT/'analysis_completion.json')['complete']
    for p,h in read(ROOT/'inputs.json').items():assert sha(p)==h
    for p,h in read(ROOT/'source_lock.json').items():assert sha(ROOT/'source_snapshot'/p)==h
    pool={r['key']:{key(p['x']):p['value'] for p in r['points']} for r in read(SOURCE/'pooled_points.json')};bycase={}
    rows=read(ROOT/'round_rows.json');agents=read(ROOT/'agent_decomposition.json');starts=read(ROOT/'start_quality.json');teams=read(ROOT/'team_metrics.json')
    cases={c['key']:c for c in read(SOURCE/'cases.json')['development']}
    for (name,si),rr in itertools.groupby(sorted(rows,key=lambda r:(r['key'],r['start'])),key=lambda r:(r['key'],r['start'])):
        rr=list(rr);source=read(SOURCE/'development'/(name+f'_s{si}_J.json'));raw={(v['i'],key(v['x'])):v['y'] for v in source['raw']};c=cases[name];w=c['alphas'];total=math.fsum(w)
        def loss(x):return float(np.dot(np.array(w)/sum(w),[abs(raw[i,key(x)]) for i in range(len(w))]))
        bycase[name,si]=rr
        for r,b in zip(rr,source['branches']):
            assert (r['i'],r['t'])==(b['i'],b['t']);match(r['j'],loss(b['x_minus'])-loss(b['x_full']));match(r['j_E'],pool[name][key(b['x_minus'])]-pool[name][key(b['x_full'])])
        for i in range(len(c['ids'])):
            v=[r['j'] for r in rr if r['i']==i];a=next(a for a in agents if a['key']==name and a['start']==si and a['i']==i)
            P=math.fsum(x for x in v if x>0);N=math.fsum(x for x in v if x<0);J=math.fsum(v);G=math.fsum(abs(x) for x in v);ordered=sorted([abs(x) for x in v],reverse=True)
            for f,x in [('P',P),('N',N),('J',J),('G',G),('cancellation',None if G<=1e-4 else (G-abs(J))/G),('top1',None if G<=1e-4 else ordered[0]/G),('top3',None if G<=1e-4 else math.fsum(ordered[:3])/G)]:match(x,a[f])
            match(J,source['values'][i])
            for s,f in [(1,'positive'),(-1,'negative'),(0,'zero')]:
                n=sum(sign(x)==s for x in v);assert n==a[f+'_rounds'];match(n/len(v),a[f+'_fraction'])
        nz=[r for r in rr if sign(r['j'])];both=[r for r in nz if sign(r['j_E'])];large=[r for r in rr if abs(r['j'])>.001];zs=[r for r in rr if sign(r['j'])==0]
        out=dict(sign=avg([sign(r['j'])==sign(r['j_E']) for r in nz]),reverse=avg([sign(r['j'])==-sign(r['j_E']) for r in nz]),external_zero=avg([sign(r['j_E'])==0 for r in nz]),
            both_nonzero_sign=avg([sign(r['j'])==sign(r['j_E']) for r in both]),severe_reverse=avg([r['j']*r['j_E']<0 and abs(r['j_E'])>.001 for r in large]),
            positive_sign=avg([sign(r['j_E'])==1 for r in nz if sign(r['j'])==1]),negative_sign=avg([sign(r['j_E'])==-1 for r in nz if sign(r['j'])==-1]),zero_external_nonzero=avg([sign(r['j_E'])!=0 for r in zs]))
        rounds=sorted(set(r['t'] for r in rr));rs=[]
        for t in rounds:
            selected=[r for r in rr if r['t']==t];value=rank([r['j'] for r in selected],[r['j_E'] for r in selected])
            if value is not None:rs.append(value)
        out.update(rank=avg(rs),rank_coverage=len(rs)/len(rounds));target=next(v for v in starts if v['key']==name and v['start']==si)
        for f,v in out.items():match(v,target[f])
        assert target['nonzero']==len(nz) and target['large']==len(large) and target['n']==len(rr)
    qfields=['sign','reverse','external_zero','both_nonzero_sign','positive_sign','negative_sign','zero_external_nonzero','severe_reverse','rank','rank_coverage']
    afields=['P','N','J','G','cancellation','top1','top3','positive_fraction','negative_fraction','zero_fraction']
    for t in teams:
        rr=[v for v in starts if v['key']==t['key']]
        for f in qfields:match(avg([v[f] for v in rr if v[f] is not None]),t[f])
        for f in afields:
            ss=[]
            for si in range(5):
                vs=[a[f] for a in agents if a['key']==t['key'] and a['start']==si and a[f] is not None]
                if vs:ss.append(avg(vs))
            match(avg(ss),t['agent_'+f])
    for summary in read(ROOT/'summary.json'):
        rr=[v for v in teams if v['scenario']==summary['scenario']];m=summary['metrics']
        for f,s in m.items():
            v=interval([r[f] for r in rr if r[f] is not None]);assert v['n']==s['n'];match(v['mean'],s['mean']);match(v['ci'],s['ci'])
        cov=m['sign']['n']>=8 and sum(r['valid_sign_starts']>=4 for r in rr)>=8
        sp=cov and m['sign']['mean']>=.85 and m['sign']['ci'][0]>.7
        rp=m['severe_reverse']['n']>=8 and sum(r['large'] for r in rr)>=20 and m['severe_reverse']['mean']<=.1 and m['severe_reverse']['ci'][1]<.2
        kp=m['rank']['n']>=8 and m['rank_coverage']['mean']>=.8 and m['rank']['mean']>=.8 and m['rank']['ci'][0]>.5
        assert [cov,sp,rp,kp,sp and rp and kp]==[summary[f] for f in ['coverage_pass','sign_pass','severe_reverse_pass','rank_pass','complete_pass']]
    for item in read(ROOT/'sign_matrix.json'):
        ratios=[];count=0
        for si in range(5):
            rr=bycase[item['key'],si];n=sum(sign(r['j'])==item['internal'] and sign(r['j_E'])==item['external'] for r in rr);count+=n;ratios.append(n/len(rr))
        assert count==item['count'];match(avg(ratios),item['fraction'])
    # Ensure representative cases follow the locked selection rules.
    for e in read(ROOT/'examples.json'):
        if e['selection']=='max_opposite':
            rr=sorted([r for r in rows if r['scenario']==e['scenario'] and sign(r['j'])*sign(r['j_E'])==-1],key=lambda r:(r['key'],r['start'],r['agent_id'],r['t']))
            expected=max(rr,key=lambda r:min(abs(r['j']),abs(r['j_E'])));assert all(e[f]==expected[f] for f in ['key','start','agent_id','t'])
        else:
            rr=sorted([a for a in agents if a['scenario']==e['scenario'] and a['G']>1e-4],key=lambda r:(r['key'],r['start'],r['agent_id']));field='G' if e['selection']=='max_gross' else 'cancellation'
            expected=max(rr,key=lambda r:r[field]);assert all(e[f]==expected[f] for f in ['key','start','agent_id'])
    result=dict(accepted=True,round_rows=len(rows),agent_start_records=len(agents),max_error=error,independent_rank_and_bootstrap=True,source_hashes_unchanged=True,new_model_inferences=0,wall_seconds=time.perf_counter()-tic)
    with (ROOT/'verification.json').open('x') as f:json.dump(result,f,indent=2)
    print(result)
if __name__=='__main__':main()
