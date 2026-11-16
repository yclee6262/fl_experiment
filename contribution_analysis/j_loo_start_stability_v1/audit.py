"""Manual ranks, independent classifications/aggregation, and report delivery."""
import itertools,statistics
from common import *
def mean(v):return sum(v)/len(v) if v else None
def cls(x):return 1 if x>1e-4 else -1 if x< -1e-4 else 0
def ranks(a):
    ordered=sorted(range(len(a)),key=lambda i:a[i]);r=[0.]*len(a);k=0
    while k<len(a):
        end=k+1
        while end<len(a) and a[ordered[end]]==a[ordered[k]]:end+=1
        for j in range(k,end):r[ordered[j]]=(k+end+1)/2
        k=end
    return r
def rank(a,b):
    if len(a)<3 or max(a)-min(a)<=1e-4 or max(b)-min(b)<=1e-4:return None
    aa=ranks(a);bb=ranks(b);ma=mean(aa);mb=mean(bb)
    return sum((x-ma)*(y-mb) for x,y in zip(aa,bb))/(sum((x-ma)**2 for x in aa)*sum((y-mb)**2 for y in bb))**.5
def interval(v):
    if not v:return dict(mean=None,ci=None,n=0)
    rng=np.random.default_rng(20261027);samples=[]
    # Same declared resampling stream, independent loop and aggregation.
    for _ in range(10000):samples.append(mean([v[i] for i in rng.integers(0,len(v),len(v))]))
    return dict(mean=mean(v),ci=np.quantile(samples,[.025,.975]).tolist(),n=len(v))
def assert_value(a,b):
    if a is None or b is None:assert a is None and b is None;return 0.
    return near(a,b)
def audit():
    check();error=0.;scores=read(ROOT/'development_scores.json');team=read(ROOT/'team_stability.json');pairs=read(ROOT/'team_comparison.json');totalpairs=0
    for c in read(ROOT/'cases.json')['development']:
        rr=[v for v in scores if v['key']==c['key']]
        for method in ['J','LOO']:
            values=[r['methods'][method]['values'] for r in rr];ranked=[];flips=[];changes=[]
            for a,b in itertools.combinations(values,2):
                v=rank(a,b)
                if v is not None:ranked.append(v)
                for x,y in zip(a,b):
                    if cls(x) and cls(y):flips.append(cls(x)!=cls(y))
                    if cls(x) or cls(y):changes.append(cls(x)!=cls(y))
                totalpairs+=1
            original=next(r for r in team if r['key']==c['key'] and r['method']==method)
            for f,v in [('rank',mean(ranked)),('rank_coverage',len(ranked)/10),('flip',mean(flips)),('class_change',mean(changes))]:error=max(error,assert_value(v,original[f]))
            assert original['nonzero_pairs']==len(flips)
            columns=list(zip(*values));spread=[max(v)-min(v) for v in columns];scale=[mean([abs(x) for x in v]) for v in columns]
            for f,v in [('spread',mean(spread)),('relative_spread',mean([x/y for x,y in zip(spread,scale) if y>1e-4])),('all_nearzero',mean([max(abs(x) for x in v)<=1e-4 for v in columns]))]:error=max(error,assert_value(v,original[f]))
        result=dict(rank=[],sign=[],nmae=[])
        for r in rr:
            j=r['methods']['J']['values'];d=r['methods']['LOO']['values'];v=rank(j,d)
            if v is not None:result['rank'].append(v)
            signs=[cls(x)==cls(y) for x,y in zip(j,d) if cls(y)]
            if signs:result['sign'].append(mean(signs))
            den=sum(abs(x) for x in d)
            if den>len(d)*1e-4:result['nmae'].append(sum(abs(x-y) for x,y in zip(j,d))/den)
        target=next(v for v in pairs if v['key']==c['key'])
        for f,v in result.items():error=max(error,assert_value(mean(v),target[f]))
    summaries=read(ROOT/'internal_summary.json')
    for item in summaries['stability']:
        rr=[r for r in team if r['scenario']==item['scenario'] and r['method']==item['method']]
        for field,summary in item['metrics'].items():
            actual=interval([r[field] for r in rr if r[field] is not None]);assert actual['n']==summary['n']
            error=max(error,assert_value(actual['mean'],summary['mean']),assert_value(actual['ci'],summary['ci']))
        cov=item['metrics'];valid=cov['rank']['n']>=8 and cov['rank_coverage']['mean']>=.8 and cov['flip']['n']>=8 and sum(v['nonzero_pairs'] for v in rr)>=20
        passed=valid and cov['rank']['mean']>=.8 and cov['rank']['ci'][0]>.5 and cov['flip']['mean']<=.1 and cov['flip']['ci'][1]<.2
        assert valid==item['coverage_pass'] and passed==item['stable_pass']
    times=read(ROOT/'development_timings.json')
    for item in read(ROOT/'costs.json'):
        for method in ['J','LOO']:
            expected=statistics.median([r['wall_seconds'] for r in times if r['key']==item['key'] and r['start']==item['start'] and r['method']==method]);error=max(error,near(expected,item['times'][method]))
    for item in summaries['replacement']:
        rr=[v for v in pairs if v['scenario']==item['scenario']]
        for f,s in item['metrics'].items():
            actual=interval([v[f] for v in rr if v[f] is not None]);assert actual['n']==s['n'];error=max(error,assert_value(actual['mean'],s['mean']),assert_value(actual['ci'],s['ci']))
        s=item['metrics'];valid=all(v['n']>=8 for v in s.values())
        effect=valid and s['rank']['mean']>=.8 and s['rank']['ci'][0]>.5 and s['sign']['mean']>=.85 and s['sign']['ci'][0]>.7 and s['nmae']['mean']<=.25 and s['nmae']['ci'][1]<.5
        assert effect==item['effect_pass'] and item['complete_pass']==(effect and item['cost_pass'])
    # Pooled ranks and sign classifications, recomputed from fixed saved rows.
    ex=read(ROOT/'external_rows.json');et=read(ROOT/'external_team.json')
    for item in et:
        a,b=item['comparison'].split(':');rr=[r for r in ex if r['key']==item['key']];rs=[];ss=[]
        for r in rr:
            v=rank(r[a],r[b])
            if v is not None:rs.append(v)
            v=[cls(x)==cls(y) for x,y in zip(r[a],r[b]) if cls(y)]
            if v:ss.append(mean(v))
        error=max(error,assert_value(mean(rs),item['rank']),assert_value(mean(ss),item['sign']))
    for item in read(ROOT/'external_summary.json'):
        rr=[r for r in et if r['scenario']==item['scenario'] and r['comparison']==item['comparison']]
        for f,s in item['metrics'].items():
            a=interval([r[f] for r in rr if r[f] is not None]);assert a['n']==s['n'];error=max(error,assert_value(a['mean'],s['mean']),assert_value(a['ci'],s['ci']))
    research=sum(read(ROOT/(p+'_completion.json'))['physical_samples'] for p in ['pilot','development'])
    reloads=sum(read(ROOT/(p+'_verification.json'))['model_reload_samples'] for p in ['pilot','development'])
    assert research<=2500000 and reloads<=1000000
    # Check executing producer/validator exactly equal the pre-run source copies.
    for p in (ROOT/'implementation').glob('*.py'):assert sha(p)==sha(Path(__file__).parent/p.name)
    save(ROOT/'audit.json',dict(accepted=True,max_error=error,rank_start_pairs=totalpairs,research_model_samples=research,verification_model_samples=reloads,
        total_participant_model_samples=research+reloads,pooled_samples=read(ROOT/'external_completion.json')['pooled_samples'],truth_loaded=False,new_training=0));check()
    print(read(ROOT/'audit.json'))
if __name__=='__main__':audit()
