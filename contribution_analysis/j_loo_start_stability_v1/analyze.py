"""Team-level start sensitivity. Pooled loads only after internal locks."""
import itertools,shutil
from scipy.stats import spearmanr
from common import *
def avg(v):return float(np.mean(v)) if v else None
def sign(v):return int(v>EPS)-int(v< -EPS)
def rho(a,b):
    if len(a)<3 or np.ptp(a)<=EPS or np.ptp(b)<=EPS:return None
    return float(spearmanr(a,b).statistic)
def boot(v):
    if not v:return dict(mean=None,ci=None,n=0)
    a=np.array(v);ix=np.random.default_rng(20261027).integers(0,len(a),(10000,len(a)))
    return dict(mean=float(a.mean()),ci=np.percentile(a[ix].mean(axis=1),[2.5,97.5]).tolist(),n=len(a))
def summarize(rows,fields):return {f:boot([r[f] for r in rows if r[f] is not None]) for f in fields}
def stability(values):
    ranks=[];flip=[];change=[];nonzero_pairs=0;possible=len(values[0])*10
    for a,b in itertools.combinations(values,2):
        v=rho(a,b)
        if v is not None:ranks.append(v)
        for x,y in zip(a,b):
            sx,sy=sign(x),sign(y)
            if sx and sy:flip.append(sx!=sy);nonzero_pairs+=1
            if sx or sy:change.append(sx!=sy)
    a=np.array(values);spread=np.ptp(a,axis=0);scale=np.mean(abs(a),axis=0)
    relative=[float(x/y) for x,y in zip(spread,scale) if y>EPS]
    return dict(rank=avg(ranks),rank_coverage=len(ranks)/10,flip=avg(flip),class_change=avg(change),nonzero_pairs=nonzero_pairs,
        possible_pairs=possible,spread=avg(spread.tolist()),relative_spread=avg(relative),all_nearzero=float(np.mean(np.max(abs(a),axis=0)<=EPS)))
def agreement(a,d):
    nz=[i for i,v in enumerate(d) if sign(v)];den=sum(abs(v) for v in d)
    return dict(rank=rho(a,d),sign=avg([sign(a[i])==sign(d[i]) for i in nz]),nmae=None if den<=len(d)*EPS else float(sum(abs(np.array(a)-d))/den))
def internal():
    check();assert read(ROOT/'development_verification.json')['accepted']
    (ROOT/'analysis_snapshot').mkdir()
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,ROOT/'analysis_snapshot'/p.name)
    save(ROOT/'analysis_source_lock.json',{p.name:sha(p) for p in (ROOT/'analysis_snapshot').glob('*.py')})
    scores=read(ROOT/'development_scores.json');timings=read(ROOT/'development_timings.json');team=[];comparison=[];cost=[]
    for c in read(ROOT/'cases.json')['development']:
        rr=[v for v in scores if v['key']==c['key']];assert len(rr)==5
        for method in ['J','LOO']:
            vals=[v['methods'][method]['values'] for v in rr]
            team.append(dict(key=c['key'],scenario=c['scenario'],method=method,**stability(vals)))
        comparisons=[agreement(v['methods']['J']['values'],v['methods']['LOO']['values']) for v in rr]
        comparison.append(dict(key=c['key'],scenario=c['scenario'],**{f:avg([r[f] for r in comparisons if r[f] is not None]) for f in ['rank','sign','nmae']}))
        for v in rr:
            times={m:float(np.median([r['wall_seconds'] for r in timings if r['key']==c['key'] and r['start']==v['start'] and r['method']==m])) for m in ['J','LOO']}
            cost.append(dict(key=c['key'],scenario=c['scenario'],start=v['start'],times=times,
                methods={m:{k:v['methods'][m][k] for k in ['samples','baseline_samples','extra_samples','python_peak_bytes']} for m in ['J','LOO']},
                query_ratio=v['methods']['J']['samples']/v['methods']['LOO']['samples'],time_ratio=times['J']/times['LOO']))
    summaries=[];costs=[];replacements=[]
    for sc in ['clean','controlled_poison']:
        for method in ['J','LOO']:
            rr=[v for v in team if v['scenario']==sc and v['method']==method]
            ss=summarize(rr,['rank','rank_coverage','flip','class_change','spread','relative_spread','all_nearzero'])
            coverage=ss['rank']['n']>=8 and ss['rank_coverage']['mean']>=.8 and ss['flip']['n']>=8 and sum(v['nonzero_pairs'] for v in rr)>=20
            stable=coverage and ss['rank']['mean']>=.8 and ss['rank']['ci'][0]>.5 and ss['flip']['mean']<=.1 and ss['flip']['ci'][1]<.2
            summaries.append(dict(scenario=sc,method=method,metrics=ss,nonzero_pairs=sum(v['nonzero_pairs'] for v in rr),coverage_pass=coverage,stable_pass=stable))
        rr=[v for v in cost if v['scenario']==sc];ct={m:{f:sum(v['methods'][m][f] for v in rr) for f in ['samples','baseline_samples','extra_samples']} for m in ['J','LOO']}
        for m in ct:
            ct[m]['total_median_seconds']=sum(v['times'][m] for v in rr)
            ct[m]['median_python_peak_bytes']=float(np.median([v['methods'][m]['python_peak_bytes'] for v in rr]))
        query_ratio=ct['J']['samples']/ct['LOO']['samples'];time_ratio=ct['J']['total_median_seconds']/ct['LOO']['total_median_seconds']
        costs.append(dict(scenario=sc,methods=ct,query_ratio=query_ratio,time_ratio=time_ratio,cost_pass=query_ratio<=.7))
        rr=[v for v in comparison if v['scenario']==sc];ss=summarize(rr,['rank','sign','nmae'])
        valid=all(v['n']>=8 for v in ss.values())
        effect=valid and ss['rank']['mean']>=.8 and ss['rank']['ci'][0]>.5 and ss['sign']['mean']>=.85 and ss['sign']['ci'][0]>.7 and ss['nmae']['mean']<=.25 and ss['nmae']['ci'][1]<.5
        replacements.append(dict(scenario=sc,metrics=ss,effect_pass=effect,cost_pass=query_ratio<=.7,complete_pass=effect and query_ratio<=.7))
    out=dict(stability=summaries,costs=costs,replacement=replacements,old_data_exploration=True)
    save(ROOT/'team_stability.json',team);save(ROOT/'team_comparison.json',comparison);save(ROOT/'costs.json',cost);save(ROOT/'internal_summary.json',out)
    save(ROOT/'internal_locked.json',dict(hashes={p:sha(ROOT/p) for p in ['team_stability.json','team_comparison.json','costs.json','internal_summary.json']},pooled_loaded=False))
    print(json.dumps(out,ensure_ascii=False,indent=2),flush=True)

def external():
    import torch
    from agent_client import PretrainedModel
    check();assert read(ROOT/'development_verification.json')['accepted']
    for p,h in read(ROOT/'internal_locked.json')['hashes'].items():assert sha(ROOT/p)==h
    points=[];rows=[];samples=0;tic=time.perf_counter()
    for c in read(ROOT/'cases.json')['development']:
        net=PretrainedModel(2);net.load_state_dict(torch.load(c['evaluator'],map_location='cpu',weights_only=True)['state_dict']);net.eval();cache={}
        def E(x):
            nonlocal samples
            k=np.asarray(x,np.float64).tobytes().hex()
            if k not in cache:
                if samples+2>30000 or time.perf_counter()-tic>300:raise RuntimeError('pooled_budget')
                with torch.no_grad():v=abs(float(net(torch.tensor(x,dtype=torch.float32)[None,:]).item()))
                cache[k]=dict(x=x,value=v);samples+=1
            return cache[k]['value']
        for si in range(5):
            rj=read(ROOT/'development'/(c['key']+f'_s{si}_J.json'));rd=read(ROOT/'development'/(c['key']+f'_s{si}_LOO.json'))
            je=np.zeros(len(c['ids']));de=np.zeros(len(c['ids']))
            for b in rj['branches']:je[b['i']]+=E(b['x_minus'])-E(b['x_full'])
            for b in rd['branches']:de[b['i']]=E(b['x_minus'])-E(b['x_full'])
            rows.append(dict(key=c['key'],scenario=c['scenario'],start=si,J=rj['values'],D_L=rd['values'],J_E=je,D_E=de))
        reload=PretrainedModel(2);reload.load_state_dict(torch.load(c['evaluator'],map_location='cpu',weights_only=True)['state_dict']);reload.eval()
        if samples+len(cache)>30000:raise RuntimeError('pooled_reload_budget')
        for v in cache.values():
            with torch.no_grad():y=abs(float(reload(torch.tensor(v['x'],dtype=torch.float32)[None,:]).item()))
            assert y==v['value'];samples+=1
        points.append(dict(key=c['key'],points=list(cache.values())))
    teams=[];summaries=[]
    for c in read(ROOT/'cases.json')['development']:
        rr=[v for v in rows if v['key']==c['key']]
        for a,b in [('J','J_E'),('D_L','D_E'),('J','D_E')]:
            vals=[agreement(v[a],v[b]) for v in rr]
            teams.append(dict(key=c['key'],scenario=c['scenario'],comparison=a+':'+b,rank=avg([v['rank'] for v in vals if v['rank'] is not None]),sign=avg([v['sign'] for v in vals if v['sign'] is not None])))
    for sc in ['clean','controlled_poison']:
        for comp in ['J:J_E','D_L:D_E','J:D_E']:
            rr=[v for v in teams if v['scenario']==sc and v['comparison']==comp]
            summaries.append(dict(scenario=sc,comparison=comp,metrics=summarize(rr,['rank','sign'])))
    save(ROOT/'pooled_points.json',points);save(ROOT/'external_rows.json',rows);save(ROOT/'external_team.json',teams);save(ROOT/'external_summary.json',summaries)
    save(ROOT/'external_completion.json',dict(complete=True,pooled_samples=samples,reload_error=0,wall_seconds=time.perf_counter()-tic,truth_loaded=False))
    print(json.dumps(summaries,ensure_ascii=False,indent=2))
if __name__=='__main__':
    setup()
    if sys.argv[1]=='internal':internal()
    else:external()
