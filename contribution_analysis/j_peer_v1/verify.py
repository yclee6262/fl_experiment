"""Independent raw-reply, accounting and manual statistics audit."""
import itertools,math,time
from common import *
error=0.
def mean(v):return math.fsum(v)/len(v) if v else None
def sign(x):return 1 if x>1e-4 else -1 if x< -1e-4 else 0
def rank(x,y):
    if len(x)<3 or max(x)-min(x)<=1e-4 or max(y)-min(y)<=1e-4:return None
    def ranks(x):return [1+sum(v<a for v in x)+(sum(v==a for v in x)-1)/2 for a in x]
    a=ranks(x);b=ranks(y);ma=mean(a);mb=mean(b)
    return math.fsum((x-ma)*(y-mb) for x,y in zip(a,b))/(math.fsum((x-ma)**2 for x in a)*math.fsum((y-mb)**2 for y in b))**.5
def match(a,b):
    global error
    if a is None or b is None:assert a is None and b is None;return
    error=max(error,near(a,b))
def boot(v):
    if not v:return dict(n=0,mean=None,ci=None)
    rng=np.random.default_rng(20261027);a=[mean([v[i] for i in rng.integers(0,len(v),len(v))]) for _ in range(10000)]
    return dict(n=len(v),mean=mean(v),ci=np.quantile(a,[.025,.975]).tolist())
def check_boot(v,expected):
    actual=boot(v);assert actual['n']==expected['n'];match(actual['mean'],expected['mean']);match(actual['ci'],expected['ci'])
def main():
    tic=time.perf_counter()
    for p,h in read(ROOT/'inputs.json').items():assert sha(p)==h
    for p,h in read(ROOT/'source_lock.json').items():assert sha(ROOT/'source_snapshot'/p)==h
    rows=read(ROOT/'round_scores.json');agents=read(ROOT/'agent_scores.json');costs=read(ROOT/'cost_ledgers.json');starts=read(ROOT/'start_metrics.json');teams=read(ROOT/'team_metrics.json');stable=read(ROOT/'team_stability.json')
    old={(c['key'],c['start']):c for c in read(SRC/'development_scores.json')};ext={(c['key'],c['start']):c for c in read(SRC/'external_rows.json')};checked=[]
    for c in read(SRC/'cases.json')['development']:
        for si in range(5):
            r=read(SRC/'development'/(c['key']+f'_s{si}_J.json'));raw={key(v['i'],v['x']):v['y'] for v in r['raw']};rr=[v for v in rows if v['key']==c['key'] and v['start']==si];n=len(c['ids'])
            for a,b in zip(rr,r['branches']):
                assert (a['i'],a['t'])==(b['i'],b['t']);ids=[k for k in range(n) if k!=a['i']];den=math.fsum(c['alphas'][k] for k in ids)
                va=math.fsum(c['alphas'][k]*abs(raw[key(k,b['x_minus'])]) for k in ids)/den;vb=math.fsum(c['alphas'][k]*abs(raw[key(k,b['x_full'])]) for k in ids)/den
                match(va-vb,a['j_peer']);assert a['j']==b['score']
            aa=[a for a in agents if a['key']==c['key'] and a['start']==si]
            for i,a in enumerate(aa):
                assert a['agent_id']==c['ids'][i] and a['J']==r['values'][i];match(math.fsum(v['j_peer'] for v in rr if v['i']==i),a['J_peer'])
                assert a['D_L']==old[c['key'],si]['methods']['LOO']['values'][i];assert a['D_E']==ext[c['key'],si]['D_E'][i] and a['J_E']==ext[c['key'],si]['J_E'][i]
            # Independently build original and peer billing from native prefix and raw coordinates.
            counts={};peer_events=[]
            for peer in [False,True]:
                prefix=[dict(stage=e['stage'],key=e['key']) for e in r['events'] if e['stage']!='common'];events=list(prefix)
                for b in r['branches']:
                    for p in [b['x_minus'],b['x_full']]:
                        for k in range(n):
                            if not peer or k!=b['i']:events.append(dict(stage='common',key=key(k,p)))
                known=set();billed=0
                for e in events:
                    hit=e['stage']!='baseline' and e['key'] in known;e['hit']=hit;billed+=not hit;known.add(e['key'])
                counts['J_peer' if peer else 'J']=billed
                if peer:peer_events=events
            item=next(v for v in costs if v['key']==c['key'] and v['start']==si)
            assert counts['J']==r['samples']==item['J'];assert counts['J_peer']==item['J_peer'];assert peer_events==item['peer_events']
            assert item['LOO']==old[c['key'],si]['methods']['LOO']['samples'];assert item['baseline']==sum(e['stage']=='baseline' for e in r['events']);assert item['peer_requests']==len(peer_events)
            checked.append(dict(key=c['key'],start=si,J=counts['J'],J_peer=counts['J_peer']))
            actual={}
            for method in ['J','J_peer']:
                for ref in ['D_L','J_E','D_E']:
                    x=[a[method] for a in aa];y=[a[ref] for a in aa];nz=[i for i,v in enumerate(y) if sign(v)];den=math.fsum(abs(v) for v in y)
                    actual[method+':'+ref+':rank']=rank(x,y);actual[method+':'+ref+':sign']=mean([sign(x[i])==sign(y[i]) for i in nz]);actual[method+':'+ref+':nmae']=None if den<=len(y)*EPS else math.fsum(abs(x-y) for x,y in zip(x,y))/den
                field='j' if method=='J' else 'j_peer';nz=[a for a in rr if sign(a['j_E'])];actual[method+':local:sign']=mean([sign(a[field])==sign(a['j_E']) for a in nz]);ranks=[];ts=sorted(set(a['t'] for a in rr))
                for t in ts:
                    selected=[a for a in rr if a['t']==t];v=rank([a[field] for a in selected],[a['j_E'] for a in selected])
                    if v is not None:ranks.append(v)
                actual[method+':local:rank']=mean(ranks);actual[method+':local:coverage']=len(ranks)/len(ts)
            original=next(v for v in starts if v['key']==c['key'] and v['start']==si)
            for f,v in actual.items():match(v,original['metrics'][f])
    for t in teams:
        rr=[r for r in starts if r['key']==t['key']]
        for f,v in t['metrics'].items():match(mean([r['metrics'][f] for r in rr if r['metrics'][f] is not None]),v)
    for s in stable:
        vec=[[a[s['method']] for a in agents if a['key']==s['key'] and a['start']==si] for si in range(5)];rs=[];flips=[]
        for a,b in itertools.combinations(vec,2):
            v=rank(a,b)
            if v is not None:rs.append(v)
            flips.extend(sign(x)!=sign(y) for x,y in zip(a,b) if sign(x) and sign(y))
        match(mean(rs),s['rank']);match(mean(flips),s['flip']);match(len(rs)/10,s['coverage'])
    for summary in read(ROOT/'summary.json'):
        tt=[t for t in teams if t['scenario']==summary['scenario']];m=summary['metrics'];cc=[c for c in costs if c['scenario']==summary['scenario']]
        for f,v in m.items():check_boot([t['metrics'][f] for t in tt if t['metrics'][f] is not None],v)
        for f,v in summary['paired_peer_minus_J'].items():check_boot([t['metrics']['J_peer:'+f]-t['metrics']['J:'+f] for t in tt if t['metrics']['J_peer:'+f] is not None and t['metrics']['J:'+f] is not None],v)
        for f,v in summary['counts'].items():assert sum(c[f] for c in cc)==v
        for method in ['J','J_peer']:
            ss=summary['stability'][method]
            for f,v in ss.items():check_boot([s[f] for s in stable if s['scenario']==summary['scenario'] and s['method']==method and s[f] is not None],v)
            r=m[method+':D_L:rank'];s=m[method+':D_L:sign'];e=m[method+':D_L:nmae'];local=m[method+':local:sign'];lr=m[method+':local:rank'];lc=m[method+':local:coverage'];counts=summary['counts']
            expected=dict(full_loo_screen=min(r['n'],s['n'],e['n'])>=8 and r['mean']>=.8 and r['ci'][0]>.5 and s['mean']>=.85 and s['ci'][0]>.7 and e['mean']<=.25 and e['ci'][1]<.5 and counts[method]<=.7*counts['LOO'],local_quality_screen=min(local['n'],lr['n'])>=8 and local['mean']>=.85 and local['ci'][0]>.7 and lr['mean']>=.8 and lr['ci'][0]>.5 and lc['mean']>=.8,stability_screen=min(ss['rank']['n'],ss['flip']['n'])>=8 and ss['rank']['mean']>=.8 and ss['rank']['ci'][0]>.5 and ss['flip']['mean']<=.1 and ss['flip']['ci'][1]<.2 and ss['coverage']['mean']>=.8)
            assert expected==summary['gates'][method]
    save(ROOT/'verification.json',dict(accepted=True,max_error=error,branches=len(rows),agent_start_records=len(agents),independent_billing=True,independent_rank_and_bootstrap=True,new_model_inferences=0,wall_seconds=time.perf_counter()-tic));print(read(ROOT/'verification.json'))
if __name__=='__main__':main()
