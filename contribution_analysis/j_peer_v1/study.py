"""Frozen endpoint trial of an explicit J_peer estimator, not a new search."""
import itertools,shutil,time,signal
from scipy.stats import spearmanr
from common import *
from kernel import score_peer
def sign(v):return int(v>EPS)-int(v< -EPS)
def avg(v):return float(np.mean(v)) if v else None
def rho(x,y):return None if len(x)<3 or np.ptp(x)<=EPS or np.ptp(y)<=EPS else float(spearmanr(x,y).statistic)
def boot(v):
    if not v:return dict(mean=None,ci=None,n=0)
    a=np.array(v);ix=np.random.default_rng(20261027).integers(0,len(a),(10000,len(a)))
    return dict(mean=float(a.mean()),ci=np.percentile(a[ix].mean(axis=1),[2.5,97.5]).tolist(),n=len(a))
def compare(a,b):
    mask=[i for i,v in enumerate(b) if sign(v)];den=sum(abs(v) for v in b)
    return dict(rank=rho(a,b),sign=avg([sign(a[i])==sign(b[i]) for i in mask]),nmae=None if den<=len(b)*EPS else float(sum(abs(np.asarray(a)-b))/den))
def ledger(record,peer):
    seen=set();events=[]
    def event(stage,k):
        hit=stage!='baseline' and k in seen;events.append(dict(stage=stage,key=k,hit=hit));seen.add(k)
    for e in record['events']:
        if e['stage']!='common':event(e['stage'],e['key'])
    n=len(record['values'])
    for b in record['branches']:
        panel=[k for k in range(n) if not peer or k!=b['i']]
        for x in [b['x_minus'],b['x_full']]:
            for k in panel:event('common',key(k,x))
    return dict(samples=sum(not e['hit'] for e in events),baseline_samples=sum(e['stage']=='baseline' for e in events),requests=len(events),events=events)
def main():
    tic=time.perf_counter();ROOT.mkdir(exist_ok=False);(ROOT/'source_snapshot').mkdir();inputs={str(PLAN):sha(PLAN)}
    for source in [SRC,DIAG]:
        receipt=read(source/'completion_receipt.json');assert receipt['complete'] and receipt['independent_verification_accepted'];inputs[str(source/'completion_receipt.json')]=sha(source/'completion_receipt.json')
        for p,h in receipt['hashes'].items():assert sha(source/p)==h;inputs[str(source/p)]=h
        assert sha(receipt['report_path'])==receipt['report_sha256'];inputs[receipt['report_path']]=receipt['report_sha256']
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,ROOT/'source_snapshot'/p.name)
    shutil.copy2(PLAN,ROOT/PLAN.name);save(ROOT/'inputs.json',inputs);save(ROOT/'source_lock.json',{p.name:sha(p) for p in (ROOT/'source_snapshot').glob('*.py')})
    save(ROOT/'registration.json',dict(old_data_exploration=True,new_confirmation=False,original_J_preserved=True,peer_formula='two renormalized endpoint losses, then subtract; sum in original round order',new_inferences=0))
    old={(r['key'],r['start']):r for r in read(SRC/'development_scores.json')};ext={(r['key'],r['start']):r for r in read(SRC/'external_rows.json')}
    diag={(r['key'],r['start'],r['i'],r['t']):r for r in read(DIAG/'round_rows.json')};rows=[];agents=[];cost=[];startmetrics=[];err=0.
    for c in read(SRC/'cases.json')['development']:
        for si in range(5):
            r=read(SRC/'development'/(c['key']+f'_s{si}_J.json'));raw={key(v['i'],v['x']):v['y'] for v in r['raw']};current=[];peer_totals=[0.]*len(c['ids'])
            for b in r['branches']:
                a=[abs(raw[key(k,b['x_minus'])]) for k in range(len(c['ids']))];f=[abs(raw[key(k,b['x_full'])]) for k in range(len(c['ids']))];value=score_peer(a,f,c['alphas'],b['i']);prior=diag[c['key'],si,b['i'],b['t']]
                err=max(err,near(value,prior['other_mean']));peer_totals[b['i']]+=value
                row=dict(key=c['key'],scenario=c['scenario'],start=si,i=b['i'],agent_id=b['agent_id'],t=b['t'],j=b['score'],j_peer=value,j_E=prior['j_E']);current.append(row);rows.append(row)
            for i,aid in enumerate(c['ids']):agents.append(dict(key=c['key'],scenario=c['scenario'],start=si,agent_id=aid,J=r['values'][i],J_peer=peer_totals[i],D_L=old[c['key'],si]['methods']['LOO']['values'][i],J_E=ext[c['key'],si]['J_E'][i],D_E=ext[c['key'],si]['D_E'][i]))
            full=ledger(r,False);new=ledger(r,True);assert full['samples']==r['samples'];assert full['events']==[{k:e[k] for k in ['stage','key','hit']} for e in r['events']]
            cost.append(dict(key=c['key'],scenario=c['scenario'],start=si,J=full['samples'],J_peer=new['samples'],LOO=old[c['key'],si]['methods']['LOO']['samples'],baseline=new['baseline_samples'],peer_requests=new['requests'],peer_events=new['events']))
            aa=[a for a in agents if a['key']==c['key'] and a['start']==si];metrics={}
            for method in ['J','J_peer']:
                for ref in ['D_L','J_E','D_E']:
                    for f,v in compare([a[method] for a in aa],[a[ref] for a in aa]).items():metrics[method+':'+ref+':'+f]=v
                nonzero=[x for x in current if sign(x['j_E'])];field='j' if method=='J' else 'j_peer'
                metrics[method+':local:sign']=avg([sign(x[field])==sign(x['j_E']) for x in nonzero]);ranked=[];ts=sorted(set(x['t'] for x in current))
                for t in ts:
                    rr=[x for x in current if x['t']==t];v=rho([x[field] for x in rr],[x['j_E'] for x in rr])
                    if v is not None:ranked.append(v)
                metrics[method+':local:rank']=avg(ranked);metrics[method+':local:coverage']=len(ranked)/len(ts)
            startmetrics.append(dict(key=c['key'],scenario=c['scenario'],start=si,metrics=metrics))
    fields=list(startmetrics[0]['metrics']);teams=[];stable=[]
    for c in read(SRC/'cases.json')['development']:
        rr=[s for s in startmetrics if s['key']==c['key']];m={f:avg([s['metrics'][f] for s in rr if s['metrics'][f] is not None]) for f in fields};teams.append(dict(key=c['key'],scenario=c['scenario'],metrics=m))
        for method in ['J','J_peer']:
            vectors=[[a[method] for a in agents if a['key']==c['key'] and a['start']==si] for si in range(5)];ranked=[];flips=[]
            for a,b in itertools.combinations(vectors,2):
                v=rho(a,b)
                if v is not None:ranked.append(v)
                flips.extend(sign(x)!=sign(y) for x,y in zip(a,b) if sign(x) and sign(y))
            stable.append(dict(key=c['key'],scenario=c['scenario'],method=method,rank=avg(ranked),flip=avg(flips),coverage=len(ranked)/10))
    summaries=[]
    for sc in ['clean','controlled_poison']:
        tt=[t for t in teams if t['scenario']==sc];m={f:boot([t['metrics'][f] for t in tt if t['metrics'][f] is not None]) for f in fields}
        paired={f:boot([t['metrics']['J_peer:'+f]-t['metrics']['J:'+f] for t in tt if t['metrics']['J_peer:'+f] is not None and t['metrics']['J:'+f] is not None]) for f in ['D_L:rank','D_L:sign','D_L:nmae','J_E:rank','D_E:rank','local:sign']}
        cc=[c for c in cost if c['scenario']==sc];counts={f:sum(c[f] for c in cc) for f in ['J','J_peer','LOO','baseline']};st={}
        for method in ['J','J_peer']:
            rr=[s for s in stable if s['scenario']==sc and s['method']==method];st[method]={f:boot([s[f] for s in rr if s[f] is not None]) for f in ['rank','flip','coverage']}
        def gates(method):
            r=m[method+':D_L:rank'];s=m[method+':D_L:sign'];e=m[method+':D_L:nmae'];local=m[method+':local:sign'];lr=m[method+':local:rank'];lc=m[method+':local:coverage'];ss=st[method]
            return dict(full_loo_screen=min(r['n'],s['n'],e['n'])>=8 and r['mean']>=.8 and r['ci'][0]>.5 and s['mean']>=.85 and s['ci'][0]>.7 and e['mean']<=.25 and e['ci'][1]<.5 and counts[method]<=.7*counts['LOO'],
                local_quality_screen=min(local['n'],lr['n'])>=8 and local['mean']>=.85 and local['ci'][0]>.7 and lr['mean']>=.8 and lr['ci'][0]>.5 and lc['mean']>=.8,
                stability_screen=min(ss['rank']['n'],ss['flip']['n'])>=8 and ss['rank']['mean']>=.8 and ss['rank']['ci'][0]>.5 and ss['flip']['mean']<=.1 and ss['flip']['ci'][1]<.2 and ss['coverage']['mean']>=.8)
        summaries.append(dict(scenario=sc,metrics=m,paired_peer_minus_J=paired,counts=counts,peer_vs_J_ratio=counts['J_peer']/counts['J'],peer_vs_LOO_ratio=counts['J_peer']/counts['LOO'],stability=st,gates={method:gates(method) for method in ['J','J_peer']}))
    for name,v in [('round_scores',rows),('agent_scores',agents),('cost_ledgers',cost),('start_metrics',startmetrics),('team_metrics',teams),('team_stability',stable),('summary',summaries)]:save(ROOT/(name+'.json'),v)
    for p,h in inputs.items():assert sha(p)==h
    save(ROOT/'analysis_completion.json',dict(complete=True,branches=len(rows),agent_start_records=len(agents),prior_peer_mean_max_error=err,new_searches=0,new_model_inferences=0,new_pooled_inferences=0,truth_loaded=False,wall_seconds=time.perf_counter()-tic))
    for s in summaries:print(s['scenario'],{k:s['metrics'][k] for k in ['J_peer:D_L:rank','J_peer:D_L:sign','J_peer:D_L:nmae','J_peer:J_E:rank','J_peer:D_E:rank']},s['counts'],s['gates'])
if __name__=='__main__':signal.alarm(300);main()
