"""Fixed 24 comparisons; no external evaluator; outcomes read after prediction lock."""
import csv,time
import numpy as np
from scipy.stats import spearmanr
from common import *
COMBOS=[('N0','model'),('N1','anchor'),('N1','model'),('N1','joint')]
def sg(x):return int(x>1e-4)-int(x< -1e-4)
def interval(v):
    a=np.asarray([x for x in v if x is not None],float)
    if not len(a):return dict(mean=None,ci=None,n=0)
    rng=np.random.default_rng(20261021);m=a[rng.integers(0,len(a),(10000,len(a)))].mean(axis=1)
    return dict(mean=float(a.mean()),ci=np.percentile(m,[.1,99.9]).tolist(),n=len(a))
def corr(x,y):
    if len(x)<3 or np.ptp(x)<=1e-4 or np.ptp(y)<=1e-4:return None
    return float(spearmanr(x,y).statistic)
def metrics(rr,T):
    ranks=[];oracle_rankable=0
    for t in range(T):
        rows=[r for r in rr if r['t']==t];p=[r['predicted'] for r in rows];y=[r['exact'] for r in rows]
        if len(y)>=3 and np.ptp(y)>1e-4:oracle_rankable+=1
        v=corr(p,y)
        if v is not None:ranks.append(v)
    ys=np.array([r['exact'] for r in rr]);ps=np.array([r['predicted'] for r in rr]);signs=[r for r in rr if sg(r['exact'])]
    pos=[r for r in rr if sg(r['exact'])==1];neg=[r for r in rr if sg(r['exact'])==-1];zero=[r for r in rr if sg(r['exact'])==0]
    den=float(np.abs(ys).sum());error=np.abs(ps-ys)
    rate=lambda rows:None if not rows else float(np.mean([sg(r['predicted'])==sg(r['exact']) for r in rows]))
    return dict(rho=None if not ranks else float(np.mean(ranks)),rankable_rounds=len(ranks),oracle_rankable_rounds=oracle_rankable,
        all_round_coverage=len(ranks)/T,oracle_rank_coverage=None if not oracle_rankable else len(ranks)/oracle_rankable,
        signs=rate(signs),positive_recall=rate(pos),negative_recall=rate(neg),positive_n=len(pos),negative_n=len(neg),
        zero_false_positive=None if not zero else float(np.mean([sg(r['predicted'])!=0 for r in zero])),zero_n=len(zero),
        nmae=None if den<=len(rr)*1e-4 else float(error.sum()/den),mae=float(error.mean()),bias=float(np.mean(ps-ys)),
        median_abs_error=float(np.median(error)),p90_abs_error=float(np.percentile(error,90)),
        predictions_near_zero=int(sum(sg(v)==0 for v in ps)),exact_nonzero_pred_zero=int(sum(sg(r['exact'])!=0 and sg(r['predicted'])==0 for r in rr)),
        zero_prediction_nmae=None if den<=len(rr)*1e-4 else 1.,complete=True,n=len(rr),rounds=T)

def analyze():
    check();assert read(ROOT/'verification.json')['accepted'];assert read(ROOT/'timing.json')['complete']
    for p,h in read(ROOT/'predictions_locked.json')['hashes'].items():assert sha(ROOT/p)==h
    tic=time.perf_counter();rows=[];teams=[];costs=read(ROOT/'reference_costs.json');timing=read(ROOT/'timing.json');examples=[];strata=[];totals=[]
    for c in read(ROOT/'cases.json')['development']:
        r=read(ROOT/'development'/(c['key']+'.json'));oracle=read(ORACLE/'development'/(c['key']+'.json'))
        for p in r['predictions']:
            g=next(g for g in oracle['groups'] if g['t']==p['t'] and g['i']==p['i']);b=g['roles'][p['role']];data=r['views'][p['t']]
            exact=b['score'];hat=p['score'];gf=np.asarray(g['full']['g']);gr=np.asarray(b['decision']['g'])
            norm=np.linalg.norm(gf)*np.linalg.norm(gr)
            angle=None if norm<=1e-16 else float(np.degrees(np.arccos(np.clip(np.dot(gf,gr)/norm,-1,1))))
            trials=b['decision']['trials'];first=bool(trials and trials[0]['accepted']);fullfirst=bool(g['full']['trials'] and g['full']['trials'][0]['accepted'])
            rec=dict(key=c['key'],scenario=c['scenario'],seed=c['seed'],t=p['t'],i=p['i'],agent_id=p['agent_id'],method=p['method'],role=p['role'],
                predicted=hat,exact=exact,error=hat-exact,abs_error=abs(hat-exact),angle=angle,mode=data['mode'],
                full_accepted=g['full']['accepted'],minus_accepted=b['decision']['accepted'],full_first_accept=fullfirst,minus_first_accept=first,
                minus_attempts=len(trials),full_attempts=len(g['full']['trials']),both_first_accept=first and fullfirst,
                boundary_limited=False,endpoint=p['endpoint'],oracle_endpoint=b['endpoint'],full_endpoint=data['full_endpoint'])
            if p['method']=='N1':
                lx=float(np.dot(np.asarray(c['alphas']),np.asarray(b['common_losses'])))
                # Algebraic truncation diagnostic from saved common losses and predicted score.
                rec['truncation_net_error']=hat-exact
                if p['candidate'] is not None:
                    rec['boundary_limited']=np.linalg.norm(np.asarray(p['candidate'])-np.asarray(data['x']))<data['eta']-1e-10
            rows.append(rec)
        for method,role in COMBOS:
            rr=[x for x in rows if x['key']==c['key'] and x['method']==method and x['role']==role]
            teams.append(dict(key=c['key'],scenario=c['scenario'],seed=c['seed'],method=method,role=role,**metrics(rr,len(r['views']))))
            for i in c['ids']:
                v=[x for x in rr if x['agent_id']==i]
                totals.append(dict(key=c['key'],method=method,role=role,agent_id=i,predicted_sum=sum(x['predicted'] for x in v),exact_sum=sum(x['exact'] for x in v),
                    predicted_positive=sum(max(0.,x['predicted']) for x in v),predicted_negative=sum(min(0.,x['predicted']) for x in v)))
    summary=[]
    for scenario in ['clean','controlled_poison']:
        for method,role in COMBOS:
            rr=[r for r in rows if r['scenario']==scenario and r['method']==method and r['role']==role]
            ts=[t for t in teams if t['scenario']==scenario and t['method']==method and t['role']==role]
            measures={k:interval([t[k] for t in ts]) for k in ['rho','signs','nmae','zero_false_positive','positive_recall','negative_recall','mae','bias']}
            rank_cov=sum(t['rankable_rounds'] for t in ts)/sum(t['rounds'] for t in ts)
            orac_cov=sum(t['rankable_rounds'] for t in ts)/max(1,sum(t['oracle_rankable_rounds'] for t in ts))
            gate={}
            gate['rank']=measures['rho']['n']>=8 and measures['rho']['mean']>=.8 and measures['rho']['ci'][0]>.5
            gate['sign']=measures['signs']['n']>=8 and measures['signs']['mean']>=.85 and measures['signs']['ci'][0]>.7
            gate['nmae']=measures['nmae']['n']>=8 and measures['nmae']['mean']<=.25 and measures['nmae']['ci'][1]<.5
            gate['positive_recall']=measures['positive_recall']['mean'] is not None and measures['positive_recall']['mean']>=.75 and sum(t['positive_n']>0 for t in ts)>=3 and sum(t['positive_n'] for t in ts)>=20
            gate['negative_recall']=measures['negative_recall']['mean'] is not None and measures['negative_recall']['mean']>=.75 and sum(t['negative_n']>0 for t in ts)>=3 and sum(t['negative_n'] for t in ts)>=20
            gate['zero_false_positive']=measures['zero_false_positive']['n']>=5 and measures['zero_false_positive']['mean']<=.1
            gate['coverage']=rank_cov>=.5 and orac_cov>=.8
            if method=='N0':cost=dict(new_samples=0,query_gate=True,time_gate=timing['N0_time_gate'])
            else:
                cs=[x for x in costs if x['phase']=='development' and x['scenario']==scenario and x['role']==role]
                new=sum(x['estimate_new_samples'] for x in cs);ref=sum(x['oracle_new_samples'] for x in cs);base=sum(x['baseline_samples'] for x in cs)
                cost=dict(new_samples=new,exact_new_samples=ref,baseline_samples=base,new_ratio=new/ref if ref else None,
                    total_ratio=(base+new)/(base+ref),query_gate=ref>0 and new/ref<=.3 and (base+new)/(base+ref)<=.7,time_gate=timing['N1_time_gate'])
            gate['cost']=cost['query_gate'] and cost['time_gate']
            summary.append(dict(scenario=scenario,method=method,role=role,n=len(rr),metrics=measures,rank_coverage=rank_cov,
                oracle_rank_coverage=orac_cov,predicted_near_zero=sum(sg(r['predicted'])==0 for r in rr),
                exact_nonzero_pred_zero=sum(sg(r['predicted'])==0 and sg(r['exact'])!=0 for r in rr),cost=cost,gates=gate,passed=all(gate.values())))
            subsets={
                'max_abs_error':rr,
                'max_sign_reversal':[r for r in rr if sg(r['exact'])*sg(r['predicted'])==-1],
                'missed_negative':[r for r in rr if sg(r['exact'])==-1 and sg(r['predicted'])!=-1],
                'zero_false_positive':[r for r in rr if sg(r['exact'])==0 and sg(r['predicted'])!=0]}
            for name,values in subsets.items():
                selected=sorted(values,key=lambda r:(-r['abs_error'],r['seed'],r['t'],r['agent_id']))[0] if values else None
                examples.append(dict(scenario=scenario,method=method,role=role,selection=name,case=selected))
            groups={
                'same_direction':[r for r in rr if r['angle'] is not None and r['angle']<=1],
                'changed_direction':[r for r in rr if r['angle'] is not None and r['angle']>1],
                'both_first_accept':[r for r in rr if r['both_first_accept']],
                'later_annealing':[r for r in rr if r['minus_attempts']>1 or r['full_attempts']>1],
                'full_no_update':[r for r in rr if not r['full_accepted']],
                'secant':[r for r in rr if r['mode']=='secant'],
                'dynamic':[r for r in rr if r['mode']=='dynamic'],
                'boundary':[r for r in rr if r['boundary_limited']]}
            for group,values in groups.items():
                strata.append(dict(scenario=scenario,method=method,role=role,group=group,n=len(values),teams=len(set(x['key'] for x in values)),
                    enough=len(values)>=20 and len(set(x['key'] for x in values))>=3,
                    mae=None if not values else float(np.mean([x['abs_error'] for x in values])),
                    max_abs_error=None if not values else max(x['abs_error'] for x in values)))
    paired=[]
    for scenario in ['clean','controlled_poison']:
        for metric in ['rho','nmae','signs']:
            diffs=[]
            for seed in range(2026101610,2026101620):
                pair=[t for t in teams if t['scenario']==scenario and t['seed']==seed and t['role']=='model']
                a=next(t for t in pair if t['method']=='N0')[metric];b=next(t for t in pair if t['method']=='N1')[metric]
                if a is not None and b is not None:diffs.append(b-a)
            paired.append(dict(scenario=scenario,metric=metric,N1_minus_N0=interval(diffs)))
    save(ROOT/'comparisons.json',rows);save(ROOT/'team_metrics.json',teams);save(ROOT/'participant_totals.json',totals)
    save(ROOT/'summary.json',summary);save(ROOT/'examples.json',examples);save(ROOT/'strata.json',strata);save(ROOT/'paired.json',paired)
    with (ROOT/'comparisons.csv').open('x',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0])+[k for k in rows[-1] if k not in rows[0]]);wr.writeheader();wr.writerows(rows)
    save(ROOT/'analysis_completion.json',dict(complete=True,rows=len(rows),teams=len(teams),wall_seconds=time.perf_counter()-tic,
        new_model_samples=0,pooled_loaded=False,truth_loaded=False,any_method_established=any(x['passed'] for x in summary)))
if __name__=='__main__':analyze()
