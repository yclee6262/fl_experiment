"""Read-only aggregation, decomposition audit, and allocation diagnostics."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from contribution_analysis.model_pool_v_cm_validation_v1.run import OUT,OLD,save,plain
import numpy as np
T=1e-10
def shares(x,tau=T):
    w=np.where(np.asarray(x)>tau,x,0.)
    p=w/w.sum() if w.sum() else w
    assert np.min(p)>=0 and abs(p.sum()+(0 if w.sum() else 1)-1)<1e-10
    return p
def sums(r,m,k='total'):
    return np.array([sum(z[m][k] for z in r['rows'] if z['i']==i) for i in range(r['n'])])
def table(zs):
    out={'units':len(zs)}
    for m in ['V','CM']:
        vals=np.array([z[m]['total'] for z in zs]);peer=np.array([z[m]['peer'] for z in zs])
        out[m]=dict(positive=int(sum(vals>T)),negative=int(sum(vals< -T)),zero=int(sum(abs(vals)<=T)),
            positive_peer_negative=int(sum((vals>T)&(peer< -T))),positive_peer_nonpositive=int(sum((vals>T)&(peer<=T))))
    out['Vpositive_CMnegative']=sum(z['V']['total']>T and z['CM']['total']< -T for z in zs)
    out['Vzero_CMnonzero']=sum(abs(z['V']['total'])<=T and abs(z['CM']['total'])>T for z in zs)
    return out
def aggregate(rr):
    rows=[z for r in rr for z in r['rows']];cum=[]
    for r in rr:
        for i in range(r['n']):cum.append({m:{k:float(sums(r,m,k)[i]) for k in ['total','self','peer']} for m in ['V','CM']})
    allocation={}
    for m in ['V','CM']:
        ps=[shares(sums(r,m)) for r in rr]
        allocation[m]=dict(no_distribution=sum(p.sum()==0 for p in ps),mean_max_share=float(np.mean([max(p) for p in ps])),max_share=float(max(max(p) for p in ps)),
            mean_share_to_negative_peer=float(np.mean([sum(p[sums(r,m,'peer')< -T]) for p,r in zip(ps,rr)])),
            mean_share_to_negative_CM=float(np.mean([sum(p[sums(r,'CM')< -T]) for p,r in zip(ps,rr)])),
            max_threshold_TV=float(max(np.abs(shares(sums(r,m),tau)-shares(sums(r,m))).sum()/2 for r in rr for tau in [1e-12,1e-8])))
    d=[z for z in rows if z['V']['total']>T and z['CM']['total']< -T]
    return dict(teams=len(rr),round=table(rows),cumulative=table(cum),allocation=allocation,
        mean_allocation_TV=float(np.mean([np.abs(shares(sums(r,'V'))-shares(sums(r,'CM'))).sum()/2 for r in rr])),
        disagreement=dict(count=len(d),CM_peer_negative=sum(z['CM']['peer']< -T for z in d),CM_self_negative=sum(z['CM']['self']< -T for z in d),
            V_peer_negative=sum(z['V']['peer']< -T for z in d),different_endpoints=sum(z['CM_V_endpoint_distance']>T for z in d),
            median_angle_degrees=None if not [z['angle_degrees'] for z in d if z['angle_degrees'] is not None] else float(np.median([z['angle_degrees'] for z in d if z['angle_degrees'] is not None]))))
def main():
    comp=json.loads((OUT/'completion.json').read_text());assert comp['complete']
    reg=json.loads((OUT/'registration.json').read_text());rr=[];maxerr=0.
    for c in reg['known']+reg['fresh']:
        r=json.loads((OUT/'records'/f"{c['key']}_30.json").read_text());rr.append(r)
        assert r['samples']==sum(r['stages'].values()) and r['baseline_only_V_new_calls']==0
        for z in r['rows']:
            a=np.array(r['weights']);f=np.array(z['full_losses'])
            for m,field in [('V','minusV_losses'),('CM','minusM_losses')]:
                d=np.array(z[field])-f;v=z[m]
                maxerr=max(maxerr,abs(v['self']+v['peer']-v['total']),abs(a@d-v['total']))
                assert abs(a[z['i']]*d[z['i']]-v['self'])<T
                assert abs(sum(a[j]*d[j] for j in range(len(a)) if j!=z['i'])-v['peer'])<T
        if c in reg['known']:
            ref=json.loads((OLD/'records'/f"{c['key']}_CMN.json").read_text())
            assert len(ref['rows'])==len(r['rows']) and r['baseline_digest']==ref['baseline_digest']
            for z,p in zip(r['rows'],ref['rows']):
                for m in ['V','CM']:assert abs(z[m]['total']-p[m])<T
    assert maxerr<T
    result={phase:aggregate([r for r in rr if r['phase']==phase]) for phase in ['historical','known_analytic80','new_confirmation']}
    for n in [3,5]:
        for sc in ['general','heterogeneous_bias']:
            keys={c['key'] for c in reg['fresh'] if c['n']==n and c['scenario']==sc}
            result[f'new_n{n}_{sc}']=aggregate([r for r in rr if r['key'] in keys])
    costs=[];examples=[]
    for c in reg['fresh']:
        cr=json.loads((OUT/'records'/f"{c['key']}_cost.json").read_text());row={k:cr[k]['samples'] for k in cr}
        row.update(key=c['key'],B=cr['ACM']['stages']['baseline']);row['stages']={k:cr[k]['stages'] for k in cr};costs.append(row)
        assert cr['AV']['samples']==sum(x['samples'] for x in cr['AV']['stages'].values())
        # Match standalone and combined scores; anchor evaluation cannot change a CM endpoint.
        r=next(r for r in rr if r['key']==c['key'])
        for x,y in zip(r['rows'],cr['ACM']['rows']):
            for m in ['V','CM']:assert abs(x[m]['total']-y[m]['total'])<T
        ds=[z for z in r['rows'] if z['V']['total']>T and z['CM']['total']< -T]
        if ds:examples.append(dict(key=c['key'],weights=r['weights'],row=max(ds,key=lambda z:z['V']['total']-z['CM']['total'])))
    result['costs']=dict(teams=len(costs),totals={k:sum(c[k] for c in costs) for k in ['B','AV','ACM','J','LOO']},
        cheaper_than_LOO={k:sum(c[k]<c['LOO'] for c in costs) for k in ['AV','ACM','J']},
        stages={k:{stage:sum((c['stages'][k].get(stage,0) if k!='AV' else c['stages'][k].get(stage,{}).get('samples',0)) for c in costs) for stage in sorted({s for c in costs for s in c['stages'][k]})} for k in ['AV','ACM','J','LOO']})
    sensitivity=[]
    for c in reg['fresh']:
        if not c['sensitivity']:continue
        base=next(r for r in rr if r['key']==c['key'])
        for it in [10,60]:
            r=json.loads((OUT/'records'/f"{c['key']}_{it}.json").read_text())
            sensitivity.append(dict(key=c['key'],iterations=it,metrics={m:dict(TV=float(np.abs(shares(sums(r,m))-shares(sums(base,m))).sum()/2),
                sign_changes=int(sum(np.sign(np.where(abs(sums(r,m))>T,sums(r,m),0))!=np.sign(np.where(abs(sums(base,m))>T,sums(base,m),0))))) for m in ['V','CM']}))
    result['sensitivity']={str(it):{m:dict(mean_TV=float(np.mean([z['metrics'][m]['TV'] for z in sensitivity if z['iterations']==it])),max_TV=max(z['metrics'][m]['TV'] for z in sensitivity if z['iterations']==it),sign_changes=sum(z['metrics'][m]['sign_changes'] for z in sensitivity if z['iterations']==it)) for m in ['V','CM']} for it in [10,60]}
    for p,h in json.loads((OUT/'source_lock.json').read_text()).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    save(OUT/'analysis.json',result);save(OUT/'examples.json',examples[:8]);save(OUT/'sensitivity.json',sensitivity)
    save(OUT/'audit.json',dict(passed=True,known_replay_teams=100,new_confirmation_teams=24,decomposition_max_error=maxerr,source_lock=True,budget_conservation=True,standalone_combined_scores_equal=True))
    print(json.dumps(plain(result),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
