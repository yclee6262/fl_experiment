"""Independent formulas and original control interpreter; no estimator imports."""
import copy,time
from types import SimpleNamespace
import numpy as np
from common import *

def kx(x):return np.asarray(x,np.float64).tobytes().hex()
def bnd(x,d,h):
    limits=[h]
    for a,b in zip(x,d):
        if b>0:limits.append((1.-a)/b)
        elif b<0:limits.append((-1.-a)/b)
    return max(0.,min(limits))

def verify():
    from support import load_server
    from independent_step import one_round
    check();lock=read(ROOT/'predictions_locked.json')
    for p,h in lock['hashes'].items():assert sha(ROOT/p)==h
    assert read(ORACLE/'verification.json')['accepted']
    tic=time.perf_counter();limit=Limit(100000,600);extra_limit=Limit(60000,900)
    maxerr=0.;pred_count=0;oracle_count=0;rawerr=0.;costs=[];notes=[]
    for phase,cs in read(ROOT/'cases.json').items():
        for c in cs:
            print('verify',c['key'],flush=True);r=read(ROOT/phase/(c['key']+'.json'));s=load_server({k:v for k,v in c.items() if k!='evaluator'})
            baseline={str(v['i'])+':'+kx(v['x']):v for v in r['normal_replies']}
            extra={str(v['i'])+':'+kx(v['x']):v for v in r['new_replies']}
            for source,budget in [(baseline,limit),(extra,extra_limit)]:
                for v in source.values():
                    budget.before();start=time.perf_counter();y=float(s.trusted_agents[v['i']].api_predict(np.asarray(v['x']))[0]);budget.charge(time.perf_counter()-start)
                    error=abs(y-v['y']);rawerr=max(rawerr,error);assert error==0.
            a=np.asarray(c['alphas']);a=a/a.sum();I=np.asarray(c['candidates']);n=len(a)
            for p in r['predictions']:
                data=r['views'][p['t']];x=np.asarray(data['x']);i=p['i'];role=p['role']
                allowed={k:v for k,v in baseline.items() if v['available_round']<=p['t']}
                def vec(z,aug=False):
                    arr=[]
                    for j in range(n):
                        k=str(j)+':'+kx(z)
                        if k in allowed:arr.append(abs(allowed[k]['y']))
                        else:
                            assert aug,'future_or_unknown_reply'
                            arr.append(abs(extra[k]['y']) if k in extra else float('nan'))
                    return np.asarray(arr)
                panel=[j for j in range(n) if role=='anchor' or j!=i];w=a[panel]/sum(a[panel]);end=x.copy()
                if p['method']=='N0':
                    for z in data['trials']:
                        if float(sum(w[k]*vec(z)[j] for k,j in enumerate(panel)))<float(sum(w[k]*vec(x)[j] for k,j in enumerate(panel))):end=np.asarray(z);break
                    predicted=float(np.dot(a,vec(end))-np.dot(a,vec(data['full_endpoint'])))
                else:
                    ks=[j for j in range(n) if role=='model' or j!=i];aw=a[ks]/sum(a[ks]);g=np.zeros_like(x)
                    for j,weight in zip(ks,aw):
                        dist=np.linalg.norm(I[j]-x)
                        if dist<1e-8:continue
                        u=(I[j]-x)/dist
                        if data['mode']=='secant':z=I[j];h=dist
                        else:
                            h=bnd(x,u,data['delta'])
                            if h<=1e-12:continue
                            z=x+h*u
                        # Dot of model differences instead of estimator's difference of dots.
                        slope=float(np.dot(w,vec(z)[panel]-vec(x)[panel]))/h
                        g+=weight*slope*u
                    maxerr=max(maxerr,near(g,p['g']))
                    # Recreate endpoint using estimator's saved direction after verifying it;
                    # separately require independently formed direction same accept/endpoint tolerance.
                    gn=np.linalg.norm(g);candidate=None
                    if gn>1e-8:
                        u=g/gn;candidate=x-bnd(x,-u,data['eta'])*u
                        # Tiny arithmetic differences can change raw-byte keys. Compare planned candidate,
                        # then use its exact API input for independent acceptance, never nearby cache lookup.
                        maxerr=max(maxerr,near(candidate,p['candidate']));z=np.asarray(p['candidate'])
                        if float(np.dot(w,vec(z,True)[panel]))<float(np.dot(w,vec(x)[panel])):end=z
                    else:assert p['candidate'] is None
                    xf=x.copy()
                    if data['trials']:
                        z=np.asarray(data['trials'][0])
                        if float(np.dot(a,vec(z)))<float(np.dot(a,vec(x))):xf=z
                    predicted=float(np.dot(a,vec(end,True))-np.dot(a,vec(xf)))
                    maxerr=max(maxerr,near(xf,p['full_one_endpoint']))
                maxerr=max(maxerr,near(predicted,p['score']),near(end,p['endpoint']));pred_count+=1
            # Independently reconstruct each mode's chronological billing (no future baseline keys).
            for mode in ['standalone_anchor','standalone_model','standalone_joint','shared']:
                own=set();charged=0;requests=0
                for e in r['query_events']:
                    if e['mode']!=mode:continue
                    requests+=1;k=str(e['i'])+':'+kx(e['x'])
                    allowed=k in baseline and baseline[k]['available_round']<=e['t']
                    if not allowed and k not in own:charged+=1;own.add(k)
                assert charged==r['cost'][mode]['samples'] and requests==r['cost'][mode]['requests']
            # Oracle labels only enter now. Rebuild all branches from frozen raw replies.
            oracle=read(ORACLE/phase/(c['key']+'.json'));old=read(PREV/phase/(c['key']+'.json'))
            raw={str(v['i'])+':'+kx(v['x']):v for v in old['replies']+oracle['new_replies']}
            for role in ['anchor','model','joint']:
                own={};requests=0;charged=0
                for t,data in enumerate(r['views']):
                    allowed={k:v for k,v in baseline.items() if v['available_round']<=t}
                    def rawreply(j,z):
                        nonlocal requests,charged
                        requests+=1;k=str(j)+':'+kx(z)
                        assert k in raw,('missing_oracle_reply',c['key'],t,role,j)
                        if k not in allowed and k not in own:charged+=1;own[k]=raw[k]
                        return abs(raw[k]['y'])
                    for i in sorted(range(n),key=lambda j:c['ids'][j]):
                        group=next(g for g in oracle['groups'] if g['t']==t and g['i']==i);ref=group['roles'][role]
                        panel=[j for j in range(n) if role=='anchor' or j!=i];w=a[panel]/sum(a[panel]);anchors=[j for j in range(n) if role=='model' or j!=i]
                        geom=SimpleNamespace(I_list=I[anchors],alphas=a[anchors]/sum(a[anchors]),feasible_lower=-1.,feasible_upper=1.)
                        def objective(z):return float(np.dot(w,[rawreply(j,z) for j in panel]))
                        pre=copy.deepcopy(group['full']['pre']);lx=objective(np.asarray(data['x']))
                        pre.update(S_current=np.asarray(data['x']),best_loss=lx,anchor_losses=[objective(z) for z in geom.I_list])
                        out=one_round(geom,pre,objective)
                        maxerr=max(maxerr,near(out['post']['S_current'],ref['endpoint']),near(out['g'],ref['decision']['g']))
                        assert out['accepted']==ref['decision']['accepted'] and len(out['trials'])==len(ref['decision']['trials'])
                        # Common full panel check is separately billed, including excluded scorer.
                        lm=np.array([rawreply(j,ref['endpoint']) for j in range(n)])
                        lf=np.array([rawreply(j,data['full_endpoint']) for j in range(n)])
                        maxerr=max(maxerr,near(float(np.dot(a,lm)-np.dot(a,lf)),ref['score']));oracle_count+=1
                costs.append(dict(key=c['key'],scenario=c['scenario'],phase=phase,role=role,
                    baseline_samples=r['provenance']['query_count'],oracle_new_samples=charged,oracle_requests=requests,
                    estimate_new_samples=r['cost']['standalone_'+role]['samples']))
            notes.append(dict(key=c['key'],rounds=len(r['views']),predictions=len(r['predictions'])))
    save(ROOT/'reference_costs.json',costs)
    save(ROOT/'verification.json',dict(accepted=True,predictions=pred_count,oracle_branches=oracle_count,max_independent_error=maxerr,
        raw_reply_reload_error=rawerr,baseline_reply_samples=limit.samples,new_endpoint_reply_samples=extra_limit.samples,
        baseline_model_seconds=limit.model_seconds,endpoint_model_seconds=extra_limit.model_seconds,
        wall_seconds=time.perf_counter()-tic,cases=notes,pooled_loaded=False,truth_loaded=False))
if __name__=='__main__':setup();verify()
