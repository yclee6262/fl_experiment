"""Independent saved-reply, cache, geometry and payout audit. No model execution."""
import json, math, hashlib, sys, time
from pathlib import Path
from fractions import Fraction as F
import numpy as np

W=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def near(a,b,tol=1e-10):
    e=float(np.max(np.abs(np.asarray(a,float)-np.asarray(b,float))))
    assert e<=tol,(e,a,b)
    return e
def payout(weights,budget):
    total=sum(weights.values(),F(0))
    if not total:return {o:0 for o in weights},budget
    qs={o:budget*w/total for o,w in weights.items()}
    pay={o:int(v) for o,v in qs.items()}
    for o in sorted(qs,key=lambda o:(-(qs[o]-pay[o]),o))[:budget-sum(pay.values())]:pay[o]+=1
    return pay,0
def main(root,out):
    root=Path(root);out=Path(out);out.mkdir(exist_ok=False);started=time.perf_counter()
    reg=read(root/'registration.json');case_map={c['key']:c for c in reg['cases']+reg['development']}
    totals={'records':0,'logical_requests':0,'physical_samples_one_pass':0,'owner_rounds':0,'payout_pools':0,'historical_loo':0,'historical_baselines':0,'historical_model_branches':0}
    details=[];maxerr=0.
    for p in sorted((root/'records').glob('*.json')):
        if p.name.endswith('.ledger.json'):continue
        d=read(p);c=case_map[d['key']];n=len(c['ids']);owners=d['owners'];a=np.array(d['weights']);cache={};lossmap={};stage={}
        for e in d['events']:
            key=e['key'];point=np.asarray(e['x'],np.float64)
            assert key.endswith(':'+str(e['model'])+':'+point.tobytes().hex())
            assert bool(e['hit'])==(key in cache),'cache truth mismatch'
            if key in cache:assert cache[key]==e['loss']
            assert math.isfinite(e['loss']);cache[key]=e['loss'];lossmap[e['model'],tuple(e['x'])]=e['loss']
            counts=stage.setdefault(e['stage'],dict(requests=0,samples=0,hits=0));counts['requests']+=1;counts['samples']+=not e['hit'];counts['hits']+=e['hit']
        assert len(cache)==d['samples'];assert len(d['events'])==d['requests'];assert len(d['events'])-len(cache)==d['hits']
        for i in range(n):assert sum(e['model']==i and not e['hit'] for e in d['events'])==d['per_model_samples'][i]
        def losses(x):return [lossmap[i,tuple(x)] for i in range(n)]
        def diff(x,y):
            ws=[F(str(v)) for v in a];z=sum(ws)
            return float(sum(w*(F(str(b))-F(str(v))) for w,b,v in zip(ws,losses(x),losses(y)))/z)
        historical=W/'contribution_analysis/custom_direction_experiment_v2_20261020/baselines'/f"{d['key']}.json"
        if historical.exists():
            old=read(historical)['result']
            for k,v in d['base'].items():assert v==old[k],('baseline',d['key'],k)
            totals['historical_baselines']+=1
        if d['arm']=='LOO':
            for r in d['rows']:
                near(r['full_losses'],losses(r['full']),0);near(r['minus_losses'],losses(r['minus']),0)
                maxerr=max(maxerr,near(r['D'],diff(r['minus'],r['full']),1e-12))
                oldp=W/'contribution_analysis/custom_direction_experiment_v2_20261020/removals'/f"{d['key']}_{c['ids'][r['i']]}.json"
                if oldp.exists():
                    old=read(oldp)['result']
                    for k,v in r['result'].items():assert v==old[k],('loo',d['key'],r['i'],k)
                    totals['historical_loo']+=1
        else:
            T=len(d['full']);assert len(d['rows'])==n*T
            assert {(r['t'],r['i']) for r in d['rows']}=={(t,i)for t in range(T)for i in range(n)}
            grid=[[x,y]for x in [-1,-.5,0,.5,1]for y in [-1,-.5,0,.5,1]]
            probes=np.asarray([losses(x)for x in grid]);near(d['probe'],probes,0)
            scale=np.quantile(probes,.9,axis=0)-np.quantile(probes,.1,axis=0);near(scale,d['scale'],0);assert min(scale)>1e-12
            us={}
            for t,fe in enumerate(d['full']):
                pre=fe['pre'];x=np.array(pre['S_current']);lx=np.array(losses(x));H=np.zeros((n,len(x)));features={}
                for k,z in enumerate(c['candidates']):
                    distance=np.linalg.norm(np.array(z)-x)
                    if distance<1e-8:continue
                    u=(np.array(z)-x)/distance
                    if pre['current_method']=='secant':h=float(distance);point=np.array(z)
                    else:
                        limits=[((-1-x[j])/u[j] if u[j]<0 else (1-x[j])/u[j]) for j in range(len(x)) if abs(u[j])>1e-12]
                        h=min([pre['delta']]+limits);h=max(0.,h);point=x+h*u
                    if h<=1e-12:continue
                    lp=np.array(losses(point));slope=math.fsum(w*(b-v)/h for w,v,b in zip(a,lx,lp));H[k]+=a[k]*slope*u
                    geom=(tuple(point),h,tuple(u));col=(lp-lx)/(scale*h)
                    if geom in features:near(features[geom],col,0)
                    features[geom]=col
                near(H,fe['H']);near(H.sum(0),fe['g']);g=np.array(list(features.values())).T
                us[t]=[min(math.sqrt(sum((g[i,k]-g[j,k])**2 for k in range(g.shape[1]))/g.shape[1])for j in range(n)if i!=j)for i in range(n)]
            for r,b in zip(d['rows'],d['branches']):
                t,i=r['t'],r['i'];assert(b['t'],b['i'])==(t,i)
                for label,pos in [('full','full'),('minusA','minusA'),('minusM','minusM')]:near(r[label+'_losses'],losses(r[pos]),0)
                maxerr=max(maxerr,near(r['C_A'],diff(r['minusA'],r['full']),1e-12),near(r['C_M'],diff(r['minusM'],r['full']),1e-12))
                pre=d['full'][t]['pre'];I=np.linalg.norm(np.array(r['full'])-r['minusM'])/pre['eta'];U=us[t][i];N=U/(10+U)*I/(1+I)
                near(U,r['U']);near(I,r['I']);near(N,r['N']);near(r['full'],d['full'][t]['post']['S_current'],0)
                # Independent anchor trial geometry / strict acceptance / first acceptance.
                gd=np.array(d['full'][t]['g'])-np.array(d['full'][t]['H'][i]);norm=np.linalg.norm(gd);x=np.array(pre['S_current']);end=x.copy();before=float(a@np.array(losses(x)));eta=pre['eta']
                for k,tr in enumerate(b['anchor']['trials']):
                    assert norm>1e-8;direction=-gd/norm
                    limits=[((-1-x[j])/direction[j] if direction[j]<0 else (1-x[j])/direction[j])for j in range(len(x))if abs(direction[j])>1e-12]
                    h=max(0.,min([eta/2**k]+limits));near(tr['x'],x+h*direction)
                    value=float(a@np.array(losses(tr['x'])));near(value,tr['loss']);assert tr['accepted']==(value<before)
                    if tr['accepted']:assert k==len(b['anchor']['trials'])-1;end=np.array(tr['x'])
                near(end,r['minusA']);near(b['model_final']['S_current'],r['minusM'],0)
            model_history=W/'contribution_analysis/combined_contribution_layer2_experiment_v1_20261021/development'/f"{d['key']}.json"
            if model_history.exists():
                groups={(g['t'],g['i']):g for g in read(model_history)['groups']}
                for r in d['rows']:
                    old=groups[r['t'],r['i']]['roles']['model'];near(r['minusM'],old['endpoint'],0);near(r['C_M'],old['score'],1e-12);totals['historical_model_branches']+=1
            tau=max(F('1e-8'),T*F('1e-10'));SA={o:sum((F(str(r['C_A']))for r in d['rows']if r['owner']==o),F(0))for o in owners}
            SM={o:sum((F(str(r['C_M']))for r in d['rows']if r['owner']==o),F(0))for o in owners};RN={o:sum((F(str(r['N']))for r in d['rows']if r['owner']==o),F(0))for o in owners}
            WA={o:SA[o]if SA[o]>max(F('1e-8'),sum((F(str(r['rho']))for r in d['rows']if r['owner']==o),F(0)))else F(0)for o in owners}
            WC={o:SM[o]if SM[o]>tau else F(0)for o in owners};WN={o:RN[o]if SM[o]>tau and RN[o]>tau else F(0)for o in owners}
            for weights,budget,actual in [(WA,500,d['allocation']['anchor']),(WC,400,d['allocation']['model']['arms']['C']),(WN,100,d['allocation']['model']['arms']['N'])]:
                pay,reserve=payout(weights,budget);assert pay==actual.get('payments',actual.get('amounts'));assert reserve==actual.get('reserved',actual.get('reserve'));totals['payout_pools']+=1
            totals['owner_rounds']+=len(d['rows'])
            oldp=W/'contribution_analysis/anchor_pool_end_to_end_experiment_v1_20261028'/d['key']/'base.json'
            if oldp.exists():
                old=read(oldp);oldrows={(b['t'],b['owner']):b for b in old['branches']}
                for r in d['rows']:
                    b=oldrows[r['t'],r['owner']];near(r['C_A'],b['C'],1e-12);near(r['minusA'],b['decision']['x'],1e-12)
                assert d['allocation']['anchor']['payments']==old['allocation']['payments']
        totals['records']+=1;totals['logical_requests']+=d['requests'];totals['physical_samples_one_pass']+=d['samples']
        details.append(dict(key=d['key'],arm=d['arm'],samples=d['samples'],stages=stage))
    lock=read(root/'source_lock.json')
    for p,h in lock.items():
        snapshot=root/'source_snapshot'/('__'.join(Path(p).relative_to(W).parts))
        assert sha(snapshot if snapshot.exists() else p)==h,('source drift',p)
    result=dict(passed=True,totals=totals,max_direct_score_difference=maxerr,details=details,source_hashes_verified=len(lock),new_model_calls=0,new_solver_calls=0,wall_seconds=time.perf_counter()-started)
    (out/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items()if k!='details'},indent=2))
if __name__=='__main__':main(sys.argv[1],sys.argv[2])
