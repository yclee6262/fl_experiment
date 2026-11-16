"""Q/V fixed-rule experiment: historical regression plus independent analytic cases."""
import sys, json, copy, time, hashlib, shutil
from pathlib import Path
W=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(W))
from contribution_analysis.model_pool_observed_vd_cost_v1.run import (
    CA,FROZEN,Replies,load_server,HostServer,run,endpoint,construct_directions,observe_vd,decide,plain)
import numpy as np
OUT=CA/'model_qv_experiment_20261104_attempt_02'
TOL=1e-10

def save(p,x):
    p.write_text(json.dumps(plain(x),indent=2,ensure_ascii=False,allow_nan=False))

class AnalyticAgent:
    def __init__(self,coef): self.coef=np.asarray(coef)
    def api_predict(self,x):
        x=np.asarray(x); b=self.coef
        return np.array([b[0]+b[1]*x[0]+b[2]*x[1]+b[3]*x[0]*x[1]+b[4]*x@x])

def analytic_server(c):
    s=HostServer.__new__(HostServer)
    s.target_T=0.;s.n_features=2;s.constraint_mode='box_constrained'
    s.feasible_lower=-1.;s.feasible_upper=1.;s.feasibility_tolerance=1e-8
    s.I_list=np.asarray(c['candidates']);s.alphas=np.asarray(c['alphas'])
    s.trusted_agents=[AnalyticAgent(z) for z in c['coefficients']]
    return s

def new_cases():
    cases=[]
    for group in ['general','duplicate','distinct_aligned','distinct_biased']:
        for k in range(10):
            seed=2026110400+k; rng=np.random.default_rng(seed)
            co=np.array([[.2,1.,.2,.1,.1],[.25,.8,.3,-.1,.1],[.15,1.2,.1,.1,.05],[.2,.9,.2,0.,.1]])
            co+=rng.normal(0,.08,co.shape)
            if group=='duplicate': co[1]=co[0].copy()
            if group=='distinct_aligned': co[3]=[.1,.15,1.3,.1,.1]
            if group=='distinct_biased': co[3]=[.8,-.15,-1.3,.1,.1]
            cases.append(dict(key=f'{group}_{seed}',scenario=group,phase='independent_analytic',seed=seed,
                coefficients=co.tolist(),candidates=rng.uniform(-.8,.8,(4,2)).tolist(),alphas=[.25]*4,
                ids=list(range(4)),fixed_start=rng.uniform(-.5,.5,2).tolist()))
    return cases

def execute(c,s,arm):
    start=time.perf_counter(); a=np.asarray(s.alphas);a=a/a.sum();ids=list(range(len(a)))
    q=Replies(s,c['key']); q.stage='baseline'
    base,tr=run(s,q.objective(ids),30,initial=np.asarray(c['fixed_start']))
    baseline_samples=sum(not z['hit'] for z in q.events)
    rows=[]; scales=None;removals=[]
    if arm=='QV':
        q.stage='scale25'
        probes=np.array([q.losses(np.array([x,y])) for x in [-1,-.5,0,.5,1] for y in [-1,-.5,0,.5,1]])
        scales=np.quantile(probes,.9,axis=0)-np.quantile(probes,.1,axis=0)
        if (scales<=1e-12).any(): raise ValueError('undefined_scale')
    if arm in ['AV','QV']:
        for t in range(len(tr.pre)):
            f=endpoint(tr,t);p=f['pre'];x=np.asarray(p['S_current']);q.stage='features'
            cur,ds,H,G=construct_directions(s,q,a,f)
            q.stage='V';before=len(q.events)
            vv=observe_vd(q,tr,f,a,cur,G)
            assert all(z['hit'] for z in q.events[before:]),'V cache miss'
            lf=float(a@q.losses(f['post']['S_current']))
            U=None
            if arm=='QV':
                if not ds:raise ValueError('undefined_directions')
                slopes=np.array([(np.asarray(d['losses'])-cur)/(scales*d['h']) for d in ds]).T
                dist=np.sqrt(np.mean((slopes[:,None,:]-slopes[None,:,:])**2,axis=2))
                np.fill_diagonal(dist,np.inf);U=dist.min(1)
            for i in ids:
                q.stage='anchor'
                alt=decide(s,q.objective(ids),x,np.asarray(f['g'])-H[i],p['best_loss'],p['eta'],p['current_method'])
                ca=float(alt['loss']-lf)
                row=dict(t=t,i=i,V=vv[i]['V'],CA=ca)
                if arm=='QV':
                    q.stage='direction'
                    alt=decide(s,q.objective(ids),x,np.asarray(f['g'])-G[i],p['best_loss'],p['eta'],p['current_method'])
                    cd=float(alt['loss']-lf);row.update(U=float(U[i]),Cdir=cd,Q=float(U[i]/(10+U[i])*cd))
                rows.append(row)
    if arm=='LOO':
        for i in ids:
            panel=[j for j in ids if j!=i]; b=copy.copy(s)
            b.I_list=np.asarray(s.I_list)[panel];b.alphas=a[panel]/a[panel].sum()
            q.stage='LOO'
            r=HostServer._solve_subspace_custom(b,b.I_list,q.objective(panel),num_iterations=30,direction_weights=b.alphas,initial_solution=np.asarray(c['fixed_start']))
            q.stage='common';score=float(a@(q.losses(r['solution'])-q.losses(base['solution'])))
            removals.append(dict(i=i,result=r,score=score))
    stages={z:dict(requests=sum(e['stage']==z for e in q.events),samples=sum(e['stage']==z and not e['hit'] for e in q.events)) for z in sorted(set(e['stage'] for e in q.events))}
    return dict(arm=arm,key=c['key'],scenario=c['scenario'],base=base,rows=rows,removals=removals,
        samples=sum(not e['hit'] for e in q.events),baseline_samples=baseline_samples,stages=stages,
        core_seconds=time.perf_counter()-start,model_seconds=sum(q.seconds),scales=scales)

def allocation(rows,n,lam):
    Q=np.array([sum(r.get('Q',0.) for r in rows if r['i']==i) for i in range(n)])
    V=np.array([sum(r['V'] for r in rows if r['i']==i) for i in range(n)])
    C=np.array([sum(r.get('Cdir',0.) for r in rows if r['i']==i) for i in range(n)])
    wq=np.where(Q>TOL,Q,0);wv=np.where(V>TOL,V,0)
    p=(lam*wq/wq.sum() if wq.sum() else np.zeros(n))+((1-lam)*wv/wv.sum() if wv.sum() else np.zeros(n))
    assert np.min(p)>=0 and p.sum()<=1+1e-12
    return dict(Q=Q,V=V,Cdir=C,shares=p,reserved=float(1-p.sum()),positive_Q_negative_C=int(sum((Q>TOL)&(C< -TOL))))

def main():
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    OUT.mkdir(exist_ok=False);(OUT/'records').mkdir();(OUT/'source_snapshot').mkdir()
    hist=json.loads((CA/'combined_contribution_layer2_experiment_v1_20261021/cases.json').read_text())['development']
    for c in hist:c['phase']='historical'
    cases=hist+new_cases();save(OUT/'registration.json',dict(cases=cases,repeats=3,warmups=1,iterations=30,
        Q='sum U/(10+U)*Cdir',allocation_weights=[.25,.5,.75],tolerance=TOL,
        analytic_scope='independent synthetic mechanism and cost only; aligned/biased labels describe coefficients, not proven benefit',
        cost_gate=dict(reduction=.1,team_fraction=.8)))
    sources=[Path(__file__),CA/'model_pool_observed_vd_cost_v1/run.py',CA/'model_step_decision_v1/engine.py',CA/'three_pool_t2_anchor_v1/core.py']+list(FROZEN.glob('*.py'))
    locks={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    save(OUT/'source_lock.json',locks)
    for p in sources:shutil.copy2(p,OUT/'source_snapshot'/'__'.join(p.relative_to(W).parts))
    results=[]; timings=[]; total_physical=0
    for ci,c in enumerate(cases):
        total_physical+=Replies.total;Replies.total=0
        assert total_physical<3000000,'global physical sample cap'
        print(c['key'],flush=True)
        s=load_server(c) if c['phase']=='historical' else analytic_server(c)
        arms=['base','AV','QV','LOO'];saved={}
        for rep in range(4):
            order=arms[(ci+rep)%4:]+arms[:(ci+rep)%4]
            for arm in order:
                r=execute(c,s,arm)
                if rep==0:saved[arm]=r;save(OUT/'records'/f"{c['key']}_{arm}.json",r)
                else:
                    assert plain(r['base'])==plain(saved[arm]['base'])
                    assert plain(r['rows'])==plain(saved[arm]['rows']) and r['samples']==saved[arm]['samples']
                    timings.append(dict(key=c['key'],arm=arm,repeat=rep,core_seconds=r['core_seconds'],model_seconds=r['model_seconds']))
        assert all(plain(saved[a]['base'])==plain(saved['base']['base']) for a in arms)
        assert [r['CA'] for r in saved['AV']['rows']]==[r['CA'] for r in saved['QV']['rows']]
        if c['phase']=='historical':
            ref=json.loads((CA/'model_pool_observed_vd_cost_20261103_attempt_07/records'/f"{c['key']}_observed.json").read_text())
            assert saved['AV']['samples']==ref['samples']
            assert all(abs(x['CA']-y['C_A'])<TOL for x,y in zip(saved['AV']['rows'],ref['rows']))
        if c['scenario']=='duplicate':
            assert all(abs(r['U'])<TOL for r in saved['QV']['rows'] if r['i'] in [0,1])
        alloc={str(l):allocation(saved['QV']['rows'],len(c['ids']),l) for l in [.25,.5,.75]}
        results.append(dict(key=c['key'],phase=c['phase'],scenario=c['scenario'],cost={a:saved[a]['samples'] for a in arms},
            allocations=alloc,rounds=len(saved['QV']['rows'])//len(c['ids'])))
        save(OUT/'progress.json',results)
    for p,h in locks.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    save(OUT/'timing.json',timings);save(OUT/'completion.json',dict(complete=True,teams=len(cases),historical=20,independent_analytic=40,physical_samples_all_repeats=total_physical+Replies.total,new_model_training=0,pooled=0,truth=0,payment=0))

if __name__=='__main__':main()
