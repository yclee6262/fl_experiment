"""Endpoint diagnostics and fixed-seed confirmation; historical solvers untouched."""
import sys, copy, json, time, hashlib, signal
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from contribution_analysis.anchor_ca_model_v_cost_v1.run import (
    CA, W, FROZEN, Replies, load_server, run, endpoint, construct_directions,
    decide, plain, analytic_server, digest, execute as av_execute)
from contribution_analysis.model_pool_combinations_v1.run import execute as reference_execute
from contribution_analysis.model_step_decision_v1.engine import branch_state
import numpy as np

OUT=CA/'model_pool_v_cm_validation_20261107_attempt_02'
OLD=CA/'model_pool_combinations_20261105_attempt_01'
def save(p,v):p.write_text(json.dumps(plain(v),ensure_ascii=False,indent=2,allow_nan=False))
class Meter(Replies):
    physical=0
    def loss(self,i,x):
        if Meter.physical>=2000000:raise RuntimeError('physical cap')
        Replies.total=0
        v=super().loss(i,x);Meter.physical+=not self.events[-1]['hit'];return v

def split(delta,a,i):
    total=float(a@delta);self=float(a[i]*delta[i]);peer=float(sum(a[j]*delta[j] for j in range(len(a)) if j!=i))
    assert abs(total-self-peer)<1e-10
    return dict(total=total,self=self,peer=peer,peer_normalized=peer/(1-a[i]))

def fixtures():
    # Finite loss tables isolate observer arithmetic from the solver.
    a=np.ones(3)/3;current=np.array([3.,3.,3.])
    def observe(full,trials,i):
        keep=[j for j in range(3) if j!=i];selected=current
        for val in trials:
            if np.mean(val[keep])<np.mean(current[keep]):selected=val;break
        return split(selected-full,a,i)
    same=np.array([2.,2.,2.]);assert observe(same,[same],0)['total']==0
    # Full panel rejects [8,2,2], while the other panel accepts it.
    trial=np.array([8.,2.,2.]);z=observe(current,[trial],0)
    assert z['total']>0 and z['peer']<0 and z['self']>0
    # V alternative=current; CM alternative improves full objective.
    full=np.array([0.,4.,4.]);v=observe(full,[full],0)
    cm=split(np.array([1.,1.,1.])-full,a,0)
    assert v['total']>0 and cm['total']<0
    # Identical endpoint evaluations give identical value, independent of identity.
    d=np.array([.2,.2,.2]);assert split(d,a,0)['total']==split(d,a,1)['total']
    return dict(passed=True,scope='four endpoint-table arithmetic fixtures, not search validation',restraint=z,opposite=dict(V=v,CM=cm))

def execute(c,s,it,anchor=False):
    start=time.perf_counter();a=np.asarray(s.alphas);a=a/a.sum();ids=list(range(len(a)))
    q=Meter(s,c['key']);q.stage='baseline'
    base,tr=run(s,q.objective(ids),it,initial=np.asarray(c['fixed_start']))
    bd=digest(q.events)
    obs={(e['model'],np.asarray(e['x'],np.float64).tobytes()):e['loss'] for e in q.events}
    def read(x):return np.array([obs[(i,np.asarray(x,np.float64).tobytes())] for i in ids])
    rows=[]
    for t in range(len(tr.pre)):
        f=endpoint(tr,t);p=f['pre'];x=np.asarray(p['S_current']);xf=np.asarray(f['post']['S_current']);cur=read(x);lf=read(xf)
        trials=[z for z in tr.trials if z['iteration']==p['next_iteration']]
        q.stage='features';before=len(q.events);_,_,H,G=construct_directions(s,q,a,f)
        assert all(e['hit'] for e in q.events[before:])
        for i in ids:
            panel=[j for j in ids if j!=i];w=a[panel]/a[panel].sum();lv=cur;xv=x;choice=None
            for k,z in enumerate(trials):
                val=read(z['x'])
                if w@val[panel]<w@cur[panel]:lv=val;xv=np.asarray(z['x']);choice=k;break
            q.stage='model';objective=q.objective(panel);state=branch_state(p,objective,s.I_list)
            _,bt=run(s,objective,1,resume=state);xm=np.asarray(bt.final['S_current'])
            q.stage='common';lm=q.losses(xm)
            g=np.asarray(f['g']);gm=g-G[i];angle=None
            if np.linalg.norm(g)>1e-10 and np.linalg.norm(gm)>1e-10:
                angle=float(np.arccos(np.clip((g@gm)/np.linalg.norm(g)/np.linalg.norm(gm),-1,1))*180/np.pi)
            row=dict(t=t,i=i,x=x,full=xf,minusV=xv,minusM=xm,full_losses=lf,minusV_losses=lv,minusM_losses=lm,
                V=split(lv-lf,a,i),CM=split(lm-lf,a,i),angle_degrees=angle,eta=p['eta'],mode=p['current_method'],
                full_moved=bool(np.linalg.norm(xf-x)>1e-10),V_choice=choice,CM_moved=bool(np.linalg.norm(xm-x)>1e-10),
                CM_V_endpoint_distance=float(np.linalg.norm(xm-xv)),original_gain=float(a@(cur-lf)))
            if anchor:
                q.stage='anchor';alt=decide(s,q.objective(ids),x,g-H[i],p['best_loss'],p['eta'],p['current_method'])
                row['CA']=float(alt['loss']-a@lf)
            rows.append(row)
    stages={st:sum(not e['hit'] and e['stage']==st for e in q.events) for st in sorted({e['stage'] for e in q.events})}
    return dict(key=c['key'],phase=c['phase'],n=len(a),iterations=it,base=base,rows=rows,weights=a,
        samples=sum(stages.values()),stages=stages,baseline_digest=bd,baseline_only_V_new_calls=0,seconds=time.perf_counter()-start)

def main():
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    OUT.mkdir(exist_ok=False);(OUT/'records').mkdir();(OUT/'source_snapshot').mkdir()
    known=json.loads((OLD/'registration.json').read_text())['cases'];fresh=[]
    base=np.array([[.2,1.,.2,.1,.1],[.25,.8,.3,-.1,.1],[.15,1.2,.1,.1,.05],[.2,.9,.2,0.,.1],[.2,1.1,.15,.05,.1]])
    for n in [3,5]:
        for sc in ['general','heterogeneous_bias']:
            for k in range(6):
                seed=2026110700+len(fresh);rng=np.random.default_rng(seed);co=base[:n]+rng.normal(0,.08,(n,5))
                if sc=='heterogeneous_bias':co[-1]=[.8,-.15,-1.3,.1,.1]
                fresh.append(dict(key=f'confirm_n{n}_{sc}_{seed}',phase='new_confirmation',scenario=sc,n=n,seed=seed,
                    coefficients=co.tolist(),candidates=rng.uniform(-.8,.8,(n,2)).tolist(),alphas=[1/n]*n,ids=list(range(n)),fixed_start=rng.uniform(-.5,.5,2).tolist(),sensitivity=k<2))
    jobs=[(c,30) for c in known+fresh]+[(c,it) for c in fresh if c['sensitivity'] for it in [10,60]]
    save(OUT/'registration.json',dict(known=known,fresh=fresh,jobs=[dict(key=c['key'],iterations=it) for c,it in jobs],tau=[1e-12,1e-10,1e-8],max_physical=2000000,max_seconds=900))
    paths=[Path(__file__),Path(__file__).with_name('analyze.py'),W/'docs/model_pool_v_cm_validation_design_20261107.md',CA/'model_pool_combinations_v1/run.py',CA/'anchor_ca_model_v_cost_v1/run.py',CA/'model_qv_v1/run.py',CA/'model_pool_observed_vd_cost_v1/run.py',CA/'method5_cost_v1/run.py',CA/'model_step_decision_v1/engine.py',CA/'three_pool_t2_anchor_v1/core.py',CA/'custom_realized_gain_v1/runtime.py']+list(FROZEN.glob('*.py'))
    lock={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};save(OUT/'source_lock.json',lock)
    for p in paths: (OUT/'source_snapshot'/'__'.join(p.relative_to(W).parts)).write_bytes(p.read_bytes())
    save(OUT/'fixtures.json',fixtures());start=time.perf_counter();progress=[]
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('wall cap')));signal.alarm(900)
    try:
        for c,it in jobs:
            print(c['key'],it,flush=True);s=load_server(c) if c['phase']=='historical' else analytic_server(c)
            r=execute(c,s,it);save(OUT/'records'/f"{c['key']}_{it}.json",r)
            if c in known:
                ref=json.loads((OLD/'records'/f"{c['key']}_CMN.json").read_text())
                assert r['baseline_digest']==ref['baseline_digest'] and plain(r['base'])==ref['base'] and len(r['rows'])==len(ref['rows'])
                for z,o in zip(r['rows'],ref['rows']):
                    assert z['t']==o['t'] and z['i']==o['i']
                    for m in ['V','CM']:assert abs(z[m]['total']-o[m])<=1e-10
            if c in fresh and it==30:
                # All query implementations use this capped meter; do not mutate historical source.
                av_execute.__globals__['Meter']=Meter;reference_execute.__globals__['Meter']=Meter
                av=av_execute(c,s,'AV',30);cm=execute(c,s,30,True);j=reference_execute(c,s,'J');loo=reference_execute(c,s,'LOO')
                assert all(plain(z['base'])==plain(r['base']) for z in [av,cm,j,loo])
                for z,v in zip(cm['rows'],av['rows']):assert abs(z['CA']-v['CA'])<1e-10 and abs(z['V']['total']-v['V'])<1e-10
                save(OUT/'records'/f"{c['key']}_cost.json",dict(AV=av,ACM=cm,J=j,LOO=loo))
            progress.append(dict(key=c['key'],iterations=it));save(OUT/'progress.json',progress)
        for p,h in lock.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
        save(OUT/'completion.json',dict(complete=True,jobs=len(progress),physical=Meter.physical,seconds=time.perf_counter()-start))
    except Exception as e:save(OUT/'failure.json',dict(error=repr(e),jobs=len(progress),physical=Meter.physical));raise
    finally:signal.alarm(0)
if __name__=='__main__':main()
