"""Paired modular pool combinations with common baseline and exact shared cache."""
import sys,json,copy,time,hashlib,shutil,signal
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from contribution_analysis.anchor_ca_model_v_cost_v1.run import (CA,W,FROZEN,Replies,load_server,HostServer,run,endpoint,construct_directions,decide,plain,analytic_server,make_cases,digest)
from contribution_analysis.model_step_decision_v1.engine import branch_state
import numpy as np
OUT=CA/'model_pool_combinations_20261105_attempt_01'
ARMS=['B','VU','CMN','VN','A_VU','A_CMN','A_VN','J','LOO']
def save(p,v):p.write_text(json.dumps(plain(v),ensure_ascii=False,indent=2,allow_nan=False))
class Meter(Replies):
    physical=0
    def loss(self,i,x):
        if Meter.physical>=12000000:raise RuntimeError('sample cap')
        Replies.total=0;v=super().loss(i,x);Meter.physical+=not self.events[-1]['hit'];return v
def execute(c,s,arm):
    began=time.perf_counter();a=np.asarray(s.alphas);a=a/a.sum();ids=list(range(len(a)))
    q=Meter(s,c['key']);q.stage='baseline'
    if arm=='LOO':
        base=HostServer._solve_subspace_custom(s,s.I_list,q.objective(ids),num_iterations=30,direction_weights=a,initial_solution=np.asarray(c['fixed_start']));tr=None
    else:base,tr=run(s,q.objective(ids),30,initial=np.asarray(c['fixed_start']))
    bd=digest(q.events);obs={(e['model'],np.asarray(e['x'],np.float64).tobytes()):e['loss'] for e in q.events}
    def read(x):return np.array([obs[(i,np.asarray(x,np.float64).tobytes())] for i in ids])
    pool=arm.removeprefix('A_');haspool=pool in ['VU','CMN','VN'];hasmodel=pool in ['CMN','VN'];rows=[];rem=[]
    if haspool:
        q.stage='scale'
        probes=np.array([q.losses(np.array([x,y])) for x in [-1,-.5,0,.5,1] for y in [-1,-.5,0,.5,1]])
        scale=np.quantile(probes,.9,axis=0)-np.quantile(probes,.1,axis=0)
        if (scale<=1e-12).any():raise ValueError('undefined scale')
    if haspool or arm=='J':
        for t in range(len(tr.pre)):
            f=endpoint(tr,t);p=f['pre'];x=np.asarray(p['S_current']);cur=read(x);lf=read(f['post']['S_current']);full_loss=float(a@lf)
            vv=np.zeros(len(a));U=np.zeros(len(a));H=None
            if haspool:
                q.stage='features';_,ds,H,G=construct_directions(s,q,a,f)
                if not ds:raise ValueError('undefined directions')
                sl=np.array([(np.asarray(d['losses'])-cur)/(scale*d['h']) for d in ds]).T
                dist=np.sqrt(np.mean((sl[:,None,:]-sl[None,:,:])**2,axis=2));np.fill_diagonal(dist,np.inf);U=dist.min(1)
                trials=[read(z['x']) for z in tr.trials if z['iteration']==p['next_iteration']]
                for i in ids:
                    panel=[j for j in ids if j!=i];w=a[panel]/a[panel].sum();selected=cur
                    for val in trials:
                        if float(w@val[panel])<float(w@cur[panel]):selected=val;break
                    vv[i]=float(a@(selected-lf))
            for i in ids:
                row=dict(t=t,i=i,V=float(vv[i]),U=float(U[i]))
                if arm.startswith('A_'):
                    q.stage='anchor';alt=decide(s,q.objective(ids),x,np.asarray(f['g'])-H[i],p['best_loss'],p['eta'],p['current_method'])
                    row['CA']=float(alt['loss']-full_loss)
                if hasmodel or arm=='J':
                    panel=[j for j in ids if j!=i];b=copy.copy(s)
                    if arm=='J':b.I_list=np.asarray(s.I_list)[panel];b.alphas=a[panel]/a[panel].sum()
                    q.stage='joint' if arm=='J' else 'model'
                    objective=q.objective(panel);state=branch_state(p,objective,b.I_list)
                    _,bt=run(b,objective,1,resume=state);xm=bt.final['S_current']
                    q.stage='common';lm=q.losses(xm);C=float(a@(lm-lf))
                    if arm=='J':row['J']=C
                    else:
                        I=float(np.linalg.norm(np.asarray(f['post']['S_current'])-xm)/p['eta'])
                        row.update(CM=C,I=I,N=float(U[i]/(10+U[i])*I/(1+I)))
                rows.append(row)
    if arm=='LOO':
        for i in ids:
            panel=[j for j in ids if j!=i];b=copy.copy(s);b.I_list=np.asarray(s.I_list)[panel];b.alphas=a[panel]/a[panel].sum()
            q.stage='LOO';r=HostServer._solve_subspace_custom(b,b.I_list,q.objective(panel),num_iterations=30,direction_weights=b.alphas,initial_solution=np.asarray(c['fixed_start']))
            q.stage='common';C=float(a@(q.losses(r['solution'])-q.losses(base['solution'])));rem.append(dict(i=i,result=r,score=C))
    out=dict(key=c['key'],arm=arm,base=base,rows=rows,removed=rem,samples=sum(not e['hit'] for e in q.events),requests=len(q.events),baseline_digest=bd,
        query_digest=digest(q.events),stages={st:sum(not e['hit'] and e['stage']==st for e in q.events) for st in sorted(set(e['stage'] for e in q.events))},V_new_calls=0,
        core_seconds=time.perf_counter()-began,model_seconds=sum(q.seconds))
    return out
def allocation(rows,n,pool,beta):
    perf='CM' if pool=='CMN' else 'V';info='U' if pool=='VU' else 'N'
    S=np.array([sum(r[perf] for r in rows if r['i']==i) for i in range(n)])
    R=np.array([sum(r[info] for r in rows if r['i']==i) for i in range(n)])
    C=np.array([sum(r.get('CM',0) for r in rows if r['i']==i) for i in range(n)])
    w=np.where(S>1e-10,S,0.);v=np.where((S>1e-10)&(R>1e-10),R,0.)
    pp=(1-beta)*w/w.sum() if w.sum() else np.zeros(n)
    ip=beta*v/v.sum() if v.sum() else np.zeros(n);p=pp+ip
    assert p.min()>=0 and p.sum()<=1+1e-12
    return dict(S=S,R=R,CM=C,shares=p,info_shares=ip,reserve=max(0.,float(1-p.sum())))
def validate(c,rr):
    assert all(plain(r['base'])==plain(rr['B']['base']) for r in rr.values())
    assert len({r['baseline_digest'] for r in rr.values()})==1
    for pre in ['','A_']:
        assert rr[pre+'CMN']['query_digest']==rr[pre+'VN']['query_digest']
        assert plain(rr[pre+'CMN']['rows'])==plain(rr[pre+'VN']['rows'])
    if c['phase']=='historical':
        ref=json.loads((CA/'method5_cost_experiment_20261030_attempt_04/records'/f"{c['key']}_New.json").read_text())
        for row,z in zip(rr['A_CMN']['rows'],ref['rows']):
            for a,b in [('CA','C_A'),('CM','C_M'),('U','U'),('N','N')]:assert abs(row[a]-z[b])<1e-10,(a,row,z)
    refroot=CA/'anchor_ca_model_v_cost_20261104_attempt_01/records'
    old=json.loads((refroot/f"{c['key']}_30_L.json").read_text())
    assert rr['LOO']['samples']==old['samples'] and plain(rr['LOO']['removed'])==old['removed']
def main():
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    OUT.mkdir(exist_ok=False);(OUT/'records').mkdir();(OUT/'source_snapshot').mkdir()
    old,fresh=make_cases();cases=[c for c in old if c['phase']=='historical']+fresh
    for c in cases:
        if c['phase']=='new_analytic':c['phase']='known_analytic80'
    save(OUT/'registration.json',dict(cases=cases,arms=ARMS,iterations=30,repeats=3,warmups=1,beta=[.1,.2,.3],tau=1e-10,max_physical=12000000,max_seconds=1200))
    paths=[Path(__file__),W/'docs/model_pool_three_combinations_design_20261105.md',CA/'anchor_ca_model_v_cost_v1/run.py',CA/'model_qv_v1/run.py',CA/'model_pool_observed_vd_cost_v1/run.py',CA/'method5_cost_v1/run.py',CA/'model_step_decision_v1/engine.py',CA/'three_pool_t2_anchor_v1/core.py']+list(FROZEN.glob('*.py'))
    locks={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};save(OUT/'source_lock.json',locks)
    for p in paths:shutil.copy2(p,OUT/'source_snapshot'/'__'.join(p.relative_to(W).parts))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('wall cap')));signal.alarm(1200)
    progress=[];timing=[];start=time.perf_counter()
    try:
        for ci,c in enumerate(cases):
            print(ci,c['key'],flush=True);s=load_server(c) if c['phase']=='historical' else analytic_server(c);saved={}
            for rep in range(4):
                k=(ci+rep)%len(ARMS)
                for arm in ARMS[k:]+ARMS[:k]:
                    r=execute(c,s,arm)
                    if rep==0:saved[arm]=r;save(OUT/'records'/f"{c['key']}_{arm}.json",r)
                    else:
                        assert plain(r['rows'])==plain(saved[arm]['rows']) and r['query_digest']==saved[arm]['query_digest']
                        assert plain(r['base'])==plain(saved[arm]['base']) and r['samples']==saved[arm]['samples']
                        timing.append(dict(key=c['key'],arm=arm,repeat=rep,core_seconds=r['core_seconds'],model_seconds=r['model_seconds']))
            validate(c,saved)
            alloc={pool:{str(b):allocation(saved[pool]['rows'],len(c['ids']),pool,b) for b in [.1,.2,.3]} for pool in ['VU','CMN','VN']}
            progress.append(dict(key=c['key'],phase=c['phase'],scenario=c['scenario'],n=len(c['ids']),cost={a:r['samples'] for a,r in saved.items()},allocations=alloc))
            save(OUT/'progress.json',progress);save(OUT/'timing.json',timing)
        for p,h in locks.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
        save(OUT/'completion.json',dict(complete=True,teams=len(cases),physical_all_repeats=Meter.physical,wall_seconds=time.perf_counter()-start))
    except Exception as e:save(OUT/'failure.json',dict(error=repr(e),teams=len(progress),physical=Meter.physical));raise
    finally:signal.alarm(0)
if __name__=='__main__':main()
