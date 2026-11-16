"""Frozen B/A/AV/LOO cost experiment; V reads baseline records only."""
import sys,copy,json,time,hashlib,shutil,signal,resource
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from contribution_analysis.model_qv_v1.run import (CA,W,FROZEN,Replies,load_server,HostServer,run,
    endpoint,construct_directions,decide,plain,analytic_server,new_cases)
import numpy as np
OUT=CA/'anchor_ca_model_v_cost_20261104_attempt_01'
class Meter(Replies):
    total_all=0
    def loss(self,i,x):
        if Meter.total_all>=5000000:raise RuntimeError('physical_sample_cap')
        Replies.total=0
        z=super().loss(i,x)
        Meter.total_all+=not self.events[-1]['hit']
        return z
def save(p,v):p.write_text(json.dumps(plain(v),ensure_ascii=False,indent=2,allow_nan=False))
def digest(events):
    h=hashlib.sha256()
    for e in events:h.update(str(e['model']).encode()+np.asarray(e['x'],np.float64).tobytes()+str(e['hit']).encode())
    return h.hexdigest()
def make_cases():
    old=json.loads((CA/'combined_contribution_layer2_experiment_v1_20261021/cases.json').read_text())['development']
    for c in old:c['phase']='historical'
    known=new_cases()
    for c in known:c['phase']='known_analytic'
    fresh=[];index=0
    base=np.array([[.2,1.,.2,.1,.1],[.25,.8,.3,-.1,.1],[.15,1.2,.1,.1,.05],[.2,.9,.2,0.,.1],[.2,1.1,.15,.05,.1]])
    for n in [3,5]:
        for group in ['general','heterogeneous_bias']:
            for k in range(20):
                seed=2026110500+index;index+=1;rng=np.random.default_rng(seed)
                co=base[:n]+rng.normal(0,.08,(n,5))
                if group=='heterogeneous_bias':co[-1]=[.8,-.15,-1.3,.1,.1]
                fresh.append(dict(key=f'new_n{n}_{group}_{seed}',phase='new_analytic',scenario=group,n=n,seed=seed,
                    coefficients=co.tolist(),candidates=rng.uniform(-.8,.8,(n,2)).tolist(),alphas=[1/n]*n,
                    ids=list(range(n)),fixed_start=rng.uniform(-.5,.5,2).tolist()))
    return old+known,fresh
def execute(c,s,arm,iterations):
    start=time.perf_counter();a=np.asarray(s.alphas);a=a/a.sum();ids=list(range(len(a)))
    q=Meter(s,c['key']);q.stage='baseline'
    if arm=='L':
        base=HostServer._solve_subspace_custom(s,s.I_list,q.objective(ids),num_iterations=iterations,direction_weights=a,initial_solution=np.asarray(c['fixed_start']));tr=None
    else:base,tr=run(s,q.objective(ids),iterations,initial=np.asarray(c['fixed_start']))
    baseline_digest=digest(q.events);baseline_samples=sum(not z['hit'] for z in q.events)
    # Immutable baseline-only values; anchor branch points never enter V.
    observed={(e['model'],np.asarray(e['x'],np.float64).tobytes()):e['loss'] for e in q.events} if arm=='AV' else None
    def read(x):return np.array([observed[(i,np.asarray(x,np.float64).tobytes())] for i in ids])
    rows=[];removed=[];trial_counts=[];vreads=0
    if arm in ['A','AV']:
        for t in range(len(tr.pre)):
            f=endpoint(tr,t);p=f['pre'];x=np.asarray(p['S_current']);q.stage='features'
            cur,ds,H,G=construct_directions(s,q,a,f)
            q.stage='anchor';lf=float(a@q.losses(f['post']['S_current']))
            vv=[None]*len(a)
            if arm=='AV':
                trials=[z for z in tr.trials if z['iteration']==p['next_iteration']]
                current=read(x);final=read(f['post']['S_current']);vreads+=2*len(a)
                tl=[read(z['x']) for z in trials];vreads+=len(tl)*len(a)
                for i in ids:
                    panel=[j for j in ids if j!=i];w=a[panel]/a[panel].sum();chosen=current
                    for val in tl:
                        if float(w@val[panel])<float(w@current[panel]):chosen=val;break
                    vv[i]=float(a@(chosen-final))
            for i in ids:
                q.stage='anchor'
                alt=decide(s,q.objective(ids),x,np.asarray(f['g'])-H[i],p['best_loss'],p['eta'],p['current_method'])
                rows.append(dict(t=t,i=i,CA=float(alt['loss']-lf),V=vv[i],trials=len(alt['trials']),stop=alt['stop']))
            trial_counts.append(sum(z['iteration']==p['next_iteration'] for z in tr.trials))
    if arm=='L':
        for i in ids:
            panel=[j for j in ids if j!=i];b=copy.copy(s);b.I_list=np.asarray(s.I_list)[panel];b.alphas=a[panel]/a[panel].sum()
            q.stage='LOO'
            r=HostServer._solve_subspace_custom(b,b.I_list,q.objective(panel),num_iterations=iterations,direction_weights=b.alphas,initial_solution=np.asarray(c['fixed_start']))
            q.stage='common';score=float(a@(q.losses(r['solution'])-q.losses(base['solution'])))
            removed.append(dict(i=i,result=r,score=score))
    result=dict(key=c['key'],arm=arm,iterations=iterations,base=base,rows=rows,removed=removed,
        samples=sum(not e['hit'] for e in q.events),requests=len(q.events),hits=sum(e['hit'] for e in q.events),
        baseline_samples=baseline_samples,baseline_digest=baseline_digest,query_digest=digest(q.events),
        stages={stage:dict(samples=sum(not e['hit'] and e['stage']==stage for e in q.events),requests=sum(e['stage']==stage for e in q.events)) for stage in sorted(set(e['stage'] for e in q.events))},
        V_new_calls=0,V_record_values_read=vreads,rounds=None if tr is None else len(tr.pre),baseline_trials=trial_counts,
        model_seconds=sum(q.seconds),process_peak_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    result['scores']={str(i):dict(SA=sum(r['CA'] for r in rows if r['i']==i),SV=sum(r['V'] for r in rows if r['i']==i and r['V'] is not None)) for i in ids} if arm in ['A','AV'] else {}
    result['core_seconds']=time.perf_counter()-start
    return result
def validate(c,rr,iterations):
    ref=plain(rr['B']['base'])
    assert all(plain(r['base'])==ref for r in rr.values()),'observer changed core result'
    assert len(set(r['baseline_digest'] for r in rr.values()))==1,'observer changed baseline queries'
    assert rr['A']['query_digest']==rr['AV']['query_digest'] and rr['A']['samples']==rr['AV']['samples']
    assert [r['CA'] for r in rr['A']['rows']]==[r['CA'] for r in rr['AV']['rows']]
    if iterations==30 and c['phase']!='new_analytic':
        old=CA/'model_qv_experiment_20261104_attempt_02/records'
        prev=json.loads((old/(c['key']+'_AV.json')).read_text())
        assert rr['AV']['samples']==prev['samples']
        assert all(abs(r['CA']-p['CA'])<1e-10 and abs(r['V']-p['V'])<1e-10 for r,p in zip(rr['AV']['rows'],prev['rows']))
        loo=json.loads((old/(c['key']+'_LOO.json')).read_text())
        assert rr['L']['samples']==loo['samples']
        assert plain(rr['L']['removed'])==loo['removals']
def main():
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    OUT.mkdir(exist_ok=False);(OUT/'records').mkdir();(OUT/'source_snapshot').mkdir()
    known,fresh=make_cases();jobs=[(c,30) for c in known+fresh]+[(c,it) for it in [10,60] for c in fresh]
    save(OUT/'registration.json',dict(known=known,fresh=fresh,jobs=[dict(key=c['key'],iterations=it) for c,it in jobs],
        repeats=3,warmups=1,max_physical=5000000,wall_cap_seconds=1800,arms=['B','A','AV','L'],
        generator='first n of frozen five base coefficient rows plus N(0,.08); biased last row fixed; disjoint seeds',bootstrap_seed=20261104,bootstrap_repeats=10000))
    paths=[Path(__file__),W/'docs/anchor_ca_model_v_cost_design_20261104.md',CA/'model_qv_v1/run.py',CA/'model_pool_observed_vd_cost_v1/run.py',CA/'method5_cost_v1/run.py',CA/'model_step_decision_v1/engine.py',CA/'three_pool_t2_anchor_v1/core.py',CA/'custom_realized_gain_v1/runtime.py']+list(FROZEN.glob('*.py'))
    paths+=list({Path(c['models']) for c in known if 'models' in c})
    locks={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};save(OUT/'source_lock.json',locks)
    for p in paths:
        if p.suffix in ['.py','.md']:shutil.copy2(p,OUT/'source_snapshot'/'__'.join(p.relative_to(W).parts))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('wall cap')));signal.alarm(1800)
    progress=[];timing=[];started=time.perf_counter()
    try:
        for ci,(c,it) in enumerate(jobs):
            print(ci,c['key'],it,flush=True);loadbeg=time.perf_counter()
            s=load_server(c) if c['phase']=='historical' else analytic_server(c);loadtime=time.perf_counter()-loadbeg
            arms=['B','A','AV','L'];saved={}
            for rep in range(4):
                rotate=(ci+rep)%4
                for arm in arms[rotate:]+arms[:rotate]:
                    r=execute(c,s,arm,it)
                    if rep==0:saved[arm]=r;save(OUT/'records'/f"{c['key']}_{it}_{arm}.json",r)
                    else:
                        assert r['samples']==saved[arm]['samples'] and r['query_digest']==saved[arm]['query_digest']
                        assert plain(r['rows'])==plain(saved[arm]['rows']) and plain(r['removed'])==plain(saved[arm]['removed'])
                        assert plain(r['base'])==plain(saved[arm]['base'])
                        before=time.perf_counter();save(OUT/'temporary_scores.json',dict(base=r['base'],scores=r['scores'],removed=r['removed']))
                        timing.append(dict(key=c['key'],iterations=it,arm=arm,repeat=rep,core_seconds=r['core_seconds'],model_seconds=r['model_seconds'],output_seconds=time.perf_counter()-before,load_seconds=loadtime,process_peak_memory_bytes=r['process_peak_memory_bytes']))
            validate(c,saved,it)
            progress.append(dict(key=c['key'],phase=c['phase'],scenario=c['scenario'],n=len(c['ids']),iterations=it,cost={a:r['samples'] for a,r in saved.items()},rounds=saved['AV']['rounds'],anchor_trials=sum(r['trials'] for r in saved['AV']['rows'])))
            save(OUT/'progress.json',progress);save(OUT/'timing.json',timing)
            if ci==1:save(OUT/'development_gate.json',dict(passed=True,teams=2))
        for p,h in locks.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
        save(OUT/'completion.json',dict(complete=True,jobs=len(jobs),unique_teams=140,physical_samples_all_repeats=Meter.total_all,wall_seconds=time.perf_counter()-started,all_hard_gates_passed=True))
    except Exception as e:save(OUT/'failure.json',dict(error=repr(e),completed_jobs=len(progress),physical_samples=Meter.total_all));raise
    finally:signal.alarm(0)
if __name__=='__main__':main()
