"""Fixed-start sensitivity; frozen historical solver and joint intervention."""
import copy,shutil,signal,tracemalloc,resource
from datetime import datetime,timezone
from common import *

def prepare():
    ROOT.mkdir(exist_ok=False)
    for name in ['frozen','implementation','pilot','development']:(ROOT/name).mkdir()
    for p in (MOD/'source_snapshot').glob('*.py'):shutil.copy2(p,ROOT/'frozen'/p.name)
    # Copy the previously independently checked interpreter, not the new producer.
    p=CA/'path_intervention_v1/verify.py'
    (ROOT/'frozen/independent_interpreter.py').write_text(p.read_text().split('def verify():')[0])
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,ROOT/'implementation'/f.name)
    shutil.copy2(PLAN,ROOT/PLAN.name)
    source=read(L2/'cases.json');cases=dict(pilot=[source['pilot'][0],next(c for c in source['pilot'] if c['scenario']=='controlled_poison')],development=source['development'])
    assert len(cases['development'])==20
    for cs in cases.values():
        for c in cs:starts(c)
    save(ROOT/'cases.json',cases)
    paths=[PLAN,L2/'cases.json',p,FOUR/'scores_and_costs.json']
    for cs in cases.values():
        for c in cs:paths.extend([Path(c['models']),Path(c['evaluator'])])
    save(ROOT/'inputs.json',{str(p):sha(p) for p in paths})
    save(ROOT/'source_lock.json',{str(p.relative_to(ROOT)):sha(p) for folder in ['frozen','implementation'] for p in (ROOT/folder).glob('*.py')})
    save(ROOT/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),new_confirmation=False,old_data_exploration=True,
        starts=['original','x_plus_025','x_minus_025','y_plus_025','y_minus_025'],repetitions=4,epsilon=EPS,
        solver_iterations=30,decision_models_frozen=True,truth_loaded=False,alpha_updates=False,plan_sha256=sha(PLAN)))

def pack(r,tr):
    steps=[]
    for k,pre in enumerate(tr.pre):
        post=next((p for p in tr.post if p['next_iteration']==pre['next_iteration']+1),tr.final)
        trials=[p for p in tr.trials if p['iteration']==pre['next_iteration']];g=tr.gradients[k]['g']
        reason='zero_direction' if np.linalg.norm(g)<=1e-8 else 'dynamic_no_improvement' if post['halted'] else None
        steps.append(dict(pre=pre,post=post,g=g,trials=trials,stop_reason=reason,accepted=any(v['accepted'] for v in trials)))
    return dict(result=r,final=tr.final,steps=steps)

def evaluate(s,c,x,scheme,budget):
    from engine import run,branch_state
    from host_server import HostServer
    q=Queries(s,budget);ids=list(range(len(s.alphas)));begin=budget.samples;tic=time.perf_counter()
    if scheme=='J':
        base,tr=run(s,q.objective(ids),30,initial=x);trace=pack(base,tr)
    else:
        base=HostServer._solve_subspace_custom(s,s.I_list,q.objective(ids),num_iterations=30,direction_weights=s.alphas,initial_solution=x);trace=None
    baseline_seconds=time.perf_counter()-tic;baseline_samples=budget.samples-begin
    branches=[];values=np.zeros(len(ids));q.stage='decision'
    prelist=tr.pre if scheme=='J' else [None]
    for t,pre in enumerate(prelist):
        for i in ids:
            panel=[j for j in ids if j!=i];b=copy.copy(s);b.I_list=np.asarray(s.I_list)[panel];w=np.asarray(s.alphas)[panel];b.alphas=w/w.sum()
            q.context=dict(t=t,i=i);q.stage='decision';a=len(q.events);f=q.objective(panel)
            if scheme=='J':
                state=branch_state(pre,f,b.I_list);r,bt=run(b,f,1,resume=state);btpack=pack(r,bt);fullx=trace['steps'][t]['post']['S_current']
            else:
                r=HostServer._solve_subspace_custom(b,b.I_list,f,num_iterations=30,direction_weights=b.alphas,initial_solution=x);btpack=None;fullx=base['solution']
            assert all(int(e['key'].split(':')[0])!=i for e in q.events[a:])
            branches.append(dict(t=t,i=i,agent_id=c['ids'][i],result=r,trace=btpack,x_full=fullx,x_minus=r['solution'],decision_locked=True))
    # All decision endpoints exist before supplemental full-panel scores.
    q.stage='common';q.context={}
    for b in branches:
        value=q.full(b['x_minus'])-q.full(b['x_full']);b['score']=value;values[b['i']]+=value
    wall=time.perf_counter()-tic
    return dict(scheme=scheme,start=x,baseline=base,baseline_trace=trace,branches=branches,values=values,
        events=q.events,raw=list(q.raw.values()),samples=budget.samples-begin,baseline_samples=baseline_samples,
        extra_samples=budget.samples-begin-baseline_samples,wall_seconds=wall,baseline_seconds=baseline_seconds,model_seconds=q.model_seconds)

def phase(name):
    from study import load_server
    check();budget=Budget(2500000,600);done=[];timings=[];cases=read(ROOT/'cases.json')[name]
    for ci,c in enumerate(cases):
        s=load_server({k:v for k,v in c.items() if k!='evaluator'})
        for si,x in enumerate(starts(c)):
            print(name,c['key'],si,flush=True);saved={}
            for rep in range(4):
                order=['J','LOO'] if (ci+si+rep)%2==0 else ['LOO','J']
                for method in order:
                    if rep==0:tracemalloc.start()
                    r=evaluate(s,c,x,method,budget)
                    if rep==0:
                        _,peak=tracemalloc.get_traced_memory();tracemalloc.stop();r['python_peak_bytes']=peak
                        save(ROOT/name/(c['key']+f'_s{si}_{method}.json'),r);saved[method]=r
                    else:
                        near(r['values'],saved[method]['values']);assert r['samples']==saved[method]['samples']
                        assert r['events']==saved[method]['events'];assert plain(r['baseline'])==plain(saved[method]['baseline'])
                        timings.append(dict(key=c['key'],scenario=c['scenario'],start=si,method=method,rep=rep,wall_seconds=r['wall_seconds'],model_seconds=r['model_seconds']))
            near(saved['J']['baseline']['solution'],saved['LOO']['baseline']['solution'])
            done.append(dict(key=c['key'],scenario=c['scenario'],seed=c['seed'],start=si,methods={k:{f:v for f,v in r.items() if f in ['values','samples','baseline_samples','extra_samples','python_peak_bytes']} for k,r in saved.items()}))
            if sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())>1_000_000_000:raise RuntimeError('output_cap')
    save(ROOT/(name+'_scores.json'),done);save(ROOT/(name+'_timings.json'),timings)
    save(ROOT/(name+'_locked.json'),dict(hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/name).glob('*.json')},scores_sha256=sha(ROOT/(name+'_scores.json')),pooled_loaded=False))
    save(ROOT/(name+'_completion.json'),dict(complete=True,cases=len(done),physical_samples=budget.samples,wall_seconds=time.perf_counter()-budget.start,process_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));check()

def main():
    command=sys.argv[1]
    if command=='prepare':prepare();return
    setup();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('phase_wall_cap')));signal.alarm(600)
    try:
        if command=='development':assert read(ROOT/'pilot_verification.json')['accepted']
        phase(command)
    except Exception as e:
        save(ROOT/(command+'_failure.json'),dict(error=repr(e)));raise
    finally:signal.alarm(0)
if __name__=='__main__':main()
