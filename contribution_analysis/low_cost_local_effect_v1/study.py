"""Baseline-only views and predictions. Oracle labels are not loaded here."""
import argparse,copy,shutil,time,signal
from pathlib import Path
import numpy as np
from common import *
from core import View,n0,n1,key,ROLES

def prepare():
    ROOT.mkdir(exist_ok=False);snap=ROOT/'source_snapshot';snap.mkdir()
    origin=PREV/'source_snapshot'
    for p in origin.glob('*.py'):
        assert sha(p)==read(PREV/'source_lock.json')[p.name]
        if p.name not in ['study.py','analyze.py','verify.py','test_engine.py']:shutil.copy2(p,snap/p.name)
    shutil.copy2(origin/'study.py',snap/'support.py')
    shutil.copy2(ORACLE/'source_snapshot/independent_step.py',snap/'independent_step.py')
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,snap/p.name)
    cases=read(PREV/'cases.json');inputs=[DESIGN,PREV/'source_lock.json',ORACLE/'completion_receipt.json',ORACLE/'verification.json',ORACLE/'scores_and_points_locked.json']
    # Hash oracle files without parsing their contents into the estimator.
    for phase,rows in cases.items():
        (ROOT/phase).mkdir()
        for c in rows:
            inputs.extend([PREV/phase/(c['key']+'.json'),ORACLE/phase/(c['key']+'.json'),Path(c['models'])])
    for n in read(ORACLE/'completion_receipt.json')['hashes']:
        p=Path(n) if Path(n).is_absolute() else WORK/n
        assert sha(p)==read(ORACLE/'completion_receipt.json')['hashes'][n]
    save(ROOT/'inputs.json',{str(p):sha(p) for p in inputs})
    save(ROOT/'cases.json',cases);shutil.copy2(DESIGN,ROOT/DESIGN.name)
    save(ROOT/'source_lock.json',{p.name:sha(p) for p in snap.glob('*.py')})
    save(ROOT/'registration.json',dict(authorization='根據實驗規劃文件去實作實驗以及整理結論',
        new_confirmation=False,combinations=['N0_model','N1_anchor','N1_model','N1_joint'],
        deadzone=1e-4,bootstrap_seed=20261021,bootstrap_repeats=10000,interval_percentiles=[.1,99.9],
        pooled_loaded=False,truth_loaded=False,technical_tests=read(Path(__file__).parent/'technical_results.json'),
        limits=dict(pilot=20000,development=40000,baseline_verify=100000,endpoint_verify=60000,timing=400000,total=620000)))

def baseline_view(c,old):
    from engine import run,Trace
    import engine
    from support import load_server
    from host_server import HostServer
    # No models needed to replay known replies; load_server is intentionally not called.
    s=HostServer.__new__(HostServer);s.I_list=np.asarray(c['candidates']);s.alphas=np.asarray(c['alphas'])
    s.constraint_mode='box_constrained';s.feasible_lower=-1.;s.feasible_upper=1.;s.feasibility_tolerance=1e-8
    ev=[e for e in old['events'] if e['stage']=='baseline'];cursor=0;ends=[];starts=[]
    def obj(x):
        nonlocal cursor
        ys=[]
        for i in range(len(c['ids'])):
            e=ev[cursor];assert e['i']==i and key(e['x'])==key(x),'baseline_sequence'
            ys.append(abs(e['y']));cursor+=1
        w=np.asarray(c['alphas']);w=w/w.sum();return float(np.dot(w,ys))
    class Mark(Trace):
        def begin(self,v):starts.append(cursor);super().begin(v)
        def finish(self,v):ends.append(cursor);super().finish(v)
        def end(self,v):
            if len(ends)<len(self.pre):ends.append(cursor)
            super().end(v)
    oldtrace=engine.Trace;engine.Trace=Mark
    try:base,tr=run(s,obj,30,initial=c['fixed_start'])
    finally:engine.Trace=oldtrace
    assert cursor==len(ev) and plain(base)==old['baseline']
    # Independent original, uninstrumented baseline replays the same full sequence.
    cursor=0;control=HostServer._solve_subspace_custom(s,s.I_list,obj,num_iterations=30,direction_weights=s.alphas,initial_solution=np.asarray(c['fixed_start']))
    assert cursor==len(ev) and plain(control)==plain(base)
    assert len(ends)==len(old['full'])
    normal={};previous=0;views=[];normal_by_round=[]
    for t,full in enumerate(old['full']):
        for e in ev[previous:ends[t]]:
            k=str(e['i'])+':'+key(e['x'])
            if k not in normal:normal[k]=dict(i=e['i'],x=e['x'],y=e['y'],source='baseline',available_round=t)
            else:assert normal[k]['y']==e['y']
        previous=ends[t];pre=full['pre']
        data=dict(case=c['key'],round=t,x=pre['S_current'],mode=pre['current_method'],eta=pre['eta'],delta=pre['delta'],
            anchors=c['candidates'],a=c['alphas'],alpha=c['alphas'],trials=[z['x'] for z in full['trials']],full_endpoint=full['post']['S_current'])
        views.append(data);normal_by_round.append(dict(normal))
    return views,normal_by_round,dict(query_count=len(ev),round_end_query_offsets=ends,round_start_query_offsets=starts,
        baseline_replay_exact=True,uninstrumented_replay_exact=True,events=ev)

class Queries:
    def __init__(self,server,limit):
        self.server=server;self.limit=limit;self.actual={};self.all_events=[];self.mode_cache={};self.cost={};self.normal={};self.mode=''
    def at(self,normal,mode,t):self.normal=normal;self.mode=mode;self.t=t
    def query(self,i,x):
        k=str(i)+':'+key(x);own=self.mode_cache.setdefault(self.mode,{})
        cost=self.cost.setdefault(self.mode,dict(requests=0,samples=0,cache_hits=0,model_seconds=0.))
        cost['requests']+=1
        if k in self.normal:y=self.normal[k]['y'];cost['cache_hits']+=1;source='baseline'
        elif k in own:y=own[k];cost['cache_hits']+=1;source='own_previous'
        else:
            cost['samples']+=1;source='charged_new'
            if k in self.actual:y=self.actual[k]['y']
            else:
                self.limit.before();tic=time.perf_counter();y=float(self.server.trusted_agents[i].api_predict(np.asarray(x))[0]);dt=time.perf_counter()-tic
                self.limit.charge(dt);cost['model_seconds']+=dt
                self.actual[k]=dict(i=i,x=plain(x),y=y,api_input_float32=np.asarray(x,dtype=np.float32).tolist())
            own[k]=y
        self.all_events.append(dict(mode=self.mode,t=self.t,i=i,x=plain(x),source=source,y=y))
        return y

def record(c,old,limit):
    from support import load_server
    tic=time.perf_counter();views,normals,provenance=baseline_view(c,old)
    clean={k:v for k,v in c.items() if k!='evaluator'};server=load_server(clean);queries=Queries(server,limit);pred=[]
    for t,(vdata,normal) in enumerate(zip(views,normals)):
        v=View(vdata,normal)
        for i in sorted(range(v.n),key=lambda i:c['ids'][i]):
            pred.append(dict(t=t,i=i,agent_id=c['ids'][i],method='N0',role='model',**n0(v,i)))
        for role in ROLES:
            queries.at(normal,'standalone_'+role,t)
            for i in sorted(range(v.n),key=lambda i:c['ids'][i]):
                pred.append(dict(t=t,i=i,agent_id=c['ids'][i],method='N1',role=role,**n1(v,i,role,queries.query)))
        queries.at(normal,'shared',t)
        for i in sorted(range(v.n),key=lambda i:c['ids'][i]):
            for role in ROLES:
                r=n1(v,i,role,queries.query)
                ref=next(p for p in pred if p['t']==t and p['i']==i and p['method']=='N1' and p['role']==role)
                assert r['score']==ref['score'] and r['endpoint']==ref['endpoint']
    return dict(case=clean,views=views,normal_replies=list(normals[-1].values()),provenance=provenance,
        predictions=pred,cost=queries.cost,new_replies=list(queries.actual.values()),query_events=queries.all_events,
        wall_seconds=time.perf_counter()-tic,oracle_loaded=False,pooled_loaded=False,truth_loaded=False)

def phase_run(phase):
    check();limit=Limit(20000 if phase=='pilot' else 40000,180 if phase=='pilot' else 600);tic=time.perf_counter();done=[]
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('stage_wall_limit')));signal.alarm(limit.seconds)
    try:
        for c in read(ROOT/'cases.json')[phase]:
            print(phase,c['key'],flush=True);old=read(PREV/phase/(c['key']+'.json'))
            r=record(c,old,limit);save(ROOT/phase/(c['key']+'.json'),r);done.append(c['key'])
        save(ROOT/(phase+'_completion.json'),dict(complete=True,cases=done,new_samples=limit.samples,model_seconds=limit.model_seconds,wall_seconds=time.perf_counter()-tic))
        if phase=='pilot':save(ROOT/'development_budget.json',dict(samples=40000,seconds=600,pilot_within_cap=True,effects_not_read=True))
        else:save(ROOT/'predictions_locked.json',dict(oracle_loaded=False,pooled_loaded=False,truth_loaded=False,
            hashes={str(p.relative_to(ROOT)):sha(p) for phase in ['pilot','development'] for p in (ROOT/phase).glob('*.json')}))
    except Exception as e:
        save(ROOT/(phase+'_failure.json'),dict(error=repr(e),completed=done,samples=limit.samples));raise
    finally:signal.alarm(0)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','pilot','development']);args=ap.parse_args()
    if args.mode=='prepare':prepare()
    else:setup();phase_run(args.mode)
