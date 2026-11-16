"""Fixed warmup/pair timing with real predictions, no oracle or evaluator."""
import time,signal
import numpy as np
from common import *
from core import View,key,n0,n1,ROLES

def single(c,mode,limit):
    from support import load_server
    import engine
    from engine import Trace,run
    from host_server import HostServer
    clean={k:v for k,v in c.items() if k!='evaluator'}
    tic=time.perf_counter();s=load_server(clean);load_seconds=time.perf_counter()-tic
    normal={};extra={};n=len(c['ids']);w=np.asarray(c['alphas']);w=w/w.sum()
    model_s=0.;overhead=0.;samples=0;extras=0;baselines=0
    def raw(i,x):
        nonlocal model_s,samples
        limit.before();t=time.perf_counter();y=float(s.trusted_agents[i].api_predict(np.asarray(x))[0]);dt=time.perf_counter()-t
        limit.charge(dt);model_s+=dt;samples+=1;return y
    def obj(x):
        nonlocal baselines
        vals=[]
        for j in range(n):
            y=raw(j,x);baselines+=1;vals.append(abs(y))
            if mode!='baseline':normal[str(j)+':'+key(x)]=dict(i=j,x=plain(x),y=y,source='baseline',available_round=0)
        return float(np.dot(w,vals))
    def query(j,x):
        nonlocal extras
        k=str(j)+':'+key(x)
        if k in normal:return normal[k]['y']
        if k in extra:return extra[k]
        y=raw(j,x);extras+=1;extra[k]=y;return y
    class Online(Trace):
        def score(self,v):
            nonlocal overhead
            begin=time.perf_counter();prev_model=model_s;t=len(self.pre)-1;pre=self.pre[-1]
            d=dict(case=c['key'],round=t,x=pre['S_current'],mode=pre['current_method'],eta=pre['eta'],delta=pre['delta'],anchors=c['candidates'],
                alpha=c['alphas'],a=c['alphas'],trials=[plain(a['x']) for a in self.trials if a['iteration']==t],full_endpoint=plain(v['S_current']))
            view=View(d,normal)
            if mode=='N0':
                for i in range(n):n0(view,i)
            else:
                for i in sorted(range(n),key=lambda i:c['ids'][i]):
                    for role in ROLES:n1(view,i,role,query)
            overhead+=time.perf_counter()-begin-(model_s-prev_model)
        def finish(self,v):super().finish(v);self.score(v)
        def end(self,v):
            if len(self.post)<len(self.pre):self.score(v)
            super().end(v)
    start=time.perf_counter()
    if mode=='baseline':
        r=HostServer._solve_subspace_custom(s,s.I_list,obj,num_iterations=30,direction_weights=s.alphas,initial_solution=np.asarray(c['fixed_start']))
    else:
        original=engine.Trace;engine.Trace=Online
        try:r,_=run(s,obj,30,initial=c['fixed_start'])
        finally:engine.Trace=original
    solver_seconds=time.perf_counter()-start
    start=time.perf_counter();encoded=json.dumps(plain(dict(result=r,normal=list(normal.values()),extra=extra)),allow_nan=False);serialization=time.perf_counter()-start
    start=time.perf_counter()
    path=ROOT/'timing_last_payload.tmp'
    with path.open('w') as f:f.write(encoded)
    io_seconds=time.perf_counter()-start
    return dict(mode=mode,load_seconds=load_seconds,solver_seconds=solver_seconds,serialization_seconds=serialization,io_seconds=io_seconds,
        wall_seconds=solver_seconds+serialization+io_seconds,model_seconds=model_s,samples=samples,baseline_samples=baselines,
        extra_samples=extras,explicit_scoring_nonmodel_seconds=overhead,result=r)

def timing():
    check();limit=Limit(400000,600);allrows=[]
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('timing_budget')));signal.alarm(600)
    try:
        for c in read(ROOT/'cases.json')['pilot']:
            print('timing',c['key'],flush=True)
            for rep in range(12):
                modes=['baseline','N0','N1'] if rep%2==0 else ['N1','N0','baseline']
                rows=[single(c,m,limit) for m in modes]
                for row in rows:
                    row.update(key=c['key'],repeat=rep,warmup=rep<2)
                    if rep>=2:allrows.append(row)
                assert all(plain(r['result'])==plain(rows[0]['result']) for r in rows)
            save(ROOT/('timing_'+c['key']+'.json'),[r for r in allrows if r['key']==c['key']])
        team=[]
        for c in read(ROOT/'cases.json')['pilot']:
            for mode in ['N0','N1']:
                rr=[r for r in allrows if r['key']==c['key']];rat=[];nonmodel=[]
                for rep in range(2,12):
                    a=next(r for r in rr if r['repeat']==rep and r['mode']==mode);b=next(r for r in rr if r['repeat']==rep and r['mode']=='baseline')
                    rat.append(a['wall_seconds']/b['wall_seconds']-1)
                    # Includes recording/serialization/I/O, subtracts directly measured model time on both sides.
                    nonmodel.append(((a['wall_seconds']-a['model_seconds'])-(b['wall_seconds']-b['model_seconds']))/b['wall_seconds'])
                team.append(dict(key=c['key'],method=mode,median_wall_increment=float(np.median(rat)),
                    median_nonmodel_increment=float(np.median(nonmodel)),wall_pair_increments=rat,nonmodel_pair_increments=nonmodel))
        save(ROOT/'timing.json',dict(complete=True,teams=team,model_samples=limit.samples,model_seconds=limit.model_seconds,
            elapsed_seconds=time.perf_counter()-limit.start,measured_repeats=10,warmup_repeats=2,
            N0_time_gate=all(r['median_wall_increment']<=.1 for r in team if r['method']=='N0'),
            N1_time_gate=all(r['median_nonmodel_increment']<=.2 for r in team if r['method']=='N1')))
    finally:signal.alarm(0)
if __name__=='__main__':setup();timing()
