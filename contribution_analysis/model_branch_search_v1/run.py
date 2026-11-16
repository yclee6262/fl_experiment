"""Frozen secant/tangent branches with a complete-F outer controller."""
import sys,json,copy,time,hashlib,signal
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from contribution_analysis.model_pool_v_cm_validation_v1.run import (CA,W,FROZEN,load_server,analytic_server,plain)
from contribution_analysis.model_step_decision_v1.engine import run,branch_state,Trace,compile_engine
from host_server import HostServer
import numpy as np

OUT=CA/'model_branch_search_20261107_attempt_02'
ARMS=['O','S','D','P']
class BudgetEnd(Exception):pass
def save(p,v):p.write_text(json.dumps(plain(v),ensure_ascii=False,indent=2,allow_nan=False))
class Replies:
    total=0
    def __init__(self,s,budget):self.s=s;self.budget=budget;self.cache={};self.events=[];self.count=0;self.stage='init'
    def loss(self,i,x):
        x=np.asarray(x,np.float64);key=(i,x.tobytes());hit=key in self.cache
        if not hit:
            if self.count>=self.budget:raise BudgetEnd()
            if Replies.total>=5000000:raise RuntimeError('global cap')
            v=abs(float(self.s.trusted_agents[i].api_predict(x)[0])-self.s.target_T)
            if not np.isfinite(v):raise ValueError('nonfinite loss')
            self.cache[key]=v;self.count+=1;Replies.total+=1
        v=self.cache[key];self.events.append(dict(stage=self.stage,i=i,x=x,loss=v,hit=hit));return v
    def losses(self,x):return np.array([self.loss(i,x) for i in range(len(self.s.alphas))])
    def objective(self,ids):
        ids=list(ids);a=np.asarray(self.s.alphas)[ids];a=a/a.sum()
        return lambda x:float(a@np.array([self.loss(i,x) for i in ids]))

def step(s,obj,tr,it=1,initial=None,resume=None):
    return compile_engine(HostServer._solve_subspace_custom)(s,s.I_list,obj,num_iterations=it,direction_weights=s.alphas,
        initial_solution=initial,_resume=copy.deepcopy(resume),_trace=tr)
def choose(options,before,epsilon):
    best=min(z['loss'] for z in options)
    if before-best<=epsilon:return options[0]
    return min((z for z in options[1:] if z['loss']<=best+epsilon),key=lambda z:z['priority'])

def execute(c,s,arm,rounds,budget=10**9):
    q=Replies(s,budget);a=np.asarray(s.alphas);a=a/a.sum();ids=list(range(len(a)));F=q.objective(ids)
    start=time.perf_counter();x=np.array(c['fixed_start']);initial=F(x);value=initial;rows=[];partial=False;reason='round_limit';pending=[];state=None
    if arm=='O':
        q.stage='full';tr=Trace()
        try:
            result=step(s,F,tr,rounds,initial=x);x=np.array(result['solution']);value=result['loss'];reason='solver_stop' if tr.final['halted'] else 'round_limit'
        except BudgetEnd:
            partial=True;reason='budget'
            accepted=[z for z in tr.trials if z['accepted']]
            if accepted:x=np.array(accepted[-1]['x']);value=accepted[-1]['loss']
        rows=[dict(t=z['next_iteration'],x=z['S_current'],loss=z['best_loss'],eta=z['eta'],mode=z['current_method']) for z in tr.post]
    else:
        q.stage='init';_,tr=run(s,F,0,initial=x);state=tr.final
        for t in range(rounds):
            pre=copy.deepcopy(state);before=float(pre['best_loss']);eps=1e-10*max(1,abs(before))
            pending=[dict(label='stay',x=np.array(pre['S_current']),loss=before,priority=-1)];full=None;branches=[]
            try:
                q.stage='full';ft=Trace();step(s,F,ft,resume=pre);sf=ft.final
                xf=np.array(sf['S_current']);lf=float(sf['best_loss']);full=dict(label='full',x=xf,loss=lf,priority=0);pending.append(full)
                if arm in ['D','P']:
                    offset=(c.get('seed',0)+t)%len(ids)
                    for i in ids[offset:]+ids[:offset]:
                        keep=[j for j in ids if j!=i]
                        if a[keep].sum()<=0:raise ValueError('invalid remaining weights')
                        q.stage='model';obj=q.objective(keep);local=branch_state(pre,obj,s.I_list);bt=Trace();step(s,obj,bt,resume=local)
                        xm=np.array(bt.final['S_current']);q.stage='common';lm=float(a@q.losses(xm))
                        z=dict(label=f'minus_{i}',i=i,x=xm,loss=lm,priority=i+1);branches.append(z)
                        if arm=='P':pending.append(z)
                picked=(full if full['loss']<before else pending[0]) if arm in ['S','D'] else choose(pending,before,eps)
                improved=before-picked['loss']>(0 if arm in ['S','D'] else eps)
                state=copy.deepcopy(sf);state.update(S_current=picked['x'].copy(),best_loss=picked['loss'],
                    global_best_S=picked['x'].copy(),global_best_loss=picked['loss'],
                    history=pre['history']+[picked['loss']],states=pre['states']+[picked['label']],
                    anchor_losses=pre['anchor_losses'],next_iteration=t+1,halted=bool(sf['halted'] and not improved))
                x=np.array(picked['x']);value=picked['loss'];assert value<=before+eps
                rows.append(dict(t=t,pre=pre['S_current'],before=before,x=x,loss=value,eta=state['eta'],mode=state['current_method'],
                    chosen=picked['label'],full_loss=lf,full_x=xf,branches=branches,G=lf-value,rescued=bool(sf['halted'] and improved),state=state))
                if state['halted']:reason='solver_stop';break
            except BudgetEnd:
                # Only endpoints whose full objective is known may be delivered.
                picked=(full if full is not None and full['loss']<before else pending[0]) if arm in ['S','D'] else choose(pending,before,eps)
                x=np.array(picked['x']);value=picked['loss'];partial=True;reason='budget';break
    assert np.isfinite(value) and s.box_violation(x)<=s.feasibility_tolerance
    stages={st:sum(not e['hit'] and e['stage']==st for e in q.events) for st in sorted({e['stage'] for e in q.events})}
    assert sum(stages.values())==q.count<=budget
    return dict(key=c['key'],arm=arm,round_limit=rounds,budget=budget,initial=initial,x=x,loss=value,rows=rows,
        samples=q.count,stages=stages,requests=len(q.events),partial_round=partial,reason=reason,
        completed_candidates=pending,seconds=time.perf_counter()-start,events=q.events)

def gates(rr):
    S,D,O=[rr[k] for k in ['S','D','O']]
    assert len(S['rows'])==len(D['rows'])
    for s,d in zip(S['rows'],D['rows']):
        for k in ['x','loss','eta','mode','state']:
            assert plain(s[k])==plain(d[k]),('SD',k)
    assert abs(O['loss']-S['loss'])<=1e-8*max(1,O['initial']),('OS loss',O['loss'],S['loss'])
    assert np.linalg.norm(np.array(O['x'])-S['x'])<=1e-7,('OS point',O['x'],S['x'])
    for r in rr.values():
        prev=r['initial']
        for z in r['rows']:
            assert z['loss']<=prev+1e-10*max(1,abs(prev));prev=z['loss']
    return dict(passed=True,OS_loss_difference=abs(O['loss']-S['loss']),SD_equal=True)

def cases():
    old=json.loads((CA/'model_pool_v_cm_validation_20261107_attempt_02/registration.json').read_text())
    dev=[c for k in [(3,'general'),(3,'heterogeneous_bias'),(5,'general'),(5,'heterogeneous_bias')] for c in [c for c in old['fresh'] if (c['n'],c['scenario'])==k][:2]]
    hist=[c for c in old['known'] if c['phase']=='historical'];dev+=hist[:2]+hist[10:12]
    base=np.array([[.2,1.,.2,.1,.1],[.25,.8,.3,-.1,.1],[.15,1.2,.1,.1,.05],[.2,.9,.2,0.,.1],[.2,1.1,.15,.05,.1]])
    fresh=[]
    for n in [3,5]:
        for sc in ['general','heterogeneous_bias']:
            for k in range(20):
                seed=202611072000+len(fresh);rng=np.random.default_rng(seed);co=base[:n]+rng.normal(0,.08,(n,5))
                if sc=='heterogeneous_bias':co[-1]=[.8,-.15,-1.3,.1,.1]
                fresh.append(dict(key=f'branch_n{n}_{sc}_{seed}',phase='confirmation',scenario=sc,n=n,seed=seed,
                    coefficients=co.tolist(),candidates=rng.uniform(-.8,.8,(n,2)).tolist(),alphas=[1/n]*n,ids=list(range(n)),fixed_start=rng.uniform(-.5,.5,2).tolist()))
    # Reject accidental seed reuse in stored registrations, excluding this new output.
    needles={c['seed'] for c in fresh}
    for p in CA.glob('*/registration.json'):
        if p.parent==OUT:continue
        data=p.read_text()
        for seed in needles:assert str(seed) not in data,('seed reused',p,seed)
    return dev,fresh

def main():
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    OUT.mkdir(exist_ok=False);(OUT/'records').mkdir();(OUT/'source_snapshot').mkdir()
    controls=json.loads(Path('/private/tmp/branch_search_controls_20261107.json').read_text());assert controls['passed'];save(OUT/'edge_controls.json',controls)
    Replies.total=controls['physical']
    dev,fresh=cases();save(OUT/'registration.json',dict(development=dev,confirmation=fresh,arms=ARMS,rounds=30,budgets_per_model=[200,500,1000],primary_budget=500,max_physical=5000000,max_seconds=1800))
    paths=[Path(__file__),Path(__file__).with_name('analyze.py'),Path(__file__).with_name('test_controls.py'),W/'docs/model_branch_candidate_search_design_20261107.md',W/'docs/model_branch_candidate_search_amendment_20261107.md',CA/'model_step_decision_v1/engine.py',CA/'model_pool_v_cm_validation_v1/run.py',CA/'model_qv_v1/run.py',CA/'custom_realized_gain_v1/runtime.py']+list(FROZEN.glob('*.py'))
    locks={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};save(OUT/'source_lock.json',locks)
    for p in paths:(OUT/'source_snapshot'/'__'.join(p.relative_to(W).parts)).write_bytes(p.read_bytes())
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('wall cap')));signal.alarm(1800)
    progress=[];start=time.perf_counter()
    try:
        for phase,cc in [('development',dev),('confirmation',fresh)]:
            for c in cc:
                print(phase,c['key'],flush=True);s=load_server(c) if c['phase']=='historical' else analytic_server(c);n=len(s.alphas)
                group={}
                for label,rounds,budget in [('round30',30,10**9)]+[(f'budget{b}',300,b*n) for b in [200,500,1000]]:
                    rr={arm:execute(c,s,arm,rounds,budget) for arm in ARMS}
                    if label=='round30':gate=gates(rr)
                    # Full ledgers kept, each execution starts a fresh independent cache.
                    save(OUT/'records'/f"{c['key']}_{label}.json",rr)
                    group[label]={a:{k:r[k] for k in ['loss','initial','samples','stages','reason','partial_round','x','seconds']} for a,r in rr.items()}
                progress.append(dict(key=c['key'],phase=phase,n=n,scenario=c['scenario'],gate=gate,results=group));save(OUT/'progress.json',progress)
            if phase=='development':save(OUT/'development_gate.json',dict(passed=True,teams=len(dev)))
        for p,h in locks.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
        save(OUT/'completion.json',dict(complete=True,teams=len(progress),physical=Replies.total,seconds=time.perf_counter()-start))
    except Exception as e:save(OUT/'failure.json',dict(error=repr(e),teams=len(progress),physical=Replies.total));raise
    finally:signal.alarm(0)
if __name__=='__main__':main()
