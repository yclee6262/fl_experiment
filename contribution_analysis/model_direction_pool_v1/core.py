"""Post-trajectory directional features and one-round removal branches."""
import copy
from fractions import Fraction as F
import numpy as np
from contribution_analysis.model_step_decision_v1.engine import run,branch_state
from contribution_analysis.three_pool_t3_model_v1.core import server_for,evaluate
from contribution_analysis.model_consensus_pool_v1.core import canonical_hash
from contribution_analysis.custom_realized_gain_v1.schema import plain

class Replies:
    def __init__(self,c):
        self.c=c;self.context=canonical_hash(dict(models=c['models'],loss='absolute_prediction_target0',dtype='float64',deterministic=True))
        self.events=[];self.cache={};self.stage='control';self.cache_enabled=False
    def loss(self,i,x):
        x=np.asarray(x,dtype=np.float64);key=(self.context,i,x.tobytes().hex());hit=self.cache_enabled and key in self.cache
        value=self.cache[key] if hit else abs(evaluate(self.c['models'][i],x))
        if not np.isfinite(value):raise ValueError('nonfinite_loss')
        if key in self.cache and not hit:assert self.cache[key]==value,'nondeterministic_loss'
        self.cache[key]=value;self.events.append(dict(stage=self.stage,model=i,x=x.copy(),loss=value,cache_hit=hit));return value
    def losses(self,x):return np.array([self.loss(i,x) for i in range(len(self.c['models']))])
    def objective(self,ids):
        ids=list(ids);w=np.array([float(F(z)) for z in self.c['weights']])[ids];w=w/w.sum()
        return lambda x:float(w@np.array([self.loss(i,x) for i in ids]))

def features(s,pre,replies,scale):
    x=np.asarray(pre['S_current']);directions=[];excluded=[]
    for k,a in enumerate(s.I_list):
        distance=float(np.linalg.norm(a-x))
        if distance<1e-8:excluded.append(dict(anchor=k,reason='distance_below_solver_threshold'));continue
        u=(a-x)/distance;h=distance if pre['current_method']=='secant' else s._bounded_step(x,u,pre['delta'])
        if h<=1e-12:excluded.append(dict(anchor=k,reason='step_below_solver_threshold'));continue
        point=a if pre['current_method']=='secant' else x+h*u
        directions.append(dict(anchor=k,distance=distance,h=float(h),unit=u,point=point))
    if not directions:return dict(status='undefined_empty_directions',directions=[],excluded=excluded,g=None,U=None)
    if not np.isfinite(scale).all() or (scale<=1e-12).any():return dict(status='undefined_scale',directions=directions,excluded=excluded,g=None,U=None)
    base=replies.losses(x)
    g=np.column_stack([(replies.losses(d['point'])-base)/(scale*d['h']) for d in directions]);n=len(g)
    if n<2:return dict(status='undefined_no_peer',directions=directions,excluded=excluded,g=g,U=None)
    dist=np.sqrt(np.mean((g[:,None,:]-g[None,:,:])**2,axis=2));np.fill_diagonal(dist,np.inf)
    return dict(status='available',directions=directions,excluded=excluded,g=g,U=np.min(dist,axis=1))

def event(trace,t):
    pre=trace.pre[t];post=next((s for s in trace.post if s['next_iteration']==pre['next_iteration']+1),trace.final)
    return dict(pre=pre,post=post,trials=[r for r in trace.trials if r['iteration']==pre['next_iteration']])

def execute(c,iterations=1,radius=None,u_reference=10.):
    from host_server import HostServer
    if (radius is not None and radius<=0) or u_reference<=0:raise ValueError('positive_fixed_scale_required')
    s=server_for(c);r=Replies(c);ids=list(range(len(c['models'])));start=np.array([float(F(z)) for z in c['start']]);obj=r.objective(ids)
    # Search-independent scales are locked before either trajectory.
    r.stage='probe';grid=[np.array([x,y]) for x in [-1,-.5,0,.5,1] for y in [-1,-.5,0,.5,1]]
    probes=np.array([r.losses(x) for x in grid]).T;scale=np.quantile(probes,.9,axis=1)-np.quantile(probes,.1,axis=1)
    r.stage='control';begin=len(r.events)
    original=HostServer._solve_subspace_custom(s,s.I_list,obj,num_iterations=iterations,direction_weights=s.alphas,initial_solution=start)
    control=copy.deepcopy(r.events[begin:]);r.stage='full';begin=len(r.events)
    result,trace=run(s,obj,iterations,initial=start);full_events=r.events[begin:]
    assert plain(result)==plain(original),'trajectory_changed'
    strip=lambda events:[dict(model=e['model'],x=plain(e['x']),loss=e['loss']) for e in events]
    assert strip(control)==strip(full_events),'queries_changed'
    full=[event(trace,t) for t in range(len(trace.pre))];branches=[]
    r.cache_enabled=True
    for t,fe in enumerate(full):
        for i in ids:
            remaining=[j for j in ids if j!=i];r.stage='minus';begin=len(r.events);f=r.objective(remaining)
            state=branch_state(fe['pre'],f,s.I_list);_,bt=run(s,f,1,resume=state)
            assert all(e['model']!=i for e in r.events[begin:]),'removed_model_called'
            branches.append(dict(round=t,model=i,decision=event(bt,0),full=fe['post']['S_current'].copy(),minus=bt.final['S_current'].copy()))
    # Every endpoint is frozen before supplemental direction/peer scoring.
    r.stage='score';rows=[];direction=[];w=np.array([float(F(z)) for z in c['weights']]);w/=w.sum()
    for t,fe in enumerate(full):
        ft=features(s,fe['pre'],r,scale);direction.append(ft)
        for b in [b for b in branches if b['round']==t]:
            i=b['model'];other=[j for j in ids if j!=i];lf=r.losses(b['full']);lm=r.losses(b['minus'])
            I=float(np.linalg.norm(b['full']-b['minus'])/(fe['pre']['eta'] if radius is None else radius))
            P=None if (scale<=1e-12).any() else float((w[other]@((lm-lf)/scale)[other])/w[other].sum())
            if ft['status']!='available':U=N=None
            else:
                U=float(ft['U'][i]);N=(U/u_reference)/(1+U/u_reference)*I/(1+I);P=float((w[other]@((lm-lf)/scale)[other])/w[other].sum())
            rows.append(dict(round=t,mode=fe['pre']['current_method'],model=i,owner=c['owners'][i],U=U,I=I,N=N,P=P,status=ft['status'],full=b['full'],minus=b['minus']))
    # Cache is checked against fresh deterministic scoring after score construction.
    audit_calls=0
    for (ctx,i,point),cached in r.cache.items():
        x=np.frombuffer(bytes.fromhex(point),dtype=np.float64);assert cached==abs(evaluate(c['models'][i],x));audit_calls+=1
    costs={stage:dict(requests=sum(e['stage']==stage for e in r.events),evaluations=sum(e['stage']==stage and not e['cache_hit'] for e in r.events),hits=sum(e['stage']==stage and e['cache_hit'] for e in r.events)) for stage in sorted({e['stage'] for e in r.events})}
    return plain(dict(fixture=c,iterations=iterations,full_result=result,full=full,branches=branches,directions=direction,rows=rows,probe=probes,scale=scale,events=r.events,cost=costs,cache_context=r.context,cache_audit_evaluations=audit_calls,solves=2+len(branches),trajectory_exact=True,query_sequence_exact=True))
