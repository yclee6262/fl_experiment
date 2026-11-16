"""Adapter around the existing model-step AST engine; v and a kept distinct."""
import copy
from fractions import Fraction
import time
import numpy as np
from contribution_analysis.model_step_decision_v1.engine import run,branch_state
from contribution_analysis.custom_realized_gain_v1.schema import plain


def evaluate(p,x):
    l=np.array([float(Fraction(z)) for z in p['linear']]);Q=np.array([[float(Fraction(z)) for z in row] for row in p['quadratic']])
    return float(float(Fraction(p['offset']))+l@x+x@Q@x)


def server_for(c):
    from host_server import HostServer
    s=HostServer.__new__(HostServer);s.target_T=0.;s.n_features=2;s.constraint_mode='box_constrained'
    s.feasible_lower=-1.;s.feasible_upper=1.;s.feasibility_tolerance=1e-8
    s.I_list=np.array([[float(Fraction(z)) for z in row] for row in c['anchors']])
    # Legacy engine.run forwards this field only as anchor direction weights.
    s.alphas=np.array([float(Fraction(z)) for z in c['anchor_weights']]);return s


class Panel:
    def __init__(self,c):
        self.c=c;self.v=np.array([float(Fraction(z)) for z in c['weights']]);self.v/=self.v.sum()
        if not np.isfinite(self.v).all() or (self.v<0).any():raise ValueError('invalid_weights')
        self.events=[];self.stage='control';self.locked=False
    def reply(self,i,x):
        if self.stage=='common' and not self.locked:raise ValueError('score_before_lock')
        if self.stage=='minus' and i in self.c['target']:raise ValueError('excluded_model_called')
        y=evaluate(self.c['models'][i],np.asarray(x,float))
        if not np.isfinite(y):raise ValueError('nonfinite_reply')
        self.events.append(dict(stage=self.stage,model=i,x=np.asarray(x).copy(),reply=y));return y
    def objective(self,ids):
        ids=list(ids);w=self.v[ids]
        if not len(ids) or w.sum()<=0:raise ValueError('no_remaining_weight')
        w=w/w.sum()
        def f(x):return float(np.dot(w,[abs(self.reply(i,x)) for i in ids]))
        return f
    def losses(self,x):return np.array([abs(self.reply(i,x)) for i in range(len(self.v))])


def decision(tr):
    return dict(pre=tr.pre[0],post=tr.final,trials=tr.trials,accepted=any(r['accepted'] for r in tr.trials),g=tr.gradients[0]['g'])


def execute(c):
    from host_server import HostServer
    s=server_for(c);p=Panel(c);start=np.array([float(Fraction(z)) for z in c['start']]);allids=list(range(len(c['models'])))
    t=time.perf_counter();obj=p.objective(allids)
    control=HostServer._solve_subspace_custom(s,s.I_list,obj,num_iterations=1,direction_weights=s.alphas,initial_solution=start)
    control_events=copy.deepcopy(p.events);p.stage='full';begin=len(p.events)
    full_result,ft=run(s,obj,1,initial=start);full=decision(ft);full_events=p.events[begin:]
    if plain(control)!=plain(full_result):raise ValueError('original_solver_result_changed')
    strip=lambda es:[{k:v for k,v in plain(e).items() if k!='stage'} for e in es]
    if strip(control_events)!=strip(full_events):raise ValueError('original_query_sequence_changed')
    ids=[i for i in allids if i not in c['target']];remaining=float(p.v[ids].sum());minus=None
    if remaining>0:
        p.stage='minus';f=p.objective(ids);pre=branch_state(full['pre'],f,s.I_list)
        _,bt=run(s,f,1,resume=pre);minus=decision(bt)
    lock=dict(full=np.asarray(full['post']['S_current']).copy(),minus=None if minus is None else np.asarray(minus['post']['S_current']).copy(),event_index=len(p.events))
    p.locked=True;p.stage='common'
    lf=p.losses(lock['full']);initial_losses=p.losses(start);E=p.v*(initial_losses-lf);target_E=float(E[c['target']].sum())
    C=own=others=peer=D=None;lm=None
    if minus is not None:
        lm=p.losses(lock['minus']);delta=lm-lf;C=float(p.v@lm-p.v@lf)
        own=float(p.v[c['target']]@delta[c['target']]);others=float(p.v[ids]@delta[ids]);peer=others/remaining
        if abs(C-own-others)>1e-8:raise ValueError('decomposition_error')
        D=evaluate(c['reference'],lock['minus'])-evaluate(c['reference'],lock['full'])
    grid=[np.array([x,0.]) for x in np.linspace(-.5,.5,21)]
    rmse=[float(np.sqrt(np.mean([(evaluate(model,z)-evaluate(c['reference'],z))**2 for z in grid]))) for model in c['models']]
    stage_counts={stage:sum(e['stage']==stage for e in p.events) for stage in ['control','full','minus','common']}
    return dict(fixture=c,control=control,full_result=full_result,full=full,minus=minus,lock=lock,events=p.events,
        score=dict(status='available' if minus is not None else 'undefined_no_remaining_weight',C=C,own=own,others=others,peer=peer,
                   E_model=E,E_target=target_E,D_analytic=D,RMSE=rmse,losses_full=lf,losses_minus=lm,payment=None),
        anchor_direction_weights=s.alphas,model_weights=p.v,remaining_model_indices=ids,stage_calls=stage_counts,
        analytic_diagnostic_model_evaluations=len(c['models'])*21,analytic_reference_points=21,seconds=time.perf_counter()-t)
