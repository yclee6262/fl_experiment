import numpy as np
import pytest
from experiments.contribution.loo_allocation_validation.allocation import allocate, compare, panel_weights
from experiments.contribution.loo_allocation_validation.methods import Observer
from experiments.contribution.loo_allocation_validation.verify import audit_events


def test_hand_allocation_and_scale():
    np.testing.assert_allclose(allocate([3,2,-1])['q'],[.6,.4,0])
    assert compare([30,20,-7],[3,2,-1])['tv']==0


def test_reserve_invalid_and_boundary():
    for x in [[0,0],[-1,-2],[1e-8,-1]]:assert allocate(x)['status']=='reserved'
    assert allocate([1e-7,0])['fragile']
    assert allocate([float('nan'),1])['status']=='invalid'
    assert compare([0,0],[1,0])['augmented_tv']==1
    assert compare([-1,0],[0,0])['both_reserved']
    assert compare(None,[1,0])['augmented_tv'] is None


def test_peer_weights():
    w=panel_weights([.2,.3,.5],1)
    assert w[1]==0
    assert w@np.array([1,100,3])==pytest.approx(17/7)
    with pytest.raises(ValueError):panel_weights([1.,0],0)


def test_rejected_and_bounded_accepted_displacement():
    ob=Observer(2,2,'CUM_A');ob.begin({'S_current':np.array([.95,0])})
    ob.h[:]=[[-2,0],[1,0]];ob.gradient({'gradient':np.array([-1,0]),'gradient_norm':1.})
    ob.trial({'candidate_loss':2,'best_loss':1,'candidate_S':np.array([1.,0])})
    assert ob.accepted==0
    ob.trial({'candidate_loss':.5,'best_loss':1,'candidate_S':np.array([1.,0])})
    np.testing.assert_allclose(ob.total.sum(0),[.05,0])
    assert Observer(2,2,'LAST_J').last is None


def test_cache_audit_and_removed_owner():
    events=[dict(key='a',stage='baseline',owner=None,model=0,hit=False,y=2),dict(key='a',stage='peer',owner=1,model=0,hit=True,y=2)]
    assert audit_events(events)['peer']['hits']==1
    events[1]['hit']=False
    with pytest.raises(ValueError):audit_events(events)


def test_budget_counts_prior_attempts():
    from experiments.contribution.loo_allocation_validation.run import Budget
    b=Budget(prior_samples=400000)
    with pytest.raises(RuntimeError):b.charge()
    b=Budget(prior_seconds=901)
    with pytest.raises(RuntimeError):b.charge()


def test_float64_cache_identity():
    from experiments.contribution.loo_allocation_validation.methods import Queries
    from experiments.contribution.loo_allocation_validation.run import Budget
    from types import SimpleNamespace
    s=SimpleNamespace(target_T=0.,alphas=np.array([.5,.5]),trusted_agents=[SimpleNamespace(api_predict=lambda x:[1.])]*2)
    q=Queries(s,Budget(),'version');x=np.array([.1,.2]);other=x.copy();other[0]=np.nextafter(other[0],1.)
    q.reply(0,x);q.stage='peer';q.reply(0,x);q.reply(0,other)
    assert [e['hit'] for e in q.events]==[False,True,False]
    assert len(q.cache)==2


def test_scaling_label_dimension_and_interaction():
    from experiments.contribution.loo_allocation_validation.scaling_data import labels
    assert labels(np.zeros((3,10)),.3).shape==(3,)
    np.testing.assert_array_equal(labels(np.zeros((3,10)),.3),0.)
    x=np.full((2,5),.2)
    assert np.all(labels(x,1.2)>labels(x,.3))


def test_multidimensional_joint_removal_and_no_motion():
    from pathlib import Path
    from types import SimpleNamespace
    from experiments.contribution.loo_allocation_validation.methods import module_at, observed_solver, execute
    from experiments.contribution.loo_allocation_validation.run import Budget
    from experiments.contribution.loo_allocation_validation.verify import independent_scores
    frozen=Path(__file__).resolve().parents[1]/'contribution_analysis/model_step_decision_experiment_v1_20261021/source_snapshot'
    host=module_at('test_scaling_host',frozen/'host_server.py')
    engine=module_at('test_scaling_engine',frozen/'engine.py')
    solver=host.HostServer._solve_subspace_custom;observed=observed_solver(solver)
    s=host.HostServer.__new__(host.HostServer);s.n_features=5;s.target_T=0.
    s.constraint_mode='box_constrained';s.feasible_lower=-1.;s.feasible_upper=1.;s.feasibility_tolerance=1e-8
    s.I_list=np.array([np.full(5,v) for v in [-.5,-.2,0.,.2,.4,.6]])
    s.alphas=np.ones(6)/6
    s.trusted_agents=[SimpleNamespace(api_predict=lambda x,b=b:np.array([np.sum(x)/np.sqrt(5)+b])) for b in np.linspace(-.1,.1,6)]
    case=dict(ids=list(range(6)),fixed_start=np.full(5,.3))
    outputs=[execute(s,case,m,Budget(),'synthetic',solver,observed,engine) for m in ['FULL_LOO','LAST_J','CUM_A']]
    for v in outputs:
        audit_events(v['events']);independent_scores(v,s.alphas,0.)
        np.testing.assert_array_equal(v['base']['solution'],outputs[0]['base']['solution'])
        assert v['x'].shape==(6,5)
    s.trusted_agents=[SimpleNamespace(api_predict=lambda x:np.array([1.])) for _ in range(6)]
    v=execute(s,case,'LAST_J',Budget(),'constant',solver,observed,engine)
    assert v['x'] is None and v['peer'] is None
