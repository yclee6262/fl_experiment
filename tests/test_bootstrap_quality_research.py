import numpy as np
import pytest
from bootstrap_quality_research import calibration, solve, json_numpy
from analyze_bootstrap_quality_research import valid, paired_stats, power_plan, summary


def test_nested_calibration_is_independent_reproducible_and_exact_stage0_labels():
    x,y=calibration(12,3,100)
    small_x,small_y=calibration(12,3,20)
    np.testing.assert_array_equal(x[:20],small_x)
    np.testing.assert_array_equal(y[:20],small_y)
    assert not np.array_equal(x,calibration(12,4,100)[0])
    np.testing.assert_allclose(y,x.sum(axis=1)+x[:,0]*x[:,1])


def test_seed_level_pairs_do_not_treat_blind_draws_as_separate_samples():
    result=paired_stats({1:[-.1]*10,2:[.2]*10})
    assert result['n_seeds']==2
    assert result['mean']==pytest.approx(.05)
    assert result['seed_SD']==pytest.approx(np.std([-.1,.2],ddof=1))
    assert paired_stats({})['CI95']==(None,None)


def test_valid_requires_actual_finite_feasible_X_and_quality():
    row={'status':'valid','X':[0.,1.+.5e-8],'pooled_error':.1}
    assert valid(row)
    for change in [{'X':[]},{'X':[0.,1.+2e-8]},{'X':[np.nan,0.]},
                   {'pooled_error':float('inf')},{'pooled_error':None},{'status':'solver_failure'}]:
        assert not valid({**row,**change})


def test_power_rule_caps_without_claiming_adequate_power():
    result={'comparisons':[{'target':0.,'n':20,'seed_SD':.2}]}
    planned=power_plan(result)
    assert planned['formal_N']==120 and planned['capped']
    assert planned['normal_approx_required_N']>120
    assert power_plan({'comparisons':[{'target':0.,'n':20,'seed_SD':0.}]})['formal_N']==30


def test_solver_failure_keeps_consumed_queries_and_initialization(monkeypatch):
    import bootstrap_quality_research as module
    class Agent:
        agent_id=1
        def api_predict(self,x):return np.ones(len(x))
    class Server:
        trusted_agents=[Agent()]
        I_list=[np.zeros(2)]
        target_T=0.
    def failed(server,*args,**kwargs):
        server.trusted_agents[0].api_predict(np.zeros((4,2)))
        raise ValueError('expected infeasible subspace')
    monkeypatch.setattr(module,'run_stage3',failed)
    row=solve(Server(),'bootstrap_quality',np.ones(1),np.zeros((20,2)),
              np.zeros(20),np.ones((20,1)),8,1000)
    assert row['status']=='solver_failure' and row['X'] is None
    assert row['stage3_requests']==1 and row['stage3_input_samples']==4
    assert row['initialization_seconds']>0 and row['initialization_requests']==0
    assert row['pooled_error'] is None


def test_unexpected_program_errors_are_not_mislabelled_solver_failure(monkeypatch):
    import bootstrap_quality_research as module
    class Agent:
        agent_id=1
        def api_predict(self,x):return np.ones(len(x))
    class Server:
        trusted_agents=[Agent()]
        I_list=[np.zeros(2)]
        target_T=0.
    def failed(*args,**kwargs):raise RuntimeError('program bug')
    monkeypatch.setattr(module,'run_stage3',failed)
    with pytest.raises(RuntimeError):
        solve(Server(),'A_uniform',np.ones(1),np.zeros((20,2)),np.zeros(20),
              np.ones((20,1)),8,1000)
