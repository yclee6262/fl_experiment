"""Hard query accounting, observed delivery, feasibility and error regression."""
import json
import numpy as np
import pytest
import core
from host_server import HostServer
from experiments.baselines.paired_benchmark.search import run_search, PanelOracle, BudgetExhausted, METHODS


class Agent:
    def __init__(self, fail=False, nan=False):
        self.calls=0
        self.fail=fail
        self.nan=nan
    def api_predict(self,x):
        assert np.asarray(x).dtype==np.float32
        assert np.isfinite(x).all() and np.all(np.abs(x)<=1)
        self.calls+=1
        if self.fail:raise RuntimeError('model down')
        return np.array([np.nan if self.nan else np.sum(x*x)],dtype=np.float32)


def server():
    s=HostServer(0.,2,constraint_mode='box_constrained',coefficient_eps=1e-5)
    s.trusted_agents=[Agent(),Agent(),Agent()]
    s.alphas=[1/3]*3
    s.I_list=[np.array([.8,.6]),np.array([.4,.6]),np.array([-.2,.3])]
    return s


@pytest.mark.parametrize('method',METHODS)
def test_hard_budget_and_best_observed_delivery(method):
    s=server();result=run_search(s,method,9)
    assert result['panel_evaluations']<=9
    assert result['model_queries']<=27
    assert sum(a.calls for a in s.trusted_agents)==result['model_queries']==result['input_samples']
    valid=[e for e in result['events'] if e.get('loss') is not None]
    if valid:
        assert result['status']=='valid'
        assert result['loss']==min(e['loss'] for e in valid)
        assert any(e['X']==result['X'] and e['loss']==result['loss'] for e in valid)
    assert result['transmitted_bytes']==result['model_queries']*12
    json.dumps(result,allow_nan=False)


def test_best_of_quota_not_transferred_and_tie_favors_coefficient():
    s=server()
    def short(anchors,objective,**kwargs):
        objective(np.zeros(2))
        return dict(success=True,message='short')
    def long(anchors,objective,**kwargs):
        for _ in range(100):objective(np.zeros(2))
    s._solve_subspace_bfgs=short
    s._solve_subspace_custom=long
    result=run_search(s,'best_of',11)
    assert result['per_engine_budget']==5 and result['unused_split_panels']==1
    assert [r['panel_evaluations'] for r in result['engine_details']]==[1,5]
    assert result['panel_evaluations']==6
    assert result['chosen_engine']=='coefficient'
    assert result['budget_exhausted']


def test_zero_and_insufficient_anchor_budget_retain_status():
    s=server();r=run_search(s,'best_anchor',0)
    assert r['status']=='no_valid_endpoint' and r['X'] is None
    assert r['stop_reason']=='budget_exhausted' and r['model_queries']==0
    s=server();r=run_search(s,'best_anchor',1)
    assert r['status']=='valid' and r['stop_reason']=='budget_exhausted'
    assert r['panel_evaluations']==1


def test_outside_inputs_rejected_before_api_and_cap_checked_before_panel():
    s=server();o=PanelOracle(s,1,'test')
    o(np.array([2.,0.]));o(np.array([np.nan,0.]))
    assert o.panel_evaluations==o.model_queries==0
    o(np.zeros(2))
    with pytest.raises(BudgetExhausted):o(np.zeros(2))
    assert o.panel_evaluations==1 and o.model_queries==3
    assert sum(a.calls for a in s.trusted_agents)==3


def test_model_failure_and_nonfinite_response_remain_json_safe():
    s=server();s.trusted_agents[1]=Agent(fail=True)
    r=run_search(s,'anchor_mean',10)
    assert r['status']=='no_valid_endpoint' and r['stop_reason']=='solver_or_model_error'
    assert r['model_queries']==2 and r['input_payload_bytes']==16 and r['output_payload_bytes']==4
    assert r['transmitted_bytes']==20
    assert r['events'][0]['kind']=='model_error'
    json.dumps(r,allow_nan=False)
    s=server();s.trusted_agents[1]=Agent(nan=True)
    r=run_search(s,'anchor_mean',10)
    assert r['status']=='no_valid_endpoint'
    assert r['events'][0]['kind']=='nonfinite_prediction'
    assert r['model_queries']==2
    json.dumps(r,allow_nan=False)


def test_invalid_method_budget_and_domain_rejected():
    for bad in [-1,1.5,True]:
        with pytest.raises(ValueError):run_search(server(),'custom',bad)
    with pytest.raises(ValueError):run_search(server(),'typo',10)
    s=server();s.constraint_mode='legacy_unconstrained'
    with pytest.raises(ValueError):run_search(s,'custom',10)
