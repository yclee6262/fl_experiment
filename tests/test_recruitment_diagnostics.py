import numpy as np
from experiments.coalition.recruitment_diagnostics.analyze import sequential,core_select,percentile,correlation


def candidate(i,q,bid,residual):
    return dict(agent_id=i,quality_share=q,bid=bid,residual=np.array(residual))


def test_actual_core_rule_and_budget_match():
    # Quality shares must be inverse-MSE normalized for the HostServer check.
    residuals=[[.1,.1],[.2,-.2],[.3,.3],[.4,-.4]]
    raw=np.array([1/(np.mean(np.array(x)**2)+1e-5) for x in residuals]);q=raw/raw.sum()
    cs=[candidate(i+1,q[i],b,residuals[i]) for i,b in enumerate([1.5,.5,.7,.4])]
    for budget in [1.,2.,4.]:
        for eta in [0.,.5]:
            got=sequential(cs,budget,eta)
            assert got==core_select(cs,budget,eta)
            assert len(got)<=3 and sum(cs[i]['bid'] for i in got)<=budget


def test_quality_only_differs_from_quality_per_bid():
    cs=[candidate(1,.6,2,[1,0]),candidate(2,.4,.5,[0,1])]
    assert sequential(cs,2,0)==(1,)
    assert sequential(cs,2,0,True)==(0,)


def test_percentile_ties_and_undefined_correlations():
    assert percentile(2,[1,2,2,4])==.5
    assert correlation([1,1],[2,3]) is None
    assert correlation([1],[2]) is None
    assert correlation([1,2,3],[3,2,1])==1.
