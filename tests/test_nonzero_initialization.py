import numpy as np
from core.host_server import HostServer
from experiments.alpha.nonzero_targets.run import h,witness
from experiments.baselines.paired_benchmark.search import PanelOracle


def test_all_fixed_targets_have_finite_box_witnesses():
    for target in [0.,-.5,.5,2.7]:
        x=np.array(witness(target))
        assert max(abs(x))<=1 and abs(h(x)-target)<1e-12
    assert h([0.,0.])==0 and abs(h([0.,0.])-2.7)==2.7


def test_panel_oracle_uses_nonzero_target_and_saved_losses():
    class Model:
        def api_predict(self,x):return np.array([np.asarray(x).sum()])
    s=HostServer(.5,2,constraint_mode='box_constrained')
    s.trusted_agents=[Model(),Model()];s.alphas=[.5,.5]
    objective=PanelOracle(s,2,'test')
    assert objective(np.array([0.,0.]))==.5
    assert objective(np.array([.5,0.]))==0.
    assert objective.best['X']==[.5,0.]
    assert objective.model_queries==4


def test_paired_bootstrap_preserves_constant_differences_and_rejects_empty():
    import pytest
    from experiments.alpha.nonzero_targets.analyze import bootstrap
    a=np.array([[1.,3.],[5.,7.],[9.,11.]])
    b=bootstrap(a,replicates=250)
    np.testing.assert_allclose(b[:,1]-b[:,0],2.)
    with pytest.raises(ValueError):bootstrap(np.empty((0,2)),replicates=10)
