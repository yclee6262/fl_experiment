import numpy as np
import pytest
from unified_alpha_methods import METHODS, initial_alpha
from initial_alpha_methods import initial_alpha as legacy

class Agent:
    def __init__(self,index): self.agent_id=index; self.calls=0
    def api_predict(self,x):
        self.calls+=1; a=np.atleast_2d(x)
        return a[:,0]*.5 + self.agent_id*.04*a[:,1] + self.agent_id*.03

def inputs():
    x=np.random.default_rng(17).uniform(-1,1,(20,2));y=.5*x[:,0]+.1*x[:,1]
    agents=[Agent(i) for i in [1,2,3]];p=np.column_stack([a.api_predict(x) for a in agents]);mse=((p-y[:,None])**2).mean(axis=0)
    return agents,mse,x,y,np.array([[.1,.2],[.2,.1],[-.1,.2]]),p

@pytest.mark.parametrize('method',METHODS)
def test_all_methods_reuse_preserves_formula_and_order(method):
    a,mse,x,y,proposal,p=inputs()
    fresh,_=initial_alpha(method,a,mse,x,y,proposal,0,bootstrap_replicates=30,bootstrap_seed=9)
    for agent in a:agent.calls=0
    reused,audit=initial_alpha(method,a,mse,x,y,proposal,0,bootstrap_replicates=30,bootstrap_seed=9,cached_blind_predictions=p)
    np.testing.assert_array_equal(fresh,reused)
    assert audit['agent_ids']==[1,2,3]
    assert sum(agent.calls for agent in a)==(6 if method=='target_peer_shrink' else 0)

def test_direct_legacy_ensemble_is_not_silently_replaced():
    a,mse,x,y,proposal,p=inputs()
    expected,_=legacy('ensemble_shrink',a,mse,x,y,proposal,0)
    actual,_=initial_alpha('ensemble_shrink',a,mse,x,y,proposal,0,cached_blind_predictions=p)
    np.testing.assert_array_equal(expected,actual)

def test_sparse_and_stacked_relationships():
    a,mse,x,y,proposal,p=inputs(); kw={'cached_blind_predictions':p}
    b,_=initial_alpha('B_stage0',a,mse,x,y,proposal,0,**kw)
    full,_=initial_alpha('stacked_full',a,mse,x,y,proposal,0,**kw)
    half,_=initial_alpha('B_stacked_half',a,mse,x,y,proposal,0,**kw)
    sparse,_=initial_alpha('top_two_expertise',a,mse,x,y,proposal,0,**kw)
    np.testing.assert_allclose(half,.5*b+.5*full)
    assert np.count_nonzero(sparse)==2

def test_registry_and_cache_validation():
    assert len(METHODS)==25 and len(set(METHODS))==25
    a,mse,x,y,proposal,p=inputs()
    with pytest.raises(ValueError,match='cache'):
        initial_alpha('A_uniform',a,mse,x,y,proposal,0,cached_blind_predictions=p[:,:2])
