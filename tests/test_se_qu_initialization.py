import numpy as np
import pytest
from se_qu_initialization import initializer
from test_pure_es_initialization import Agent
from pure_es_initialization import initializer as pure
from initial_alpha_methods import initial_alpha

def test_component_alignment_and_cached_only_inputs():
    y=np.array([0.,.2,-.1,.4]);p=np.array([[.1,-.1,.2],[.1,.3,.4],[-.2,.1,-.1],[.3,.6,.5]])
    x=np.arange(8).reshape(4,2);agents=[Agent(i,p[:,i]) for i in range(3)]
    mse=((p-y[:,None])**2).mean(axis=0);proposals=np.zeros((3,2))
    peer=np.array([[0.,.4,.1],[.2,0.,.3],[.3,.6,0.]])
    def forbidden(*args): raise AssertionError('Unnecessary model call with cached evidence')
    for a in agents:a.api_predict=forbidden
    args=(agents,mse,x,y,proposals,0);kw=dict(cached_blind_predictions=p)
    e,_=pure(peer)('stacked_full',*args,**kw);s,_=pure(peer)('pure_support',*args,**kw)
    q,_=initial_alpha('B_stage0',*args,**kw)
    se,_=initializer(peer)('support_ensemble_mix50',*args,**kw)
    qu,_=initializer(peer)('quality_shrink',*args,**kw)
    np.testing.assert_allclose(se,(s+e)/2,atol=1e-12)
    np.testing.assert_allclose(qu,(q+1/3)/2,atol=1e-12)
    perm=np.array([2,0,1])
    for method,expected in [('support_ensemble_mix50',se),('quality_shrink',qu)]:
        w,_=initializer(peer[perm][:,perm])(method,[agents[i] for i in perm],mse[perm],x,y,proposals[perm],0,cached_blind_predictions=p[:,perm])
        np.testing.assert_allclose(w,expected[perm],atol=2e-6)
    with pytest.raises(ValueError):initializer(peer)('typo',*args,**kw)
