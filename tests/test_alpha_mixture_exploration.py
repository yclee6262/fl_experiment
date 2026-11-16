import numpy as np
import pytest
from alpha_mixture_exploration import initializer, candidates
from test_pure_es_initialization import Agent

def test_endpoints_cache_and_inactive_support():
    p=np.array([[.1,-.1,.2],[.1,.3,.4],[-.2,.1,-.1],[.3,.6,.5]])
    y=np.array([0.,.2,-.1,.4]);x=np.arange(8).reshape(4,2)
    agents=[Agent(i,p[:,i]) for i in range(3)]
    for a in agents:
        a.api_predict=lambda *_: (_ for _ in ()).throw(AssertionError('cache not used'))
    args=(agents,((p-y[:,None])**2).mean(0),x,y,np.zeros((3,2)),0)
    peer=np.array([[np.nan,.4,.1],[.2,np.nan,.3],[.3,.6,np.nan]])
    basis=[initializer(w,peer)('x',*args,cached_blind_predictions=p)[0] for w in np.eye(4)]
    for w in candidates().values():
        actual,_=initializer(w,peer)('x',*args,cached_blind_predictions=p)
        np.testing.assert_allclose(actual,np.array(w)@basis,atol=1e-12)
    initializer([.5,0,.5,0],None)('x',*args,cached_blind_predictions=p)
    with pytest.raises(ValueError):initializer([0,0,0,1],np.full((3,3),np.nan))('x',*args,cached_blind_predictions=p)
    for w in [[-1,1,1,0],[1,1,1,1],[np.nan,0,0,1]]:
        with pytest.raises(ValueError):initializer(w,peer)

def test_e_fit_failure_preserves_mixture(monkeypatch):
    import initial_alpha_methods as legacy
    from types import SimpleNamespace
    monkeypatch.setattr(legacy,'minimize',lambda *a,**k:SimpleNamespace(success=False))
    p=np.array([[1.,2.,3.],[2.,3.,4.]]);y=np.zeros(2);x=np.zeros((2,2));mse=(p*p).mean(0)
    agents=[Agent(i,p[:,i]) for i in range(3)]
    out,a=initializer([.5,0,.5,0],None)('x',agents,mse,x,y,x.tolist()+[[0,0]],0,cached_blind_predictions=p)
    q=1/(mse+1e-5);q/=q.sum()
    np.testing.assert_allclose(out,.5/3+.5*q)
    assert a['component_audits']['E']['fallback_reason']
