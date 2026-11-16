import numpy as np
import pytest
from support_quality_mix_experiment import mixed_weights
from peer_support_alpha import support_signal

def test_exact_equal_mix_and_no_uniform_shrink():
    p=np.array([[0.,.3,.1],[.4,0.,.2],[.5,.7,0.]])
    mse=np.array([.01,.1,1.]);s,_=support_signal(p,0);b=1/(mse+1e-5);b/=b.sum()
    alpha,a=mixed_weights(mse,p,0)
    np.testing.assert_allclose(alpha,.5*s+.5*b)
    assert not np.allclose(alpha,.5*np.full(3,1/3)+.5*alpha)
    assert sum(alpha)==pytest.approx(1) and min(alpha)>=0
    assert a['uniform_fraction']==0

def test_order_equivariance_and_diagonal_exclusion():
    p=np.array([[0.,.3,.1],[.4,0.,.2],[.5,.7,0.]])
    mse=np.array([.01,.1,1.]);w,_=mixed_weights(mse,p,0)
    np.fill_diagonal(p,1e10);w2,_=mixed_weights(mse,p,0)
    np.testing.assert_allclose(w,w2)
    perm=[2,0,1];wp,_=mixed_weights(mse[perm],p[np.ix_(perm,perm)],0)
    np.testing.assert_allclose(wp,w[perm])

@pytest.mark.parametrize('mse',[[.1,-.2,.3],[.1,np.nan,.3],[.1,np.inf,.3]])
def test_invalid_quality(mse):
    with pytest.raises(ValueError):mixed_weights(mse,np.zeros((3,3)),0)
