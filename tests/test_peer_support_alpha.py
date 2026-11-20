import numpy as np
import pytest
from peer_support_alpha import support_signal,weights

def test_self_predictions_excluded():
 p=np.array([[100,.2,.6],[.1,200,.5],[.2,.1,300.]])
 s,_=support_signal(p,0);np.fill_diagonal(p,-1000);s2,_=support_signal(p,0);assert np.array_equal(s,s2)
def test_permutation_alignment():
 p=np.array([[0,.2,.6],[.1,0,.5],[.2,.1,0]]);m=[.01,.02,.03];k=[2,0,1]
 w,_=weights('peer_support_shrink',m,p,0);w2,_=weights('peer_support_shrink',np.array(m)[k],p[np.ix_(k,k)],0);assert np.allclose(w2,w[k])
def test_support_monotonic_with_shared_scale():
 p=np.array([[0,2,4],[1,0,4],[1,2,0]])
 w,_=weights('peer_support_shrink',[1,1,1],p,0);assert w[0]>w[1]>w[2] and np.isclose(w.sum(),1) and min(w)>=1/6
 # Improve only candidate 3 enough to become strongest; others still unchanged.
 p[:,2]=.1;w2,_=weights('peer_support_shrink',[1,1,1],p,0);assert w2[2]>w[2]
def test_zero_errors_equal_support():
 s,_=support_signal(np.zeros((3,3)),0);assert np.allclose(s,[1/3]*3)
def test_failure_not_silently_dropped():
 with pytest.raises(ValueError):support_signal([[0,float('nan')],[1,0]],0)
 with pytest.raises(ValueError):support_signal([[0]],0)
def test_repeat_and_boundary_proposals():
 p=np.array([[0,0,.1],[.2,.2,0],[.3,.3,0]])
 s,_=support_signal(p,0);assert np.isfinite(s).all() and np.isclose(s.sum(),1)
 # Self-exclusion gives different judge sets even for duplicated columns.
 assert s[0]!=s[1]
def test_invalid_quality():
 with pytest.raises(ValueError):weights('peer_support_shrink',[1,-1],np.zeros((2,2)),0)
