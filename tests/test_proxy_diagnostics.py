import numpy as np
import pytest
from experiments.contribution.proxy_diagnostics.analyze import share,metrics,projection

def test_signed_projection_conserves_gain_and_rejects_bad_decomposition():
 H=np.array([[3.,0.],[-1.,0.],[0.,1.]])
 g=H.sum(0);v,zero=projection(H,g,.2)
 assert not zero and v[1]<0 and abs(v.sum()-.2)<1e-12
 with pytest.raises(ValueError):projection(H,[1.,1.],.2)
 v,zero=projection([[1.,0.],[-1.,0.]],[0.,0.],0)
 assert zero and np.array_equal(v,[0,0])
 with pytest.raises(ValueError):projection([[1.,0.],[-1.,0.]],[0.,0.],.2)

def test_zero_score_is_reserve_not_uniform_or_missing_rank_zero():
 np.testing.assert_array_equal(share([0,0,0]),[0,0,0,1])
 m=metrics([0,0,0],[1,-1,0])
 assert m['spearman'] is None and m['both_zero']==1 and m['one_zero']==2 and m['opposite']==0
 assert m['TV']==1 and m['proxy_reserved'] and not m['reference_reserved']

def test_signed_sums_precede_positive_part():
 # Two rounds cancel for owner 0; clipping each round would assign nonzero mass.
 scores=np.array([[2.,1.],[-3.,0.]]).sum(0)
 np.testing.assert_array_equal(share(scores),[0,1,0])
