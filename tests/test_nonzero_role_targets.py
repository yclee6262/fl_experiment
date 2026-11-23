import numpy as np
from experiments.contribution.nonzero_targets.run import witness,truth,split,execute
from experiments.contribution.nonzero_targets.analyze import share,rho


def test_nonzero_witness_and_peer_score():
 for target in [0.,-.5,.5,2.7]:
  x=witness(target);assert max(abs(np.array(x)))<=1 and truth(x,target)<1e-12
 a=np.array([.2,.3,.5]);d=np.array([1.,2.,-3.]);s=split(d,a,1)
 assert abs(s['F']-a@d)<1e-12
 assert abs(s['Fminus']-(.2*1+.5*-3)/.7)<1e-12


def test_reserve_and_undefined_rank_remain_explicit():
 np.testing.assert_array_equal(share([-1.,0.],1e-10),[0.,0.,1.])
 assert rho([0,0,0],[1,2,3]) is None


def test_empty_ratio_keeps_missing_denominator():
 from experiments.contribution.nonzero_targets.analyze import ratio_ci
 r=ratio_ci([],[])
 assert r['ratio'] is None and r['denominator']==0 and r['zero_denominator_draws']==200000
