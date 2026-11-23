import numpy as np
import pytest
from experiments.contribution.role_statistics.analyze import rank_metrics,ratio_interval,allocate,interval,unique


def test_constant_rank_null_but_sign_and_ties_retained():
    r=rank_metrics([0.,0.,0.],[1.,2.,3.])
    assert r['spearman'] is None and r['top1'] is None
    assert r['sign_agreement']==0 and r['x_zero']==3 and r['x_tied_pairs']==3


def test_cluster_ratio_uses_resampled_teams_and_zero_denominators():
    # Explicit resample rows: all first, one each, all second.
    w=np.array([[2,0],[1,1],[0,2]])
    r=ratio_interval([0,2],[0,4],w)
    assert r['ratio']==.5 and r['zero_denominator_draws']==1
    assert r['positive_denominator_teams']==1 and r['ci95_low']==.5 and r['ci95_high']==.5


def test_payment_reserve_and_integer_ties():
    p,r,s=allocate([-1,0],1e-10,500,[2,1])
    assert p==[0,0] and r==500 and s[-1]==1
    p,r,s=allocate([1,1,1],1e-10,500,[3,2,1])
    assert p==[166,167,167] and sum(p)+r==500
    with pytest.raises(ValueError):unique([{'a':1},{'a':1}],['a'])
