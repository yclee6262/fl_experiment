import numpy as np
import pytest
from experiments.alpha.analysis.analyze_revised_initializers import indexed, seed_matrix, agreement, LABELS, SCENARIOS, bootstrap


def row(s,seed,m,value):
    return dict(scenario=s,seed=seed,method=m,status='valid',X=[0.,0.],pooled_error=value,truth_error=value)


def test_seed_aggregation_and_alignment():
    rr=[row(s,k,m,float(k+j)) for k in [8,3] for j,s in enumerate(SCENARIOS) for m in LABELS]
    ix=indexed(list(reversed(rr)))
    keys,mat=seed_matrix(ix,[3,8],SCENARIOS,'truth_error')
    assert keys==[3,8]
    np.testing.assert_array_equal(mat[:,0],[4.,9.])
    bad=ix[SCENARIOS[1],3,next(iter(LABELS))];bad['status']='solver_failure'
    keys,mat=seed_matrix(ix,[3,8],SCENARIOS,'truth_error')
    assert keys==[8] and mat.shape==(1,10)
    with pytest.raises(ValueError): indexed(rr+[rr[0]])


def test_direction_ties_not_counted_as_conflicts():
    got=agreement([1,-1,0,0,1],[1,1,0,1,0])
    assert got['both_non_tie']==2 and got['agree_non_tie']==1
    assert got['conflict_non_tie']==1 and got['both_tie']==1 and got['one_tie']==2


def test_shared_bootstrap_keeps_pair_identity():
    mat=np.array([[1.,3.],[2.,4.],[8.,10.]])
    samples=bootstrap(mat,replicates=200)
    np.testing.assert_allclose(samples[:,1]-samples[:,0],2.)
