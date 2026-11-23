import numpy as np
import pytest
from experiments.baselines.paired_benchmark.analyze import indexed,paired_deltas,interval,is_valid


def row(seed,method,value,status='valid'):
    return dict(seed=seed,dimension=2,rho=.3,method=method,panel_budget=400,X=[0.,0.],truth_error=value,status=status)


def test_seed_pairing_preserves_missing_failures():
    data=[row(9,'best_of',2),row(4,'best_of',3),row(9,'nelder_mead',1),row(4,'nelder_mead',9,'no_valid_endpoint')]
    ix=indexed(list(reversed(data)))
    seeds,d=paired_deltas(ix,[4,9],2,.3,('best_of',400),('nelder_mead',400))
    assert seeds==[9];np.testing.assert_array_equal(d,[1.])
    assert not is_valid(row(3,'best_of',0,'no_coalition'))
    with pytest.raises(ValueError):indexed(data+[data[0]])


def test_constant_paired_delta_and_empty_intervals():
    got=interval([2.,2.,2.],reps=200,family=30)
    assert got['mean']==got['ci95_low']==got['family_high']==2
    none=interval([],reps=200)
    assert none['n']==0 and none['mean'] is None and none['ci95_low'] is None


def test_smoke_cannot_be_used_as_formal_evidence(tmp_path):
    from experiments.baselines.paired_benchmark.analyze import validate_source
    with pytest.raises(ValueError,match='Smoke'):
        validate_source(tmp_path,{'smoke':True,'status':'complete'})
