import numpy as np
from audit_peer_support_completion import ordinary_interval,interval_matches

def test_empty_interval_is_not_zero_success():
 assert ordinary_interval([],{'batch_ci_rng':1,'batch_ci_replicates':100})==[None,None]
 assert interval_matches([None,None],[None,None])
 assert not interval_matches([0,0],[None,None])
def test_interval_nonfinite_rejected():
 assert not interval_matches([float('nan'),0],[-1,0])
 assert not interval_matches(None,[-1,0])
def test_seed_unit_reference_reproduces_interval():
 values=[-.2,-.1,.4];reg={'batch_ci_rng':123,'batch_ci_replicates':1000}
 rng=np.random.default_rng(123);idx=rng.integers(0,3,size=(1000,3));expected=np.quantile(np.array(values)[idx].mean(axis=1),[.025,.975])
 assert interval_matches(ordinary_interval(values,reg),expected)
