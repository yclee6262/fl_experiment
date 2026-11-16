import numpy as np
import pytest
from conservative_shrinkage import crossfit_weights,risk_weights

def test_disjoint_balanced_validation_and_reproduction():
 y=np.linspace(-1,1,20);p=np.column_stack([y+.1,y-.2,y+.3]);w,a=crossfit_weights(p,y,12,100,2)
 w2,b=crossfit_weights(p,y,12,100,2);np.testing.assert_array_equal(w,w2)
 assert np.all(w>=0) and np.isclose(w.sum(),1)
 counts=np.zeros(20,int)
 for f in a['fold_audits']:
  train,hold=f['train_indices'],f['heldout_indices'];assert len(train)==16 and len(hold)==4 and set(train).isdisjoint(hold)
  counts[hold]+=1
  q,r=risk_weights(p[train],y[train],100,np.random.SeedSequence([12,20261010,f['repeat'],f['fold'],1]))
  np.testing.assert_array_equal(q,f['risk_weight'])
  changed=y.copy();changed[hold]+=1000
  q2,_=risk_weights(p[train],changed[train],100,np.random.SeedSequence([12,20261010,f['repeat'],f['fold'],1]));np.testing.assert_array_equal(q,q2)
  assert f['heldout_losses']['selected']<=min(f['heldout_losses']['quality'],f['heldout_losses']['uniform'])+1e-12
 np.testing.assert_array_equal(counts,np.full(20,2))

def test_no_shrink_ablation_matches_same_fold_risk():
 p=np.random.default_rng(1).normal(size=(20,3));y=np.zeros(20)
 w,a=crossfit_weights(p,y,3,100,2,False);_,b=crossfit_weights(p,y,3,100,2,True)
 for x,z in zip(a['fold_audits'],b['fold_audits']):np.testing.assert_array_equal(x['risk_weight'],z['risk_weight']);assert x['lambda']==0
 np.testing.assert_allclose(w,np.mean([x['risk_weight'] for x in a['fold_audits']],axis=0))

def test_identical_and_single_models():
 for n in [1,3]:
  w,a=crossfit_weights(np.ones((20,n)),np.zeros(20),1,30,1)
  np.testing.assert_allclose(w,np.full(n,1/n));assert a['lambda_mean']==1

def test_input_errors():
 with pytest.raises(ValueError):crossfit_weights(np.zeros((19,3)),np.zeros(19))
 with pytest.raises(ValueError):crossfit_weights(np.full((20,3),np.nan),np.zeros(20))
 with pytest.raises(ValueError):crossfit_weights(np.zeros((20,3)),np.zeros(20),repeats=0)

def test_quantile_definition():
 w,r=risk_weights(np.column_stack([np.ones(16),np.ones(16)*2]),np.zeros(16),30,2)
 np.testing.assert_array_equal(r,[1,4]);assert w[0]>w[1]
