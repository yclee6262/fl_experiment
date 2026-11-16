import numpy as np
from types import SimpleNamespace
from repeated_alpha_candidates import METHODS,CANDIDATES,labelled_inputs,initial_alpha
from analyze_repeated_alpha_confirmation import contrasts,stable,valid

def server():
    return SimpleNamespace(I_list=[np.array([.1,.2]),np.array([-.1,.3]),np.array([.2,-.2])],
        test_X=np.random.default_rng(3).uniform(-1,1,(20,2)),test_y=np.zeros(20))

def test_same_additional_labels_for_controls():
    s=server();x9,y9,n=labelled_inputs('proposal_local_9',s,7);x29,y29,k=labelled_inputs('B_augmented29',s,7)
    np.testing.assert_array_equal(x9,x29[20:]);np.testing.assert_array_equal(y9,y29[20:]);assert n==k==9
    x100,y100,n=labelled_inputs('local_bias_risk_100',s,7);xb,yb,k=labelled_inputs('B_augmented100',s,7)
    np.testing.assert_array_equal(x100,xb);np.testing.assert_array_equal(y100,yb);assert n==k==80

def test_all_original_candidates_kept_and_family_fixed():
    assert len(METHODS)==29 and len(CANDIDATES)==27
    assert len(contrasts())==53 and len(set(contrasts()))==53

def test_matched_precision_control_uses_all_rows():
    agents=[SimpleNamespace(agent_id=i) for i in [1,2,20]];y=np.zeros(29);p=np.tile([.1,.2,.3],(29,1))
    w,a=initial_alpha('B_augmented29',agents,[9,9,9],np.zeros((29,2)),y,np.zeros((3,2)),0,cached_blind_predictions=p)
    score=1/(np.array([.01,.04,.09])+1e-5);np.testing.assert_allclose(w,score/score.sum());assert a['label_count']==29

def test_pooled_win_needs_replication_and_failures_protected():
    good=[{'mean_delta':-.01,'CI95':[-.02,-.001]}]*3
    assert stable([-.02,-.001],good,True,True)
    assert not stable([-.02,.001],good,True,True)
    assert not stable([-.02,-.001],good,False,True)
    assert not stable([-.02,-.001],good,True,False)
    bad=good[:2]+[{'mean_delta':.01,'CI95':[-.01,.03]}]
    assert not stable([-.02,-.001],bad,True,True)

def test_invalid_delivery_not_counted_as_quality():
    assert not valid({'status':'valid','X':[float('nan'),0],'pooled_error':0})
    assert not valid({'status':'solver_failure','X':None,'pooled_error':None})
    assert not valid({'status':'valid','X':[1.1,0],'pooled_error':0})
