from audit_repeated_alpha_evidence import delivery_errors,evaluate_claim

def test_nonfinite_or_empty_delivery_cannot_be_accepted():
    base={'status':'valid','X':[0,0],'alpha':[.2,.3,.5],'pooled_error':.01,'truth_error':.02}
    assert delivery_errors(base)==[]
    assert delivery_errors(dict(base,pooled_error=float('nan')))
    assert delivery_errors(dict(base,pooled_error=float('inf')))
    assert delivery_errors(dict(base,X=None))
    assert delivery_errors(dict(base,alpha=[1,-.1,.1]))
    assert delivery_errors(dict(base,alpha=[0,0,0]))

def test_expected_failures_stay_failures_without_fake_quality():
    assert not delivery_errors({'status':'solver_failure','pooled_error':None,'truth_error':None})
    assert delivery_errors({'status':'qualification_failure','pooled_error':0,'truth_error':None})

def test_claim_needs_all_batches_and_expected_sample_size():
    r={'n':564,'excluded':[],'CI_simultaneous':[-.01,-.001],'method_failures':0,'baseline_failures':0,
       'batches':[{'mean_delta':-.01,'CI95':[-.02,-.001]}]*3,'stable_advantage':True}
    assert not evaluate_claim(r,564)
    assert evaluate_claim(dict(r,n=300),564)
    assert evaluate_claim(dict(r,batches=r['batches'][:2]),564)
    assert evaluate_claim(dict(r,method_failures=1),564)
    assert evaluate_claim(dict(r,CI_simultaneous=[-.01,.001]),564)

def test_partial_improvement_does_not_become_stable_claim():
    r={'n':564,'excluded':[],'CI_simultaneous':[-.01,-.001],'method_failures':0,'baseline_failures':0,
       'batches':[{'mean_delta':-.01,'CI95':[-.02,-.001]}]*2+[{'mean_delta':.001,'CI95':[-.01,.01]}],
       'stable_advantage':True}
    assert evaluate_claim(r,564)
    r['stable_advantage']=False
    assert not evaluate_claim(r,564)
