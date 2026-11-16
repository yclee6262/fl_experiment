import numpy as np
import pytest
from oob_shrinkage import bootstrap_oob_quality, tune_shrinkage
from initial_alpha_methods import initial_alpha


def test_analytic_minimum_matches_dense_grid_and_endpoints():
    rng=np.random.default_rng(81)
    for _ in range(15):
        p=rng.normal(size=(7,3));y=rng.normal(size=7);q=rng.dirichlet(np.ones(3))
        lam,w,loss=tune_shrinkage(p,y,q)
        grid=np.linspace(0,1,10001)
        mixtures=(1-grid[:,None])*q+grid[:,None]/3
        minimum=np.min(np.mean((p@mixtures.T-y[:,None])**2,axis=0))
        assert loss['selected']<=minimum+1e-12
        assert loss['selected']<=min(loss['quality'],loss['uniform'])+1e-12
        assert 0<=lam<=1 and np.all(w>=0) and w.sum()==pytest.approx(1)


def test_oob_rows_are_disjoint_and_lambda_does_not_use_training_labels():
    rng=np.random.default_rng(4);p=rng.normal(size=(20,3));y=rng.normal(size=20)
    seed=77;rep=15
    a,audit=bootstrap_oob_quality(p,y,rep,seed)
    samples=np.random.default_rng(seed).integers(0,20,size=(rep,20));ws=[];ls=[];sizes=[]
    for sample in samples:
        hold=np.setdiff1d(np.arange(20),sample)
        if len(hold)<2:continue
        assert not set(hold)&set(sample)
        scores=1/(((p[sample]-y[sample,None])**2).mean(axis=0)+1e-5)
        lam,w,_=tune_shrinkage(p[hold],y[hold],scores/scores.sum())
        ws.append(w);ls.append(lam);sizes.append(len(hold))
    expected=np.mean(ws,axis=0);expected/=expected.sum()
    np.testing.assert_array_equal(a,expected)
    assert audit['lambda_values']==ls and audit['oob_sizes']==sizes


def test_flat_tie_single_agent_and_no_eligible_oob_are_explicit():
    assert tune_shrinkage(np.ones((3,2)),np.zeros(3),np.array([.9,.1]))[0]==1
    a,audit=bootstrap_oob_quality(np.ones((20,1)),np.zeros(20),10,3)
    np.testing.assert_array_equal(a,[1.])
    a,audit=bootstrap_oob_quality(np.ones((2,2)),np.zeros(2),10,3)
    np.testing.assert_array_equal(a,[.5,.5])
    assert audit['accepted_replicates']==0 and audit['fallback_reason']=='no_eligible_oob_use_uniform'


def test_cached_and_fresh_match_without_new_calls_and_rng_is_local():
    class Agent:
        def __init__(self,i):self.agent_id=i;self.calls=0
        def api_predict(self,x):self.calls+=1;return x[:,0]*self.agent_id
    agents=[Agent(1),Agent(2),Agent(3)];x=np.arange(40).reshape(20,2)/40;y=x[:,1]
    p=np.column_stack([a.api_predict(x) for a in agents]);mse=((p-y[:,None])**2).mean(axis=0)
    np.random.seed(4);state=np.random.get_state()
    fresh,_=initial_alpha('bootstrap_oob_quality_shrink',agents,mse,x,y,[None]*3,0,bootstrap_replicates=20,bootstrap_seed=11)
    calls=sum(a.calls for a in agents)
    reused,audit=initial_alpha('bootstrap_oob_quality_shrink',agents,mse,x,y,[None]*3,0,bootstrap_replicates=20,bootstrap_seed=11,cached_blind_predictions=p)
    np.testing.assert_array_equal(fresh,reused)
    assert sum(a.calls for a in agents)==calls and audit['blind_prediction_source']=='qualification_reuse'
    np.testing.assert_array_equal(state[1],np.random.get_state()[1])


@pytest.mark.parametrize('bad',[np.nan,np.inf])
def test_nonfinite_input_is_rejected(bad):
    with pytest.raises(ValueError):bootstrap_oob_quality([[bad,1],[0,2]],[0,0],10,2)


def test_endpoint_choices_use_only_held_out_outcomes():
    p=np.array([[1.,0.],[1.,0.]])
    assert tune_shrinkage(p,np.array([.9,.9]),np.array([.9,.1]))[0]==pytest.approx(0.)
    assert tune_shrinkage(p,np.array([.5,.5]),np.array([.9,.1]))[0]==pytest.approx(1.)


def test_all_failure_analysis_keeps_rows_and_undefined_ci(tmp_path,monkeypatch):
    import json
    import analyze_oob_shrinkage as module
    methods=['A_uniform','B_stage0',module.PRIMARY]
    manifest={'scenarios':['reference_clean'],'seeds':[1,2],'draws':[-1],
              'sizes':[20],'targets':[0.],'methods':methods}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    # Both A and B missing seed 2 must not become a deceptively complete pair.
    rows=[{'scenario':'reference_clean','seed':1,'draw':-1,'blind_n':20,'target':0.,
           'method':m,'status':'solver_failure','X':None,'pooled_error':None,
           'stage3_requests':7} for m in methods]
    (tmp_path/'results.jsonl').write_text('\n'.join(map(json.dumps,rows)))
    from research_analysis import audit_coverage
    expected=[('reference_clean',s,-1,20,0.,m) for s in [1,2] for m in methods]
    coverage=audit_coverage(rows,expected,module.KEYS)
    monkeypatch.setattr(module,'verify',lambda *args:{'passed':coverage['complete'],'errors':[], 'coverage':coverage})
    out=module.analyze(tmp_path)
    assert not out['verification']['passed']
    assert all(c['n_seeds']==0 and c['CI95']==(None,None) and not c['confirmed_advantage'] for c in out['comparisons'])
    assert all(c['method_failures']==2 for c in out['comparisons'])
    assert all(g['stage3_requests']['mean']==7 for g in out['summaries'])
    assert (tmp_path/'results.csv').exists() and (tmp_path/'paired_differences.csv').exists()
