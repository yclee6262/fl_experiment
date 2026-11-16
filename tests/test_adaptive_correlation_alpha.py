import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from adaptive_correlation_alpha import (correlation_weights, fit_matrix, initial_alpha,
                                      choose_lambda, PRIMARY)
from initial_alpha_methods import initial_alpha as legacy

class Agent:
    def __init__(self, ident, column):self.agent_id=ident;self.column=column;self.calls=0
    def api_predict(self, x):self.calls+=1;return self.column.copy()

class Tests(unittest.TestCase):
    def setUp(self):
        self.y=np.linspace(-1,1,20)
        self.p=self.y[:,None]+np.random.default_rng(7).normal(size=(20,3))*.08
        self.x=np.zeros((20,2));self.mse=((self.p-self.y[:,None])**2).mean(axis=0)
        self.agents=[Agent(i,self.p[:,i]) for i in range(3)];self.proposals=[np.zeros(2)]*3
    def test_fixed_exact_legacy_and_diagonal_B(self):
        expected,_=legacy('corr_precision_0p5',self.agents,self.mse,self.x,self.y,self.proposals,0)
        observed,_=correlation_weights(self.p,self.y,adaptive=False)
        np.testing.assert_array_equal(expected,observed)
        w,_=fit_matrix(self.p-self.y[:,None],1)
        b,_=legacy('B_stage0',self.agents,self.mse,self.x,self.y,self.proposals,0)
        np.testing.assert_array_equal(w,b)
    def test_train_only_matrices_and_heldout_loss(self):
        w,a=correlation_weights(self.p,self.y,seed=123)
        self.assertEqual(len(a['fold_audits']),25)
        counts=np.zeros(20,int)
        for f in a['fold_audits']:
            tr=f['train_indices'];ho=f['heldout_indices']
            self.assertEqual(len(tr),16);self.assertEqual(len(ho),4);self.assertFalse(set(tr)&set(ho))
            r=self.p[tr]-self.y[tr,None]
            np.testing.assert_array_equal(f['second_moment'],r.T@r/16)
            self.assertAlmostEqual(f['heldout_mse'],np.mean((self.p[ho]@f['weight']-self.y[ho])**2))
            if f['lambda']==0:counts[ho]+=1
        np.testing.assert_array_equal(counts,np.ones(20))
        self.assertAlmostEqual(w.sum(),1);self.assertGreaterEqual(w.min(),0)
        self.assertEqual(a['selected_lambda'],choose_lambda(a['candidate_cv_mse']))
    def test_target_and_cache_invariance(self):
        kw=dict(cached_blind_predictions=self.p,bootstrap_seed=19)
        a,_=initial_alpha(PRIMARY,self.agents,self.mse,self.x,self.y,self.proposals,0,**kw)
        b,_=initial_alpha(PRIMARY,self.agents,self.mse,self.x,self.y,self.proposals,999,**kw)
        np.testing.assert_array_equal(a,b);self.assertEqual(sum(a.calls for a in self.agents),0)
        fresh,_=initial_alpha(PRIMARY,self.agents,self.mse,self.x,self.y,self.proposals,0,bootstrap_seed=19)
        np.testing.assert_array_equal(a,fresh);self.assertEqual(sum(a.calls for a in self.agents),3)
    def test_nonfinite_shape_and_mse_rejected(self):
        for p in [self.p[:-1],self.p*np.nan,self.p*np.inf]:
            with self.assertRaises(ValueError):correlation_weights(p,self.y)
        with self.assertRaises(ValueError):initial_alpha(PRIMARY,self.agents,self.mse+1,self.x,self.y,self.proposals,0,cached_blind_predictions=self.p)
    def test_bias_retained_rank_deficiency_and_single_model(self):
        for p in [np.tile(self.y[:,None]+.1,(1,3)),self.y[:,None]+.2,np.zeros((20,3))]:
            w,a=correlation_weights(p,self.y)
            self.assertTrue(np.isfinite(w).all());self.assertAlmostEqual(w.sum(),1)
        r=np.full((16,3),.2)
        _,a=fit_matrix(r,.5);np.testing.assert_allclose(a['second_moment'],.04,atol=1e-14)
    def test_ties_prefer_diagonal_and_bad_candidates(self):
        self.assertEqual(choose_lambda({'0':1.,'0.5':1+5e-13,'1':1.}),1.)
        self.assertEqual(choose_lambda({'0':None,'0.25':float('nan'),'1':.8}),1.)
        self.assertIsNone(choose_lambda({'0':None}))
    def test_fold_failure_excludes_candidate_and_final_fallback(self):
        original=fit_matrix
        def failure(r,lam):
            if lam==.25:return None,{'success':False,'lambda':lam,'message':'forced failure'}
            return original(r,lam)
        with patch('adaptive_correlation_alpha.fit_matrix',side_effect=failure):
            _,a=correlation_weights(self.p,self.y)
            self.assertIsNone(a['candidate_cv_mse']['0.25']);self.assertNotEqual(a['selected_lambda'],.25)
        def final_failure(r,lam):
            if len(r)==20 and lam!=1:return None,{'success':False,'lambda':lam,'message':'forced final failure'}
            return original(r,lam)
        with patch('adaptive_correlation_alpha.fit_matrix',side_effect=final_failure),patch('adaptive_correlation_alpha.choose_lambda',return_value=.5):
            w,a=correlation_weights(self.p,self.y)
            self.assertEqual(a['fallback_reason'],'final_matrix_solver_failure_use_B')
            np.testing.assert_array_equal(w,original(self.p-self.y[:,None],1)[0])
    def test_optimizer_failures_and_infeasible_results(self):
        for result in [SimpleNamespace(success=False,x=np.ones(3)/3,message='failure'),
                       SimpleNamespace(success=True,x=np.array([np.nan,0,1]),message='nan'),
                       SimpleNamespace(success=True,x=np.array([-.1,.1,1]),message='invalid')]:
            with patch('adaptive_correlation_alpha.minimize',return_value=result):
                w,a=fit_matrix(self.p-self.y[:,None],.5)
                self.assertIsNone(w);self.assertFalse(a['success'])

if __name__=='__main__':unittest.main()
