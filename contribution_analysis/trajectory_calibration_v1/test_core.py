import unittest
import numpy as np
from host_server import HostServer
from recorder import solve,features
import rules


class Agent:
    def __init__(self,i):self.agent_id=i
    def api_predict(self,x):
        x=np.asarray(x,dtype=np.float32)
        return np.atleast_1d(np.sum(x,axis=-1)+.01*self.agent_id)


class CoreTests(unittest.TestCase):
    def server(self):
        s=HostServer(0,2,constraint_mode='box_constrained')
        s.trusted_agents=[Agent(i) for i in [1,2,3]]
        s.I_list=np.array([[.2,.1],[-.3,.4],[.1,-.2]])
        s.alphas=[1/3]*3
        return s

    def test_both_history_modes_no_extra_query_or_changed_X(self):
        for mode in ['objective','diagnostic']:
            a=self.server();b=self.server()
            control=solve(a,False,mode);record=solve(b,True,mode)
            self.assertEqual(control['counter'],record['counter'])
            np.testing.assert_array_equal(control['result']['solution'],record['result']['solution'])
            self.assertEqual(control['result']['loss'],record['result']['loss'])
            self.assertEqual(control['result']['history'],record['result']['history'])
            self.assertTrue(np.isfinite(features(record['trace'],b.I_list,b.alphas)).all())

    def test_permutation_equivariance_of_features(self):
        s=self.server();r=solve(s,True)
        z=features(r['trace'],s.I_list,s.alphas);p=[2,0,1]
        trace=dict(r['trace']);trace['responses']=trace['responses'][:,p]
        zp=features(trace,s.I_list[p],np.array(s.alphas)[p])
        np.testing.assert_allclose(zp,z[p],atol=1e-12)

    def test_missing_replies_not_zero_filled(self):
        with self.assertRaises(ValueError):features(dict(states=[[0,0]],responses=[[0,0]]),self.server().I_list,[1/3]*3)

    def test_saved_initial_solution_is_not_recomputed(self):
        s=self.server();saved=np.array([.010000000149,.020000000298])
        r=solve(s,True,initial_solution=saved)
        np.testing.assert_array_equal(r['fixed_start'],saved)
        np.testing.assert_allclose(r['trace']['states'][0],saved,atol=1e-12)

    def test_query_limit_does_not_leak_wrappers(self):
        s=self.server();original=[a.api_predict for a in s.trusted_agents]
        with self.assertRaisesRegex(RuntimeError,'model_call_budget'):
            solve(s,True,max_calls=1)
        self.assertEqual([a.api_predict for a in s.trusted_agents],original)

    def test_tree_leaves_require_five_groups(self):
        X=np.arange(40.)[:,None];y=(X[:,0]>19)*1.;g=np.repeat(np.arange(10),4);w=np.ones(40)
        t=rules.fit(X,y,g,w)
        def check(n):
            if 'feature' in n:check(n['left']);check(n['right'])
            else:self.assertGreaterEqual(n['n_groups'],5)
        check(t);np.testing.assert_array_equal(rules.predict(t,X),y)

    def test_constant_uses_training_data_only(self):
        model=rules.constant_fit(np.array([1.,3.,7.]),np.array([3,3,5]),np.ones(3))
        np.testing.assert_array_equal(rules.constant_predict(model,np.array([5,3])),[7.,2.])


if __name__=='__main__':unittest.main()
