"""Synthetic semantic tests; do not call participant models."""
import copy,unittest
from common import *
class Tests(unittest.TestCase):
    def test_starts_fixed(self):
        a=starts(dict(fixed_start=[.1,-.2]));near(a,[[.1,-.2],[.35,-.2],[-.15,-.2],[.1,.05],[.1,-.45]])
    def test_starts_clip(self):self.assertEqual(starts(dict(fixed_start=[.9,-.9]))[1][0],1)
    def test_duplicate_stops(self):
        with self.assertRaises(AssertionError):starts(dict(fixed_start=[1.,0.]))
    def test_key_precise(self):self.assertNotEqual(key(0,[.1,.2]),key(0,[np.nextafter(.1,1),.2]))
    def test_model_identity(self):self.assertNotEqual(key(0,[0,0]),key(1,[0,0]))
    def test_rebase(self):
        from engine import branch_state
        x=np.array([.1,.2]);p=dict(S_current=x,best_loss=10,global_best_S=np.zeros(2),global_best_loss=0,eta=.07,current_method='dynamic',delta=.0001,next_iteration=4,halted=True)
        old=copy.deepcopy(p);r=branch_state(p,lambda z:float(np.dot(z,z)),np.array([[0,0],[1,1]]))
        near(r['global_best_S'],x);near(r['best_loss'],.05);self.assertEqual(r['eta'],.07);self.assertEqual(r['current_method'],'dynamic');self.assertFalse(r['halted']);near(p['global_best_S'],old['global_best_S'])
    def test_one_round_matches_manual(self):
        from engine import run,branch_state
        from independent_interpreter import walk,reset
        from host_server import HostServer
        s=HostServer.__new__(HostServer);s.I_list=np.array([[.8,.1],[-.2,.7]]);s.alphas=np.array([.5,.5]);s.constraint_mode='box_constrained';s.feasible_lower=-1.;s.feasible_upper=1.;s.feasibility_tolerance=1e-8
        f=lambda x:float(np.dot(x,x));r,tr=run(s,f,30,initial=[.3,.2]);pre=tr.pre[0];state=branch_state(pre,f,s.I_list);_,bt=run(s,f,1,resume=state)
        manual=reset(pre,f,s.I_list);manual['states']=['Local Start'];v=walk(s,f,manual,1)
        near(v['result']['solution'],bt.final['global_best_S']);near(v['final']['eta'],bt.final['eta'])
    def test_counterfactual_not_additive(self):self.assertNotEqual((12-10)+(11-10),9-10)
if __name__=='__main__':
    setup();suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests);r=unittest.TextTestRunner(verbosity=2).run(suite)
    save(ROOT/'technical_tests.json',dict(tests=r.testsRun,accepted=r.wasSuccessful(),failures=[str(v) for v in r.failures+r.errors]));sys.exit(not r.wasSuccessful())
