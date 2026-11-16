import copy
import unittest
import numpy as np
from host_server import HostServer
from state_engine import run,relocate

class Tests(unittest.TestCase):
    def server(self):
        s=HostServer(0.,2,constraint_mode='box_constrained')
        s.I_list=np.array([[.8,.1],[-.6,.7],[.3,-.7]])
        s.alphas=[.2,.3,.5]
        return s
    def test_plain_matches_original_and_resume(self):
        s=self.server();fn=lambda x:float(np.sum((np.asarray(x)-[.2,.1])**2))
        a=s._solve_subspace_custom(s.I_list,fn,num_iterations=12,direction_weights=s.alphas,initial_solution=np.array([.4,.4]))
        b,tr=run(s,fn,12,initial=[.4,.4])
        np.testing.assert_array_equal(a['solution'],b['solution']);self.assertEqual(a['history'],b['history'])
        resumed,rt=run(s,fn,10,resume=tr.post[1])
        np.testing.assert_array_equal(resumed['solution'],a['solution']);self.assertEqual(resumed['history'],a['history'])
        self.assertEqual(rt.pre[0]['eta'],tr.pre[2]['eta'])
    def test_terminal_constant_loss_stays_terminal(self):
        s=self.server();r,tr=run(s,lambda x:1.,5,initial=[.1,.1])
        self.assertTrue(tr.final['halted']);self.assertEqual(len(tr.post),0)
        again,rt=run(s,lambda x:1.,5,resume=tr.final)
        self.assertEqual(len(rt.pre),0);np.testing.assert_array_equal(again['solution'],r['solution'])
    def test_failed_secant_switch_and_dynamic_stop(self):
        s=self.server();fn=lambda x:float(np.linalg.norm(np.asarray(x)))
        r,tr=run(s,fn,8,initial=[0.,0.])
        self.assertEqual(tr.post[0]['current_method'],'dynamic');self.assertEqual(tr.post[0]['eta'],.5)
        self.assertTrue(tr.final['halted'])
    def test_relocate_preserves_controller_not_original_best(self):
        s=self.server();r,tr=run(s,lambda x:float(np.sum(np.asarray(x)**2)),1,initial=[.4,.4])
        state=tr.post[0];before=copy.deepcopy(state);a=relocate(state,[.3,.3],.18)
        self.assertEqual(a['eta'],state['eta']);self.assertEqual(a['current_method'],state['current_method'])
        self.assertEqual(a['global_best_loss'],.18);np.testing.assert_array_equal(a['global_best_S'],[.3,.3])
        np.testing.assert_array_equal(state['S_current'],before['S_current']);self.assertEqual(a['history'][-1],.18)

if __name__=='__main__':unittest.main()
