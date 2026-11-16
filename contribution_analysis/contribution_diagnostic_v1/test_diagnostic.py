import unittest
import numpy as np
from host_server import HostServer
from core import anchor_only,fixed_evaluate
from diagnostic_stats import pair,boot

class Agent:
    def __init__(self,i,bias):self.agent_id=i;self.bias=bias;self.calls=0
    def api_predict(self,x):self.calls+=1;return np.array([(float(np.asarray(x)[0])-.2)**2+self.bias])

def make():
    server=HostServer(0.,1,constraint_mode='box_constrained');server.trusted_agents=[Agent(i,b) for i,b in enumerate([1.,2.,9.])]
    server.alphas=[1/3]*3;server.I_list=np.array([[-.8],[.3],[.9]])
    case=dict(ids=[0,1,2],alphas=[1/3]*3,candidates=server.I_list.tolist(),fixed_start=[.6])
    return server,case

class Tests(unittest.TestCase):
    def test_fixed_panel_includes_removed_model(self):
        s,_=make();r=fixed_evaluate(s.trusted_agents,s.alphas,np.array([.2]))
        self.assertAlmostEqual(r['loss'],4.);self.assertEqual(r['calls'],3);self.assertNotAlmostEqual(r['loss'],1.5)
    def test_only_anchor_removed_models_alpha_start_preserved(self):
        s,c=make();r=anchor_only(s,c,1)
        self.assertEqual(r['scoring_model_ids'],[0,1,2]);self.assertEqual(r['remaining_anchor_ids'],[0,2])
        np.testing.assert_array_equal(r['scoring_alphas'],[1/3]*3);np.testing.assert_array_equal(r['fixed_start'],[.6])
        self.assertEqual(r['result']['chosen_engine'],'custom');self.assertEqual(r['counter']['calls'],sum(a.calls for a in s.trusted_agents))
    def test_original_solver_matches_direct_candidate_only_call(self):
        s,c=make();r=anchor_only(s,c,1);s2,_=make()
        expected=s2.optimize_candidate_subspace(candidates=np.delete(s2.I_list,1,axis=0),strategy='custom',mode='optimization',custom_iterations=30,initial_solution=np.array([.6]))
        np.testing.assert_array_equal(r['result']['solution'],expected['solution']);self.assertEqual(r['result']['history'],expected['history'])
    def test_nonuniform_not_silently_misrepresented(self):
        s,c=make();s.alphas=[.2,.3,.5]
        with self.assertRaises(ValueError):anchor_only(s,c,0)
    def test_near_flat_not_fabricated_as_zero(self):
        r=pair([.1,.3,.2],[1e-6,3e-6,2e-6],'A','anchor_L');self.assertIsNone(r['spearman']);self.assertAlmostEqual(r['raw_spearman'],1.)
    def test_exact_tie_and_undefined_bootstrap(self):
        self.assertIsNone(pair([1,1,1],[.1,.2,.3],'A','full_E')['spearman'])
        r=boot([None,None]);self.assertIsNone(r['mean']);self.assertEqual(r['undefined_bootstraps'],10000)

if __name__=='__main__':unittest.main()
