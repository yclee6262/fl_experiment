import unittest
import numpy as np
from contribution_analysis.three_pool_t2_anchor_v1.core import decide,owner_vectors


class Box:
    def _bounded_step(self,x,d,eta):
        return max(0,min([eta]+[(1-a)/b if b>0 else (-1-a)/b for a,b in zip(x,d) if b]))


class CoreTests(unittest.TestCase):
    def test_zero_no_trial(self):
        def never(x):raise AssertionError('unexpected_query')
        r=decide(Box(),never,[0,0],[0,0],1.,.1);self.assertEqual(r['trials'],[]);self.assertIsNone(r['q']);self.assertEqual(r['loss'],1.)
    def test_nonfinite_rejected(self):
        with self.assertRaisesRegex(ValueError,'invalid_state'):decide(Box(),lambda x:1,[0,0],[float('nan'),0],1.,.1)
    def test_wrong_shape_rejected(self):
        with self.assertRaisesRegex(ValueError,'invalid_state'):decide(Box(),lambda x:1,[0,0],[1],1.,.1)
    def test_dynamic_failure_stops(self):
        r=decide(Box(),lambda x:1+x[0]**2,[0,0],[1,0],1.,.1,'dynamic');self.assertEqual(len(r['trials']),10);self.assertEqual(r['stop'],'no_improvement')
    def test_secant_failure_switches(self):
        r=decide(Box(),lambda x:1+x[0]**2,[0,0],[1,0],1.,.1);self.assertTrue(r['mode_switch']);self.assertEqual(r['eta_after'],.5)
    def test_backtracking(self):
        r=decide(Box(),lambda x:(x[0]+.01)**2,[0,0],[1,0],.0001,.1);self.assertEqual(len(r['trials']),4);self.assertTrue(r['accepted'])
    def test_boundary(self):
        r=decide(Box(),lambda x:1-x[0],[.98,0],[-1,0],.02,.1);self.assertAlmostEqual(r['x'][0],1);self.assertAlmostEqual(r['trials'][0]['actual_step'],.02)
    def test_group_owner(self):
        m=dict(anchor_owners=['same','same'],anchors=[[1,0],[0,1]],direction_weights=[.5,.5],model_weights=[1.])
        r=dict(evaluations=[dict(individual_losses=[1]),dict(individual_losses=[3]),dict(individual_losses=[5])])
        it=dict(x_before=[0,0],current_eval_id=0,g_solver=[1,2],directions=[dict(index=k,skip_reason=None,source_eval_id=k+1,actual_h=1,u=u) for k,u in enumerate([[1,0],[0,1]])])
        owners,H=owner_vectors(m,r,it);self.assertEqual(owners,['same']);np.testing.assert_allclose(H,[[1,2]])
        it['directions'].pop()
        with self.assertRaisesRegex(ValueError,'missing_direction'):owner_vectors(m,r,it)


if __name__=='__main__':unittest.main()
