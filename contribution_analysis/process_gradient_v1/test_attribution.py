import unittest
import numpy as np
from attribution import attribute


class LinearAPI:
    def __init__(self, a, b): self.a=np.array(a); self.b=b; self.calls=0; self.samples=0
    def api_predict(self, x):
        self.calls+=1; self.samples+=len(x)
        return np.asarray(x)@self.a+self.b


class Tests(unittest.TestCase):
    def test_linear_work_and_opposing_models(self):
        agents=[LinearAPI([1.,0.],3.),LinearAPI([-1.,0.],3.)]
        trace=dict(states=[[0.,0.],[.1,0.],[.3,0.]], responses=[[3.,3.],[3.1,2.9],[3.3,2.7]])
        r=attribute(agents,trace,[.5,.5])
        np.testing.assert_allclose(r['gradient_work'],[-.15,.15],atol=1e-10)
        np.testing.assert_allclose(r['endpoint_work'],r['gradient_work'],atol=1e-10)
        self.assertEqual((r['calls'],r['samples']),(2,16))
        self.assertEqual(sum(a.samples for a in agents),16)

    def test_boundary_uses_actual_one_sided_width(self):
        a=LinearAPI([2.],3.)
        r=attribute([a],dict(states=[[1.],[.9]],responses=[[5.],[4.8]]),[1.])
        np.testing.assert_allclose(r['gradient_work'],[.2],atol=1e-10)
        self.assertLessEqual(r['fd_points'].max(),1.)

    def test_no_movement_does_not_query(self):
        a=LinearAPI([1.],3.)
        r=attribute([a],dict(states=[[0.]],responses=[[3.]]),[1.])
        np.testing.assert_array_equal(r['gradient_work'],[0.])
        self.assertEqual(a.calls,0)

    def test_endpoint_telescopes_but_gradient_work_can_fail_at_kink(self):
        a=LinearAPI([1.],0.)
        r=attribute([a],dict(states=[[-.1],[.1]],responses=[[-.1],[.1]]),[1.])
        self.assertAlmostEqual(r['endpoint_work'][0],0.)
        self.assertAlmostEqual(r['gradient_work'][0],.2)

    def test_invalid_and_missing_data_fail(self):
        for trace,alpha in [(dict(states=[[0.]],responses=[]),[1.]),
                            (dict(states=[[0.]],responses=[[1.]]),[.5]),
                            (dict(states=[[float('nan')]],responses=[[1.]]),[1.])]:
            with self.assertRaises(ValueError): attribute([LinearAPI([1.],1.)],trace,alpha)

    def test_target_and_weight_scaling(self):
        a=LinearAPI([1.],0.)
        r=attribute([a],dict(states=[[0.],[.1]],responses=[[0.],[.1]]),[1.],target=1.)
        np.testing.assert_allclose(r['gradient_work'],[.1],atol=1e-10)


if __name__=='__main__': unittest.main()
