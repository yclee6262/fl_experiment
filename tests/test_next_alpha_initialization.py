import unittest
import numpy as np
from next_alpha_initialization import next_weights,combination
class NextTests(unittest.TestCase):
    def test_complementary(self):
        y=np.linspace(-1,1,20);p=np.column_stack([y+1,y-1,y+2])
        w,ok=combination(p,y);self.assertTrue(ok);self.assertLess(np.mean((p@w-y)**2),1e-10)
    def test_determinism_and_simplex(self):
        y=np.linspace(-1,1,20);p=np.column_stack([y+.1,y-.2,y+.3])
        for method in ['bootstrap_oob_ensemble_shrink','bootstrap_conservative_quality']:
            w,a=next_weights(method,p,y,30,8);w2,b=next_weights(method,p,y,30,8)
            np.testing.assert_array_equal(w,w2);self.assertTrue((w>=0).all());self.assertAlmostEqual(w.sum(),1)
    def test_conservative_definition(self):
        y=np.zeros(20);p=np.column_stack([np.ones(20),np.ones(20)*2])
        w,a=next_weights('bootstrap_conservative_quality',p,y,30)
        np.testing.assert_allclose(a['risk_mse'],[1,4]);self.assertGreater(w[0],w[1])
    def test_invalid_and_single(self):
        with self.assertRaises(ValueError):next_weights('bootstrap_conservative_quality',[[np.nan]],[0])
        w,a=next_weights('bootstrap_oob_ensemble_shrink',np.zeros((20,1)),np.zeros(20),20)
        np.testing.assert_array_equal(w,[1.])
    def test_identical_models(self):
        w,a=next_weights('bootstrap_oob_ensemble_shrink',np.ones((20,3)),np.zeros(20),20)
        np.testing.assert_allclose(w,[1/3]*3)
if __name__=='__main__':unittest.main()
