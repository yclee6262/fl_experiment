import unittest
import numpy as np
from core import features,fit,predict
class Tests(unittest.TestCase):
 def test_basis(self):np.testing.assert_array_equal(features([[2,3]],2),[[1,2,3,4,6,9]])
 def test_generic_quadratic_fit(self):
  rng=np.random.default_rng(2);x=rng.uniform(-1,1,(500,2));beta=np.array([.2,2,-3,.5,.3,-.2]);y=features(x,2)@beta;s=fit(x,y,[0,0],2);self.assertAlmostEqual(predict(s,[.12,-.2]),float(features([[.12,-.2]],2)[0]@beta),places=5)
 def test_linear_fit(self):
  rng=np.random.default_rng(2);x=rng.uniform(-1,1,(500,2));y=2*x[:,0]+3*x[:,1]-.4;s=fit(x,y,[.1,.1],1);self.assertAlmostEqual(predict(s,[.1,.2]),.4,places=5)
 def test_missing_observations(self):
  with self.assertRaisesRegex(ValueError,'insufficient'):fit(np.zeros((20,2)),np.zeros(20),[0,0],2)
if __name__=='__main__':unittest.main()
