import unittest
import numpy as np
from kernel import select,quality,bank
class Tests(unittest.TestCase):
 def test_remove_model_and_candidate(self):
  p=np.array([[0,0],[1,0],[0,1]],float);l=np.array([[0,9,9],[9,0,0],[2,2,2]],float)
  j,x,w=select(p,l,p,[1/3]*3,0);self.assertEqual(j,1);np.testing.assert_allclose(w,[0,.5,.5])
 def test_excluded_anchor_never_selected(self):
  p=np.array([[0,0],[1,0],[0,1]],float);l=np.array([[0,0,0],[2,2,2],[3,3,3]],float)
  self.assertEqual(select(p,l,p,[1/3]*3,0)[0],1)
 def test_tie_deterministic(self):
  p=np.array([[0,0],[1,0],[0,1]],float);l=np.ones((3,3));self.assertEqual(select(p,l,p,[1/3]*3,2)[0],0)
 def test_signed_quality(self):
  q=quality([-1,0,2]);self.assertEqual(q['median_abs'],1);self.assertEqual(q['abs_median_signed'],0);self.assertAlmostEqual(q['abs_mean_signed'],1/3)
 def test_dedup(self):
  r=dict(iterations=[dict(x=[0,0],current_losses=[1,2],directions=[dict(point=[0,0],reply_losses=[1,2])],trials=[])])
  self.assertEqual(len(bank(r)[0]),1)
if __name__=='__main__':unittest.main()
