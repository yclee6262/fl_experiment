import unittest
from kernel import score_peer
class Tests(unittest.TestCase):
    def test_uniform(self):self.assertEqual(score_peer([8,2,3],[2,3,4],[1,1,1],0),-1)
    def test_excluded_invariant(self):self.assertEqual(score_peer([800,2,3],[200,3,4],[1,1,1],0),-1)
    def test_nonuniform(self):self.assertAlmostEqual(score_peer([9,5,2],[3,1,1],[.2,.2,.6],0),1.75)
    def test_positive(self):self.assertEqual(score_peer([3,3],[1,1],[1,1],0),2)
    def test_negative(self):self.assertEqual(score_peer([1,1],[3,3],[1,1],1),-2)
    def test_empty(self):
        with self.assertRaises(ValueError):score_peer([1],[0],[1],0)
    def test_zero_remaining(self):
        with self.assertRaises(ValueError):score_peer([1,2],[0,0],[1,0],0)
    def test_negative_weight(self):
        with self.assertRaises(ValueError):score_peer([1,2],[0,0],[1,-1],0)
    def test_nonfinite(self):
        with self.assertRaises(ValueError):score_peer([1,float('nan')],[0,0],[1,1],0)
if __name__=='__main__':unittest.main(verbosity=2)
