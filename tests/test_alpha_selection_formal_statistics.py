import unittest
import numpy as np
from alpha_selection_formal_statistics import intervals,stable_advantage,power_screen
from research_analysis import audit_coverage

class FormalStats(unittest.TestCase):
 def test_bonferroni_not_narrower(self):
  a=np.random.default_rng(4).normal(size=(30,11));r=intervals(a,4000)
  for ordinary,corrected in zip(r['ordinary95'],r['simultaneous']):
   self.assertLessEqual(corrected[0],ordinary[0]);self.assertGreaterEqual(corrected[1],ordinary[1])
 def test_shared_resampling_identical_columns(self):
  a=np.arange(9)[:,None];r=intervals(np.repeat(a,2,axis=1),1000)
  self.assertEqual(r['ordinary95'][0],r['ordinary95'][1])
 def test_failures_and_batch_consistency_gate(self):
  args=([-2,-1],[-1,-1,-1],[[-2,-1],[-2,-1],[-2,.1]],True,True)
  self.assertTrue(stable_advantage(*args))
  self.assertFalse(stable_advantage(*args[:-2],False,True))
  self.assertFalse(stable_advantage(*args[:-1],False))
  self.assertFalse(stable_advantage([-2,-1],[-1,1,-1],args[2],True,True))
 def test_positive_difference_not_advantage(self):
  self.assertFalse(stable_advantage([1,2],[-1]*3,[[-2,-1]]*3,True,True))
 def test_qualification_rows_are_not_missing(self):
  r=[{'scenario':'a','seed':1,'method':'C','status':'qualification_failure'}]
  self.assertTrue(audit_coverage(r,[('a',1,'C')],['scenario','seed','method'])['complete'])
  self.assertFalse(audit_coverage(r+r,[('a',1,'C')],['scenario','seed','method'])['complete'])
 def test_empty_and_nonfinite_rejected(self):
  for a in [np.empty((0,11)),np.full((3,11),np.nan)]:
   with self.assertRaises(ValueError):intervals(a,100)
 def test_power_reproducible_and_bounded(self):
  a=np.random.default_rng(2).normal(0,.01,(30,2));x=power_screen(a,(30,),simulations=30)
  self.assertEqual(x,power_screen(a,(30,),simulations=30))
  self.assertTrue(all(0<=p<=1 for p in x[0]['approx_stable_power_by_contrast']))
if __name__=='__main__':unittest.main()
