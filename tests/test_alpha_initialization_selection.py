import unittest
from unittest.mock import patch
import inspect
import numpy as np
import alpha_initialization_selection as s

class Selectors(unittest.TestCase):
 def setUp(self):
  self.p=np.random.default_rng(7).normal(size=(20,3));self.y=np.random.default_rng(8).normal(size=20)
  self.peer=np.array([[0,.3,.6],[.1,0,.8],[.2,.5,0]])
 def test_formulas(self):
  w,_=s.weights_from_predictions(self.p,self.y,self.peer,0)
  np.testing.assert_allclose(w['ensemble_shrink'],.5*w['stacked_full']+.5/3)
  np.testing.assert_allclose(w['support_quality_mix50'],.5*w['pure_support']+.5*w['B_stage0'])
  for v in w.values(): self.assertGreaterEqual(v.min(),0);self.assertAlmostEqual(v.sum(),1)
 def test_held_out_labels_do_not_fit_fold(self):
  a=s.select_c(self.p,self.y,self.peer,0);held=a['folds'][0]['held_out'];y=self.y.copy();y[held]+=100
  b=s.select_c(self.p,y,self.peer,0)
  self.assertEqual(a['folds'][0]['alpha'],b['folds'][0]['alpha'])
  self.assertNotEqual(a['scores'],b['scores'])
 def test_partition_and_determinism(self):
  a=s.select_c(self.p,self.y,self.peer,0);self.assertEqual(a,s.select_c(self.p,self.y,self.peer,0))
  self.assertEqual(sorted(k for f in a['folds'] for k in f['held_out']),list(range(20)))
  for f in a['folds']:self.assertFalse(set(f['train'])&set(f['held_out']))
 def test_single_label_rejected(self):
  with self.assertRaises(ValueError):s.select_c(self.p[:1],self.y[:1],self.peer,0)
 def test_tie_and_nonfinite(self):
  self.assertEqual(s.choose({'B_stage0':1.,'ensemble_shrink':1.+1e-13}), 'ensemble_shrink')
  self.assertIsNone(s.choose({'B_stage0':float('nan')}))
 def test_p_common_quality_and_invalid_candidate(self):
  rows={m:{'status':'valid','X':[0,0],'alpha':[1,0,0]} for m in s.METHODS}
  rows['ensemble_shrink']['X']=[2,0]
  predictions={m:[.1,.2,.3] for m in s.METHODS};predictions['B_stage0']=[0,0,0]
  a=s.select_p(rows,predictions,[.6,.3,.1],0);self.assertEqual(a['selected_method'],'B_stage0')
  self.assertNotIn('ensemble_shrink',a['scores'])
  rows['B_stage0']['alpha']=[0,0,1];self.assertEqual(a,s.select_p(rows,predictions,[.6,.3,.1],0))
 def test_p_failure_preserved(self):
  self.assertIsNone(s.select_p({}, {}, [.6,.3,.1],0)['selected_method'])
  self.assertEqual(s.select_p({}, {}, [float('nan'),0,1],0)['fallback_reason'],'invalid_quality')
 def test_external_data_not_in_interface(self):
  for fn in [s.select_c,s.select_p]:
   with self.assertRaises(TypeError):fn(pooled_scores=[1])
 def test_ensemble_failure_semantics(self):
  from types import SimpleNamespace
  with patch.object(s,'minimize',return_value=SimpleNamespace(success=False,x=np.zeros(3))):
   w,a=s.weights_from_predictions(self.p,self.y,self.peer,0)
   self.assertTrue(a['ensemble_fallback']);np.testing.assert_allclose(w['stacked_full'],w['B_stage0']);np.testing.assert_allclose(w['ensemble_shrink'],np.ones(3)/3)

if __name__=='__main__':unittest.main()
