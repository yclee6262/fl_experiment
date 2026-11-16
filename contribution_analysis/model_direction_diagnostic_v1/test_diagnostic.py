import unittest
from .diagnostic import canon,encoded,obs,differences,dedup,u_value,geometry,sign
class DiagnosticTests(unittest.TestCase):
 def test_float_exact(self):
  self.assertNotEqual(encoded(1.),encoded(1));self.assertNotEqual(encoded(0.),encoded(-0.))
 def test_dictionary_order(self):self.assertEqual(encoded({'b':2,'a':1}),encoded({'a':1,'b':2}))
 def test_full_diff(self):self.assertEqual(len(differences({'a':[1,2]},{'a':[3,4]})),2)
 def test_sign(self):self.assertEqual([sign(x) for x in [-1,-1e-14,1] ],[-1,0,1])
 def test_distance_preserved(self):
  a=dict(point=[1.,0.],distance=1.,h=1.,unit=[1.,0.]);b=dict(point=[2.,0.],distance=2.,h=2.,unit=[1.,0.])
  self.assertNotEqual(geometry(a),geometry(b));self.assertEqual(len(dedup([a,b],[[1,1],[2,2]])[1]),2)
 def test_duplicate_invariance(self):
  a=dict(point=[1.],distance=1.,h=1.,unit=[1.]);b=dict(point=[-1.],distance=1.,h=1.,unit=[-1.]);g=[[0,2],[1,5],[3,7]]
  expected,_=dedup([a,b],g)
  for n in [1,3,9]:self.assertEqual(expected,dedup([a]*n+[b],[[r[0]]*n+[r[1]] for r in g])[0])
 def test_same_geometry_inconsistent_g_rejected(self):
  a=dict(point=[1.],distance=1.,h=1.,unit=[1.])
  with self.assertRaises(ValueError):dedup([a,a],[[1,2],[3,4]])
 def test_equal_g_different_geometry_retained(self):
  a=dict(point=[1.],distance=1.,h=1.,unit=[1.]);b=dict(point=[2.],distance=2.,h=2.,unit=[1.])
  self.assertEqual(len(dedup([a,b],[[1,1],[2,2]])[1]),2)
 def test_empty(self):self.assertEqual(dedup([], [[],[]]),(None,[]))
 def test_nonfinite(self):
  a=dict(point=[1.],distance=1.,h=1.,unit=[1.])
  with self.assertRaises(ValueError):dedup([a],[[float('nan')],[2.]])
  a['point']=[float('inf')]
  with self.assertRaises(ValueError):dedup([a],[[1.],[2.]])
 def test_tangent_anchor_distance_ignored(self):
  a=dict(point=[.0001],distance=1.,h=.0001,unit=[1.]);b=dict(a,distance=2.)
  self.assertEqual(len(dedup([a,b],[[1,1],[2,2]])[1]),1)
if __name__=='__main__':unittest.main()
