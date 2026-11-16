import unittest
import numpy as np
from core import grid,Proxy,key
class Tests(unittest.TestCase):
 def test_nested(self):
  p=np.array([[0,0],[1,1]],float);small=grid(p,5);large=grid(p,9);self.assertTrue(all(key(x) in {key(y) for y in large} for x in small));self.assertLessEqual(abs(large).max(),1)
 def test_affine(self):
  p=grid(np.array([[0,0],[.5,.5]]),5);v=np.array([p[:,0]+p[:,1],2*p[:,0]-p[:,1]]).T
  for method in ['linear','rbf']:
   f=Proxy(p,v,method);np.testing.assert_allclose(f([.13,.22]),[.35,.04],atol=1e-6)
 def test_outside_recorded(self):
  p=grid(np.array([[0,0],[.2,.2]]),5);f=Proxy(p,p,'linear');self.assertTrue(np.isfinite(f([.9,.9])).all());self.assertEqual(f.outside_calls,1)
 def test_known_point_exact(self):
  p=grid(np.array([[0,0],[.2,.2]]),5);f=Proxy(p,p*3,'rbf');np.testing.assert_array_equal(f(p[3]),p[3]*3)
if __name__=='__main__':unittest.main()

class RoleTests(unittest.TestCase):
 def test_simulation_removes_both_roles(self):
  from core import simulate
  class F:
   calls=0;outside_calls=0
   def __call__(self,x):self.calls+=1;return np.array([100.,2.,4.])
  class Server:
   def _solve_subspace_custom(self,candidates,objective,**kw):
    np.testing.assert_array_equal(candidates,[[.1,0],[0,.1]])
    np.testing.assert_allclose(kw['direction_weights'],[.4,.6]);np.testing.assert_array_equal(kw['initial_solution'],[.01,.02]);self.loss=objective([0,0]);return dict(solution=np.array([0.,0.]),loss=self.loss)
  server=Server();case=dict(ids=[1,2,3],candidates=[[.9,.9],[.1,0],[0,.1]],alphas=[.5,.2,.3],fixed_start=[.01,.02]);r=simulate(server,F(),case,0)
  self.assertAlmostEqual(server.loss,3.2);self.assertEqual(r['remaining_ids'],[2,3]);self.assertEqual(r['proxy_calls'],1)
 def test_recorder_preserves_returns_and_restores_api(self):
  from core import record_normal
  class Agent:
   def api_predict(self,x):return np.array([float(np.sum(x))])
  class Server:
   trusted_agents=[Agent(),Agent()]
   def optimize_candidate_subspace(self,**kw):
    p=kw['initial_solution'];v=[a.api_predict(p)[0] for a in self.trusted_agents];return dict(solution=p,loss=np.mean(abs(np.array(v))),history=[1])
  s=Server();before=[a.api_predict for a in s.trusted_agents];r=record_normal(s,[.1,.2]);self.assertEqual(r['counter'],dict(calls=2,samples=2));self.assertEqual([a.api_predict for a in s.trusted_agents],before);self.assertEqual(len(r['query_order_sha256']),64)
